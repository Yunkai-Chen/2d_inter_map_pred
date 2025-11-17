#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, time, argparse
from pathlib import Path
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

# === 项目内导入 ===
from contrasive_learning.data.data_loader import (
    read_db_index, read_split_list, ProtPepFullTokenDataset, collate_full_tokens,
    read_gt_index, strip_suffix_key
)
from contrasive_learning.model.models import PairwiseModel, PairModelConfig, PairwiseCriterion

# === 可选：W&B 最小侵入记录（不上传模型/代码）===
try:
    import wandb
except Exception:
    wandb = None


def seed_all(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def adapt_batch_for_model(batch):
    """
    把 DataLoader 的键适配为模型期望的键，并基于 GT 构造监督标签 + pair_mask。
    - 输入: collate_full_tokens 输出的 batch
    - 输出: (model_batch, labels)
        model_batch:
            prot_emb [B,Tp_pad,D]
            pep_emb  [B,Tl_pad,D]
            prot_mask [B,Tp_pad] (真实残基)
            pep_mask  [B,Tl_pad] (真实残基)
        labels:
            labels_contact 或 labels_distance: [B,Tl_pad,Tp_pad]（浮点）
            pair_mask_override: [B,Tl_pad,Tp_pad] （True 的位置参与loss）
    """
    device = batch["protein_emb"].device

    prot_mask = batch["protein_masks"]["valid_real_residue"]  # [B,Tp_pad]
    pep_mask  = batch["peptide_masks"]["valid_real_residue"]  # [B,Tl_pad]

    model_batch = {
        "prot_emb": batch["protein_emb"],   # [B,Tp,D]
        "pep_emb":  batch["peptide_emb"],   # [B,Tl,D]
        "prot_mask": prot_mask,             # [B,Tp]
        "pep_mask":  pep_mask,              # [B,Tl]
    }

    labels = {}
    # 基础 pair_mask = 真实肽 × 真实蛋白
    pair_mask = pep_mask[:, :, None] & prot_mask[:, None, :]  # [B,Tl_pad,Tp_pad]

    # 若 batch 内含 GT（已经在 dataloader 里完成按样本的 [Tl_i, Tp_i] 裁剪和批内 padding）
    if "gt_map" in batch:
        gt_map  = batch["gt_map"].to(device)         # [B,Tl_gtmax,Tp_gtmax] (uint8/float)
        gt_mask = batch["gt_mask"].to(device).bool() # [B,Tl_gtmax,Tp_gtmax] (有效处 True)

        B, Tl_pad = pep_mask.shape
        _, Tp_pad = prot_mask.shape

        # 回写到 padded 网格
        gt_full      = torch.zeros((B, Tl_pad, Tp_pad), dtype=torch.float32, device=device)
        gt_full_mask = torch.zeros((B, Tl_pad, Tp_pad), dtype=torch.bool,    device=device)

        for b in range(B):
            # 真实 token 的索引，在 padded 维度里
            row_idx = torch.nonzero(pep_mask[b],  as_tuple=False).squeeze(1)  # [Tl_real]
            col_idx = torch.nonzero(prot_mask[b], as_tuple=False).squeeze(1)  # [Tp_real]

            # 该样本在压紧 GT 中的真实高宽
            tl_gt = int(gt_mask[b].any(dim=1).sum().item())
            tp_gt = int(gt_mask[b].any(dim=0).sum().item())

            # 与 embedding 真实尺寸对齐，避免越界
            tl = min(tl_gt, row_idx.numel())
            tp = min(tp_gt, col_idx.numel())
            if tl == 0 or tp == 0:
                continue

            gt_slice = gt_map[b, :tl, :tp].to(dtype=gt_full.dtype)

            rr = row_idx[:tl].unsqueeze(1)  # [tl,1]
            cc = col_idx[:tp].unsqueeze(0)  # [1,tp]

            # 高级索引写回
            gt_full[b].index_put_((rr, cc), gt_slice)
            gt_full_mask[b].index_put_((rr, cc), torch.ones_like(gt_slice, dtype=torch.bool))

        # 标签
        if batch.get("gt_kind", "contact") == "contact":
            labels["labels_contact"] = gt_full  # float: 0/1
        else:
            labels["labels_distance"] = gt_full # float: 距离

        # 只在 “真实×真实 且 GT 有值”的位置训练
        pair_mask = pair_mask & gt_full_mask

    labels["pair_mask_override"] = pair_mask  # 交给损失使用
    return model_batch, labels


@torch.no_grad()
def evaluate(model, criterion, loader, device, max_batches=None):
    model.eval()
    meters = {"loss": 0.0, "n": 0}

    for i, batch in enumerate(loader):
        if (max_batches is not None) and (i >= max_batches):
            break

        # to(device)
        for k in ["protein_emb", "peptide_emb"]:
            batch[k] = batch[k].to(device, non_blocking=True)
        for side in ["protein_masks", "peptide_masks"]:
            for kk in batch[side]:
                batch[side][kk] = batch[side][kk].to(device, non_blocking=True)
        if "gt_map" in batch:
            batch["gt_map"]  = batch["gt_map"].to(device, non_blocking=True)
            batch["gt_mask"] = batch["gt_mask"].to(device, non_blocking=True)

        model_batch, labels = adapt_batch_for_model(batch)
        out = model(model_batch)

        # 规范 scores 的方向为 [B,Tl,Tp]
        s = out["scores"]
        assert s.ndim == 3, f"scores must be 3D, got {s.shape}"
        Tl, Tp = model_batch["pep_emb"].size(1), model_batch["prot_emb"].size(1)
        if s.shape[-2:] == (Tp, Tl):        # [B,Tp,Tl] -> 转置
            s = s.transpose(1, 2).contiguous()
        elif s.shape[-2:] == (Tl, Tp):      # 已是 [B,Tl,Tp]
            pass
        else:
            raise RuntimeError(f"Unexpected scores shape {s.shape}, expected (B,Tl,Tp) or (B,Tp,Tl)")
        out["scores"] = s

        if "pair_mask_override" in labels:
            out["pair_mask"] = labels["pair_mask_override"]

        loss_dict = criterion(out, labels)
        meters["loss"] += float(loss_dict["loss"])
        meters["n"] += 1

    for k in list(meters.keys()):
        if k != "n":
            meters[k] = meters[k] / max(1, meters["n"])
    return meters


def save_checkpoint(state, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    ap = argparse.ArgumentParser()

    # 数据
    ap.add_argument("--index", required=True, help="path to batch_index.json")
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split",   default="val")

    # GT（本 trainer 走有监督，要求提供）
    ap.add_argument("--gt-index", type=str, required=True, help="results.json / cb_maps.json")
    ap.add_argument("--gt-as-contact", action="store_true", default=True)
    ap.add_argument("--gt-threshold", type=float, default=8.0)

    # 训练
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true", help="use torch.amp (mixed precision on CUDA)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    # 模型
    ap.add_argument("--head", choices=["bilinear","mlp"], default="bilinear")
    ap.add_argument("--d-model", type=int, default=1536)
    ap.add_argument("--d-proj", type=int, default=256)
    ap.add_argument("--mlp-hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--score-activation", choices=[None, "softplus", "sigmoid"], default=None)

    # 损失
    ap.add_argument("--use-bce", action="store_true", help="Use BCE for contact (default hinge if not set)")
    ap.add_argument("--m-pos", type=float, default=1.0)
    ap.add_argument("--m-neg", type=float, default=0.0)
    ap.add_argument("--pos-weight", type=float, default=None)
    ap.add_argument("--huber-delta", type=float, default=1.0)

    # 训练细节
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--val-interval", type=int, default=1)
    ap.add_argument("--save-dir", type=str, default="runs/exp1")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--ckpt-every-steps", type=int, default=0,
                    help="If >0, additionally save checkpoint every N steps")

    # W&B
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb-project", type=str, default="pep-prot")
    ap.add_argument("--wandb-run-name", type=str, default=None)
    ap.add_argument("--wandb-offline", action="store_true", help="W&B offline mode")
    ap.add_argument("--wandb-tags", nargs="*", default=None)

    args = ap.parse_args()
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    # W&B init（仅记录标量，不上传模型/代码）
    use_wandb = bool(args.wandb)
    if use_wandb:
        if wandb is None:
            raise RuntimeError("You passed --wandb but wandb is not installed. pip install wandb")
        wandb_mode = "offline" if args.wandb_offline else "online"
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=wandb_mode,
            config=vars(args),
            tags=args.wandb_tags,
            save_code=False,
            settings=wandb.Settings(code_dir=str(Path.cwd()))
        )

    # === 数据索引 ===
    key2npz = read_db_index(args.index)
    train_all = read_split_list(args.splits_dir, args.train_split)
    val_all   = read_split_list(args.splits_dir, args.val_split)

    # --- GT 映射 ---
    gt_key2npz = read_gt_index(args.gt_index)  # {stem -> npz_path}

    def filter_usable(keys):
        usable = []
        for k in keys:
            if k not in key2npz:
                continue
            if strip_suffix_key(k) not in gt_key2npz:
                continue
            usable.append(k)
        return usable

    def build_gt_map(keys):
        m = {}
        for k in keys:
            k2 = strip_suffix_key(k)
            if k2 in gt_key2npz:
                m[k] = gt_key2npz[k2]
        return m

    train_keys = filter_usable(train_all)
    val_keys   = filter_usable(val_all)
    gt_train_map = build_gt_map(train_keys)
    gt_val_map   = build_gt_map(val_keys)

    print(f"[data] train usable={len(train_keys)}  val usable={len(val_keys)}")
    print(f"[data] GT matched: train={len(gt_train_map)}/{len(train_keys)}  val={len(gt_val_map)}/{len(val_keys)}")

    if len(train_keys) == 0 or len(val_keys) == 0:
        raise ValueError("No usable samples found. Check --index, --splits-dir, and --gt-index alignment.")

    # === Dataset & Loader ===
    ds_train = ProtPepFullTokenDataset(
        key_to_npz=key2npz,
        keys=train_keys,
        strict_fixed=False,
        gt_key_to_npz=gt_train_map,
        gt_as_contact=args.gt_as_contact,
        gt_threshold=args.gt_threshold,
    )
    ds_val = ProtPepFullTokenDataset(
        key_to_npz=key2npz,
        keys=val_keys,
        strict_fixed=False,
        gt_key_to_npz=gt_val_map,
        gt_as_contact=args.gt_as_contact,
        gt_threshold=args.gt_threshold,
    )

    loader_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_full_tokens, drop_last=False
    )
    loader_val = DataLoader(
        ds_val, batch_size=max(1, args.batch_size // 2), shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_full_tokens, drop_last=False
    )

    # === 模型/准则 ===
    cfg = PairModelConfig(
        d_model=args.d_model,
        head=args.head,
        hidden=args.mlp_hidden,
        d_proj=args.d_proj,
        dropout=args.dropout,
        score_activation=(None if args.score_activation in [None, "None", "null"] else args.score_activation),
    )
    model = PairwiseModel(cfg).to(device)

    crit = PairwiseCriterion(
        use_bce_for_contact=args.use_bce,
        m_pos=args.m_pos, m_neg=args.m_neg,
        pos_weight=args.pos_weight,
        huber_delta=args.huber_delta,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # torch.amp GradScaler
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 1
    best_val = math.inf
    global_step = 0
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    # 断点续训
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        global_step = ckpt.get("global_step", 0)
        print(f"[resume] Loaded from {args.resume} at epoch {ckpt['epoch']}")

    # === 训练 ===
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        for step, batch in enumerate(loader_train, start=1):
            # to(device)
            for k in ["protein_emb", "peptide_emb"]:
                batch[k] = batch[k].to(device, non_blocking=True)
            for side in ["protein_masks", "peptide_masks"]:
                for kk in batch[side]:
                    batch[side][kk] = batch[side][kk].to(device, non_blocking=True)
            if "gt_map" in batch:
                batch["gt_map"]  = batch["gt_map"].to(device, non_blocking=True)
                batch["gt_mask"] = batch["gt_mask"].to(device, non_blocking=True)

            model_batch, labels = adapt_batch_for_model(batch)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(model_batch)

                # 统一 scores 为 [B,Tl,Tp]
                s = out["scores"]
                assert s.ndim == 3, f"scores must be 3D, got {s.shape}"
                Tl = model_batch["pep_emb"].size(1)
                Tp = model_batch["prot_emb"].size(1)
                if s.shape[-2:] == (Tp, Tl):
                    s = s.transpose(1, 2).contiguous()
                elif s.shape[-2:] == (Tl, Tp):
                    pass
                else:
                    raise RuntimeError(f"Unexpected scores shape {s.shape}, expected (B,Tl,Tp) or (B,Tp,Tl)")
                out["scores"] = s

                if "pair_mask_override" in labels:
                    out["pair_mask"] = labels["pair_mask_override"]

                loss_dict = crit(out, labels)
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            global_step += 1

            # 日志
            if (step % args.log_interval) == 0 or step == 1:
                print(f"[epoch {epoch} step {step}] loss={float(loss):.4f}")

            if use_wandb:
                # 取当前 lr
                try:
                    current_lr = opt.param_groups[0]["lr"]
                except Exception:
                    current_lr = None
                wandb.log({
                    "train/loss": float(loss),
                    "train/lr": current_lr,
                    "epoch": epoch,
                    "step": global_step,
                }, step=global_step)

            # 可选：每 N 步额外保存一次快照（本地）
            if args.ckpt_every_steps and (global_step % args.ckpt_every_steps == 0):
                save_checkpoint({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "cfg": asdict(cfg),
                    "best_val": best_val,
                }, save_dir / f"step_{global_step}.pt")

        # 验证 & 保存
        if (epoch % args.val_interval) == 0:
            val_m = evaluate(model, crit, loader_val, device)
            print(f"[epoch {epoch}] val_loss={val_m['loss']:.4f}  (time {time.time()-t0:.1f}s)")

            if use_wandb:
                wandb.log({
                    "val/loss": float(val_m["loss"]),
                    "epoch": epoch,
                    "step": global_step,
                }, step=global_step)

            # 保存最新
            save_checkpoint({
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "cfg": asdict(cfg),
                "best_val": best_val,
            }, save_dir / "last.pt")

            # 保存最佳
            if val_m["loss"] < best_val:
                best_val = val_m["loss"]
                save_checkpoint({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "cfg": asdict(cfg),
                    "best_val": best_val,
                }, save_dir / "best.pt")
                print(f"  ↳ new best! saved to {save_dir/'best.pt'}")
                if use_wandb:
                    wandb.summary["best_val_loss"] = best_val

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
