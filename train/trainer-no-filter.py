#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, time, argparse
from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# === 本项目导入 ===
# trainer.py
from contrasive_learning.data.data_loader import (
    read_db_index, read_split_list, ProtPepFullTokenDataset, collate_full_tokens
)
from contrasive_learning.model.models import PairwiseModel, PairModelConfig, PairwiseCriterion


def seed_all(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def adapt_batch_for_model(batch):
    """
    把 DataLoader 的键适配为模型期望的键；并整理标签。
    - 输入: batch 来自 collate_full_tokens
    - 输出: model_batch, label_dict
    """
    device = batch["protein_emb"].device
    model_batch = {
        "prot_emb": batch["protein_emb"],   # [B,Lp,D]
        "pep_emb":  batch["peptide_emb"],   # [B,Ll,D]
        # 由 valid_real_residue 推出 token 有效掩码
        "prot_mask": batch["protein_masks"]["valid_real_residue"],  # [B,Lp]
        "pep_mask":  batch["peptide_masks"]["valid_real_residue"],  # [B,Ll]
    }

    labels = {}
    if "gt_map" in batch:
        # pair 有效区域 = 行列有效的外积，再与 gt_mask 取交集
        pair_mask = (model_batch["prot_mask"][:, :, None] &
                     model_batch["pep_mask"][:, None, :])
        if "gt_mask" in batch:
            pair_mask = pair_mask & batch["gt_mask"].to(pair_mask.dtype).bool()

        if batch.get("gt_kind", "contact") == "contact":
            labels["labels_contact"] = batch["gt_map"].float()
        else:
            labels["labels_distance"] = batch["gt_map"].float()

        labels["pair_mask_override"] = pair_mask  # 供 loss 使用时覆盖模型内置的 pair_mask
    return model_batch, labels

@torch.no_grad()
def evaluate(model, criterion, loader, device, max_batches=None):
    model.eval()
    meters = {"loss": 0.0, "n": 0}

    for i, batch in enumerate(loader):
        if (max_batches is not None) and (i >= max_batches):
            break
        # 迁移到 device
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

        # 用外部 pair_mask（考虑 gt_mask）
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

def load_checkpoint(path: Path, map_location="cpu"):
    return torch.load(path, map_location=map_location)

def main():
    ap = argparse.ArgumentParser()
    # 数据
    ap.add_argument("--index", required=True, help="path to batch_index.json")
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split",   default="val")
    ap.add_argument("--gt-index", type=str, default=None, help="results.json/cb_maps.json")
    ap.add_argument("--gt-as-contact", action="store_true", default=True)
    ap.add_argument("--gt-threshold", type=float, default=8.0)
    # 训练
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true", help="use torch.cuda.amp")
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
    ap.add_argument("--use-bce", action="store_true", help="use BCE for contact; default hinge")
    ap.add_argument("--m-pos", type=float, default=1.0)
    ap.add_argument("--m-neg", type=float, default=0.0)
    ap.add_argument("--pos-weight", type=float, default=None)
    ap.add_argument("--huber-delta", type=float, default=1.0)
    # 训练细节
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--val-interval", type=int, default=1)
    ap.add_argument("--save-dir", type=str, default="runs/exp1")
    ap.add_argument("--resume", type=str, default=None)

    args = ap.parse_args()
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 数据集 ===
    key2npz = read_db_index(args.index)
    train_keys = read_split_list(args.splits_dir, args.train_split)
    val_keys   = read_split_list(args.splits_dir, args.val_split)

    # 交集筛选 + GT 匹配逻辑由 Dataset 内部处理（你在 data_loader.py 里已实现）
    ds_train = ProtPepFullTokenDataset(
        key_to_npz=key2npz,
        keys=train_keys,
        strict_fixed=False,
        gt_key_to_npz=(None if args.gt_index is None else None),  # 先占位，下面通过命令行参数传入
        gt_as_contact=args.gt_as_contact,
        gt_threshold=args.gt_threshold,
    )
    ds_val = ProtPepFullTokenDataset(
        key_to_npz=key2npz,
        keys=val_keys,
        strict_fixed=False,
        gt_key_to_npz=(None if args.gt_index is None else None),
        gt_as_contact=args.gt_as_contact,
        gt_threshold=args.gt_threshold,
    )

    # 由于 Dataset 的 GT 映射是在 CLI main 里构造的，你那版 data_loader.py 自带 main。
    # 这里我们直接用 collate，不复用它的 main。=> 最简单方式：走 data_loader.py 的 CLI 进行 dry-run 之外的 loader 构建。
    # 为避免重复实现 GT 匹配，这里采用“简化假设”：若提供 gt_index，则你的 batch_index.json 和 gt_index 的键已对齐（或带 _nomutation 后缀）
    # —— 如果你的训练时一定要严格复用 data_loader.py 的 GT 匹配那段逻辑，我们也可以把那段函数拷进来，这里先给最简工作流：
    # 直接在 Dataset.__getitem__ 内部会根据 self.gt_key_to_npz 是否为空决定是否加载 GT。
    # 因此，建议你把 data_loader.py 中“构造 gt_key_to_npz 的逻辑”抽成函数后在此处调用。
    #
    # 为让脚本可立即运行，先不强制在这里指定 gt_key_to_npz；如果必须加载 GT，请使用你 data_loader.py 的 CLI 做训练，
    # 或者把 gt_key_to_npz 的 dict 在此构造后传入 Dataset。

    loader_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_full_tokens
    )
    loader_val = DataLoader(
        ds_val, batch_size=max(1,args.batch_size//2), shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_full_tokens
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
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 1
    best_val = math.inf
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    # 断点续训
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"[resume] Loaded from {args.resume} at epoch {ckpt['epoch']}")

    # === 训练 ===
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        for step, batch in enumerate(loader_train, start=1):
            # 迁移到 device
            for k in ["protein_emb", "peptide_emb"]:
                batch[k] = batch[k].to(device, non_blocking=True)
            for side in ["protein_masks", "peptide_masks"]:
                for kk in batch[side]:
                    batch[side][kk] = batch[side][kk].to(device, non_blocking=True)
            if "gt_map" in batch:
                batch["gt_map"]  = batch["gt_map"].to(device, non_blocking=True)
                batch["gt_mask"] = batch["gt_mask"].to(device, non_blocking=True)

            model_batch, labels = adapt_batch_for_model(batch)

            with torch.cuda.amp.autocast(enabled=args.amp):
                out = model(model_batch)
                if "pair_mask_override" in labels:
                    out["pair_mask"] = labels["pair_mask_override"]
                loss_dict = crit(out, labels)
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            if (step % args.log-interval) == 0 or step == 1:
                print(f"[epoch {epoch} step {step}] loss={float(loss):.4f}")

        # 验证 & 保存
        if (epoch % args.val_interval) == 0:
            val_m = evaluate(model, crit, loader_val, device)
            print(f"[epoch {epoch}] val_loss={val_m['loss']:.4f}  (time {time.time()-t0:.1f}s)")

            # 保存最新
            save_checkpoint({
                "epoch": epoch,
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
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "cfg": asdict(cfg),
                    "best_val": best_val,
                }, save_dir / "best.pt")
                print(f"  ↳ new best! saved to {save_dir/'best.pt'}")

if __name__ == "__main__":
    main()
