#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trainer_fullmap.py

Full-map ONLY trainer:
  - Data: uses data_loader_fullmap.ProtPepFullMapDataset + collate_fullmap
  - Model: uses pair_model_fullmap.PairwiseModelFullMap (MLP/Bilinear/Axial)
  - Labels: contact (BCE) or distance (Huber), masked by gt_mask_full
  - Symmetry: trains on upper-triangular pairs only (excludes diagonal)

Usage (example):
  python trainer_fullmap.py \
    --index path/to/batch_index.json \
    --splits-dir path/to/splits \
    --train-split train --val-split val \
    --gt-index path/to/cb_maps.json \
    --gt-as-contact \
    --head axial --d-model 1536 --d-proj 256 \
    --batch-size 2 --epochs 5 --save-dir runs/exp_full
"""

import os, math, time, argparse
from pathlib import Path
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

# ===== data (full-map only) =====
from contrasive_learning.data.data_loader_fullmap import (
    read_db_index, read_gt_index, read_split_list,
    ProtPepFullMapDataset, collate_fullmap, strip_suffix_key

)

# ===== model (full-map only) =====
from contrasive_learning.model.models_fullmap import (
    PairwiseModelFullMap, PairModelConfig
)

# === optional: W&B ===
try:
    import wandb
except Exception:
    wandb = None

# ---------------------------
# DDP helpers
# ---------------------------
def debug_check_alignment(batch):
    """
    打印：
      - embedding pairwise 的原始(L,L)
      - embedding 1D mask 的(L,)
      - 去0后 embedding 有效方阵 (Le,Le)
      - GT map 的(Lg,Lg)
      - GT 2D mask 的(Lg,Lg)
      - 去0后 GT 有效子矩阵 (Lg_eff_row, Lg_eff_col)
    """
    full_emb  = batch["full_emb"]        # [B, L, D]
    full_mask = batch["full_mask"]       # [B, L] (bool)
    gt_map    = batch["gt_map_full"]     # [B, Lg, Lg]
    gt_mask   = batch["gt_mask_full"]    # [B, Lg, Lg] (bool)

    B = full_emb.shape[0]
    for i in range(B):
        key = batch["keys"][i]

        # ---- embedding side ----
        L = full_emb[i].shape[0]
        emb_mask_1d = full_mask[i].bool()
        Le = int(emb_mask_1d.sum().item())

        # 去0后有效方阵 = 仅保留有效 token 的行列
        # 这里只打印形状，不做真实切片构图
        emb_pairwise_shape = (L, L)
        emb_pairwise_after_mask_shape = (Le, Le)

        # ---- GT side ----
        Lg = gt_map[i].shape[-1]
        gt_mask_2d = gt_mask[i].bool()

        # 按“行/列任一为真”保留有效 index（兼容稀疏/非方块掩码）
        row_keep = gt_mask_2d.any(dim=1)
        col_keep = gt_mask_2d.any(dim=0)
        Lg_eff_row = int(row_keep.sum().item())
        Lg_eff_col = int(col_keep.sum().item())
        gt_after_mask_shape = (Lg_eff_row, Lg_eff_col)

        print(f"\n[{key}]")
        print(f"  emb_pairwise_shape:          ({L}, {L})")
        print(f"  emb_mask_shape:              ({L},)")
        print(f"  emb_after_mask_shape:        {emb_pairwise_after_mask_shape}   # Le={Le}")

        print(f"  gt_map_shape:                ({Lg}, {Lg})")
        print(f"  gt_mask_shape:               ({Lg}, {Lg})")
        print(f"  gt_after_mask_shape:         {gt_after_mask_shape}              # rows={Lg_eff_row}, cols={Lg_eff_col}")

    """
    对齐检查：打印 embedding side 与 GT side 的有效 shape 与有效元素数量
    """
    full_emb = batch["full_emb"]        # [B, L, D]
    full_mask = batch["full_mask"]      # [B, L]
    gt_map = batch["gt_map_full"]       # [B, L_gt, L_gt]
    gt_mask = batch["gt_mask_full"]     # [B, L_gt, L_gt]

    B = full_emb.shape[0]
    for i in range(B):
        key = batch["keys"][i] if "keys" in batch else f"sample_{i}"

        # --- embedding side ---
        L = full_emb[i].shape[0]
        pairwise_shape = (L, L)
        mask_emb_2d = full_mask[i][:, None] & full_mask[i][None, :]
        emb_valid_pairs = int(mask_emb_2d.sum().item())
        emb_mask_shape = tuple(full_mask[i].shape)

        # --- GT side ---
        gt_map_shape = tuple(gt_map[i].shape)
        gt_mask_shape = tuple(gt_mask[i].shape)
        gt_valid_pairs = int(gt_mask[i].sum().item())

        print(f"\n[{key}]")
        print(f"  emb_pairwise_shape: {pairwise_shape}")
        print(f"  emb_mask_shape:     {emb_mask_shape}")
        print(f"  emb_valid_pairs:    {emb_valid_pairs} / {pairwise_shape[0]**2}")

        print(f"  gt_map_shape:       {gt_map_shape}")
        print(f"  gt_mask_shape:      {gt_mask_shape}")
        print(f"  gt_valid_pairs:     {gt_valid_pairs} / {gt_map_shape[0]**2}")

def dist_is_available_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    return dist.get_world_size() if dist_is_available_and_initialized() else 1

def get_rank():
    return dist.get_rank() if dist_is_available_and_initialized() else 0

def is_main_process():
    return get_rank() == 0

def setup_distributed(backend: str | None = None):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://",
                                world_size=world_size, rank=rank)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, local_rank
    return False, 0

def cleanup_distributed():
    if dist_is_available_and_initialized():
        dist.barrier(); dist.destroy_process_group()

# ---------------------------
# misc
# ---------------------------
def seed_all(seed: int = 42, add_rank: int = 0):
    import random, numpy as np
    seed = int(seed) + int(add_rank)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# ---------------------------
# Minimal full-map criterion
# ---------------------------
# ---------------------------
# Masked full-map criterion (Plan A)
# ---------------------------
class FullMapCriterion(torch.nn.Module):
    """
    仅在 mask 所定义的有效子矩阵上计算 loss。
    mask: [B, L, L]，对称；整行整列为0的区域视为无效区。
    gt_map_full: [B, n, n]，对应去掉无效行列后的有效区域。
    """
    def __init__(self,
                 task: str = "contact",          # "contact" | "distance"
                 pos_weight: float | None = None,
                 focal_gamma: float = 0.0,
                 huber_delta: float = 1.0):
        super().__init__()
        assert task in ("contact", "distance")
        self.task = task
        self.pos_weight = pos_weight
        self.focal_gamma = float(focal_gamma)
        self.huber_delta = float(huber_delta)

    def forward(self, outputs: dict, batch: dict) -> dict:
        scores = outputs["scores"]         # [B, L, L]
        pair_mask = outputs.get("pair_mask", None)  # [B, L, L] or None
        gt_map = batch["gt_map_full"]      # [B, n, n]
        gt_mask = batch["gt_mask_full"]    # [B, L, L]，对称、稀疏

        B, L, _ = scores.shape
        losses, metrics = [], {}

        for b in range(B):
            s = scores[b]
            m = gt_mask[b] > 0  # bool [L,L]
            # 有效行列索引（整行或整列为True）
            row_valid = m.any(dim=1)
            col_valid = m.any(dim=0)
            if not torch.equal(row_valid, col_valid):
                row_valid = row_valid & col_valid
            idx = row_valid.nonzero(as_tuple=False).squeeze(-1)

            # 按索引裁出有效子矩阵
            s_sub = s.index_select(0, idx).index_select(1, idx)
            g_sub = gt_map[b]
            # 若尺寸仍不一致，取交集
            n = min(s_sub.shape[0], g_sub.shape[0])
            s_sub = s_sub[:n, :n]
            g_sub = g_sub[:n, :n]

            if self.task == "contact":
                loss = F.binary_cross_entropy_with_logits(
                    s_sub, g_sub,
                    pos_weight=(torch.tensor(self.pos_weight, device=s_sub.device, dtype=s_sub.dtype)
                                if self.pos_weight is not None else None),
                    reduction="mean"
                )
                if self.focal_gamma > 0:
                    with torch.no_grad():
                        p = torch.sigmoid(s_sub)
                    pt = torch.where(g_sub > 0.5, p, 1 - p).clamp_min(1e-6)
                    loss = (F.binary_cross_entropy_with_logits(s_sub, g_sub, reduction="none")
                            * ((1 - pt) ** self.focal_gamma)).mean()

                with torch.no_grad():
                    prob = torch.sigmoid(s_sub)
                    pos_prob = float(prob[g_sub > 0.5].mean().item()) if (g_sub > 0.5).any() else 0.0
                    neg_prob = float(prob[g_sub <= 0.5].mean().item()) if (g_sub <= 0.5).any() else 0.0
                losses.append(loss)
                metrics.setdefault("pos_prob", []).append(pos_prob)
                metrics.setdefault("neg_prob", []).append(neg_prob)

            else:  # distance task
                loss = F.huber_loss(s_sub, g_sub, delta=self.huber_delta, reduction="mean")
                with torch.no_grad():
                    mae = float((s_sub - g_sub).abs().mean().item())
                losses.append(loss)
                metrics.setdefault("mae", []).append(mae)

        # batch 平均
        loss = torch.stack(losses).mean()
        out = {"loss": loss}
        for k, v in metrics.items():
            out[k] = sum(v) / max(len(v), 1)
        return out

    

# ---------------------------
# evaluation (full-map only)
# ---------------------------
@torch.no_grad()
def evaluate(model, criterion, loader, device, task: str):
    """
    Evaluation for full-map version:
      - uses gt_mask_full to crop valid submatrix
      - computes loss in the same masked region as training

    """


    model.eval()
    meters = {"loss_sum": 0.0, "n": 0}
    if task == "distance":
        meters["mae_sum"] = 0.0

    for batch in loader:
        # Move tensors to device
        batch["full_emb"]    = batch["full_emb"].to(device, non_blocking=True)
        batch["full_mask"]   = batch["full_mask"].to(device, non_blocking=True)
        batch["gt_map_full"] = batch["gt_map_full"].to(device, non_blocking=True)
        batch["gt_mask_full"]= batch["gt_mask_full"].to(device, non_blocking=True)

        model_batch = {
            "full_emb":  batch["full_emb"],
            "full_mask": batch["full_mask"],
        }
        out = model(model_batch)
        scores = out["scores"]               # [B, L, L]
        gt_map = batch["gt_map_full"]        # [B, Lg, Lg]
        gt_mask = batch["gt_mask_full"]      # [B, Lg, Lg]

        B = scores.shape[0]
        losses, maes = [], []

        for b in range(B):
            s = scores[b]
            g = gt_map[b]
            m = gt_mask[b] > 0  # bool [L,L]

            # --- 取有效行列索引 ---
            row_valid = m.any(dim=1)
            col_valid = m.any(dim=0)
            if not torch.equal(row_valid, col_valid):
                row_valid = row_valid & col_valid
            idx = row_valid.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue  # 没有有效区域就跳过
            


            # --- 裁出有效子矩阵 ---
            s_sub = s.index_select(0, idx).index_select(1, idx)
            g_sub = g
            n = min(s_sub.shape[0], g_sub.shape[0])
            s_sub = s_sub[:n, :n]
            g_sub = g_sub[:n, :n]
            
            if meters["n"] == 0 and is_main_process():
                print(f"[DEBUG] eval sample {b}: valid_size={len(idx)}, cropped={s_sub.shape}")
            # --- 计算 loss ---
            if task == "contact":
                loss = F.binary_cross_entropy_with_logits(
                    s_sub, g_sub,
                    pos_weight=(torch.tensor(criterion.pos_weight, device=device, dtype=s_sub.dtype)
                                if criterion.pos_weight is not None else None),
                    reduction="mean"
                )
                losses.append(loss.detach())
            else:
                loss = F.huber_loss(s_sub, g_sub, delta=criterion.huber_delta, reduction="mean")
                losses.append(loss.detach())
                maes.append((s_sub - g_sub).abs().mean().detach())

        if len(losses) == 0:
            continue

        batch_loss = torch.stack(losses).mean()
        meters["loss_sum"] += float(batch_loss.item())
        meters["n"] += 1
        if task == "distance" and len(maes) > 0:
            meters["mae_sum"] += float(torch.stack(maes).mean().item())

    # --- 分布式聚合 ---
    if dist_is_available_and_initialized():
        t = torch.tensor([
            meters["loss_sum"],
            meters.get("mae_sum", 0.0),
            meters["n"]
        ], dtype=torch.float32, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        meters["loss_sum"], meters["mae_sum"], meters["n"] = map(float, t.tolist())

    out = {"loss": meters["loss_sum"] / max(1, meters["n"])}
    if task == "distance":
        out["mae"] = meters["mae_sum"] / max(1, meters["n"])
    return out

# ---------------------------
# save
# ---------------------------
def save_checkpoint(state, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

# ---------------------------
# main (full-map only)
# ---------------------------
def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--index", required=True)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split",   default="val")

    # GT / labels
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--gt-as-contact",  dest="task_contact", action="store_true",
                       help="Use thresholded contact labels")
    group.add_argument("--gt-as-distance", dest="task_contact", action="store_false",
                       help="Use raw distance labels")
    ap.add_argument("--gt-index", required=True)
    ap.add_argument("--gt-cap-pep", type=int, default=None)
    ap.add_argument("--gt-cap-pro", type=int, default=None)
    ap.add_argument("--gt-threshold", type=float, default=8.0)  # used when --gt-as-contact

    # train
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    # distributed
    ap.add_argument("--dist-backend", type=str, default=None, choices=[None, "nccl", "gloo", "mpi"])

    # model
    ap.add_argument("--head", choices=["bilinear","mlp","axial","axial_ta","ta"], default="mlp")
    ap.add_argument("--d-model", type=int, default=1536)
    ap.add_argument("--d-proj", type=int, default=256)
    ap.add_argument("--mlp-hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=2.0)
    # axial knobs
    ap.add_argument("--axial-c-pair",      type=int, default=128)
    ap.add_argument("--axial-layers",      type=int, default=2)
    ap.add_argument("--axial-heads",       type=int, default=4)
    ap.add_argument("--axial-ffn-hidden",  type=int, default=512)

    # loss
    ap.add_argument("--use-bce", action="store_true", help="(contact) use BCE; otherwise margin loss not provided here")
    ap.add_argument("--pos-weight", type=float, default=None, help="(contact) BCE pos_weight")
    ap.add_argument("--focal-gamma", type=float, default=0.0, help="(contact) focal gamma; 0 disables focal")
    ap.add_argument("--huber-delta", type=float, default=1.0, help="(distance) Huber delta")

    # logging/saving
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--val-interval", type=int, default=1)
    ap.add_argument("--save-dir", type=str, default="runs/exp_full")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--ckpt-every-steps", type=int, default=0)

    # W&B
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="pep-prot-full")
    ap.add_argument("--wandb-run-name", type=str, default=None)
    ap.add_argument("--wandb-offline", action="store_true")
    ap.add_argument("--wandb-tags", nargs="*", default=None)

    # LR schedule (warmup + cosine)
    ap.add_argument("--lr-warmup-steps", type=int, default=1000)
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)

    args = ap.parse_args()

    # DDP / device
    used_ddp, local_rank = setup_distributed(backend=args.dist_backend)
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    world_size = get_world_size()
    seed_all(args.seed, add_rank=get_rank())
    use_amp = bool(args.amp and device.type == "cuda")

    # W&B
    use_wandb = bool(args.wandb and is_main_process())
    if use_wandb:
        if wandb is None:
            raise RuntimeError("You passed --wandb but wandb is not installed.")
        wandb_mode = "offline" if args.wandb_offline else "online"
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, mode=wandb_mode,
                   config=vars(args), tags=args.wandb_tags, save_code=False,
                   settings=wandb.Settings(code_dir=str(Path.cwd())))
        print(f"[wandb] mode={wandb_mode}")

    # data indexes / splits
    # ======== data index loading ========
    key2npz = read_db_index(args.index)
    gt_key2npz = read_gt_index(args.gt_index)

    # 读取 split 列表
    train_all = read_split_list(args.splits_dir, args.train_split)
    val_all   = read_split_list(args.splits_dir, args.val_split)

    # 应用 suffix 去除逻辑：例如 "_nomutation" → ""
    train_keys = [k for k in train_all if (k in key2npz and strip_suffix_key(k) in gt_key2npz)]
    val_keys   = [k for k in val_all   if (k in key2npz and strip_suffix_key(k) in gt_key2npz)]

    # 构建 “原始 split key → GT npz 路径” 的映射
    gt_train_map = {k: gt_key2npz[strip_suffix_key(k)] for k in train_keys}
    gt_val_map   = {k: gt_key2npz[strip_suffix_key(k)] for k in val_keys}

    if is_main_process():
        print(f"[data] train usable={len(train_keys)}  val usable={len(val_keys)}")
        print(f"[dist] world_size={world_size}  per_gpu_batch={args.batch_size}  global_batch={args.batch_size*world_size}")

    # 调试输出：若为空则打印首几个样本的匹配状态
    if len(train_keys) == 0 or len(val_keys) == 0:
        if is_main_process():
            print("[debug] Checking first few keys for mismatches:")
            for k in (train_all[:5] + val_all[:5]):
                k_gt = strip_suffix_key(k)
                print(f"  {k:35s} | emb={k in key2npz} | gt_key={k_gt:25s} | gt_hit={k_gt in gt_key2npz}")
        raise ValueError("No usable samples found.")


    if is_main_process():
        print(f"[data] train usable={len(train_keys)}  val usable={len(val_keys)}")
        print(f"[dist] world_size={world_size}  per_gpu_batch={args.batch_size}  global_batch={args.batch_size*world_size}")

    if len(train_keys) == 0 or len(val_keys) == 0:
        if is_main_process(): raise ValueError("No usable samples found.")
        cleanup_distributed(); return

    # datasets / loaders (full-map only)
    # datasets / loaders (full-map only)
    ds_train = ProtPepFullMapDataset(
        key_to_npz=key2npz, keys=train_keys, gt_key_to_npz=gt_train_map,
        gt_cap_pep=args.gt_cap_pep, gt_cap_pro=args.gt_cap_pro,
        gt_threshold=args.gt_threshold, gt_as_contact=args.task_contact
    )
    ds_val = ProtPepFullMapDataset(
        key_to_npz=key2npz, keys=val_keys, gt_key_to_npz=gt_val_map,
        gt_cap_pep=args.gt_cap_pep, gt_cap_pro=args.gt_cap_pro,
        gt_threshold=args.gt_threshold, gt_as_contact=args.task_contact
    )


    sampler_train = DistributedSampler(ds_train, shuffle=True) if used_ddp else None
    sampler_val   = DistributedSampler(ds_val,   shuffle=False, drop_last=False) if used_ddp else None

    loader_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=(sampler_train is None),
        sampler=sampler_train, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), collate_fn=collate_fullmap, drop_last=False
    )
    loader_val = DataLoader(
        ds_val, batch_size=max(1, args.batch_size // 2), shuffle=False,
        sampler=sampler_val, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), collate_fn=collate_fullmap, drop_last=False
    )

    # model (full-map only)
    cfg = PairModelConfig(
        d_model=args.d_model,
        head=args.head,
        hidden=args.mlp_hidden,
        d_proj=args.d_proj,
        dropout=args.drop_out if hasattr(args, "drop_out") else args.dropout,
        out_channels=1,
        temperature=max(1e-6, args.temperature),
        axial_c_pair=args.axial_c_pair,
        axial_layers=args.axial_layers,
        axial_heads=args.axial_heads,
        axial_ffn_hidden=args.axial_ffn_hidden,
    )
    model = PairwiseModelFullMap(cfg).to(device)

    # loss
    task = "contact" if args.task_contact else "distance"
    crit = FullMapCriterion(
        task=task,
        pos_weight=args.pos_weight if args.task_contact and args.use_bce else None,
        focal_gamma=(args.focal_gamma if args.task_contact else 0.0),
        huber_delta=(args.huber_delta if not args.task_contact else 1.0)
    )

    if used_ddp and device.type == "cuda":
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # LR schedule: warmup + cosine tail
    steps_per_epoch = max(1, len(loader_train))
    total_steps = steps_per_epoch * args.epochs
    def lr_lambda(step):
        warm = max(1, args.lr_warmup_steps)
        if step < warm:
            return float(step + 1) / float(warm)
        prog = float(step - warm) / float(max(1, total_steps - warm))
        from math import pi, cos
        min_ratio = max(0.0, min(1.0, args.lr_min_ratio))
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + cos(pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start_epoch = 1
    best_val = math.inf
    global_step = 0
    save_dir = Path(args.save_dir)
    if is_main_process(): save_dir.mkdir(parents=True, exist_ok=True)

    # resume
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        (model.module if isinstance(model, DDP) else model).load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"]); scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        global_step = ckpt.get("global_step", 0)
        if is_main_process():
            print(f"[resume] Loaded from {args.resume} at epoch {ckpt['epoch']}")

    if used_ddp:
        t = torch.tensor([start_epoch, best_val if math.isfinite(best_val) else 1e9, global_step],
                         dtype=torch.float32, device=device)
        dist.broadcast(t, src=0)
        start_epoch, best_val, global_step = int(t[0].item()), float(t[1].item()), int(t[2].item())

    # === train loop ===
    for epoch in range(start_epoch, args.epochs + 1):
        if sampler_train is not None: sampler_train.set_epoch(epoch)
        if is_main_process(): t0 = time.time()
        model.train()

        for step, batch in enumerate(loader_train, start=1):
            # move to device
            batch["full_emb"]    = batch["full_emb"].to(device, non_blocking=True)
            batch["full_mask"]   = batch["full_mask"].to(device, non_blocking=True)
            batch["gt_map_full"] = batch["gt_map_full"].to(device, non_blocking=True)
            batch["gt_mask_full"]= batch["gt_mask_full"].to(device, non_blocking=True)

            if is_main_process() and epoch == start_epoch and step == 1:

                print("\n[DEBUG] Checking alignment (validation set):")
                debug_check_alignment(batch)
                print("[DEBUG] Alignment check done.\n")


            model_batch = {
                "full_emb":  batch["full_emb"],
                "full_mask": batch["full_mask"],
            }

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(model_batch)  # scores: [B, Lm, Lm]
                scores = out["scores"]
                B, Lm, _ = scores.shape
                # —— 放在 out = model(model_batch) 之后 —— 
                scores = out["scores"]           # [B, Lm, Lm]
                pair_mask = out["pair_mask"]     # [B, Lm, Lm] 或 None

                if is_main_process() and epoch == start_epoch and step == 1:
                    print("\n[DEBUG] Shapes after model forward (pairwise already built):")
                    for i in range(min(scores.shape[0], 3)):  # 打印前几个样本
                        key = batch["keys"][i] if "keys" in batch else f"sample_{i}"

                        # —— embedding side —— 
                        emb_mask_1d = batch["full_mask"][i].bool()          # [L]
                        Le = int(emb_mask_1d.sum().item())                  # 有效 token 数
                        print(f"[{key}]")
                        print(f"  emb_pairwise(scores) shape: {tuple(scores[i].shape)}")
                        if pair_mask is not None:
                            print(f"  emb_pairwise(mask)   shape: {tuple(pair_mask[i].shape)}")
                            print(f"  emb_pairwise(mask)   nonzero: {int(pair_mask[i].sum().item())} / {pair_mask[i].numel()}")
                        else:
                            print(f"  emb_pairwise(mask)   shape: None")

                        # “去掉 0 后的 shape”：对 1D mask 有效长度 Le，对 pairwise 是一个 Le×Le 的实心块
                        print(f"  emb_valid_tokens -> pairwise block: ({Le}, {Le})")

                        # —— GT side —— 
                        gt_map  = batch["gt_map_full"][i]        # [Lg, Lg]
                        gt_mask = batch["gt_mask_full"][i].bool()# [Lg, Lg]
                        Lg = gt_map.shape[0]
                        # GT 有效边长：按“行是否存在至少一个 True”统计
                        Lg_eff = int(gt_mask.any(dim=-1).sum().item())
                        print(f"  gt_map_shape:               {tuple(gt_map.shape)}")
                        print(f"  gt_mask_shape:              {tuple(gt_mask.shape)}")
                        print(f"  gt_mask nonzero:            {int(gt_mask.sum().item())} / {gt_mask.numel()}")
                        print(f"  gt_valid_pairs block ~:     ({Lg_eff}, {Lg_eff})")

                        
                    print("[DEBUG] Done.\n")






                """
                                # 选对应任务标签
                if task == "contact":
                    labels = {"labels_contact": gt_map}
                else:
                    labels = {"labels_distance": gt_map}

                loss_dict = crit(out, labels)
                loss = loss_dict["loss"]
                """



                # 直接把原始 batch 交给 criterion（里面会用 gt_map_full/gt_mask_full）
                loss_dict = crit(out, batch)
                loss = loss_dict["loss"]




            scaler.scale(loss).backward()

            # grad clip
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if scheduler is not None: scheduler.step()
            global_step += 1

            # logging
            if is_main_process() and ((step % args.log_interval) == 0 or step == 1):
                s = out["scores"].detach()
                mn, mx = float(s.min().item()), float(s.max().item())
                log_line = f"[epoch {epoch} step {step}] loss={float(loss):.4f} | scores[{mn:.2f},{mx:.2f}]"
                try: log_line += f"  lr={opt.param_groups[0]['lr']:.2e}"
                except: pass
                print(log_line)

            if is_main_process() and wandb is not None and wandb.run is not None:
                logdic = {"train/loss": float(loss), "epoch": epoch, "step": global_step}
                try: logdic["train/lr"] = opt.param_groups[0]["lr"]
                except: pass
                wandb.log(logdic, step=global_step)

            if is_main_process() and args.ckpt_every_steps and (global_step % args.ckpt_every_steps == 0):
                state_dict = (model.module if isinstance(model, DDP) else model).state_dict()
                snap = {
                    "epoch": epoch, "global_step": global_step,
                    "model": state_dict, "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "cfg": asdict(cfg), "best_val": best_val,
                }
                save_checkpoint(snap, save_dir / f"step_{global_step}.pt")

        # validation + save
        if (epoch % args.val_interval) == 0:
            val_m = evaluate(model, crit, loader_val, device, task=task)
            if is_main_process():
                elapsed = time.time() - t0
                if task == "distance":
                    print(f"[epoch {epoch}] val_loss={val_m['loss']:.4f}  val_mae={val_m.get('mae',0):.4f}  (time {elapsed:.1f}s)")
                else:
                    print(f"[epoch {epoch}] val_loss={val_m['loss']:.4f}  (time {elapsed:.1f}s)")

                if wandb is not None and wandb.run is not None:
                    logdic = {"val/loss": float(val_m["loss"]), "epoch": epoch, "step": global_step}
                    if task == "distance" and "mae" in val_m: logdic["val/mae"] = float(val_m["mae"])
                    wandb.log(logdic, step=global_step)

                # save last/best
                state_dict = (model.module if isinstance(model, DDP) else model).state_dict()
                snap = {
                    "epoch": epoch, "global_step": global_step,
                    "model": state_dict, "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "cfg": asdict(cfg), "best_val": best_val,
                }
                save_checkpoint(snap, save_dir / "last.pt")
                if val_m["loss"] < best_val:
                    best_val = val_m["loss"]
                    snap["best_val"] = best_val
                    save_checkpoint(snap, save_dir / "best.pt")
                    print(f"  ↳ new best! saved to {save_dir/'best.pt'}")
                    if wandb is not None and wandb.run is not None:
                        wandb.summary["best_val_loss"] = best_val

    if is_main_process() and wandb is not None and wandb.run is not None:
        wandb.finish()
    cleanup_distributed()

if __name__ == "__main__":
    main()
