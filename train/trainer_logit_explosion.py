#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, time, argparse
from pathlib import Path
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

# === 项目内导入 ===
from contrasive_learning.data.data_loader import (
    read_db_index, read_split_list, ProtPepFullTokenDataset, collate_full_tokens,
    read_gt_index, strip_suffix_key
)

from contrasive_learning.model.models import PairwiseModel, PairModelConfig, PairwiseCriterion, PairMLP  # ← add PairMLP

# === 可选：W&B ===
try:
    import wandb
except Exception:
    wandb = None

# ---------------------------
# 分布式辅助
# ---------------------------
def dist_is_available_and_initialized():
    return dist.is_available() and dist.is_initialized()
def get_world_size():
    return dist.get_world_size() if dist_is_available_and_initialized() else 1
def get_rank():
    return dist.get_rank() if dist_is_available_and_initialized() else 0
def is_main_process():
    return get_rank() == 0
def setup_distributed(backend: str = None):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=rank)
        torch.cuda.set_device(local_rank if torch.cuda.is_available() else 0)
        return True, local_rank
    return False, 0
def cleanup_distributed():
    if dist_is_available_and_initialized():
        dist.barrier(); dist.destroy_process_group()

# ---------------------------
# 其它工具
# ---------------------------
def seed_all(seed: int = 42, add_rank: int = 0):
    import random, numpy as np
    seed = int(seed) + int(add_rank)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def ensure_TlTp_order(t: torch.Tensor, Tl: int, Tp: int) -> torch.Tensor:
    """
    把模型输出统一成：
      - 3D: [B, Tl, Tp]
      - 4D: [B, Tl, Tp, C]
    """
    if t.ndim == 3:
        if t.shape[-2:] == (Tp, Tl):    # [B,Tp,Tl]
            return t.transpose(1, 2).contiguous()
        elif t.shape[-2:] == (Tl, Tp):  # [B,Tl,Tp]
            return t
    elif t.ndim == 4:
        if t.shape[1:3] == (Tp, Tl):    # [B,Tp,Tl,C]
            return t.permute(0, 2, 1, 3).contiguous()  # -> [B,Tl,Tp,C]
        elif t.shape[1:3] == (Tl, Tp):  # [B,Tl,Tp,C]
            return t
    raise RuntimeError(f"Unexpected output shape {tuple(t.shape)}, expect (B,Tl,Tp[,C])")

def _np_table(a, H=3, W=3, fmt="{:8.3f}"):
    import numpy as np
    a = np.asarray(a)
    h = min(H, a.shape[0])
    w = min(W, a.shape[1] if a.ndim > 1 else 1)
    lines = []
    if a.ndim == 1:
        row = " ".join(fmt.format(float(v)) for v in a[:w])
        lines.append(row)
        if a.shape[0] > h:
            lines.append(f"... ({a.shape[0]} shown {h})")
        return "\n".join(lines)
    for i in range(h):
        row = " ".join(fmt.format(float(v)) for v in a[i, :w])
        lines.append(row)
    if a.shape[0] > h or a.shape[1] > w:
        lines.append(f"... ({a.shape[0]}x{a.shape[1]} shown {h}x{w})")
    return "\n".join(lines)

def _tstats(t: torch.Tensor):
    return (float(torch.nanmin(t).item()),
            float(torch.nanmean(t).item()),
            float(torch.nanmax(t).item()))


# ---------------------------
# 距离分桶工具
# ---------------------------
def build_bin_edges(bin0_max: float, bin0_step: float,
                    bin1_max: float, bin1_step: float) -> torch.Tensor:
    """
    例如：0-8 步长0.5；8-32 步长1；>32 为∞的开区间。
    返回 edges 长度 = C+1，最后一个为 +inf。
    """
    import numpy as np
    e0 = np.arange(0.0, bin0_max, bin0_step)  # [0, 0.5, ... , 7.5]
    e1 = np.arange(bin0_max, bin1_max, bin1_step)  # [8, 9, ..., 31]
    edges = np.concatenate([e0, [bin0_max], e1, [bin1_max, np.inf]], axis=0)
    # 去重 & 保序
    edges = np.unique(edges)
    return torch.tensor(edges, dtype=torch.float32)

def bin_centers_from_edges(edges: torch.Tensor) -> torch.Tensor:
    e = edges
    centers = 0.5 * (e[:-1] + e[1:])
    # 最后一个是 [last, inf)：用 last + 4Å 作为代表值（可调）
    centers[-1] = e[-2] + 4.0
    return centers

def distances_to_soft_targets(dist: torch.Tensor,
                              edges: torch.Tensor,
                              smooth_alpha: float = 0.2) -> torch.Tensor:
    """
    把距离 [B,Tl,Tp] 映射为软标签 Q [B,Tl,Tp,C]，仅对有限值有效。
    平滑：主 bin 质量 (1-alpha)，邻居均分 alpha。
    """
    B,Tl,Tp = dist.shape
    device = dist.device
    edges = edges.to(device)
    C = edges.numel() - 1
    # bin 索引：bucketize 返回 [1..C]，减1得到 [0..C-1]
    cls = torch.bucketize(dist, edges, right=False).clamp(1, C) - 1  # [B,Tl,Tp]
    Q = torch.zeros(B, Tl, Tp, C, dtype=torch.float32, device=device)
    # 主 bin
    Q.scatter_(dim=-1, index=cls.unsqueeze(-1), value=(1.0 - smooth_alpha))
    if smooth_alpha > 0:
        w = smooth_alpha / 2.0
        left  = (cls - 1).clamp_min(0)
        right = (cls + 1).clamp_max(C-1)
        Q.scatter_(dim=-1, index=left.unsqueeze(-1),  src=torch.full_like(left.unsqueeze(-1), w, dtype=Q.dtype))
        Q.scatter_(dim=-1, index=right.unsqueeze(-1), src=torch.full_like(right.unsqueeze(-1), w, dtype=Q.dtype))
    # 对 NaN/inf 的距离清零（不参与训练）
    invalid = ~torch.isfinite(dist)
    if invalid.any():
        Q[invalid] = 0.0
    return Q

class BinnedDistanceCriterion(torch.nn.Module):
    """
    多类别分桶距离的损失：
      - 主体：soft targets 的交叉熵  L = -sum_c Q_c * log P_c
      - 可选 Focal：乘以 (1 - p_t)^gamma，其中 p_t = sum_c Q_c * P_c
      - 可选 类别权重 w_c：对每个 c 的项乘 w_c（或对整体乘 sum Q_c w_c）
    仅在 pair_mask=True 的位置聚合。
    """
    def __init__(self,
                 edges: torch.Tensor,
                 smooth_alpha: float = 0.2,
                 class_weights: torch.Tensor | None = None,
                 focal_gamma: float | None = None):
        super().__init__()
        self.register_buffer("edges", edges.float())
        self.register_buffer("class_weights", (class_weights.float() if class_weights is not None else None))
        self.smooth_alpha = float(smooth_alpha)
        self.focal_gamma = (None if focal_gamma is None else float(focal_gamma))

    def forward(self, outputs: dict, batch: dict) -> dict:
        logits = outputs["scores"]        # [B,Tl,Tp,C]
        mask   = outputs["pair_mask"].bool()  # [B,Tl,Tp]
        dist   = batch["labels_distance"].to(logits.dtype)  # [B,Tl,Tp]

        # soft targets
        Q = distances_to_soft_targets(dist, self.edges, self.smooth_alpha)  # [B,Tl,Tp,C]

        # log_probs & probs
        logp = F.log_softmax(logits, dim=-1)
        p    = logp.exp()

        # 基础 CE：sum_c (-Q_c * log p_c)
        if self.class_weights is not None:
            w = self.class_weights.to(logp.device)  # [C]
            ce = -(Q * (logp * w)).sum(dim=-1)      # [B,Tl,Tp]
        else:
            ce = -(Q * logp).sum(dim=-1)

        # Focal (可选)
        if self.focal_gamma is not None and self.focal_gamma > 0:
            pt = (Q * p).sum(dim=-1).clamp(1e-6, 1.0)   # [B,Tl,Tp]
            mod = (1.0 - pt).pow(self.focal_gamma)
            ce = ce * mod

        # 掩码聚合
        if mask is not None:
            ce = ce * mask.float()
            denom = mask.float().sum().clamp_min(1.0)
        else:
            denom = torch.tensor(ce.numel(), device=ce.device, dtype=ce.dtype).clamp_min(1.0)
        loss = ce.sum() / denom

        with torch.no_grad():
            # 用分布期望计算一个“距离回归”指标（方便监控）
            centers = bin_centers_from_edges(self.edges).to(p.device)  # [C]
            pred_mean = (p * centers).sum(dim=-1)  # [B,Tl,Tp]
            valid = mask & torch.isfinite(dist)
            mae = (pred_mean[valid] - dist[valid]).abs().mean() if valid.any() else torch.tensor(0.0, device=loss.device)

        return {"loss": loss, "mae": float(mae.item())}

# ---------------------------
# 适配 batch
# ---------------------------
def adapt_batch_for_model(batch):
    device = batch["protein_emb"].device
    prot_mask = batch["protein_masks"]["valid_real_residue"]
    pep_mask  = batch["peptide_masks"]["valid_real_residue"]
    model_batch = {
        "prot_emb": batch["protein_emb"],
        "pep_emb":  batch["peptide_emb"],
        "prot_mask": prot_mask,
        "pep_mask":  pep_mask,
    }
    labels = {}
    pair_mask = pep_mask[:, :, None] & prot_mask[:, None, :]

    if "gt_map" in batch:
        gt_map  = batch["gt_map"].to(device)         # [B,Tl_gtmax,Tp_gtmax]
        gt_mask = batch["gt_mask"].to(device).bool() # [B,Tl_gtmax,Tp_gtmax]

        B, Tl_pad = pep_mask.shape
        _, Tp_pad = prot_mask.shape
        gt_full      = torch.zeros((B, Tl_pad, Tp_pad), dtype=torch.float32, device=device)
        gt_full_mask = torch.zeros((B, Tl_pad, Tp_pad), dtype=torch.bool,    device=device)

        for b in range(B):
            row_idx = torch.nonzero(pep_mask[b],  as_tuple=False).squeeze(1)
            col_idx = torch.nonzero(prot_mask[b], as_tuple=False).squeeze(1)
            tl_gt = int(gt_mask[b].any(dim=1).sum().item())
            tp_gt = int(gt_mask[b].any(dim=0).sum().item())
            tl = min(tl_gt, row_idx.numel()); tp = min(tp_gt, col_idx.numel())
            if tl == 0 or tp == 0: continue
            gt_slice = gt_map[b, :tl, :tp].to(dtype=gt_full.dtype)
            rr = row_idx[:tl].unsqueeze(1); cc = col_idx[:tp].unsqueeze(0)
            gt_full[b].index_put_((rr, cc), gt_slice)
            gt_full_mask[b].index_put_((rr, cc), torch.ones_like(gt_slice, dtype=torch.bool))

        if batch.get("gt_kind", "contact") == "contact":
            labels["labels_contact"] = gt_full
        else:
            labels["labels_distance"] = gt_full

        pair_mask = pair_mask & gt_full_mask

    labels["pair_mask_override"] = pair_mask
    return model_batch, labels

# ---------------------------
# 评估（val）
# ---------------------------
@torch.no_grad()
def evaluate(model, criterion, loader, device, max_batches=None, is_bins: bool = False):
    model.eval()
    meters = {"loss_sum": 0.0, "n": 0, "mae_sum": 0.0, "mae_n": 0}

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

        # 统一维度
        Tl, Tp = model_batch["pep_emb"].size(1), model_batch["prot_emb"].size(1)
        s = out["scores"]
        out["scores"] = ensure_TlTp_order(s, Tl, Tp)
        out["pair_mask"] = labels["pair_mask_override"]

        loss_dict = criterion(out, {**labels})
        meters["loss_sum"] += float(loss_dict["loss"])
        meters["n"] += 1
        if is_bins and ("mae" in loss_dict):
            meters["mae_sum"] += float(loss_dict["mae"])
            meters["mae_n"] += 1

    # 分布式聚合平均
    if dist_is_available_and_initialized():
        t = torch.tensor([meters["loss_sum"], meters["n"], meters["mae_sum"], meters["mae_n"]],
                         dtype=torch.float32, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        meters["loss_sum"], meters["n"], meters["mae_sum"], meters["mae_n"] = map(float, t.tolist())

    loss_avg = meters["loss_sum"] / max(1, meters["n"])
    out = {"loss": loss_avg}
    if is_bins and meters["mae_n"] > 0:
        out["mae"] = meters["mae_sum"] / meters["mae_n"]
    return out

# ---------------------------
# 保存
# ---------------------------
def save_checkpoint(state, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()

    # 数据
    ap.add_argument("--index", required=True)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split",   default="val")

    # GT
    ap.add_argument("--gt-index", type=str, required=True)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--gt-as-contact",  dest="gt_as_contact", action="store_true",
                    help="Use thresholded contact labels")
    group.add_argument("--gt-as-distance", dest="gt_as_contact", action="store_false",
                    help="Use raw distance labels")
    ap.set_defaults(gt_as_contact=True)
    ap.add_argument("--gt-threshold", type=float, default=8.0)

    # 训练
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2, help="Per-GPU batch size")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    # 分布式
    ap.add_argument("--dist-backend", type=str, default=None, choices=[None, "nccl", "gloo", "mpi"])

    # 模型
    ap.add_argument("--head", choices=["bilinear","mlp"], default="mlp")
    ap.add_argument("--d-model", type=int, default=1536)
    ap.add_argument("--d-proj", type=int, default=256)
    ap.add_argument("--mlp-hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--score-activation", choices=[None, "softplus", "sigmoid"], default=None)
    ap.add_argument("--chunk-ll", type=int, default=0)

    # Contact/Distance（标量）损失
    ap.add_argument("--use-bce", action="store_true")
    ap.add_argument("--m-pos", type=float, default=1.0)
    ap.add_argument("--m-neg", type=float, default=0.0)
    ap.add_argument("--pos-weight", type=float, default=None)
    ap.add_argument("--huber-delta", type=float, default=1.0)

    # === 距离分桶（多分类） ===
    ap.add_argument("--distance-bins", action="store_true",
                    help="Enable distance bin classification (requires --gt-as-distance and --head mlp)")
    ap.add_argument("--bin0-max", type=float, default=8.0)
    ap.add_argument("--bin0-step", type=float, default=0.5)
    ap.add_argument("--bin1-max", type=float, default=32.0)
    ap.add_argument("--bin1-step", type=float, default=1.0)
    ap.add_argument("--bin-smooth", type=float, default=0.2, help="soft target amount (0..1)")
    ap.add_argument("--bin-focal-gamma", type=float, default=0.0, help="0 to disable focal")
    ap.add_argument("--bin-short-mult", type=float, default=2.0,
                    help="Multiply class weights for bins with center < 8Å")

    # 日志/保存
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--val-interval", type=int, default=1)
    ap.add_argument("--save-dir", type=str, default="runs/exp_bins")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--ckpt-every-steps", type=int, default=0)

    # W&B
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default="pep-prot")
    ap.add_argument("--wandb-run-name", type=str, default=None)
    ap.add_argument("--wandb-offline", action="store_true")
    ap.add_argument("--wandb-tags", nargs="*", default=None)

    # ===== Debug（只在 rank0 打印）=====
    ap.add_argument("--debug-steps", type=int, default=0, help="前 N 个 step 打印调试信息；0 关闭")
    ap.add_argument("--debug-max-tl", type=int, default=6, help="打印肽段维度的最大行数")
    ap.add_argument("--debug-max-tp", type=int, default=6, help="打印蛋白维度的最大列数")
    ap.add_argument("--debug-max-c",  type=int, default=6, help="分类头时的最大类别数")
    ap.add_argument("--debug-vals", action="store_true", help="除形状外，额外打印小片段数值")






    args = ap.parse_args()

    # 分布式
    used_ddp, local_rank = setup_distributed(backend=args.dist_backend)
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    world_size = get_world_size()
    seed_all(args.seed, add_rank=get_rank())
    use_amp = bool(args.amp and device.type == "cuda")

    # W&B（rank0）
    use_wandb = bool(args.wandb and is_main_process())
    if use_wandb:
        if wandb is None: raise RuntimeError("You passed --wandb but wandb is not installed.")
        wandb_mode = "offline" if args.wandb_offline else "online"
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, mode=wandb_mode,
                   config=vars(args), tags=args.wandb_tags, save_code=False,
                   settings=wandb.Settings(code_dir=str(Path.cwd())))
        print(f"[wandb] mode={wandb_mode}")

    # 数据索引 / splits
    key2npz = read_db_index(args.index)
    train_all = read_split_list(args.splits_dir, args.train_split)
    val_all   = read_split_list(args.splits_dir, args.val_split)
    gt_key2npz = read_gt_index(args.gt_index)

    def filter_usable(keys):
        return [k for k in keys if (k in key2npz and strip_suffix_key(k) in gt_key2npz)]
    def build_gt_map(keys):
        return {k: gt_key2npz[strip_suffix_key(k)] for k in keys if strip_suffix_key(k) in gt_key2npz}

    train_keys = filter_usable(train_all)
    val_keys   = filter_usable(val_all)
    gt_train_map = build_gt_map(train_keys)
    gt_val_map   = build_gt_map(val_keys)

    if is_main_process():
        print(f"[data] train usable={len(train_keys)}  val usable={len(val_keys)}")
        print(f"[dist] world_size={world_size}  per_gpu_batch={args.batch_size}  global_batch={args.batch_size*world_size}")

    if len(train_keys) == 0 or len(val_keys) == 0:
        if is_main_process(): raise ValueError("No usable samples found.")
        cleanup_distributed(); return

    # Dataset/Loader
    ds_train = ProtPepFullTokenDataset(
        key_to_npz=key2npz, keys=train_keys, strict_fixed=False,
        gt_key_to_npz=gt_train_map, gt_as_contact=args.gt_as_contact, gt_threshold=args.gt_threshold,
    )
    ds_val = ProtPepFullTokenDataset(
        key_to_npz=key2npz, keys=val_keys, strict_fixed=False,
        gt_key_to_npz=gt_val_map, gt_as_contact=args.gt_as_contact, gt_threshold=args.gt_threshold,
    )
    sampler_train = DistributedSampler(ds_train, shuffle=True) if used_ddp else None
    sampler_val   = DistributedSampler(ds_val,   shuffle=False, drop_last=False) if used_ddp else None
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=(sampler_train is None),
                              sampler=sampler_train, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"), collate_fn=collate_full_tokens, drop_last=False)
    loader_val = DataLoader(ds_val, batch_size=max(1, args.batch_size // 2), shuffle=False,
                            sampler=sampler_val, num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"), collate_fn=collate_full_tokens, drop_last=False)

    # === 选择任务 & 准备模型/损失 ===
    if args.distance_bins:
        # 要求：使用原始距离 + MLP 头
        if args.gt_as_contact:
            raise ValueError("distance-bins requires --gt-as-distance (i.e., do NOT pass --gt-as-contact).")
        if args.head != "mlp":
            raise ValueError("distance-bins requires --head mlp.")

        edges = build_bin_edges(args.bin0_max, args.bin0_step, args.bin1_max, args.bin1_step)
        centers = bin_centers_from_edges(edges)
        C = int(edges.numel() - 1)

        # 类别权重：<8Å 的 bin 放大
        w = torch.ones(C, dtype=torch.float32)
        w[centers < 8.0] *= float(args.bin_short_mult)

        cfg = PairModelConfig(
            d_model=args.d_model, head="mlp", hidden=args.mlp_hidden, d_proj=args.d_proj,
            dropout=args.dropout, chunk_ll=args.chunk_ll, score_activation=None, out_channels=C
        )
        model = PairwiseModel(cfg).to(device)
        crit = BinnedDistanceCriterion(edges=edges, smooth_alpha=max(0.0, min(1.0, args.bin_smooth)),
                                       class_weights=w, focal_gamma=(args.bin_focal_gamma if args.bin_focal_gamma>0 else None))
        is_bins = True
        bins_meta = {"edges": edges.cpu().tolist(), "centers": centers.cpu().tolist()}
    else:
        # 旧任务（contact/distance 标量）
        cfg = PairModelConfig(
            d_model=args.d_model, head=args.head, hidden=args.mlp_hidden, d_proj=args.d_proj,
            dropout=args.dropout, chunk_ll=args.chunk_ll,
            score_activation=(None if args.score_activation in [None, "None", "null"] else args.score_activation),
            out_channels=1
        )
        model = PairwiseModel(cfg).to(device)
        crit = PairwiseCriterion(
            use_bce_for_contact=args.use_bce,
            m_pos=args.m_pos, m_neg=args.m_neg,
            pos_weight=args.pos_weight,
            huber_delta=args.huber_delta,
        )
        is_bins = False
        bins_meta = None

    if used_ddp and device.type == "cuda":
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 1
    best_val = math.inf
    global_step = 0
    save_dir = Path(args.save_dir)
    if is_main_process(): save_dir.mkdir(parents=True, exist_ok=True)

    # 断点续训
    if args.resume and is_main_process():
        ckpt = torch.load(args.resume, map_location="cpu")
        (model.module if isinstance(model, DDP) else model).load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"]); scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        global_step = ckpt.get("global_step", 0)
        print(f"[resume] Loaded from {args.resume} at epoch {ckpt['epoch']}")

    if used_ddp:
        t = torch.tensor([start_epoch, best_val if math.isfinite(best_val) else 1e9, global_step],
                         dtype=torch.float32, device=device)
        dist.broadcast(t, src=0)
        start_epoch, best_val, global_step = int(t[0].item()), float(t[1].item()), int(t[2].item())

    # === 训练循环 ===
    for epoch in range(start_epoch, args.epochs + 1):
        if sampler_train is not None: sampler_train.set_epoch(epoch)
        if is_main_process(): t0 = time.time()
        model.train()

        for step, batch in enumerate(loader_train, start=1):
            # to(device)

            # 在训练第一步前加几行临时 log
            for name, tens in [("prot_emb", batch["protein_emb"]), ("pep_emb", batch["peptide_emb"])]:
                n_bad = (~torch.isfinite(tens)).sum().item()
                if n_bad: print(f"[warn] {name} non-finite count = {n_bad}")

            for k in ["protein_emb", "peptide_emb"]:
                batch[k] = batch[k].to(device, non_blocking=True)
            for side in ["protein_masks", "peptide_masks"]:
                for kk in batch[side]:
                    batch[side][kk] = batch[side][kk].to(device, non_blocking=True)
            if "gt_map" in batch:
                batch["gt_map"]  = batch["gt_map"].to(device, non_blocking=True)
                batch["gt_mask"] = batch["gt_mask"].to(device, non_blocking=True)

            model_batch, labels = adapt_batch_for_model(batch)


                    # ====== DEBUG 位置A：forward 之前 ======
            if is_main_process() and args.debug_steps > 0 and step <= args.debug_steps:
                P = model_batch["prot_emb"]; L = model_batch["pep_emb"]
                pm = model_batch["prot_mask"]; lm = model_batch["pep_mask"]
                print(f"[debug/in] prot_emb {tuple(P.shape)}  pep_emb {tuple(L.shape)}  "
                    f"prot_mask {tuple(pm.shape)}  pep_mask {tuple(lm.shape)}")
                if args.debug_vals:
                    tl = min(args.debug_max_tl, L.size(1))
                    tp = min(args.debug_max_tp, P.size(1))
                    print(f"[debug/in] pep_emb[0,:{tl},:8]:\n{L[0,:tl,:8].detach().float().cpu().numpy()}")
                    print(f"[debug/in] prot_emb[0,:{tp},:8]:\n{P[0,:tp,:8].detach().float().cpu().numpy()}")
                    p0 = P[0,0,:8].detach().float().cpu(); l0 = L[0,0,:8].detach().float().cpu()
                    print(f"[debug/in] pair(0,0) p[:8]: {p0.numpy()}")
                    print(f"[debug/in] pair(0,0) l[:8]: {l0.numpy()}")
                    print(f"[debug/in] |p-l|[:8]: {(p0-l0).abs().numpy()}")
                    print(f"[debug/in] p*l [:8]: {(p0*l0).numpy()}")


            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(model_batch)


                            # ====== DEBUG 位置B：forward 之后（统一轴顺序之前）======
                if is_main_process() and args.debug_steps > 0 and step <= args.debug_steps:
                    s = out["scores"]
                    print(f"[debug/out] raw scores shape: {tuple(s.shape)}  dtype={s.dtype}")
                    if s.ndim == 4:
                        tl = min(args.debug_max_tl, s.size(1))
                        tp = min(args.debug_max_tp, s.size(2))
                        cc = min(args.debug_max_c,  s.size(3))
                        logits_small = s[0,:tl,:tp,:cc].detach().float().cpu()
                        probs_small  = torch.softmax(s[0,:tl,:tp,:cc], dim=-1).detach().float().cpu()
                        mn, mx = float(logits_small.min()), float(logits_small.max())
                        print(f"[debug/out] logits head min/max = {mn:.3f}/{mx:.3f}")
                        if args.debug_vals:
                            print(f"[debug/out] logits head:\n{logits_small.numpy()}")
                            print(f"[debug/out] probs  head:\n{probs_small.numpy()}")
                    elif s.ndim == 3:
                        tl = min(args.debug_max_tl, s.size(1))
                        tp = min(args.debug_max_tp, s.size(2))
                        heat = s[0,:tl,:tp].detach().float().cpu()
                        print(f"[debug/out] scores[0,:{tl},:{tp}] min/mean/max = "
                            f"{float(heat.min()):.3f}/{float(heat.mean()):.3f}/{float(heat.max()):.3f}")

                
                Tl = model_batch["pep_emb"].size(1); Tp = model_batch["prot_emb"].size(1)
                out["scores"] = ensure_TlTp_order(out["scores"], Tl, Tp)
                out["pair_mask"] = labels["pair_mask_override"]
                loss_dict = crit(out, labels)
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            global_step += 1

            if is_main_process() and ((step % args.log_interval) == 0 or step == 1):
                if is_bins:
                    print(f"[epoch {epoch} step {step}] loss={float(loss):.4f}  mae={loss_dict.get('mae',0):.4f}")
                else:
                    print(f"[epoch {epoch} step {step}] loss={float(loss):.4f}")

            if is_main_process() and wandb is not None and wandb.run is not None:
                logdic = {"train/loss": float(loss), "epoch": epoch, "step": global_step}
                if is_bins and "mae" in loss_dict: logdic["train/mae"] = float(loss_dict["mae"])
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
                if bins_meta is not None: snap["bins"] = bins_meta
                save_checkpoint(snap, save_dir / f"step_{global_step}.pt")

        # 验证 + 保存
        if (epoch % args.val_interval) == 0:
            val_m = evaluate(model, crit, loader_val, device, is_bins=is_bins)
            if is_main_process():
                elapsed = time.time() - t0
                if is_bins:
                    print(f"[epoch {epoch}] val_loss={val_m['loss']:.4f}  val_mae={val_m.get('mae',0):.4f}  (time {elapsed:.1f}s)")
                else:
                    print(f"[epoch {epoch}] val_loss={val_m['loss']:.4f}  (time {elapsed:.1f}s)")

                if wandb is not None and wandb.run is not None:
                    logdic = {"val/loss": float(val_m["loss"]), "epoch": epoch, "step": global_step}
                    if is_bins and "mae" in val_m: logdic["val/mae"] = float(val_m["mae"])
                    wandb.log(logdic, step=global_step)

                # 保存
                state_dict = (model.module if isinstance(model, DDP) else model).state_dict()
                snap = {
                    "epoch": epoch, "global_step": global_step,
                    "model": state_dict, "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "cfg": asdict(cfg), "best_val": best_val,
                }
                if bins_meta is not None: snap["bins"] = bins_meta
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
