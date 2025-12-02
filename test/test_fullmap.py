#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_fullmap.py — Enhanced version
----------------------------------
- Evaluates full-map contact models (BCE)
- Outputs per-sample + global metrics, Top-K, summary.json
- Adaptive vmax for heatmaps
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from contrasive_learning.data.data_loader_fullmap import (
    read_db_index, read_gt_index, read_split_list,
    ProtPepFullMapDataset, collate_fullmap, strip_suffix_key
)
from contrasive_learning.model.models_fullmap import PairwiseModelFullMap, PairModelConfig
from contrasive_learning.train.trainer_fullmap import FullMapCriterion


# ---------------------------------------------------
# Utility
# ---------------------------------------------------
def save_heatmap_with_points(mat, tp_coords, fp_coords, path: Path,
                             title: str = "", vmin=None, vmax=None,
                             tp_marker="o", fp_marker="x",
                             tp_size=14, fp_size=16, tp_lw=0.6, fp_lw=0.8):
    """
    在概率热图上叠加Top-K点：TP(与GT重合)和FP(不重合)。
    coords 形如 ndarray(N,2) -> [row, col]
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4))
    plt.imshow(mat, cmap="viridis", origin="upper", vmin=vmin, vmax=vmax)
    # FP 先画，TP 后画，防止被覆盖
    if fp_coords is not None and len(fp_coords) > 0:
        plt.scatter(fp_coords[:,1], fp_coords[:,0], marker=fp_marker,
                    s=fp_size, linewidths=fp_lw, c="red", label="FP", alpha=0.9)
    if tp_coords is not None and len(tp_coords) > 0:
        plt.scatter(tp_coords[:,1], tp_coords[:,0], marker=tp_marker,
                    s=tp_size, linewidths=tp_lw, facecolors="none",
                    edgecolors="white", label="TP", alpha=0.95)
    plt.colorbar(fraction=0.03, pad=0.02)
    if title:
        plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def make_mask_from_coords(shape, coords):
    """
    根据 coords(N,2)[row,col] 生成0/1 mask
    """
    import numpy as np
    m = np.zeros(shape, dtype=np.float32)
    if coords is not None and len(coords) > 0:
        m[coords[:,0], coords[:,1]] = 1.0
    return m

def seed_all(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def save_heatmap(mat, path: Path, title: str = "", vmin=None, vmax=None):
    plt.figure(figsize=(4, 4))
    plt.imshow(mat, cmap="viridis", origin="upper", vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------
@torch.no_grad()
def run_eval(model, criterion, loader, device, out_dir: Path,
             save_prob=True, save_text=True, save_plots=True,
             bin_th=0.5, gt_threshold=8.0, plots_first=20, topk=50,
             symmetrize: bool = False):   # <-- 新增


    out_npz = out_dir / "pred_npz"
    out_txt = out_dir / "texts"
    out_fig = out_dir / "figs"
    for p in [out_npz, out_txt, out_fig]:
        p.mkdir(parents=True, exist_ok=True)

    model.eval()
    have_gt = False
    meters = {"loss": 0.0, "n": 0}
    stats_global = {"tp": 0, "fp": 0, "fn": 0, "samples": 0}
    plotted = 0

    for step, batch in enumerate(loader, start=1):
        # Move tensors to device
        for k in ["full_emb", "full_mask", "gt_map_full", "gt_mask_full"]:
            if k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

        model_batch = {"full_emb": batch["full_emb"], "full_mask": batch["full_mask"]}
        out = model(model_batch)
        scores = out["scores"]
        pair_mask = out["pair_mask"]

        # Apply symmetrization if requested (before passing to criterion)
        if symmetrize:
            scores = torch.triu(scores, diagonal=0)
            scores = 0.5 * (scores + scores.transpose(-1, -2))
            out["scores"] = scores

        # --- Output ---
        probs = torch.sigmoid(scores).detach().cpu().numpy()
        gt_np = batch["gt_map_full"].detach().cpu().numpy()
        # Keep original tensors for per-sample loss calculation
        gt_map_full = batch["gt_map_full"]
        gt_mask_full = batch["gt_mask_full"]
        keys = batch["keys"]

        for i, key in enumerate(keys):
            
            # --- 1. 获取当前样本的预测和目标 ---
            prob_i = probs[i]      # NumPy: 预测概率，形状 [L_pred, L_pred] (e.g., 684x684)
            gt_i = gt_np[i]        # NumPy: 目标，形状 [L_orig, L_orig] (e.g., 120x120)
            
            # 获取当前样本的 GT Mask (PyTorch Tensor)
            m = gt_mask_full[i].cpu().bool() 
            
            # --- 2. 【核心修复：对齐预测和目标】---
            # 复用 FullMapCriterion 的索引逻辑，确保基于有效残基长度 n_eff 裁剪
            
            # 确定有效索引 (行和列)
            row_valid = m.any(dim=1)
            col_valid = m.any(dim=0)
            if not torch.equal(row_valid, col_valid):
                # 确保行和列的有效性一致 (解决 score/mask 尺寸不一致问题)
                row_valid = row_valid & col_valid 

            idx = row_valid.nonzero(as_tuple=False).squeeze(-1)
            n_eff = idx.shape[0] # 确定最终的有效尺寸 n (L_orig)

            # 裁剪预测概率 prob_i (从 L_pred x L_pred 裁剪到 n_eff x n_eff)
            # 转化为 Tensor，使用索引裁剪，再转回 NumPy
            prob_t = torch.from_numpy(prob_i).float()
            
            # 使用索引裁剪预测，形状变为 [n_eff, n_eff]
            prob_t_sub = prob_t.index_select(0, idx).index_select(1, idx)
            prob_i = prob_t_sub.numpy() # 预测概率现在是 [n_eff, n_eff]

            # 裁剪目标 gt_i (确保与 prob_i 尺寸匹配 n_eff x n_eff)
            if gt_i.shape[0] > n_eff:
                # 裁剪目标以去除可能存在的 pad 或多余部分
                gt_i = gt_i[:n_eff, :n_eff]
            elif gt_i.shape[0] < n_eff:
                # 理论上不应发生，但如果发生，跳过该样本并打印警告
                print(f"[ERROR] GT size is too small for sample {key}. Skipping.")
                continue 
            
            # --- 对齐结束：现在 prob_i.shape == gt_i.shape == (n_eff, n_eff) ---

            # 二值化预测（基于裁剪后的 prob_i）
            pred_bin = (prob_i >= bin_th).astype(np.float32)

            # --- 保存 NumPy 文件 (使用裁剪后的数组) ---
            np.savez_compressed(out_npz / f"{key}.npz",
                                prob=prob_i, gt=gt_i, pred_bin=pred_bin)

            # === Compute loss per sample (Criterion 仍然使用原始 PyTorch Tensor) ===
            # Criterion 内部必须包含它的裁剪逻辑（如我们在前述讨论中所述）
            
            # 注意：这里我们使用原始的 PyTorch Tensor batch，让 Criterion 自己处理
            sample_out = {"scores": scores[i:i+1], "pair_mask": pair_mask[i:i+1]}
            sample_gt = {"gt_map_full": gt_map_full[i:i+1], "gt_mask_full": gt_mask_full[i:i+1]}

            loss_dict = criterion(sample_out, sample_gt)

            if loss_dict and "loss" in loss_dict:
                loss_val = float(loss_dict["loss"])
                if not np.isnan(loss_val):
                    meters["loss"] += loss_val
                    meters["n"] += 1
            have_gt = True

            # === per-sample metrics (基于裁剪后的 NumPy 数组) ===
            # 这些计算现在不会因为尺寸不匹配而报错
            tp = (pred_bin * gt_i).sum()
            fp = (pred_bin * (1 - gt_i)).sum()
            fn = ((1 - pred_bin) * gt_i).sum()
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)

            stats_global["tp"] += tp
            stats_global["fp"] += fp
            stats_global["fn"] += fn
            stats_global["samples"] += 1

            # 自动设定 Top-K
            L_total = n_eff # 使用裁剪后的有效长度作为总长度
            
            # 重新计算 pep_len 和 prot_len（依赖于 mask 的行/列和 n_eff 的关系）
            # 这里的计算是为了 Top-K 的数量，如果原始代码逻辑复杂，我们使用 n_eff
            # 沿用原始代码中获取长度的思路，但基于已裁剪的 mask 索引
            pep_len = int(m[idx].any(dim=1).sum().item())
            prot_len = int(m[idx].any(dim=0).sum().item())
            
            # 为了防止索引错误，简化 Top-K 长度的估算，使用 n_eff
            # 注意：此处 topk_dynamic 的原始计算依赖于 pep/prot len，如果计算错误可能会导致 topk 性能评估偏差。
            # 我们假设 n_eff 是正确的总长度。
            a=5
            topk_dynamic = max(1, a * (n_eff // 2) * 2) 
            
            # ... (Top-K 坐标提取和后续代码不变) ...
            flat = prob_i.ravel()
            if topk_dynamic >= flat.size:
                topk_dynamic = max(1, flat.size // 20)  # 极端小图/全图退化保护
            flat_idx = np.argpartition(flat, -topk_dynamic)[-topk_dynamic:]
            coords = np.column_stack(np.unravel_index(flat_idx, prob_i.shape))
            vals = prob_i[coords[:, 0], coords[:, 1]]
            order = np.argsort(-vals)
            coords = coords[order]
            vals = vals[order]

            # 区分Top-K里的TP/FP
            gt_i_bool = (gt_i > 0.5)
            is_tp = gt_i_bool[coords[:,0], coords[:,1]]
            tp_coords = coords[is_tp]
            fp_coords = coords[~is_tp]

            # === write text ===
            if save_text:
                txt = [
                    f"key={key}",
                    f"shape={prob_i.shape}",
                    f"min={prob_i.min():.3f} mean={prob_i.mean():.3f} max={prob_i.max():.3f}",
                    f"bin_th={bin_th:.2f}",
                    f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}",
                    f"TopK (pep+prot={topk_dynamic}) predicted contacts (row,col,prob):"
                ]
                for r, c, v in zip(coords[:topk_dynamic, 0], coords[:topk_dynamic, 1], vals[:topk_dynamic]):
                    txt.append(f"  ({r:3d},{c:3d})  {v:.3f}")
                (out_txt / f"{key}.txt").write_text("\n".join(txt))

            # === plots ===
            if save_plots and plotted < plots_first:
                vmax_use = float(np.quantile(prob_i, 0.95)) if prob_i.size > 0 else 1.0

                # 概率图
                save_heatmap(prob_i, out_fig / f"{key}_prob.png",
                            title=f"{key} pred prob", vmin=0.0, vmax=vmax_use)

                # 新增：Top-K overlay（区分TP/FP）
                save_heatmap_with_points(
                    prob_i, tp_coords, fp_coords,
                    out_fig / f"{key}_topk_overlay.png",
                    title=f"{key} TopK overlay (K={topk_dynamic})", vmin=0.0, vmax=vmax_use
                )

                # 新增：Top-K mask（仅显示前K个点）
                topk_mask = make_mask_from_coords(prob_i.shape, coords[:topk_dynamic])
                save_heatmap(topk_mask, out_fig / f"{key}_topk_mask.png",
                            title=f"{key} TopK mask (K={topk_dynamic})", vmin=0.0, vmax=1.0)

                # GT / |prob-GT|
                save_heatmap(gt_i, out_fig / f"{key}_gt.png",
                            title=f"{key} gt contact", vmin=0.0, vmax=1.0)
                diff = np.abs(prob_i - gt_i)
                save_heatmap(diff, out_fig / f"{key}_abs_err.png",
                            title=f"{key} |prob-GT|", vmin=0.0, vmax=1.0)

                # 可选：二值图（若已生成 pred_bin）
                save_heatmap(pred_bin, out_fig / f"{key}_predbin.png",
                            title=f"{key} pred bin≥{bin_th}", vmin=0.0, vmax=1.0)

                plotted += 1

    # === Summary ===
    print(f"\n[test] mean_loss={meters['loss'] / meters['n']:.6f}" if have_gt else "[test] no GT")

    if stats_global["samples"] > 0:
        tp, fp, fn = stats_global["tp"], stats_global["fp"], stats_global["fn"]
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        print(f"[stats] Global Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")

        summary = {
            "samples": int(stats_global["samples"]),
            "mean_loss": float(meters["loss"] / max(meters["n"], 1)),
            "global_precision": float(precision),
            "global_recall": float(recall),
            "global_f1": float(f1)
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[saved] summary -> {out_dir / 'summary.json'}")

    print(f"[saved] outputs -> {out_dir}")


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--gt-index", required=True)
    ap.add_argument("--gt-as-contact", action="store_true")
    ap.add_argument("--gt-threshold", type=float, default=8.0)
    ap.add_argument("--gt-cap-pep", type=int, default=80)
    ap.add_argument("--gt-cap-pro", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--head", default="axial")
    ap.add_argument("--d-model", type=int, default=1536)
    ap.add_argument("--d-proj", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--axial-c-pair", type=int, default=128)
    ap.add_argument("--axial-layers", type=int, default=2)
    ap.add_argument("--axial-heads", type=int, default=4)
    ap.add_argument("--axial-ffn-hidden", type=int, default=512)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--save-prob", action="store_true")
    ap.add_argument("--save-text", action="store_true")
    ap.add_argument("--save-plots", action="store_true")
    ap.add_argument("--bin-th", type=float, default=0.5)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--symmetrize", action="store_true",
                help="Use upper-triangular predictions and symmetrize output.")

    args = ap.parse_args()

    seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    key2npz = read_db_index(args.index)
    gt_key2npz = read_gt_index(args.gt_index)
    keys_all = read_split_list(args.splits_dir, args.split)
    keys = [k for k in keys_all if (k in key2npz and strip_suffix_key(k) in gt_key2npz)]
    gt_map = {k: gt_key2npz[strip_suffix_key(k)] for k in keys}

    ds = ProtPepFullMapDataset(
        key_to_npz=key2npz, keys=keys, gt_key_to_npz=gt_map,
        gt_cap_pep=args.gt_cap_pep, gt_cap_pro=args.gt_cap_pro,
        gt_threshold=args.gt_threshold, gt_as_contact=args.gt_as_contact
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        collate_fn=collate_fullmap)

    print(f"[data] {args.split} usable={len(keys)} samples")

    ckpt = torch.load(args.weights, map_location="cpu")
    cfg = ckpt.get("cfg", None)
    if cfg is None:
        cfg = {
            "d_model": args.d_model,
            "head": args.head,
            "hidden": 512,
            "d_proj": args.d_proj,
            "dropout": args.dropout,
            "temperature": 1.0,
            "axial_c_pair": args.axial_c_pair,
            "axial_layers": args.axial_layers,
            "axial_heads": args.axial_heads,
            "axial_ffn_hidden": args.axial_ffn_hidden,
        }

    model = PairwiseModelFullMap(PairModelConfig(**cfg)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    print(f"[weights] loaded from {args.weights}")

    crit = FullMapCriterion(task="contact" if args.gt_as_contact else "distance")

    out_dir = Path(args.out_dir)
    run_eval(
        model, crit, loader, device, out_dir=out_dir,
        save_prob=args.save_prob, save_text=args.save_text,
        save_plots=args.save_plots, bin_th=args.bin_th,
        gt_threshold=args.gt_threshold, symmetrize=args.symmetrize  # <-- 新增
    )

    

if __name__ == "__main__":
    main()
