#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # for heatmaps

# project imports
from contrasive_learning.data.data_loader import (
    read_db_index, read_split_list, ProtPepFullTokenDataset, collate_full_tokens,
    read_gt_index, strip_suffix_key
)
from contrasive_learning.model.models import PairwiseModel, PairModelConfig, PairwiseCriterion


# ---------------------------
# Utils
# ---------------------------

def seed_all(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _format_head(mat: np.ndarray, H: int = 10, fmt: str = "{:6.3f}") -> str:
    """format top-left H×H of a 2D matrix."""
    if mat.ndim != 2:
        return f"<ndim={mat.ndim} not supported>"
    r = min(H, mat.shape[0]); c = min(H, mat.shape[1])
    lines = []
    for i in range(r):
        row = " ".join(fmt.format(float(v)) for v in mat[i, :c])
        lines.append(row)
    if mat.shape[0] > r or mat.shape[1] > c:
        lines.append(f"... ({mat.shape[0]}x{mat.shape[1]} shown {r}x{c})")
    return "\n".join(lines)


def _stats_txt(x: np.ndarray, m: np.ndarray | None = None, name: str = "pred"):
    xs = x[m.astype(bool)] if m is not None else x.ravel()
    if xs.size == 0:
        return f"{name}: empty"
    return f"{name}: min/mean/max = {xs.min():.3f} / {xs.mean():.3f} / {xs.max():.3f}"


def _save_heatmap_with_marks(
    mat: np.ndarray,
    path: Path,
    title: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    rows_mark: list[int] | None = None,   # 肽方向（行）的 X 位置
    cols_mark: list[int] | None = None,   # 蛋白方向（列）的 X 位置
    row_color: str = "k",
    col_color: str = "k",
    alpha: float = 0.5,
    lw: float = 0.4,
):
    """在热图上用细线标出含 'X' 的行/列（来源于 GT 序列）"""
    plt.figure(figsize=(8, 2.5))
    if (vmin is not None) or (vmax is not None):
        plt.imshow(mat, origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
    else:
        plt.imshow(mat, origin="upper", aspect="auto")
    if title:
        plt.title(title)
    plt.colorbar(fraction=0.025, pad=0.02)

    ax = plt.gca()
    R, C = mat.shape[:2]

    # 画横线：肽端（行）X
    if rows_mark:
        for r in rows_mark:
            if 0 <= r < R:
                ax.axhline(r - 0.5, color=row_color, alpha=alpha, lw=lw)

    # 画竖线：蛋白端（列）X
    if cols_mark:
        for c in cols_mark:
            if 0 <= c < C:
                ax.axvline(c - 0.5, color=col_color, alpha=alpha, lw=lw)

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def adapt_batch_for_model(batch):
    """to model inputs + build labels & pair mask."""
    device = batch["protein_emb"].device
    prot_mask = batch["protein_masks"]["valid_real_residue"]  # [B,Tp]
    pep_mask  = batch["peptide_masks"]["valid_real_residue"]  # [B,Tl]
    model_batch = {
        "prot_emb": batch["protein_emb"],
        "pep_emb":  batch["peptide_emb"],
        "prot_mask": prot_mask,
        "pep_mask":  pep_mask,
    }
    labels = {}
    pair_mask = pep_mask[:, :, None] & prot_mask[:, None, :]  # [B,Tl,Tp]

    if "gt_map" in batch:
        gt_map  = batch["gt_map"].to(device)
        gt_mask = batch["gt_mask"].to(device).bool()

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
            if tl == 0 or tp == 0:
                continue
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
# Eval
# ---------------------------

@torch.no_grad()
def run_eval(
    model, criterion, loader, device,
    out_dir: Path,
    save_prob: bool,
    save_binary: bool,
    bin_th: float,
    use_amp: bool,
    print_first: int,
    print_head: int,
    save_text: bool,
    topk: int,
    score_activation: str | None,
    save_plots: bool = False,
    plots_first: int = 10,
    vmax: float | None = None,
    *,
    bin_edges=None,
    bin_centers=None,
    gt_threshold: float = 8.0,
):
    model.eval()
    out_npz = out_dir / "pred_npz"
    out_txt = out_dir / "texts"
    out_fig = out_dir / "figs"
    out_npz.mkdir(parents=True, exist_ok=True)
    if save_text:
        out_txt.mkdir(parents=True, exist_ok=True)
    if save_plots:
        out_fig.mkdir(parents=True, exist_ok=True)

    have_gt = False
    meters = {"loss": 0.0, "n": 0}
    printed = 0
    plotted = 0

    # skip summary
    summary = {
        "total_samples": 0,
        "with_gt": 0,
        "ok_aligned": 0,
        "skipped_misaligned": 0,
        "skipped_list": []  # [{key, pred_shape, gt_shape}]
    }

    amp_ctx = torch.amp.autocast(device_type=device.type, enabled=use_amp)

    for batch in loader:
        # to(device)
        for k in ["protein_emb", "peptide_emb"]:
            batch[k] = batch[k].to(device, non_blocking=True)
        for side in ["protein_masks", "peptide_masks"]:
            for kk in batch[side]:
                batch[side][kk] = batch[side][kk].to(device, non_blocking=True)
        if "gt_map" in batch:
            batch["gt_map"]  = batch["gt_map"].to(device, non_blocking=True)
            batch["gt_mask"] = batch["gt_mask"].to(device, non_blocking=True)
            have_gt = True

        model_batch, labels = adapt_batch_for_model(batch)

        with amp_ctx:
            out = model(model_batch)

        # ===== unify scores to [B,Tl,Tp,(C?)] =====
        s = out["scores"]
        Tl = model_batch["pep_emb"].size(1)
        Tp = model_batch["prot_emb"].size(1)
        batch_is_distance = None

        if s.ndim == 4:
            # [B, Tl, Tp, C] or [B, Tp, Tl, C] -> [B, Tl, Tp, C]
            B, X, Y, C = s.shape
            if (X, Y) == (Tp, Tl):
                s = s.permute(0, 2, 1, 3).contiguous()
            elif (X, Y) != (Tl, Tp):
                raise RuntimeError(f"Unexpected 4D scores shape {s.shape} for Tl={Tl},Tp={Tp}")
            logits = s
            probs  = torch.softmax(logits, dim=-1)  # [B,Tl,Tp,C]

            if (bin_centers is None) and (bin_edges is None):
                raise RuntimeError("bins mode needs bin_centers or bin_edges")

            _centers = bin_centers
            if (_centers is None) and (bin_edges is not None):
                _centers = []
                for i in range(len(bin_edges) - 1):
                    lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
                    _centers.append((lo + hi) / 2.0 if np.isfinite(hi) else (lo + 1.0))
            assert len(_centers) == C, f"len(bin_centers)={len(_centers)} != C={C}"
            centers_t = torch.as_tensor(_centers, device=probs.device, dtype=probs.dtype)

            is_gt_distance = bool(batch.get("gt_kind", "distance") == "distance")
            if is_gt_distance:
                # expected distance
                exp_dist = torch.tensordot(probs, centers_t, dims=([-1], [0]))  # [B,Tl,Tp]
                out["exp_dist"] = exp_dist
                out["scores"]   = exp_dist
                batch_is_distance = True
            else:
                # contact prob by summing bins <= threshold
                if bin_edges is None:
                    raise RuntimeError("need bin_edges to accumulate contact prob")
                be = torch.as_tensor(bin_edges, device=probs.device, dtype=probs.dtype)
                upper = be[1:]
                mask  = (upper <= float(gt_threshold))
                p_contact = (probs * mask.view(1, 1, 1, -1)).sum(dim=-1)
                out["p_contact"] = p_contact
                out["scores"]    = p_contact
                batch_is_distance = False

        elif s.ndim == 3:
            if s.shape[-2:] == (Tp, Tl):
                s = s.transpose(1, 2).contiguous()
            elif s.shape[-2:] != (Tl, Tp):
                raise RuntimeError(f"Unexpected 3D scores shape {s.shape}")
            out["scores"] = s

            k = batch.get("gt_kind", None)
            if isinstance(k, str):
                batch_is_distance = (k == "distance")
            else:
                batch_is_distance = (str(score_activation or "").lower() == "softplus")
        else:
            raise RuntimeError(f"Unexpected scores ndim={s.ndim}")

        # set pair_mask
        if "pair_mask_override" in labels:
            out["pair_mask"] = labels["pair_mask_override"]

        # loss if GT provided
        if have_gt:
            loss_dict = criterion(out, labels)
            meters["loss"] += float(loss_dict["loss"]); meters["n"] += 1

        scores = out["scores"]           # [B,Tl,Tp]: distance or contact-prob
        pair_m = out["pair_mask"].bool()
        keys   = batch["keys"]

        B = scores.size(0)
        for b in range(B):
            pep_idx  = torch.nonzero(model_batch["pep_mask"][b],  as_tuple=False).squeeze(1)
            prot_idx = torch.nonzero(model_batch["prot_mask"][b], as_tuple=False).squeeze(1)


            # 这里定义“真实长度”，后面给 GT 用
            tl_real = pep_idx.numel()
            tp_real = prot_idx.numel()

            sb = scores[b].index_select(0, pep_idx).index_select(1, prot_idx).detach().cpu().float().numpy()
            mb = pair_m[b].index_select(0, pep_idx).index_select(1, prot_idx).detach().cpu().numpy().astype(np.bool_)


            out_rec = {"scores": sb, "pair_mask": mb, "key": keys[b]}

            # stats counter
            summary["total_samples"] += 1

            # GT slice (aligned to available region)
            gt_map_np = None
            misaligned = False
            if have_gt:
                summary["with_gt"] += 1
                gt_map  = batch["gt_map"][b]
                gt_mask = batch["gt_mask"][b]
                tl_gt = int(gt_mask.any(dim=1).sum().item())
                tp_gt = int(gt_mask.any(dim=0).sum().item())
                tl_cut = min(tl_gt, tl_real)
                tp_cut = min(tp_gt, tp_real)
                    
                if tl_cut > 0 and tp_cut > 0:
                    gt_map_np = gt_map[:tl_cut, :tp_cut].detach().cpu().to(dtype=torch.float32).numpy()
                    out_rec["gt_map"] = gt_map_np
                # misalignment check against current pred slice
                if gt_map_np is not None:
                    pred_shape = (sb.shape[0], sb.shape[1])
                    gt_shape   = (gt_map_np.shape[0], gt_map_np.shape[1])
                    if pred_shape != gt_shape:
                        misaligned = True
                        summary["skipped_misaligned"] += 1
                        summary["skipped_list"].append({
                            "key": keys[b],
                            "pred_shape": [int(pred_shape[0]), int(pred_shape[1])],
                            "gt_shape":   [int(gt_shape[0]),   int(gt_shape[1])]
                        })

            # decide save fields
            if not batch_is_distance:
                # contact: logits(3D)→prob or 4D already prob
                prob = sb if (str(score_activation or "").lower() == "sigmoid") else _sigmoid_np(sb)
                out_rec["prob"] = prob
                if save_binary:
                    out_rec["binary"] = (prob >= float(bin_th)).astype(np.uint8)
            else:
                # distance
                out_rec["distance"] = sb

            # npz save
            np.savez_compressed(out_npz / f"{keys[b]}.npz", **out_rec)

            if gt_map_np is not None and not misaligned:
                summary["ok_aligned"] += 1

            # ----- console print -----
            if printed < print_first:
                print(f"\n=== [{printed+1}] key={keys[b]}  shape: Tl={sb.shape[0]}, Tp={sb.shape[1]}")
                if not batch_is_distance:
                    prob_show = out_rec.get("prob", sb)
                    print(f"[prob head {print_head}x{print_head}]")
                    print(_format_head(prob_show, H=print_head))
                    if gt_map_np is not None:
                        print(f"[GT head {print_head}x{print_head}]")
                        print(_format_head(gt_map_np, H=print_head, fmt='{:6.0f}'))
                else:
                    print(f"[pred distance head {print_head}x{print_head}]")
                    print(_format_head(sb, H=print_head, fmt="{:6.2f}"))
                    print(_stats_txt(sb, mb, name="pred-dist"))
                    if gt_map_np is not None:
                        print(f"[GT distance head {print_head}x{print_head}]")
                        print(_format_head(gt_map_np, H=print_head, fmt="{:6.2f}"))
                        print(_stats_txt(gt_map_np, None, name="gt-dist"))
                        if misaligned:
                            print("!! shape mismatch -> skip error metrics for this sample")
                        elif mb.sum() > 0:
                            mae = np.abs(sb - gt_map_np)[mb].mean()
                            rmse = np.sqrt(((sb - gt_map_np)[mb] ** 2).mean())
                            print(f"[error] MAE={mae:.3f} Å  RMSE={rmse:.3f} Å")
                printed += 1

            # ----- save txt summary -----
            if save_text:
                lines = [f"key: {keys[b]}", f"shape: Tl={sb.shape[0]}, Tp={sb.shape[1]}"]
                if not batch_is_distance:
                    prob_save = out_rec.get("prob", sb)
                    lines.append(f"\n[prob head {print_head}x{print_head}]")
                    lines.append(_format_head(prob_save, H=print_head))
                    if gt_map_np is not None:
                        lines.append(f"\n[GT head {print_head}x{print_head}]")
                        lines.append(_format_head(gt_map_np, H=print_head, fmt='{:6.0f}'))
                else:
                    lines.append(_stats_txt(sb, mb, name="pred-dist"))
                    if gt_map_np is not None:
                        lines.append(_stats_txt(gt_map_np, None, name="gt-dist"))
                        if misaligned:
                            lines.append("!! shape mismatch -> skip error metrics")
                        elif mb.sum() > 0:
                            mae = np.abs(sb - gt_map_np)[mb].mean()
                            rmse = np.sqrt(((sb - gt_map_np)[mb] ** 2).mean())
                            lines.append(f"MAE={mae:.4f} Å  RMSE={rmse:.4f} Å")
                (out_txt / f"{keys[b]}.txt").write_text("\n".join(lines), encoding="utf-8")

            # ===（新增）从 batch 里取 GT 序列并找出 'X' 的行/列===
            pep_seq_list = batch.get("gt_pep_seq", None)  # list[str|None] （data_loader 输出）
            pro_seq_list = batch.get("gt_pro_seq", None)

            pep_seq = None
            pro_seq = None
            if isinstance(pep_seq_list, list) and b < len(pep_seq_list):
                pep_seq = pep_seq_list[b]
            if isinstance(pro_seq_list, list) and b < len(pro_seq_list):
                pro_seq = pro_seq_list[b]

            Tl_plot, Tp_plot = sb.shape  # 当前绘图区域大小

            def _x_idx_from_seq(seq: str | None, limit: int) -> list[int]:
                if not isinstance(seq, str) or len(seq) == 0:
                    return []
                limit = min(limit, len(seq))
                return [i for i, ch in enumerate(seq[:limit]) if ch == 'X' or ch == 'x']

            rows_mark = _x_idx_from_seq(pep_seq, Tl_plot)  # 肽端行索引
            cols_mark = _x_idx_from_seq(pro_seq, Tp_plot)  # 蛋白端列索引

            # 只有 contact 模式会用到 prob_plot；提前统一定义，避免 NameError
            prob_plot = None
            if not batch_is_distance:
                prob_plot = out_rec.get(
                    "prob",
                    sb if (str(score_activation or "").lower() == "sigmoid") else _sigmoid_np(sb)
                )

            # ----- plots -----
            if save_plots and plotted < plots_first:
                if batch_is_distance:
                    vmax_use = vmax
                    if vmax_use is None:
                        vals = [sb]
                        if gt_map_np is not None:
                            vals.append(gt_map_np)
                        cat = np.concatenate([v.ravel() for v in vals]) if len(vals) else np.array([])
                        vmax_use = float(np.quantile(cat, 0.95)) if cat.size > 0 else None
                    _save_heatmap_with_marks(
                        sb, out_fig / f"{keys[b]}_pred.png",
                        title=f"{keys[b]} pred dist (Å) | X_rows={len(rows_mark)}, X_cols={len(cols_mark)}",
                        vmin=None, vmax=vmax_use, rows_mark=rows_mark, cols_mark=cols_mark
                    )
                    if gt_map_np is not None:
                        _save_heatmap_with_marks(
                            gt_map_np, out_fig / f"{keys[b]}_gt.png",
                            title=f"{keys[b]} GT dist (Å) | X_rows={len(rows_mark)}, X_cols={len(cols_mark)}",
                            vmin=None, vmax=vmax_use, rows_mark=rows_mark, cols_mark=cols_mark
                        )
                        if not misaligned:
                            diff = np.abs(sb - gt_map_np)
                            _save_heatmap_with_marks(
                                diff, out_fig / f"{keys[b]}_abs_err.png",
                                title=f"{keys[b]} |pred-GT| (Å) | X_rows={len(rows_mark)}, X_cols={len(cols_mark)}",
                                vmin=None, vmax=None, rows_mark=rows_mark, cols_mark=cols_mark
                            )
                else:
                    # pred prob
                    _save_heatmap_with_marks(
                        prob_plot, out_fig / f"{keys[b]}_prob.png",
                        title=f"{keys[b]} pred prob | X_rows={len(rows_mark)}, X_cols={len(cols_mark)}",
                        vmin=0.0, vmax=1.0, rows_mark=rows_mark, cols_mark=cols_mark
                    )

                    # GT contact
                    if gt_map_np is not None:
                        _save_heatmap_with_marks(
                            gt_map_np.astype(np.float32), out_fig / f"{keys[b]}_gt.png",
                            title=f"{keys[b]} GT contact | X_rows={len(rows_mark)}, X_cols={len(cols_mark)}",
                            vmin=0.0, vmax=1.0, rows_mark=rows_mark, cols_mark=cols_mark
                        )

                    # |prob - GT|
                    if gt_map_np is not None and not misaligned:
                        diff = np.abs(prob_plot - gt_map_np.astype(np.float32))
                        _save_heatmap_with_marks(
                            diff, out_fig / f"{keys[b]}_abs_err.png",
                            title=f"{keys[b]} |prob-GT| | X_rows={len(rows_mark)}, X_cols={len(cols_mark)}",
                            vmin=0.0, vmax=1.0, rows_mark=rows_mark, cols_mark=cols_mark
                        )

                    # binary
                    if save_binary:
                        binary = (prob_plot >= float(bin_th)).astype(np.float32)
                        _save_heatmap_with_marks(
                            binary, out_fig / f"{keys[b]}_bin_th{bin_th}.png",
                            title=f"{keys[b]} pred>= {bin_th} | X_rows={len(rows_mark)}, X_cols={len(cols_mark)}",
                            vmin=0.0, vmax=1.0, rows_mark=rows_mark, cols_mark=cols_mark
                        )

                plotted += 1

    if have_gt and meters["n"] > 0:
        print(f"\n[test] mean_loss={meters['loss']/meters['n']:.6f}")
    else:
        print("\n[test] saved predictions (no GT provided).")

    # ---- skip/misalignment summary ----
    import json
    print(f"\n[summary] total={summary['total_samples']}  with_gt={summary['with_gt']}  "
          f"ok_aligned={summary['ok_aligned']}  skipped_misaligned={summary['skipped_misaligned']}")
    (out_dir / "skip_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[saved] skip summary -> {out_dir/'skip_summary.json'}")


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--index", required=True)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)

    # GT (mutually exclusive)
    ap.add_argument("--gt-index", type=str, default=None)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--gt-as-contact",  dest="gt_as_contact", action="store_true",
                       help="Use thresholded contact labels (0/1)")
    group.add_argument("--gt-as-distance", dest="gt_as_contact", action="store_false",
                       help="Use raw distance labels (Å)")
    ap.set_defaults(gt_as_contact=True)
    ap.add_argument("--gt-threshold", type=float, default=8.0,
                    help="Contact threshold in Å for contact evaluation")

    # weights / model
    ap.add_argument("--weights", type=str, required=True, help="path to best.pt / last.pt")
    ap.add_argument("--head", choices=["bilinear","mlp","axial"], default=None)  # allow axial
    ap.add_argument("--d-model", type=int, default=1536)
    ap.add_argument("--d-proj", type=int, default=256)
    ap.add_argument("--mlp-hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--score-activation", choices=[None, "softplus", "sigmoid"], default=None)

    # criterion (only used if GT provided)
    ap.add_argument("--use-bce", action="store_true")
    ap.add_argument("--m-pos", type=float, default=1.0)
    ap.add_argument("--m-neg", type=float, default=0.0)
    ap.add_argument("--pos-weight", type=float, default=None)
    ap.add_argument("--huber-delta", type=float, default=1.0)

    # output
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--save-prob", action="store_true")
    ap.add_argument("--save-binary", action="store_true")
    ap.add_argument("--bin-th", type=float, default=0.5)

    # printing / text
    ap.add_argument("--print-first", type=int, default=3)
    ap.add_argument("--print-head", type=int, default=10)
    ap.add_argument("--save-text", action="store_true")
    ap.add_argument("--topk", type=int, default=50)

    # visualization
    ap.add_argument("--save-plots", action="store_true")
    ap.add_argument("--plots-first", type=int, default=10)
    ap.add_argument("--vmax", type=float, default=None)

    # misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")

    args = ap.parse_args()
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    # === data index / keys ===
    key2npz = read_db_index(args.index)
    keys_all = read_split_list(args.splits_dir, args.split)

    gt_map = {}
    if args.gt_index:
        gt_key2npz = read_gt_index(args.gt_index)
        keys = [k for k in keys_all if (k in key2npz and strip_suffix_key(k) in gt_key2npz)]
        gt_map = {k: gt_key2npz[strip_suffix_key(k)] for k in keys}
    else:
        keys = [k for k in keys_all if k in key2npz]

    print(f"[data] {args.split} usable={len(keys)}")
    if args.gt_index:
        print(f"[data] GT matched: {len(gt_map)}/{len(keys)}")

    ds = ProtPepFullTokenDataset(
        key_to_npz=key2npz,
        keys=keys,
        strict_fixed=False,
        gt_key_to_npz=(gt_map or None),
        gt_as_contact=args.gt_as_contact,
        gt_threshold=args.gt_threshold,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_full_tokens, drop_last=False
    )

    # === load weights / model ===
    ckpt = torch.load(args.weights, map_location="cpu")

    for name, tensor in ckpt["model"].items():
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print("Bad param:", name)

    ckpt_cfg = ckpt.get("cfg", None)
    bins_meta = ckpt.get("bins", None)  # read bins if saved during training

    if ckpt_cfg is not None:
        cfg = PairModelConfig(**ckpt_cfg)
        score_activation = (ckpt_cfg or {}).get("score_activation", None)
    else:
        cfg = PairModelConfig(
            d_model=args.d_model,
            head=(args.head or "bilinear"),
            hidden=args.mlp_hidden,
            d_proj=args.d_proj,
            dropout=args.dropout,
            score_activation=(None if args.score_activation in [None, "None", "null"] else args.score_activation),
        )
        score_activation = cfg.score_activation

    # bins (distance classification) — prefer ckpt["bins"], else fallback
    bin_edges = None
    bin_centers = None
    if bins_meta is not None:
        bin_edges = bins_meta.get("edges", None)
        bin_centers = bins_meta.get("centers", None)

    # fallback bins (if not saved)
    if (bin_edges is None) and (bin_centers is None):
        e = list(np.arange(0.0, 8.0, 0.5)) + list(np.arange(8.0, 32.0, 1.0)) + [32.0, float("inf")]
        bin_edges = e
    if (bin_centers is None) and (bin_edges is not None):
        centers = []
        for i in range(len(bin_edges) - 1):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            centers.append((lo + hi) / 2.0 if np.isfinite(hi) else (lo + 1.0))
        bin_centers = centers

    model = PairwiseModel(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    # criterion (only used when GT available)
    crit = PairwiseCriterion(
        use_bce_for_contact=args.use_bce,
        m_pos=args.m_pos, m_neg=args.m_neg,
        pos_weight=args.pos_weight,
        huber_delta=args.huber_delta,
    )

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    run_eval(
        model, crit, loader, device,
        out_dir=out_dir,
        save_prob=bool(args.save_prob),
        save_binary=bool(args.save_binary),
        bin_th=float(args.bin_th),
        use_amp=use_amp,
        print_first=int(args.print_first),
        print_head=int(args.print_head),
        save_text=bool(args.save_text),
        topk=int(args.topk),
        score_activation=score_activation,
        save_plots=bool(args.save_plots),
        plots_first=int(args.plots_first),
        vmax=(None if args.vmax is None else float(args.vmax)),
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        gt_threshold=float(args.gt_threshold),
    )


if __name__ == "__main__":
    main()
