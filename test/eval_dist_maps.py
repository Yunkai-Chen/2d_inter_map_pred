#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json, csv, sys
import numpy as np

# ---------- metrics ----------
def finite_mask(*arrs):
    m = np.ones_like(arrs[0], dtype=bool)
    for x in arrs:
        m &= np.isfinite(x)
    return m

def mae_rmse(pred, gt, mask):
    d = pred[mask] - gt[mask]
    mae = float(np.mean(np.abs(d))) if d.size else float("nan")
    rmse = float(np.sqrt(np.mean(d**2))) if d.size else float("nan")
    return mae, rmse

def pearsonr_vec(x, y):
    if x.size < 2: return float("nan")
    xm, ym = x.mean(), y.mean()
    xs, ys = x - xm, y - ym
    num = float((xs * ys).sum())
    den = float(np.sqrt((xs**2).sum()) * np.sqrt((ys**2).sum()))
    return num / den if den > 0 else float("nan")

def spearmanr_vec(x, y):
    if x.size < 2: return float("nan")
    def rankdata(v):
        order = np.argsort(v)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(v)+1, dtype=float)
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, ranks)
        avg = sums / cnt
        return avg[inv]
    rx, ry = rankdata(x), rankdata(y)
    return pearsonr_vec(rx, ry)

def contact_metrics_from_dist(pred_dist, gt_dist, mask, thr=8.0):
    m = mask & np.isfinite(pred_dist) & np.isfinite(gt_dist)
    if not m.any():
        return dict(precision=float("nan"), recall=float("nan"), f1=float("nan"),
                    tp=0, fp=0, fn=0, support=0)
    pc = (pred_dist < thr)
    gc = (gt_dist   < thr)
    tp = int(np.sum(pc & gc & m))
    fp = int(np.sum(pc & (~gc) & m))
    fn = int(np.sum((~pc) & gc & m))
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2*precision*recall / (precision + recall + 1e-8)
    return dict(precision=float(precision), recall=float(recall), f1=float(f1),
                tp=tp, fp=fp, fn=fn, support=int(np.sum(gc & m)))

def topk_precision_global_from_dist(pred_dist, gt_dist, mask, k, thr=8.0):
    m = mask & np.isfinite(pred_dist) & np.isfinite(gt_dist)
    if not m.any() or k <= 0: return float("nan")
    p = pred_dist[m].astype(float); g = gt_dist[m].astype(float)
    idx = np.argsort(p)[:min(k, p.size)]  # 距离越小越“正”
    hits = np.sum(g[idx] < thr)
    return float(hits / max(1, len(idx)))

def top1_per_row_precision_from_dist(pred_dist, gt_dist, mask, thr=8.0):
    Tl, Tp = pred_dist.shape
    correct = 0; total = 0
    for i in range(Tl):
        m_row = mask[i] & np.isfinite(pred_dist[i]) & np.isfinite(gt_dist[i])
        if not m_row.any(): continue
        j_local = int(np.argmin(pred_dist[i][m_row]))
        j_global = np.arange(Tp)[m_row][j_local]
        correct += int(gt_dist[i, j_global] < thr)
        total += 1
    return float(correct / max(1, total)), int(total)

# ---------- IO ----------
def load_pred_gt_npz(path):
    """
    适配你当前 test 的输出：
    keys 里常见：distance, pair_mask, scores, key, gt_map
    也顺手支持一些兜底键名。
    """
    data = np.load(path, allow_pickle=False)
    keys = set(data.files)
    # pred distance
    pred = None
    for k in ["distance", "pred_dist", "pred", "pred_map", "pred_distance"]:
        if k in keys:
            pred = data[k]; break
    # gt distance
    gt = None
    for k in ["gt_map", "gt_dist", "gt", "distance_map", "target"]:
        if k in keys:
            gt = data[k]; break
    # pair mask（可选）
    pm = None
    for k in ["pair_mask", "mask", "valid_mask"]:
        if k in keys:
            pm = data[k].astype(bool); break
    return pred, gt, pm, list(keys)

def eval_one_sample(pred_dist, gt_dist, pair_mask, thr):
    assert pred_dist.shape == gt_dist.shape, f"shape mismatch {pred_dist.shape} vs {gt_dist.shape}"
    base_mask = finite_mask(pred_dist, gt_dist)
    if pair_mask is not None:
        base_mask &= pair_mask.astype(bool)
    mae, rmse = mae_rmse(pred_dist, gt_dist, base_mask)
    r   = pearsonr_vec(pred_dist[base_mask], gt_dist[base_mask])
    rho = spearmanr_vec(pred_dist[base_mask], gt_dist[base_mask])
    cm  = contact_metrics_from_dist(pred_dist, gt_dist, base_mask, thr=thr)
    Tl, Tp = pred_dist.shape
    L = min(Tl, Tp)
    topL  = topk_precision_global_from_dist(pred_dist, gt_dist, base_mask, k=L,           thr=thr)
    topL2 = topk_precision_global_from_dist(pred_dist, gt_dist, base_mask, k=max(1,L//2), thr=thr)
    top1_row, top1_row_n = top1_per_row_precision_from_dist(pred_dist, gt_dist, base_mask, thr=thr)
    return {
        "shape_Tl_Tp": [int(Tl), int(Tp)],
        "thr_A": float(thr),
        "MAE_A": mae, "RMSE_A": rmse,
        "Pearson_r": r, "Spearman_rho": rho,
        "contact_precision": cm["precision"],
        "contact_recall": cm["recall"],
        "contact_f1": cm["f1"],
        "contact_tp": cm["tp"], "contact_fp": cm["fp"], "contact_fn": cm["fn"],
        "contact_support": cm["support"],
        "TopL_precision": topL,
        "TopL2_precision": topL2,
        "Top1_per_row_precision": top1_row,
        "Top1_per_row_count": top1_row_n
    }

def save_diff_heatmap(pred, gt, mask, out_path, vmin=-8, vmax=8, title=None):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib not available, skip heatmap: {e}")
        return
    diff = np.where(mask, pred - gt, np.nan)
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 2.5))
    im = plt.imshow(diff, aspect='auto', interpolation='nearest',
                    vmin=vmin, vmax=vmax, cmap='coolwarm')
    plt.title(title or "pred - gt (Å)")
    plt.colorbar(im, fraction=0.02, pad=0.02)
    plt.xlabel("Protein index (Tp)")
    plt.ylabel("Peptide index (Tl)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Batch evaluate distance maps saved as NPZ.")
    ap.add_argument("--root", required=True, help="e.g. outputs/test_bins_dist")
    ap.add_argument("--subdir", default="pred_npz", help="subfolder containing *.npz")
    ap.add_argument("--thr", type=float, default=8.0, help="contact threshold in Å")
    ap.add_argument("--save-figs", action="store_true", help="save per-sample diff heatmaps")
    ap.add_argument("--summary-csv", default="summary_metrics.csv", help="output CSV under root")
    args = ap.parse_args()

    in_dir = os.path.join(args.root, args.subdir)
    if not os.path.isdir(in_dir):
        print(f"[error] not a directory: {in_dir}", file=sys.stderr); sys.exit(1)

    figs_dir = os.path.join(args.root, "figs")
    metrics_dir = os.path.join(args.root, "metrics")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(in_dir) if f.endswith(".npz")])
    if not files:
        print(f"[warn] no .npz found under {in_dir}"); sys.exit(0)

    rows = []
    for fname in files:
        path = os.path.join(in_dir, fname)
        base = os.path.splitext(fname)[0]
        pred, gt, pm, keys = load_pred_gt_npz(path)
        if pred is None or gt is None:
            print(f"[skip] {fname}: missing 'distance' or 'gt_map' (keys={keys})")
            continue
        try:
            metrics = eval_one_sample(pred, gt, pm, thr=args.thr)
        except Exception as e:
            print(f"[error] {fname}: {e}")
            continue

        # per-sample json
        with open(os.path.join(metrics_dir, base + ".json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # heatmap
        if args.save_figs:
            mask = (np.isfinite(pred) & np.isfinite(gt) & (pm.astype(bool) if pm is not None else True))
            save_diff_heatmap(pred, gt, mask, os.path.join(figs_dir, base + "_diff.png"),
                              vmin=-8, vmax=8, title=f"{base}: pred - gt (Å)")

        row = {"id": base}; row.update(metrics); rows.append(row)
        print(f"[done] {base}  MAE={metrics['MAE_A']:.2f}  F1={metrics['contact_f1']:.3f}")

    # CSV summary
    if rows:
        csv_path = os.path.join(args.root, args.summary_csv)
        keys = sorted({k for r in rows for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"[saved] CSV summary -> {csv_path}")
    else:
        print("[warn] no samples evaluated")

if __name__ == "__main__":
    main()
