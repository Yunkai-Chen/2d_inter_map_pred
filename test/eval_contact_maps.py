#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluate contact maps saved by test.py (contact mode) and also
produce dataset-level micro/macro PR/ROC curves + global confusion matrices.

New:
- Per-sample metrics also include:
  * MCC (at main threshold and at 0.8)
  * Short/Medium/Long-range Precision (at main threshold and at 0.8)
  * Contact Density Deviation / CDD (at main threshold and at 0.8)
- Dataset-level global_summary.csv will additionally contain the mean of the above metrics.
- Fixed: ensure global confusion matrices use the exact same valid mask across thresholds
         so GT totals (TP+FN and FP+TN) are invariant.
"""

import argparse, os, json, csv, sys
import numpy as np

# ----------------- utils -----------------
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def finite_mask(*arrs):
    m = np.ones_like(arrs[0], dtype=bool)
    for x in arrs: m &= np.isfinite(x)
    return m

def safe_mean(x):
    return float(np.mean(x)) if len(x) else float("nan")

def top1_per_row_precision(prob, gt, mask):
    Tl, Tp = prob.shape
    correct, total = 0, 0
    for i in range(Tl):
        m_row = mask[i] & np.isfinite(prob[i]) & np.isfinite(gt[i])
        if not m_row.any(): continue
        j_local = int(np.argmax(prob[i][m_row]))
        j_global = np.arange(Tp)[m_row][j_local]
        correct += int(gt[i, j_global] > 0.5)
        total += 1
    return (float(correct / max(1, total)), int(total))

def topk_precision_global(prob, gt, mask, k):
    m = mask & np.isfinite(prob) & np.isfinite(gt)
    p = prob[m].astype(float); g = gt[m].astype(float)
    if p.size == 0 or k <= 0: return float("nan")
    k = min(k, p.size)
    idx = np.argpartition(-p, k-1)[:k]
    hits = np.sum(g[idx] > 0.5)
    return float(hits / max(1, k))

def confusion_at_threshold(prob, gt, mask, th):
    m = mask & np.isfinite(prob) & np.isfinite(gt)
    if not m.any():
        return dict(tp=0, fp=0, fn=0, tn=0, precision=float("nan"),
                    recall=float("nan"), f1=float("nan"), support=0)
    p_sel = prob[m]; g_sel = (gt[m] > 0.5)
    pred_bin = (p_sel >= th)
    tp = int(np.sum(pred_bin & g_sel))
    fp = int(np.sum(pred_bin & (~g_sel)))
    fn = int(np.sum((~pred_bin) & g_sel))
    tn = int(np.sum((~pred_bin) & (~g_sel)))
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=float(precision), recall=float(recall),
                f1=float(f1), support=int(np.sum(g_sel)))

def sweep_pr(prob, gt, mask, num=101):
    """Return PR-AUC & ROC-AUC via threshold sweep (no sklearn)."""
    m = mask & np.isfinite(prob) & np.isfinite(gt)
    if not m.any(): return float("nan"), float("nan")
    p = prob[m].astype(float); y = (gt[m] > 0.5).astype(int)
    # 单类样本 AUC 不定义
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan"), float("nan")

    ths = np.linspace(0.0, 1.0, num=num)
    prec, rec, tpr, fpr = [], [], [], []

    for th in ths:
        pred = (p >= th)
        tp = float(np.sum(pred & (y == 1)))
        fp = float(np.sum(pred & (y == 0)))
        fn = float(np.sum((~pred) & (y == 1)))
        tn = float(np.sum((~pred) & (y == 0)))
        prc = tp / (tp + fp + 1e-8)
        rcl = tp / (tp + fn + 1e-8)
        tpr.append(rcl)                       # TPR = recall
        fpr.append(fp / (fp + tn + 1e-8))     # FPR
        prec.append(prc); rec.append(rcl)

    # AUC via trapezoidal rule
    r_arr, p_arr = np.asarray(rec), np.asarray(prec)
    order = np.argsort(r_arr)
    pr_auc = float(np.trapz(y=p_arr[order], x=r_arr[order]))

    f_arr, t_arr = np.asarray(fpr), np.asarray(tpr)
    order2 = np.argsort(f_arr)
    roc_auc = float(np.trapz(y=t_arr[order2], x=f_arr[order2]))
    return pr_auc, roc_auc

# ----- dataset-level curves -----
def micro_curves(all_probs, all_labels):
    """
    all_probs, all_labels are 1D arrays (concatenated across all samples).
    Return dict with PR/ROC points & AUC.
    """
    p = all_probs.astype(float)
    y = all_labels.astype(int)
    # 去掉 NaN
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    if p.size == 0:
        return {"pr": ([],[]), "roc": ([],[]), "auprc": float("nan"), "auroc": float("nan")}

    # 如果全正或全负，AUC 不定义（但可以画一条退化曲线）
    all_pos = (y.sum() == y.size)
    all_neg = (y.sum() == 0)

    # 采用“排序累积”精确曲线（阈值按分数唯一值）
    order = np.argsort(-p)
    p_sorted = p[order]; y_sorted = y[order]

    tp_cum = np.cumsum(y_sorted == 1).astype(float)
    fp_cum = np.cumsum(y_sorted == 0).astype(float)
    P = float((y == 1).sum()); N = float((y == 0).sum())

    # PR
    recall = tp_cum / max(P, 1e-8)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

    # ROC
    tpr = recall
    fpr = fp_cum / max(N, 1e-8)

    # 压唯一
    def uniq_xy(x, y):
        x = np.asarray(x); y = np.asarray(y)
        order = np.argsort(x)
        x = x[order]; y = y[order]
        ux, idx = np.unique(x, return_index=True)
        y_max = []
        for i in range(len(ux)):
            start = idx[i]
            end = idx[i+1] if i+1 < len(idx) else len(x)
            y_max.append(np.max(y[start:end]))
        return ux, np.asarray(y_max)

    pr_x, pr_y = uniq_xy(recall, precision)
    auprc = float(np.trapz(y=pr_y, x=pr_x)) if not (all_pos or all_neg) else float("nan")

    roc_x, roc_y = uniq_xy(fpr, tpr)
    auroc = float(np.trapz(y=roc_y, x=roc_x)) if not (all_pos or all_neg) else float("nan")

    return {"pr": (pr_x.tolist(), pr_y.tolist()),
            "roc": (roc_x.tolist(), roc_y.tolist()),
            "auprc": auprc, "auroc": auroc}

def macro_curves(per_sample_curves, grid_n=1001):
    """
    per_sample_curves: list of dicts with keys
       {"pr":(recall[], precision[]), "roc":(fpr[], tpr[]), "valid":bool}
    在统一网格上插值后逐点平均，再积分得 AUC。
    """
    cs = [c for c in per_sample_curves if c.get("valid", False)]
    if len(cs) == 0:
        return {"pr": ([],[]), "roc": ([],[]), "auprc": float("nan"), "auroc": float("nan")}

    # PR: recall ∈ [0,1]
    rec_grid = np.linspace(0.0, 1.0, grid_n)
    pr_mat = []
    for c in cs:
        rx, py = np.asarray(c["pr"][0]), np.asarray(c["pr"][1])
        if rx.size < 2: continue
        order = np.argsort(rx)
        rx, py = rx[order], py[order]
        pr_interp = np.interp(rec_grid, rx, py, left=py[0], right=py[-1])
        pr_mat.append(pr_interp)
    if len(pr_mat) == 0:
        pr_mean = np.zeros_like(rec_grid)
        auprc = float("nan")
    else:
        pr_mean = np.mean(np.stack(pr_mat, axis=0), axis=0)
        auprc = float(np.trapz(y=pr_mean, x=rec_grid))

    # ROC: FPR ∈ [0,1]
    fpr_grid = np.linspace(0.0, 1.0, grid_n)
    tpr_mat = []
    for c in cs:
        fx, ty = np.asarray(c["roc"][0]), np.asarray(c["roc"][1])
        if fx.size < 2: continue
        order = np.argsort(fx)
        fx, ty = fx[order], ty[order]
        tpr_interp = np.interp(fpr_grid, fx, ty, left=ty[0], right=ty[-1])
        tpr_mat.append(tpr_interp)
    if len(tpr_mat) == 0:
        tpr_mean = np.zeros_like(fpr_grid)
        auroc = float("nan")
    else:
        tpr_mean = np.mean(np.stack(tpr_mat, axis=0), axis=0)
        auroc = float(np.trapz(y=tpr_mean, x=fpr_grid))

    return {"pr": (rec_grid.tolist(), pr_mean.tolist()),
            "roc": (fpr_grid.tolist(), tpr_mean.tolist()),
            "auprc": auprc, "auroc": auroc}

# ----------------- extra metrics (requested) -----------------
def mcc_from_counts(tp, fp, fn, tn):
    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-8
    return float((tp * tn - fp * fn) / denom)

def mcc_at_threshold(prob, gt, mask, th):
    cm = confusion_at_threshold(prob, gt, mask, th)
    return mcc_from_counts(cm["tp"], cm["fp"], cm["fn"], cm["tn"])

def range_precision(prob, gt, mask, cutoff=0.5):
    """Short/Medium/Long-range precision by |i-j| bins."""
    Tl, Tp = prob.shape
    i, j = np.meshgrid(np.arange(Tl), np.arange(Tp), indexing='ij')
    sep = np.abs(i - j)
    p_bin = (prob >= cutoff)
    g_bin = (gt > 0.5)
    m = mask & np.isfinite(prob) & np.isfinite(gt)

    def prec(sel):
        selm = m & sel
        if not np.any(selm):
            return float('nan')
        tp = np.sum(p_bin[selm] & g_bin[selm])
        fp = np.sum(p_bin[selm] & (~g_bin[selm]))
        denom = tp + fp
        return float(tp / (denom + 1e-8)) if denom > 0 else float('nan')

    short = prec(sep < 12)
    medium = prec((sep >= 12) & (sep < 24))
    long = prec(sep >= 24)
    return short, medium, long

def cdd_threshold(prob, gt, mask, cutoff=0.5):
    """Contact Density Deviation at a threshold: (pred_count - true_count) / true_count."""
    m = mask & np.isfinite(prob) & np.isfinite(gt)
    pred_c = int(np.sum(prob[m] >= cutoff))
    true_c = int(np.sum(gt[m] > 0.5))
    if true_c == 0:
        return float("nan")
    return float((pred_c - true_c) / (true_c + 1e-8))

# ----------------- IO helpers -----------------
def load_contact_npz(path):
    """
    Expected keys from your test.py (contact mode):
      - 'prob' OR 'scores' (logits). If only 'scores' exist, we'll sigmoid them.
      - 'gt_map' (0/1 contact labels)
      - 'pair_mask' (optional)
    """
    data = np.load(path, allow_pickle=False)
    keys = set(data.files)
    prob = None
    if "prob" in keys:
        raw = data["prob"]
        # 如果 prob 全在 0.5~1 之间，再次反 sigmoid
       
          
        raw = np.log(raw / (1 - raw + 1e-8))
        
        prob = raw

    #elif "scores" in keys:
    #    prob = sigmoid(data["scores"])
    elif "distance" in keys:
        prob = None
    gt = data["gt_map"] if "gt_map" in keys else None
    pm = data["pair_mask"].astype(bool) if "pair_mask" in keys else None
    return prob, gt, pm, list(keys)

def save_heatmap(mat, out_path, title="", vmin=None, vmax=None):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib not available, skip heatmap: {e}")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    if vmin is None and vmax is None:
        plt.imshow(mat, origin="upper", aspect="auto")
    else:
        plt.imshow(mat, origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
    plt.title(title); plt.colorbar(fraction=0.03, pad=0.03)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def plot_curve(x, y, xlabel, ylabel, title, out_path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib not available, skip curve: {e}")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(5,4))
    plt.plot(x, y)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def plot_confusion(tp, fp, fn, tn, title, out_path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[warn] matplotlib not available, skip confusion: {e}")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # 行: GT, 列: Pred
    mat = np.array([[tp, fn],[fp, tn]], dtype=float)
    plt.figure(figsize=(4,4))
    im = plt.imshow(mat, cmap='Blues', vmin=0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    for (i,j), v in np.ndenumerate(mat):
        plt.text(j, i, f"{int(v)}", ha='center', va='center', fontsize=12)
    plt.xticks([0,1], ["Pred=1","Pred=0"])
    plt.yticks([0,1], ["GT=1","GT=0"])
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

# ----------------- eval one sample -----------------
def eval_one(prob, gt, mask, th=0.5):
    assert prob.shape == gt.shape, f"shape mismatch {prob.shape} vs {gt.shape}"
    base_mask = finite_mask(prob, gt)
    if mask is not None:
        base_mask &= mask.astype(bool)

    # === confusion-based metrics at requested th ===
    cm_main = confusion_at_threshold(prob, gt, base_mask, th)

    # === confusion at fixed thresholds 0.5 & 0.8 (always saved per-sample) ===
    cm050 = confusion_at_threshold(prob, gt, base_mask, 0.5)
    cm080 = confusion_at_threshold(prob, gt, base_mask, 0.8)

    # === threshold sweeps -> AUPRC / AUROC ===
    pr_auc, roc_auc = sweep_pr(prob, gt, base_mask)

    # === top-k variants ===
    Tl, Tp = prob.shape
    L = min(Tl, Tp)
    top1_row, top1_n = top1_per_row_precision(prob, gt, base_mask)
    topL  = topk_precision_global(prob, gt, base_mask, k=L)
    topL2 = topk_precision_global(prob, gt, base_mask, k=max(1, L//2))

    # === per-sample curves (for macro aggregation) ===
    m = base_mask & np.isfinite(prob) & np.isfinite(gt)
    y_flat = (gt[m] > 0.5).astype(int)
    valid_auc = not (y_flat.size == 0 or y_flat.sum() == 0 or y_flat.sum() == y_flat.size)
    pr_x = []; pr_y = []; roc_x = []; roc_y = []
    if valid_auc:
        ths = np.linspace(0.0, 1.0, num=400)
        precision, recall, fpr, tpr = [], [], [], []
        p = prob[m].astype(float)
        for t in ths:
            pred = (p >= t)
            tp = float(np.sum(pred & (y_flat == 1)))
            fp = float(np.sum(pred & (y_flat == 0)))
            fn = float(np.sum((~pred) & (y_flat == 1)))
            tn = float(np.sum((~pred) & (y_flat == 0)))
            prc = tp / (tp + fp + 1e-8)
            rcl = tp / (tp + fn + 1e-8)
            precision.append(prc); recall.append(rcl)
            tpr.append(rcl); fpr.append(fp / (fp + tn + 1e-8))
        pr_x, pr_y = np.asarray(recall), np.asarray(precision)
        roc_x, roc_y = np.asarray(fpr), np.asarray(tpr)

    # === extra metrics at requested th and at 0.8 ===
    # MCC
    mcc_main = mcc_from_counts(cm_main["tp"], cm_main["fp"], cm_main["fn"], cm_main["tn"])
    mcc_08   = mcc_at_threshold(prob, gt, base_mask, th=0.8)

    # Range precisions
    short_p_main, med_p_main, long_p_main = range_precision(prob, gt, base_mask, cutoff=th)
    short_p_08,   med_p_08,   long_p_08   = range_precision(prob, gt, base_mask, cutoff=0.8)

    # CDD
    cdd_main = cdd_threshold(prob, gt, base_mask, cutoff=th)
    cdd_08   = cdd_threshold(prob, gt, base_mask, cutoff=0.8)

    return {
        "shape_Tl_Tp": [int(Tl), int(Tp)],
        "prob_th": float(th),

        "contact_precision": cm_main["precision"],
        "contact_recall":    cm_main["recall"],
        "contact_f1":        cm_main["f1"],
        "contact_tp": cm_main["tp"], "contact_fp": cm_main["fp"],
        "contact_fn": cm_main["fn"], "contact_support": cm_main["support"],

        "precision_t050": cm050["precision"], "recall_t050": cm050["recall"], "f1_t050": cm050["f1"],
        "tp_t050": cm050["tp"], "fp_t050": cm050["fp"], "fn_t050": cm050["fn"], "tn_t050": cm050["tn"],
        "precision_t080": cm080["precision"], "recall_t080": cm080["recall"], "f1_t080": cm080["f1"],
        "tp_t080": cm080["tp"], "fp_t080": cm080["fp"], "fn_t080": cm080["fn"], "tn_t080": cm080["tn"],

        "PR_AUC": pr_auc, "ROC_AUC": roc_auc,

        "Top1_per_row_precision": top1_row,
        "Top1_per_row_count": top1_n,
        "TopL_precision": topL,
        "TopL2_precision": topL2,

        # New per-sample metrics:
        "MCC_at_main_th": mcc_main,
        "MCC_at_0p8":     mcc_08,

        "Short_range_P_at_main_th": short_p_main,
        "Medium_range_P_at_main_th": med_p_main,
        "Long_range_P_at_main_th": long_p_main,

        "Short_range_P_at_0p8": short_p_08,
        "Medium_range_P_at_0p8": med_p_08,
        "Long_range_P_at_0p8": long_p_08,

        "CDD_at_main_th": cdd_main,
        "CDD_at_0p8":     cdd_08,

        "_macro_curves": {
            "valid": bool(valid_auc),
            "pr": (pr_x.tolist() if len(pr_x) else [], pr_y.tolist() if len(pr_y) else []),
            "roc": (roc_x.tolist() if len(roc_x) else [], roc_y.tolist() if len(roc_y) else []),
        }
    }

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Batch evaluate contact maps saved by test.py (contact mode).")
    ap.add_argument("--root", required=True, help="e.g. outputs/test_contact_mlp")
    ap.add_argument("--subdir", default="pred_npz", help="subfolder containing *.npz")
    ap.add_argument("--prob-th", type=float, default=0.5, help="probability threshold for 0/1 metrics (per-sample & global)")
    ap.add_argument("--save-figs", action="store_true", help="save per-sample heatmaps")
    ap.add_argument("--summary-csv", default="summary_contact.csv", help="output CSV under root")
    ap.add_argument("--allow-crop", action="store_true",
                    help="if shapes differ, crop both to min(Tl) x min(Tp) instead of skipping")
    ap.add_argument("--debug", action="store_true", help="print per-sample valid counts & GT distribution")
    args = ap.parse_args()

    in_dir = os.path.join(args.root, args.subdir)
    if not os.path.isdir(in_dir):
        print(f"[error] not a directory: {in_dir}", file=sys.stderr); sys.exit(1)

    figs_dir = os.path.join(args.root, "figs_contact_eval")
    metrics_dir = os.path.join(args.root, "metrics_contact_eval")
    curves_dir = os.path.join(args.root, "summary_curves")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(curves_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(in_dir) if f.endswith(".npz")])
    if not files:
        print(f"[warn] no .npz found under {in_dir}"); sys.exit(0)

    rows, skipped = [], []

    # For dataset-level micro curves & global confusion matrices
    all_probs = []
    all_labels = []

    # For macro curves, accumulate per-sample curves
    per_sample_curves = []

    # For dataset-level averages of new metrics
    mcc_main_list, mcc_08_list = [], []
    short_main_list, med_main_list, long_main_list = [], [], []
    short_08_list, med_08_list, long_08_list = [], [], []
    cdd_main_list, cdd_08_list = [], []

    for fname in files:
        path = os.path.join(in_dir, fname)
        base = os.path.splitext(fname)[0]
        prob, gt, pm, keys = load_contact_npz(path)

        if (prob is None) or (gt is None):
            skipped.append({"id": base, "reason": f"missing prob/scores or gt_map (keys={keys})"})
            print(f"[skip] {fname}: missing prob/scores or gt_map (keys={keys})")
            continue

        # shape alignment
        if prob.shape != gt.shape:
            if args.allow_crop:
                Tl = min(prob.shape[0], gt.shape[0])
                Tp = min(prob.shape[1], gt.shape[1])
                prob = prob[:Tl, :Tp]; gt = gt[:Tl, :Tp]
                if pm is not None: pm = pm[:Tl, :Tp]
                print(f"[crop] {base}: cropped to {prob.shape}")
            else:
                skipped.append({"id": base, "reason": f"shape mismatch {prob.shape} vs {gt.shape}"})
                print(f"[error] {fname}: shape mismatch {prob.shape} vs {gt.shape}")
                continue

        try:
            metrics = eval_one(prob, gt, pm, th=args.prob_th)
        except Exception as e:
            skipped.append({"id": base, "reason": f"eval error: {e}"})
            print(f"[error] {fname}: {e}")
            continue

        # write per-sample json
        with open(os.path.join(metrics_dir, base + ".json"), "w") as f:
            json.dump({"id": base, **metrics}, f, indent=2)

        # optional figs (per-sample)
        if args.save_figs:
            m = (np.isfinite(prob) & np.isfinite(gt) & (pm.astype(bool) if pm is not None else True))
            save_heatmap(prob, os.path.join(figs_dir, base + "_prob.png"),
                         title=f"{base}: pred prob", vmin=0.0, vmax=1.0)
            save_heatmap(gt.astype(np.float32), os.path.join(figs_dir, base + "_gt.png"),
                         title=f"{base}: GT contact", vmin=0.0, vmax=1.0)
            abs_err = np.abs(prob - gt.astype(np.float32))
            save_heatmap(np.where(m, abs_err, np.nan),
                         os.path.join(figs_dir, base + "_abs_err.png"),
                         title=f"{base}: |prob - GT|", vmin=0.0, vmax=1.0)
            binmap = (prob >= args.prob_th).astype(np.float32)
            save_heatmap(binmap, os.path.join(figs_dir, base + f"_bin_th{args.prob_th}.png"),
                         title=f"{base}: pred>= {args.prob_th:.2f}", vmin=0.0, vmax=1.0)

        # collect table row
        row = {"id": base}; row.update(metrics); rows.append(row)
        print(f"[done] {base}  F1={metrics['contact_f1']:.3f}  AUPRC={metrics['PR_AUC']:.3f}  AUROC={metrics['ROC_AUC']:.3f}")

        # ====== Accumulate for dataset-level curves ======
        mflat = finite_mask(prob, gt)
        if pm is not None: mflat &= pm.astype(bool)
        p_vec = prob[mflat].astype(float)
        y_vec = (gt[mflat] > 0.5).astype(int)
        if args.debug:
            print(f"[dbg] {base}  valid={p_vec.size}  GT1={int((y_vec==1).sum())}  GT0={int((y_vec==0).sum())}")
        if p_vec.size > 0:
            all_probs.append(p_vec)
            all_labels.append(y_vec)

        # macro per-sample curves（只存可用的）
        per_sample_curves.append({
            "valid": bool(metrics["_macro_curves"]["valid"]),
            "pr": tuple(metrics["_macro_curves"]["pr"]),
            "roc": tuple(metrics["_macro_curves"]["roc"]),
        })

        # accumulate new metrics for dataset-level averages
        mcc_main_list.append(metrics["MCC_at_main_th"])
        mcc_08_list.append(metrics["MCC_at_0p8"])

        short_main_list.append(metrics["Short_range_P_at_main_th"])
        med_main_list.append(metrics["Medium_range_P_at_main_th"])
        long_main_list.append(metrics["Long_range_P_at_main_th"])

        short_08_list.append(metrics["Short_range_P_at_0p8"])
        med_08_list.append(metrics["Medium_range_P_at_0p8"])
        long_08_list.append(metrics["Long_range_P_at_0p8"])

        cdd_main_list.append(metrics["CDD_at_main_th"])
        cdd_08_list.append(metrics["CDD_at_0p8"])

    # summary CSV (per-sample)
    if rows:
        csv_path = os.path.join(args.root, args.summary_csv)
        keys_csv = ["id"] + sorted({k for r in rows for k in r.keys() if k != "id" and not k.startswith("_")})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys_csv); w.writeheader(); w.writerows([{k:v for k,v in r.items() if k in keys_csv} for r in rows])
        print(f"[saved] CSV summary -> {csv_path}")
    else:
        print("[warn] no samples evaluated")

    # ====== Dataset-level curves & confusion matrices ======
    # Micro + Global Confusions (once, on the same universe)
    if len(all_probs) > 0:
        all_probs_1d = np.concatenate(all_probs, axis=0)
        all_labels_1d = np.concatenate(all_labels, axis=0)

        # ---- Unified valid mask (FIX) ----
        m_all = np.isfinite(all_probs_1d) & np.isfinite(all_labels_1d)
        all_probs_1d = all_probs_1d[m_all].astype(float)
        all_labels_1d = all_labels_1d[m_all].astype(int)

        def confusion_from_vectors(p_vec, y_vec, th: float):
            pred_bin = (p_vec >= th)
            y_pos = (y_vec == 1)
            y_neg = ~y_pos
            tp = int(np.sum(pred_bin & y_pos))
            fp = int(np.sum(pred_bin & y_neg))
            fn = int(np.sum((~pred_bin) & y_pos))
            tn = int(np.sum((~pred_bin) & y_neg))
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2 * prec * rec / (prec + rec + 1e-8)
            # sanity invariant
            assert tp + fn == int(np.sum(y_pos)), "GT=1 mismatch across thresholds"
            assert fp + tn == int(np.sum(y_neg)), "GT=0 mismatch across thresholds"
            return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=float(prec), recall=float(rec), f1=float(f1))

        # —— 全局一次性计算 —— #
        global_counts_at = {}
        for th in (0.5, 0.8):
            global_counts_at[th] = confusion_from_vectors(all_probs_1d, all_labels_1d, th)

        micro = micro_curves(all_probs_1d, all_labels_1d)

        # 保存曲线点
        with open(os.path.join(curves_dir, "global_micro_points.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["type","x","y"])
            for x,y in zip(micro["pr"][0], micro["pr"][1]): w.writerow(["PR", x, y])
            for x,y in zip(micro["roc"][0], micro["roc"][1]): w.writerow(["ROC", x, y])

        # 画图
        plot_curve(micro["pr"][0], micro["pr"][1], "Recall", "Precision",
                   f"Micro PR (AUPRC={micro['auprc']:.3f})",
                   os.path.join(curves_dir, "global_micro_pr.png"))
        plot_curve(micro["roc"][0], micro["roc"][1], "FPR", "TPR",
                   f"Micro ROC (AUROC={micro['auroc']:.3f})",
                   os.path.join(curves_dir, "global_micro_roc.png"))

        # Global confusion matrices at thresholds 0.5 & 0.8
        gcm_rows = []
        for th, c in global_counts_at.items():
            tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2*prec*rec / (prec + rec + 1e-8)
            gcm_rows.append({"type": f"GLOBAL_confusion@{th}",
                             "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                             "precision": float(prec), "recall": float(rec), "f1": float(f1)})
            # 图
            plot_confusion(tp, fp, fn, tn,
                           title=f"Global Confusion @ {th}",
                           out_path=os.path.join(curves_dir, f"global_confusion_th{th}.png"))
            print(f"[global @ {th}] TP={tp} FP={fp} FN={fn} TN={tn} | P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

        # 同时另存混淆矩阵统计
        with open(os.path.join(curves_dir, "global_confusions.csv"), "w", newline="") as f:
            keys = ["type","tp","fp","fn","tn","precision","recall","f1"]
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(gcm_rows)

    else:
        micro = {"auprc": float("nan"), "auroc": float("nan")}

    # Macro
    macro = macro_curves(per_sample_curves, grid_n=1001)
    # 保存曲线点
    with open(os.path.join(curves_dir, "global_macro_points.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["type","x","y"])
        for x,y in zip(macro["pr"][0], macro["pr"][1]): w.writerow(["PR", x, y])
        for x,y in zip(macro["roc"][0], macro["roc"][1]): w.writerow(["ROC", x, y])
    # 画图
    plot_curve(macro["pr"][0], macro["pr"][1], "Recall", "Precision",
               f"Macro PR (AUPRC={macro['auprc']:.3f})",
               os.path.join(curves_dir, "global_macro_pr.png"))
    plot_curve(macro["roc"][0], macro["roc"][1], "FPR", "TPR",
               f"Macro ROC (AUROC={macro['auroc']:.3f})",
               os.path.join(curves_dir, "global_macro_roc.png"))

    # 追加 GLOBAL 行到 summary CSV（便于统一查看）
    global_csv = os.path.join(curves_dir, "global_summary.csv")
    with open(global_csv, "w", newline="") as f:
        fieldnames = [
            "name","AUPRC_micro","AUROC_micro","AUPRC_macro","AUROC_macro",
            # new dataset-level means:
            "MCC_mean_at_main_th","MCC_mean_at_0p8",
            "ShortP_mean_at_main_th","MedP_mean_at_main_th","LongP_mean_at_main_th",
            "ShortP_mean_at_0p8","MedP_mean_at_0p8","LongP_mean_at_0p8",
            "CDD_mean_at_main_th","CDD_mean_at_0p8"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        w.writerow({
            "name":"GLOBAL",
            "AUPRC_micro": micro["auprc"],
            "AUROC_micro": micro["auroc"],
            "AUPRC_macro": macro["auprc"],
            "AUROC_macro": macro["auroc"],

            "MCC_mean_at_main_th": safe_mean([x for x in mcc_main_list if np.isfinite(x)]),
            "MCC_mean_at_0p8":     safe_mean([x for x in mcc_08_list if np.isfinite(x)]),

            "ShortP_mean_at_main_th": safe_mean([x for x in short_main_list if np.isfinite(x)]),
            "MedP_mean_at_main_th":   safe_mean([x for x in med_main_list if np.isfinite(x)]),
            "LongP_mean_at_main_th":  safe_mean([x for x in long_main_list if np.isfinite(x)]),

            "ShortP_mean_at_0p8": safe_mean([x for x in short_08_list if np.isfinite(x)]),
            "MedP_mean_at_0p8":   safe_mean([x for x in med_08_list if np.isfinite(x)]),
            "LongP_mean_at_0p8":  safe_mean([x for x in long_08_list if np.isfinite(x)]),

            "CDD_mean_at_main_th": safe_mean([x for x in cdd_main_list if np.isfinite(x)]),
            "CDD_mean_at_0p8":     safe_mean([x for x in cdd_08_list if np.isfinite(x)]),
        })

    # skip summary
    if skipped:
        with open(os.path.join(args.root, "skip_summary.json"), "w") as f:
            json.dump({"total": len(files),
                       "evaluated": len(rows),
                       "skipped": skipped}, f, indent=2)
        print(f"[summary] total={len(files)}  evaluated={len(rows)}  skipped={len(skipped)}")
        print(f"[saved] skip summary -> {os.path.join(args.root,'skip_summary.json')}")

    print(f"[done] global curves & confusions saved to: {curves_dir}")

if __name__ == "__main__":
    main()
    
