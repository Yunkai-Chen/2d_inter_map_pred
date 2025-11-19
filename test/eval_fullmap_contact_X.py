#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate full-map contact predictions (with region partition)
==============================================================
- Reads per-sample .npz files (from test_fullmap.py)
- Reads meta JSON containing peptide/protein lengths and order
- Computes metrics for:
    * overall (full map)
    * inter-region (peptide–protein)
    * intra-peptide
    * intra-protein
    * intra-both = intra_peptide ∪ intra_protein
- Supports pairing modes:
    * unique_upper (default): only count each residue pair once (upper triangle)
    * full: count full map, with symmetric inter region
- Outputs per-sample JSON/CSV + global summaries
- NEW:
    * Per-sample figs: add full & intra-both (pred/gt + optional abs-diff)
    * Global per-region confusion@{0.5,0.8}, micro/macro PR/ROC (PNG + CSV points)
    * Mirror curves both to global_curves/ and summary_curves/

Usage:
  python eval_fullmap_contact.py \
      --root outputs/fullmap_test_V2 \
      --subdir pred_npz \
      --meta-json data/full_distance_maps_v2/full_map_cb.json \
      --prob-th 0.8 \
      --pairing unique_upper \
      --save-figs
"""

import os, sys, json, csv, argparse, re
import numpy as np
import matplotlib.pyplot as plt


# ===========================
# Utility functions
# ===========================
def top1_per_row_precision(prob, gt, mask):
    Tl, Tp = prob.shape
    correct, total = 0, 0
    for i in range(Tl):
        row_mask = mask[i] & np.isfinite(prob[i]) & np.isfinite(gt[i])
        if not np.any(row_mask):
            continue
        j_local = np.argmax(prob[i][row_mask])
        j_global = np.arange(Tp)[row_mask][j_local]
        correct += int(gt[i, j_global] > 0.5)
        total += 1
    return float(correct / max(1, total))

def topk_precision_global(prob, gt, mask, k):
    m = mask & np.isfinite(prob) & np.isfinite(gt)
    p = prob[m].astype(float); g = gt[m].astype(float)
    if p.size == 0 or k <= 0:
        return float("nan")
    k = min(k, p.size)
    idx = np.argpartition(-p, k-1)[:k]
    hits = np.sum(g[idx] > 0.5)
    return float(hits / max(1, k))

def range_precision(prob, gt, mask, cutoff=0.5):
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
        return float(tp / max(tp + fp, 1e-8))
    
    short = prec(sep < 12)
    med   = prec((sep >= 12) & (sep < 24))
    long  = prec(sep >= 24)
    return short, med, long

def cdd_threshold(prob, gt, mask, cutoff=0.5):
    m = mask & np.isfinite(prob) & np.isfinite(gt)
    pred_c = int(np.sum(prob[m] >= cutoff))
    true_c = int(np.sum(gt[m] > 0.5))
    if true_c == 0:
        return float("nan")
    return float((pred_c - true_c) / (true_c + 1e-8))

def meta_key_from_filename(fname: str) -> str:
    """Turn npz filename into metadata key: only strip trailing `_nomutation`."""
    key = os.path.splitext(os.path.basename(fname))[0]
    key = re.sub(r'_nomutation$', '', key, flags=re.IGNORECASE)
    return key


def safe_div(a, b):
    return a / (b + 1e-8)


def finite_mask(*arrs):
    m = np.ones_like(arrs[0], dtype=bool)
    for x in arrs:
        m &= np.isfinite(x)
    return m


def mcc(tp, fp, fn, tn):
    """Matthews correlation with float64 to prevent overflow."""
    tp, fp, fn, tn = map(np.float64, [tp, fp, fn, tn])
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-8
    return (tp * tn - fp * fn) / denom


def sweep_pr(prob, gt, mask):
    """Compute PR-AUC and ROC-AUC with stable sorting + precision envelope."""
    prob = np.asarray(prob).ravel()
    gt = (np.asarray(gt) > 0.5).astype(int).ravel()
    mask = np.asarray(mask, dtype=bool).ravel()

    m = mask & np.isfinite(prob) & np.isfinite(gt)
    p = prob[m].astype(float)
    y = gt[m].astype(int)
    if p.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return float("nan"), float("nan")

    # sort by descending prob (same as sklearn)
    order = np.argsort(-p)
    p_sorted = p[order]
    y_sorted = y[order]

    # cumulative TP / FP
    tp_cum = np.cumsum(y_sorted == 1).astype(float)
    fp_cum = np.cumsum(y_sorted == 0).astype(float)
    P = float((y == 1).sum())
    N = float((y == 0).sum())

    recall = tp_cum / max(P, 1e-8)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

    # ---- precision envelope (monotone decreasing) ----
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    # ensure recall monotonic increasing before integration
    order_r = np.argsort(recall)
    recall = recall[order_r]
    precision = precision[order_r]

    # AUPRC
    auprc = float(np.trapz(y=precision, x=recall))

    # ROC
    tpr = tp_cum / max(P, 1e-8)
    fpr = fp_cum / max(N, 1e-8)
    fpr, tpr = np.array(fpr), np.array(tpr)
    order_f = np.argsort(fpr)
    fpr, tpr = fpr[order_f], tpr[order_f]
    auroc = float(np.trapz(y=tpr, x=fpr))

    return auprc, auroc



def curve_points(prob, gt, mask, num=400):
    """Return per-sample PR/ROC curve points for macro aggregation."""
    prob = np.asarray(prob).ravel()
    gt = (np.asarray(gt) > 0.5).astype(int).ravel()
    mask = np.asarray(mask, dtype=bool).ravel()
    m = mask & np.isfinite(prob) & np.isfinite(gt)
    p = prob[m].astype(float); y = gt[m].astype(int)
    if p.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return dict(valid=False, pr=([],[]), roc=([],[]))
    ths = np.linspace(0.0, 1.0, num=num)
    prec, rec, fpr, tpr = [], [], [], []
    for th in ths:
        pred = (p >= th)
        tp = float(np.sum(pred & (y == 1)))
        fp = float(np.sum(pred & (y == 0)))
        fn = float(np.sum((~pred) & (y == 1)))
        tn = float(np.sum((~pred) & (y == 0)))
        prc = safe_div(tp, tp + fp); rcl = safe_div(tp, tp + fn)
        tpr.append(rcl); fpr.append(safe_div(fp, fp + tn))
        prec.append(prc); rec.append(rcl)
    return dict(valid=True, pr=(rec, prec), roc=(fpr, tpr))


def save_heatmap(mat, out_path, title=""):
    plt.figure(figsize=(4, 4))
    plt.imshow(mat, cmap="viridis", origin="upper", vmin=0, vmax=1)
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


# ===========================
# Region-level evaluation
# ===========================
def region_metrics(prob, gt, mask, th=0.5):
    """Compute P/R/F1/MCC/AUC and extra metrics restricted to a region mask."""
    if not np.any(mask):
        return dict(precision=np.nan, recall=np.nan, f1=np.nan, mcc=np.nan, auprc=np.nan, auroc=np.nan,
                    top1=np.nan, topL=np.nan, topL2=np.nan,
                    range_short=np.nan, range_med=np.nan, range_long=np.nan, cdd=np.nan)

    p_sel = prob[mask]
    y_sel = (gt[mask] > 0.5)
    pred_sel = (p_sel >= th)

    tp = np.sum(pred_sel & y_sel)
    fp = np.sum(pred_sel & (~y_sel))
    fn = np.sum((~pred_sel) & y_sel)
    tn = np.sum((~pred_sel) & (~y_sel))

    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    MCC = mcc(tp, fp, fn, tn)
    auprc, auroc = sweep_pr(prob, gt, mask)

    # ---- extra metrics ----
    L = prob.shape[0]
    top1 = top1_per_row_precision(prob, gt, mask)
    topL = topk_precision_global(prob, gt, mask, k=L)
    topL2 = topk_precision_global(prob, gt, mask, k=max(1, L//2))
    shortP, medP, longP = range_precision(prob, gt, mask, cutoff=th)
    cdd = cdd_threshold(prob, gt, mask, cutoff=th)

    return dict(precision=prec, recall=rec, f1=f1, mcc=MCC,
                auprc=auprc, auroc=auroc,
                top1=top1, topL=topL, topL2=topL2,
                range_short=shortP, range_med=medP, range_long=longP,
                cdd=cdd)



# ===========================
# Main
# ===========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing outputs/")
    ap.add_argument("--subdir", default="pred_npz", help="subfolder containing *.npz")
    ap.add_argument("--meta-json", required=True, help="metadata JSON with peptide/protein lengths")
    ap.add_argument("--prob-th", type=float, default=0.5)
    ap.add_argument("--pairing", choices=["unique_upper", "full"], default="unique_upper",
                    help="unique_upper: only count upper-triangular unique pairs; full: use full symmetric map.")
    ap.add_argument("--save-figs", action="store_true")
    args = ap.parse_args()

    # --- load metadata ---
    with open(args.meta_json, "r") as f:
        meta_data = json.load(f)

    in_dir = os.path.join(args.root, args.subdir)
    files = sorted([f for f in os.listdir(in_dir) if f.endswith(".npz")])
    if not files:
        print(f"[error] no npz found under {in_dir}")
        sys.exit(1)

    results = []

    # region accumulators (micro) & per-sample curves (macro)
    region_all_probs = {k: [] for k in ["full","inter","intra_pep","intra_pro","intra_both"]}
    region_all_labels = {k: [] for k in ["full","inter","intra_pep","intra_pro","intra_both"]}
    region_per_sample_curves = {k: [] for k in ["full","inter","intra_pep","intra_pro","intra_both"]}

    for f in files:
        path = os.path.join(in_dir, f)
        raw_key = os.path.splitext(f)[0]
        key = meta_key_from_filename(f)

        if key not in meta_data:
            print(f("[skip] {raw_key}: not found in metadata (tried '{key}')"))
            continue

        meta = meta_data[key].get("metadata", {})
        pep_len = int(meta.get("peptide_length", 0))
        prot_len = int(meta.get("protein_length", 0))
        order = meta.get("order", "peptide_protein")

        data = np.load(path, allow_pickle=False)
        if "prob" not in data or "gt" not in data:
            print(f"[skip] {f}: missing prob/gt keys")
            continue
        prob, gt = data["prob"], data["gt"]
        if prob.shape != gt.shape or prob.ndim != 2:
            print(f"[skip] {f}: shape mismatch {prob.shape} vs {gt.shape}")
            continue

        L = prob.shape[0]
        if pep_len + prot_len != L:
            pep_len = int(np.clip(pep_len, 0, L))
            prot_len = L - pep_len

        # ---- Construct region masks ----
        inter_mask = np.zeros((L, L), dtype=bool)
        intra_pep_mask = np.zeros((L, L), dtype=bool)
        intra_pro_mask = np.zeros((L, L), dtype=bool)
        if order == "peptide_protein":
            inter_mask[:pep_len, pep_len:] = True
            intra_pep_mask[:pep_len, :pep_len] = True
            intra_pro_mask[pep_len:, pep_len:] = True
        else:  # protein_peptide
            inter_mask[prot_len:, :prot_len] = True
            intra_pep_mask[prot_len:, prot_len:] = True
            intra_pro_mask[:prot_len, :prot_len] = True

        intra_both_mask = intra_pep_mask | intra_pro_mask

        # ---- Pairing mode ----
        if args.pairing == "unique_upper":
            tri = np.triu(np.ones((L, L), dtype=bool), k=1)
            inter_mask &= tri
            intra_pep_mask &= tri
            intra_pro_mask &= tri
            intra_both_mask &= tri
            full_mask = finite_mask(prob, gt) & tri
        else:
            inter_mask = inter_mask | inter_mask.T
            intra_both_mask = intra_both_mask | intra_both_mask.T
            full_mask = finite_mask(prob, gt)

        # ---- Compute per-region metrics ----
        m_full  = region_metrics(prob, gt, full_mask, th=args.prob_th)
        m_inter = region_metrics(prob, gt, inter_mask, th=args.prob_th)
        m_pep   = region_metrics(prob, gt, intra_pep_mask, th=args.prob_th)
        m_pro   = region_metrics(prob, gt, intra_pro_mask, th=args.prob_th)
        m_both  = region_metrics(prob, gt, intra_both_mask, th=args.prob_th)

        row = dict(
            id=key,
            pep_len=pep_len,
            prot_len=prot_len,
            pairing=args.pairing,

            # ===== full =====
            precision_all=m_full["precision"], recall_all=m_full["recall"], f1_all=m_full["f1"],
            mcc_all=m_full["mcc"], auprc_all=m_full["auprc"], auroc_all=m_full["auroc"],
            top1_all=m_full["top1"], topL_all=m_full["topL"], topL2_all=m_full["topL2"],
            range_short_all=m_full["range_short"], range_med_all=m_full["range_med"], range_long_all=m_full["range_long"],
            cdd_all=m_full["cdd"],

            # ===== inter =====
            precision_inter=m_inter["precision"], recall_inter=m_inter["recall"], f1_inter=m_inter["f1"],
            mcc_inter=m_inter["mcc"], auprc_inter=m_inter["auprc"], auroc_inter=m_inter["auroc"],
            top1_inter=m_inter["top1"], topL_inter=m_inter["topL"], topL2_inter=m_inter["topL2"],
            range_short_inter=m_inter["range_short"], range_med_inter=m_inter["range_med"], range_long_inter=m_inter["range_long"],
            cdd_inter=m_inter["cdd"],

            # ===== intra-peptide =====
            precision_pep=m_pep["precision"], recall_pep=m_pep["recall"], f1_pep=m_pep["f1"],
            mcc_pep=m_pep["mcc"], auprc_pep=m_pep["auprc"], auroc_pep=m_pep["auroc"],
            top1_pep=m_pep["top1"], topL_pep=m_pep["topL"], topL2_pep=m_pep["topL2"],
            range_short_pep=m_pep["range_short"], range_med_pep=m_pep["range_med"], range_long_pep=m_pep["range_long"],
            cdd_pep=m_pep["cdd"],

            # ===== intra-protein =====
            precision_pro=m_pro["precision"], recall_pro=m_pro["recall"], f1_pro=m_pro["f1"],
            mcc_pro=m_pro["mcc"], auprc_pro=m_pro["auprc"], auroc_pro=m_pro["auroc"],
            top1_pro=m_pro["top1"], topL_pro=m_pro["topL"], topL2_pro=m_pro["topL2"],
            range_short_pro=m_pro["range_short"], range_med_pro=m_pro["range_med"], range_long_pro=m_pro["range_long"],
            cdd_pro=m_pro["cdd"],

            # ===== intra-both =====
            precision_intra_both=m_both["precision"], recall_intra_both=m_both["recall"], f1_intra_both=m_both["f1"],
            mcc_intra_both=m_both["mcc"], auprc_intra_both=m_both["auprc"], auroc_intra_both=m_both["auroc"],
            top1_intra_both=m_both["top1"], topL_intra_both=m_both["topL"], topL2_intra_both=m_both["topL2"],
            range_short_intra_both=m_both["range_short"], range_med_intra_both=m_both["range_med"], range_long_intra_both=m_both["range_long"],
            cdd_intra_both=m_both["cdd"],
        )
        results.append(row)

        # ---- accumulate vectors for global (micro) & points for macro ----
        reg_masks = {
            "full": full_mask, "inter": inter_mask,
            "intra_pep": intra_pep_mask, "intra_pro": intra_pro_mask,
            "intra_both": intra_both_mask
        }
        fm = finite_mask(prob, gt)
        for name, rmask in reg_masks.items():
            msel = rmask & fm
            if not np.any(msel):  # keep empty to preserve shapes in concat step
                region_per_sample_curves[name].append(dict(valid=False, pr=([],[]), roc=([],[])))
                continue
            region_all_probs[name].append(prob[msel].astype(float))
            region_all_labels[name].append((gt[msel] > 0.5).astype(int))
            region_per_sample_curves[name].append(curve_points(prob, gt, rmask, num=400))

        print(f"[done] {key}: F1_all={m_full['f1']:.3f}  F1_inter={m_inter['f1']:.3f}")

        # ---- figs ----
        if args.save_figs:
            out_fig = os.path.join(args.root, "figs_eval_regions")
            os.makedirs(out_fig, exist_ok=True)

            def crop_and_plot(mat, region_mask, title, fname):
                if not np.any(region_mask):
                    return
                sub = mat * region_mask
                save_heatmap(sub, os.path.join(out_fig, fname), title)

            # full / intra_both pred & gt
            crop_and_plot(prob, full_mask,       f"{key} full_pred",        f"{key}_full_pred.png")
            crop_and_plot(gt,   full_mask,       f"{key} full_gt",          f"{key}_full_gt.png")
            crop_and_plot(prob, intra_both_mask, f"{key} intra_both_pred",  f"{key}_intra_both_pred.png")
            crop_and_plot(gt,   intra_both_mask, f"{key} intra_both_gt",    f"{key}_intra_both_gt.png")

            # regions pred & gt
            crop_and_plot(prob, inter_mask,      f"{key} inter_pred",       f"{key}_inter_pred.png")
            crop_and_plot(gt,   inter_mask,      f"{key} inter_gt",         f"{key}_inter_gt.png")
            crop_and_plot(prob, intra_pep_mask,  f"{key} intra_pep_pred",   f"{key}_intra_pep_pred.png")
            crop_and_plot(gt,   intra_pep_mask,  f"{key} intra_pep_gt",     f"{key}_intra_pep_gt.png")
            crop_and_plot(prob, intra_pro_mask,  f"{key} intra_pro_pred",   f"{key}_intra_pro_pred.png")
            crop_and_plot(gt,   intra_pro_mask,  f"{key} intra_pro_gt",     f"{key}_intra_pro_gt.png")

            # optional abs diff
            diff = np.abs(prob - gt)
            crop_and_plot(diff, full_mask,       f"{key} full_abs_diff",     f"{key}_full_abs_diff.png")
            crop_and_plot(diff, intra_both_mask, f"{key} intra_both_abs_diff", f"{key}_intra_both_abs_diff.png")
            crop_and_plot(diff, inter_mask,      f"{key} inter_abs_diff",    f"{key}_inter_abs_diff.png")
            crop_and_plot(diff, intra_pep_mask,  f"{key} intra_pep_abs_diff",f"{key}_intra_pep_abs_diff.png")
            crop_and_plot(diff, intra_pro_mask,  f"{key} intra_pro_abs_diff",f"{key}_intra_pro_abs_diff.png")

        # ---- per-sample JSON ----
        per_sample_dir = os.path.join(args.root, "metrics_eval_regions")
        os.makedirs(per_sample_dir, exist_ok=True)
        with open(os.path.join(per_sample_dir, f"{row['id']}.json"), "w") as jf:
            json.dump(row, jf, indent=2)

    # ---- per-sample CSV ----
    csv_path = os.path.join(args.root, "summary_fullmap_regions.csv")
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(results)
        print(f"[saved] per-sample summary -> {csv_path}")
    else:
        print("[warn] no samples evaluated, skip summary CSV")

    # ===========================
    # Global (per-region) summaries
    # ===========================
    # dirs
    conf_dir = os.path.join(args.root, "global_confusions")
    curve_dir = os.path.join(args.root, "global_curves")
    summary_curves_dir = os.path.join(args.root, "summary_curves")  # mirror
    os.makedirs(conf_dir, exist_ok=True)
    os.makedirs(curve_dir, exist_ok=True)
    os.makedirs(summary_curves_dir, exist_ok=True)

    # helper: plot/save curve to two dirs
    def save_curve_png_csv(x, y, xlabel, ylabel, title, stem, out_dirs):
        for d in out_dirs:
            os.makedirs(d, exist_ok=True)
            plt.figure(figsize=(5,4)); plt.plot(x, y)
            plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
            plt.tight_layout(); plt.savefig(os.path.join(d, f"{stem}.png"), dpi=200); plt.close()
            with open(os.path.join(d, f"{stem}_points.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["x","y"])
                for xi, yi in zip(x, y): w.writerow([xi, yi])

    # helper: confusion
    def confusion_from_vectors(p_vec, y_vec, th):
        pred = (p_vec >= th)
        y_pos = (y_vec == 1); y_neg = ~y_pos
        tp = int(np.sum(pred & y_pos)); fp = int(np.sum(pred & y_neg))
        fn = int(np.sum((~pred) & y_pos)); tn = int(np.sum((~pred) & y_neg))
        prec = safe_div(tp, tp + fp); rec = safe_div(tp, tp + fn)
        f1 = safe_div(2*prec*rec, (prec+rec))
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=float(prec), recall=float(rec), f1=float(f1))

    # collect and compute per-region
    region_order = ["full","inter","intra_pep","intra_pro","intra_both"]
    conf_rows_all = []
    global_summary_rows = []
    for name in region_order:
        if len(region_all_probs[name]) == 0:
            # nothing valid for this region
            continue
        p_all = np.concatenate(region_all_probs[name], axis=0)
        y_all = np.concatenate(region_all_labels[name], axis=0)
        mask_all = np.isfinite(p_all) & np.isfinite(y_all)
        p_all = p_all[mask_all].astype(float); y_all = y_all[mask_all].astype(int)

        # confusions @ 0.5 / 0.8
        for th in (0.5, 0.8):
            c = confusion_from_vectors(p_all, y_all, th)
            conf_rows_all.append({"region": name, "threshold": th, **c})
            # plot to both dirs
            for d in [conf_dir, summary_curves_dir]:
                plt.figure(figsize=(4,4))
                mat = np.array([[c["tp"], c["fn"]],[c["fp"], c["tn"]]], dtype=float)
                im = plt.imshow(mat, cmap="Blues", vmin=0); plt.colorbar(im, fraction=0.046, pad=0.04)
                for (i,j), v in np.ndenumerate(mat): plt.text(j, i, f"{int(v)}", ha="center", va="center")
                plt.xticks([0,1], ["Pred=1","Pred=0"]); plt.yticks([0,1], ["GT=1","GT=0"])
                plt.title(f"Confusion ({name}) @ {th}")
                plt.tight_layout(); plt.savefig(os.path.join(d, f"confusion_{name}_th{th}.png"), dpi=200); plt.close()

        # micro curves
        micro = micro_curves(p_all, y_all)
        save_curve_png_csv(micro["pr"][0], micro["pr"][1],
                           "Recall","Precision",
                           f"Micro PR ({name}) AUPRC={micro['auprc']:.3f}",
                           f"micro_pr_{name}",
                           [curve_dir, summary_curves_dir])
        save_curve_png_csv(micro["roc"][0], micro["roc"][1],
                           "FPR","TPR",
                           f"Micro ROC ({name}) AUROC={micro['auroc']:.3f}",
                           f"micro_roc_{name}",
                           [curve_dir, summary_curves_dir])

        # macro curves
        macro = macro_curves(region_per_sample_curves[name], grid_n=1001)
        save_curve_png_csv(macro["pr"][0], macro["pr"][1],
                           "Recall","Precision",
                           f"Macro PR ({name}) AUPRC={macro['auprc']:.3f}",
                           f"macro_pr_{name}",
                           [curve_dir, summary_curves_dir])
        save_curve_png_csv(macro["roc"][0], macro["roc"][1],
                           "FPR","TPR",
                           f"Macro ROC ({name}) AUROC={macro['auroc']:.3f}",
                           f"macro_roc_{name}",
                           [curve_dir, summary_curves_dir])

        global_summary_rows.append(dict(
            region=name,
            AUPRC_micro=micro["auprc"], AUROC_micro=micro["auroc"],
            AUPRC_macro=macro["auprc"], AUROC_macro=macro["auroc"],
        ))

    # write combined confusions CSV
    if conf_rows_all:
        with open(os.path.join(conf_dir, "confusions_all_regions.csv"), "w", newline="") as f:
            keys = ["region","threshold","tp","fp","fn","tn","precision","recall","f1"]
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(conf_rows_all)
        # mirror to summary_curves
        with open(os.path.join(summary_curves_dir, "confusions_all_regions.csv"), "w", newline="") as f:
            keys = ["region","threshold","tp","fp","fn","tn","precision","recall","f1"]
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(conf_rows_all)
        print(f"[saved] global confusions per region -> {conf_dir}")

    # write global summary CSV (per region)
    if global_summary_rows:
        with open(os.path.join(curve_dir, "global_summary_per_region.csv"), "w", newline="") as f:
            keys = ["region","AUPRC_micro","AUROC_micro","AUPRC_macro","AUROC_macro"]
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(global_summary_rows)
        with open(os.path.join(summary_curves_dir, "global_summary_per_region.csv"), "w", newline="") as f:
            keys = ["region","AUPRC_micro","AUROC_micro","AUPRC_macro","AUROC_macro"]
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(global_summary_rows)
        print(f"[saved] global curves per region -> {curve_dir}")

    # ---- Skip summary ----
    evaluated_ids = set(r["id"] for r in results)
    skipped = [f for f in files if meta_key_from_filename(os.path.splitext(f)[0]) not in evaluated_ids]
    if skipped:
        skip_json = os.path.join(args.root, "skip_summary.json")
        with open(skip_json, "w") as f:
            json.dump({"total": len(files), "evaluated": len(results), "skipped": skipped}, f, indent=2)
        print(f"[saved] skip summary -> {skip_json}")


# ----- reuse from earlier code blocks -----
def micro_curves(all_probs, all_labels):
    p = all_probs.astype(float)
    y = all_labels.astype(int)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    if p.size == 0:
        return {"pr": ([],[]), "roc": ([],[]), "auprc": float("nan"), "auroc": float("nan")}
    all_pos = (y.sum() == y.size)
    all_neg = (y.sum() == 0)
    order = np.argsort(-p)
    p_sorted = p[order]; y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted == 1).astype(float)
    fp_cum = np.cumsum(y_sorted == 0).astype(float)
    P = float((y == 1).sum()); N = float((y == 0).sum())
    recall = tp_cum / max(P, 1e-8)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)
    tpr = recall
    fpr = fp_cum / max(N, 1e-8)

    def uniq_xy(x, y):
        x = np.asarray(x); y = np.asarray(y)
        order = np.argsort(x); x = x[order]; y = y[order]
        ux, idx = np.unique(x, return_index=True)
        y_max = []
        for i in range(len(ux)):
            start = idx[i]; end = idx[i+1] if i+1 < len(idx) else len(x)
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
    cs = [c for c in per_sample_curves if c.get("valid", False)]
    if len(cs) == 0:
        return {"pr": ([],[]), "roc": ([],[]), "auprc": float("nan"), "auroc": float("nan")}
    rec_grid = np.linspace(0.0, 1.0, grid_n)
    pr_mat = []
    for c in cs:
        rx, py = np.asarray(c["pr"][0]), np.asarray(c["pr"][1])
        if rx.size < 2: continue
        order = np.argsort(rx); rx, py = rx[order], py[order]
        pr_interp = np.interp(rec_grid, rx, py, left=py[0], right=py[-1])
        pr_mat.append(pr_interp)
    pr_mean = np.mean(np.stack(pr_mat, axis=0), axis=0) if pr_mat else np.zeros_like(rec_grid)
    auprc = float(np.trapz(y=pr_mean, x=rec_grid)) if pr_mat else float("nan")

    fpr_grid = np.linspace(0.0, 1.0, grid_n)
    tpr_mat = []
    for c in cs:
        fx, ty = np.asarray(c["roc"][0]), np.asarray(c["roc"][1])
        if fx.size < 2: continue
        order = np.argsort(fx); fx, ty = fx[order], ty[order]
        tpr_interp = np.interp(fpr_grid, fx, ty, left=ty[0], right=ty[-1])
        tpr_mat.append(tpr_interp)
    tpr_mean = np.mean(np.stack(tpr_mat, axis=0), axis=0) if tpr_mat else np.zeros_like(fpr_grid)
    auroc = float(np.trapz(y=tpr_mean, x=fpr_grid)) if tpr_mat else float("nan")
    return {"pr": (rec_grid.tolist(), pr_mean.tolist()),
            "roc": (fpr_grid.tolist(), tpr_mean.tolist()),
            "auprc": auprc, "auroc": auroc}


if __name__ == "__main__":
    main()
