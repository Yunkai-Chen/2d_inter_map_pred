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
- Supports pairing modes:
    * unique_upper (default): only count each residue pair once (upper triangle)
    * full: count full map, with symmetric inter region
- Outputs per-sample CSV + global summary

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
    """Compute PR-AUC and ROC-AUC with a simple threshold sweep."""
    prob = np.asarray(prob).ravel()
    gt = (np.asarray(gt) > 0.5).astype(int).ravel()
    mask = np.asarray(mask, dtype=bool).ravel()

    m = mask & np.isfinite(prob) & np.isfinite(gt)
    p = prob[m].astype(float)
    y = gt[m].astype(int)
    if p.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return float("nan"), float("nan")

    ths = np.linspace(0.0, 1.0, 101)
    prec, rec, fpr, tpr = [], [], [], []
    for th in ths:
        pred = (p >= th)
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        fn = np.sum((~pred) & (y == 1))
        tn = np.sum((~pred) & (y == 0))
        prec.append(safe_div(tp, tp + fp))
        rec.append(safe_div(tp, tp + fn))
        tpr.append(safe_div(tp, tp + fn))
        fpr.append(safe_div(fp, fp + tn))

    prec, rec = np.array(prec), np.array(rec)
    auprc = float(np.trapz(y=prec, x=rec))
    tpr, fpr = np.array(tpr), np.array(fpr)
    auroc = float(np.trapz(y=tpr, x=fpr))
    return auprc, auroc


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
    all_probs, all_labels = [], []

    for f in files:
        path = os.path.join(in_dir, f)
        raw_key = os.path.splitext(f)[0]
        key = meta_key_from_filename(f)

        if key not in meta_data:
            print(f"[skip] {raw_key}: not found in metadata (tried '{key}')")
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

        # ---- Add combined intra mask ----
        intra_both_mask = intra_pep_mask | intra_pro_mask

        # ---- Pairing mode ----
        if args.pairing == "unique_upper":
            tri = np.triu(np.ones((L, L), dtype=bool), k=1)
            inter_mask &= tri
            intra_pep_mask &= tri
            intra_pro_mask &= tri
            overall_mask = finite_mask(prob, gt) & tri
        else:
            inter_mask = inter_mask | inter_mask.T
            overall_mask = finite_mask(prob, gt)

        # ---- Compute metrics ----
        m_overall = region_metrics(prob, gt, overall_mask, th=args.prob_th)
        m_inter = region_metrics(prob, gt, inter_mask, th=args.prob_th)
        m_pep = region_metrics(prob, gt, intra_pep_mask, th=args.prob_th)
        m_pro = region_metrics(prob, gt, intra_pro_mask, th=args.prob_th)
        m_intra_both = region_metrics(prob, gt, intra_both_mask, th=args.prob_th)

        row = dict(
            id=key,
            pep_len=pep_len,
            prot_len=prot_len,
            pairing=args.pairing,

            # ===== overall (full map) =====
            precision_all=m_overall["precision"],
            recall_all=m_overall["recall"],
            f1_all=m_overall["f1"],
            mcc_all=m_overall["mcc"],
            auprc_all=m_overall["auprc"],
            auroc_all=m_overall["auroc"],
            top1_all=m_overall["top1"],
            topL_all=m_overall["topL"],
            topL2_all=m_overall["topL2"],
            range_short_all=m_overall["range_short"],
            range_med_all=m_overall["range_med"],
            range_long_all=m_overall["range_long"],
            cdd_all=m_overall["cdd"],

            # ===== inter (peptide–protein) =====
            precision_inter=m_inter["precision"],
            recall_inter=m_inter["recall"],
            f1_inter=m_inter["f1"],
            mcc_inter=m_inter["mcc"],
            auprc_inter=m_inter["auprc"],
            auroc_inter=m_inter["auroc"],
            top1_inter=m_inter["top1"],
            topL_inter=m_inter["topL"],
            topL2_inter=m_inter["topL2"],
            range_short_inter=m_inter["range_short"],
            range_med_inter=m_inter["range_med"],
            range_long_inter=m_inter["range_long"],
            cdd_inter=m_inter["cdd"],

            # ===== intra-peptide =====
            precision_pep=m_pep["precision"],
            recall_pep=m_pep["recall"],
            f1_pep=m_pep["f1"],
            mcc_pep=m_pep["mcc"],
            auprc_pep=m_pep["auprc"],
            auroc_pep=m_pep["auroc"],
            top1_pep=m_pep["top1"],
            topL_pep=m_pep["topL"],
            topL2_pep=m_pep["topL2"],
            range_short_pep=m_pep["range_short"],
            range_med_pep=m_pep["range_med"],
            range_long_pep=m_pep["range_long"],
            cdd_pep=m_pep["cdd"],

            # ===== intra-protein =====
            precision_pro=m_pro["precision"],
            recall_pro=m_pro["recall"],
            f1_pro=m_pro["f1"],
            mcc_pro=m_pro["mcc"],
            auprc_pro=m_pro["auprc"],
            auroc_pro=m_pro["auroc"],
            top1_pro=m_pro["top1"],
            topL_pro=m_pro["topL"],
            topL2_pro=m_pro["topL2"],
            range_short_pro=m_pro["range_short"],
            range_med_pro=m_pro["range_med"],
            range_long_pro=m_pro["range_long"],
            cdd_pro=m_pro["cdd"],

            # ===== intra-both (protein+peptide) =====
            precision_intra_both=m_intra_both["precision"],
            recall_intra_both=m_intra_both["recall"],
            f1_intra_both=m_intra_both["f1"],
            mcc_intra_both=m_intra_both["mcc"],
            auprc_intra_both=m_intra_both["auprc"],
            auroc_intra_both=m_intra_both["auroc"],
            top1_intra_both=m_intra_both["top1"],
            topL_intra_both=m_intra_both["topL"],
            topL2_intra_both=m_intra_both["topL2"],
            range_short_intra_both=m_intra_both["range_short"],
            range_med_intra_both=m_intra_both["range_med"],
            range_long_intra_both=m_intra_both["range_long"],
            cdd_intra_both=m_intra_both["cdd"],

        )

        results.append(row)

        all_probs.append(prob[overall_mask])
        all_labels.append((gt[overall_mask] > 0.5).astype(int))

        print(f"[done] {key}: F1_all={m_overall['f1']:.3f}  F1_inter={m_inter['f1']:.3f}")

        if args.save_figs:
            out_fig = os.path.join(args.root, "figs_eval_regions")
            os.makedirs(out_fig, exist_ok=True)

            # 定义一个函数：裁出并绘图某个区域
            def crop_and_plot(mat, region_mask, title, fname):
                if not np.any(region_mask):
                    return
                sub = mat * region_mask  # 区域内的值，其他为 0
                save_heatmap(sub, os.path.join(out_fig, fname), title)

            # ---- 绘制 full ----
            crop_and_plot(prob, overall_mask, f"{key} full_pred", f"{key}_full_pred.png")
            crop_and_plot(gt, overall_mask, f"{key} full_gt", f"{key}_full_gt.png")

            # ---- 绘制 intra_both ----
            crop_and_plot(prob, intra_both_mask, f"{key} intra_both_pred", f"{key}_intra_both_pred.png")
            crop_and_plot(gt, intra_both_mask, f"{key} intra_both_gt", f"{key}_intra_both_gt.png")


            # ---- 绘制区域的预测与GT ----
            crop_and_plot(prob, inter_mask, f"{key} inter_pred", f"{key}_inter_pred.png")
            crop_and_plot(gt, inter_mask, f"{key} inter_gt", f"{key}_inter_gt.png")

            crop_and_plot(prob, intra_pep_mask, f"{key} intra_pep_pred", f"{key}_intra_pep_pred.png")
            crop_and_plot(gt, intra_pep_mask, f"{key} intra_pep_gt", f"{key}_intra_pep_gt.png")

            crop_and_plot(prob, intra_pro_mask, f"{key} intra_pro_pred", f"{key}_intra_pro_pred.png")
            crop_and_plot(gt, intra_pro_mask, f"{key} intra_pro_gt", f"{key}_intra_pro_gt.png")

            # ---- 可选：预测 - GT 的差值可视化 ----
            diff = np.abs(prob - gt)
            crop_and_plot(diff, inter_mask, f"{key} inter_abs_diff", f"{key}_inter_abs_diff.png")
            crop_and_plot(diff, intra_pep_mask, f"{key} intra_pep_abs_diff", f"{key}_intra_pep_abs_diff.png")
            crop_and_plot(diff, intra_pro_mask, f"{key} intra_pro_abs_diff", f"{key}_intra_pro_abs_diff.png")


        # ---- Per-sample JSON ----
        per_sample_dir = os.path.join(args.root, "metrics_eval_regions")
        os.makedirs(per_sample_dir, exist_ok=True)
        for r in results:
            json_path = os.path.join(per_sample_dir, f"{r['id']}.json")
            with open(json_path, "w") as f:
                json.dump(r, f, indent=2)
        print(f"[saved] per-sample JSONs -> {per_sample_dir}")

        # ---- Per-sample CSV ----
        csv_path = os.path.join(args.root, "summary_fullmap_regions.csv")
        if results:
            keys = list(results[0].keys())
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(results)
            print(f"[saved] per-sample summary -> {csv_path}")
        else:
            print("[warn] no samples evaluated, skip summary CSV")

        # ---- Global (micro) metrics + confusion matrices + curves ----
        if all_probs:
            p = np.concatenate(all_probs)
            y = np.concatenate(all_labels)
            m = np.isfinite(p) & np.isfinite(y)
            p, y = p[m], y[m]

            def confusion_from_vectors(p_vec, y_vec, th):
                pred = (p_vec >= th)
                tp = np.sum(pred & (y_vec == 1))
                fp = np.sum(pred & (y_vec == 0))
                fn = np.sum((~pred) & (y_vec == 1))
                tn = np.sum((~pred) & (y_vec == 0))
                prec = safe_div(tp, tp + fp)
                rec = safe_div(tp, tp + fn)
                f1 = safe_div(2 * prec * rec, prec + rec)
                return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                            precision=float(prec), recall=float(rec), f1=float(f1))

            # 混淆矩阵 @ 0.5 / 0.8
            conf_dir = os.path.join(args.root, "global_confusions")
            os.makedirs(conf_dir, exist_ok=True)
            conf_rows = []
            for th in (0.5, 0.8):
                c = confusion_from_vectors(p, y, th)
                conf_rows.append({"threshold": th, **c})
                plt.figure(figsize=(4, 4))
                mat = np.array([[c["tp"], c["fn"]], [c["fp"], c["tn"]]])
                plt.imshow(mat, cmap="Blues")
                for (i, j), v in np.ndenumerate(mat):
                    plt.text(j, i, str(v), ha="center", va="center", color="black")
                plt.xticks([0, 1], ["Pred=1", "Pred=0"])
                plt.yticks([0, 1], ["GT=1", "GT=0"])
                plt.title(f"Confusion @ {th}")
                plt.tight_layout()
                plt.savefig(os.path.join(conf_dir, f"confusion_th{th}.png"), dpi=200)
                plt.close()
            with open(os.path.join(conf_dir, "confusions.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=conf_rows[0].keys())
                w.writeheader(); w.writerows(conf_rows)
            print(f"[saved] global confusions -> {conf_dir}")

            # ---- Micro PR/ROC ----
            ths = np.linspace(0.0, 1.0, 101)
            precs, recs, fprs, tprs = [], [], [], []
            for th in ths:
                pred = (p >= th)
                tp = np.sum(pred & (y == 1))
                fp = np.sum(pred & (y == 0))
                fn = np.sum((~pred) & (y == 1))
                tn = np.sum((~pred) & (y == 0))
                precs.append(safe_div(tp, tp + fp))
                recs.append(safe_div(tp, tp + fn))
                tprs.append(safe_div(tp, tp + fn))
                fprs.append(safe_div(fp, fp + tn))
            auprc = float(np.trapz(y=precs, x=recs))
            auroc = float(np.trapz(y=tprs, x=fprs))

            curve_dir = os.path.join(args.root, "global_curves")
            os.makedirs(curve_dir, exist_ok=True)
            plt.figure(); plt.plot(recs, precs); plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.title(f"Micro PR (AUPRC={auprc:.3f})"); plt.tight_layout()
            plt.savefig(os.path.join(curve_dir, "micro_pr.png"), dpi=200); plt.close()
            plt.figure(); plt.plot(fprs, tprs); plt.xlabel("FPR"); plt.ylabel("TPR")
            plt.title(f"Micro ROC (AUROC={auroc:.3f})"); plt.tight_layout()
            plt.savefig(os.path.join(curve_dir, "micro_roc.png"), dpi=200); plt.close()

            # 保存曲线点
            with open(os.path.join(curve_dir, "micro_points.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["recall", "precision", "fpr", "tpr"])
                for r, p1, f, t in zip(recs, precs, fprs, tprs): w.writerow([r, p1, f, t])

            # ---- Macro PR/ROC ----
            # 每个样本独立计算，再平均（简化）
            macro_precs, macro_recs = [], []
            macro_tprs, macro_fprs = [], []
            for prob_i, gt_i in zip(all_probs, all_labels):
                if prob_i.size == 0: continue
                auprc_i, auroc_i = sweep_pr(prob_i, gt_i, np.ones_like(gt_i, bool))
                macro_precs.append(auprc_i); macro_tprs.append(auroc_i)
            auprc_macro = np.nanmean(macro_precs)
            auroc_macro = np.nanmean(macro_tprs)
            with open(os.path.join(curve_dir, "global_macro_summary.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["AUPRC_macro", "AUROC_macro"])
                w.writeheader(); w.writerow({"AUPRC_macro": auprc_macro, "AUROC_macro": auroc_macro})
            print(f"[saved] global curves -> {curve_dir}")

            # ---- Global summary CSV ----
            summary = dict(
                AUPRC_micro=auprc, AUROC_micro=auroc,
                AUPRC_macro=auprc_macro, AUROC_macro=auroc_macro,
                conf_dir=conf_dir, curve_dir=curve_dir
            )
            with open(os.path.join(args.root, "global_summary_all.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=summary.keys())
                w.writeheader(); w.writerow(summary)
            print(f"[saved] global summary -> {args.root}/global_summary_all.csv")

        else:
            print("[warn] no valid samples for global summary")

        # ---- Skip summary ----
        skipped = [f for f in files if not any(f.startswith(r["id"]) for r in results)]
        if skipped:
            skip_json = os.path.join(args.root, "skip_summary.json")
            with open(skip_json, "w") as f:
                json.dump({"total": len(files), "evaluated": len(results), "skipped": skipped}, f, indent=2)
            print(f"[saved] skip summary -> {skip_json}")



if __name__ == "__main__":
    main()
