#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_fullmap_txt_metrics.py

Batch analyze metrics from test_fullmap.py output (.txt files).

Extracts per-sample:
  - Precision, Recall, F1
  - (Optional) TopK contact count
Then produces:
  - CSV summary
  - F1 histogram
  - Precision–Recall scatter
  - TopK contact count distribution (optional)
"""

import os, re, csv, argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Analyze per-sample metrics from test_fullmap outputs (.txt).")
    ap.add_argument("--root", required=True, help="Folder containing .txt files (e.g. outputs/fullmap_test_V2/texts)")
    ap.add_argument("--save-dir", default=None, help="Where to save summary and plots (default: root)")
    ap.add_argument("--save-topk", action="store_true", help="Also plot TopK contact count distribution")
    args = ap.parse_args()

    root = args.root
    out_dir = args.save_dir or root
    os.makedirs(out_dir, exist_ok=True)

    pattern_metrics = re.compile(r"Precision=(\d+\.\d+)\s+Recall=(\d+\.\d+)\s+F1=(\d+\.\d+)")
    pattern_topk = re.compile(r"TopK\s*\(pep\+prot=(\d+)\)")

    rows = []
    precisions, recalls, f1s, topks = [], [], [], []

    files = sorted([f for f in os.listdir(root) if f.endswith(".txt")])
    if not files:
        print(f"[warn] no .txt found under {root}")
        return

    for fname in files:
        path = os.path.join(root, fname)
        text = open(path).read()
        m = pattern_metrics.search(text)
        if not m:
            print(f"[skip] {fname}: no metrics found")
            continue
        p, r, f1 = map(float, m.groups())
        precisions.append(p); recalls.append(r); f1s.append(f1)

        m_topk = pattern_topk.search(text)
        topk = int(m_topk.group(1)) if m_topk else np.nan
        topks.append(topk)

        rows.append({"id": fname.replace(".txt",""), "Precision":p, "Recall":r, "F1":f1, "TopK_len":topk})

    # ====== Save summary CSV ======
    csv_path = os.path.join(out_dir, "metrics_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","Precision","Recall","F1","TopK_len"])
        w.writeheader(); w.writerows(rows)
    print(f"[saved] CSV summary -> {csv_path}")

    # ====== Compute macro means ======
    mean_p = np.nanmean(precisions)
    mean_r = np.nanmean(recalls)
    mean_f1 = np.nanmean(f1s)
    print(f"[macro mean] Precision={mean_p:.3f}  Recall={mean_r:.3f}  F1={mean_f1:.3f}")

    # ====== F1 distribution ======
    plt.figure(figsize=(6,4))
    plt.hist(f1s, bins=30, color='steelblue', alpha=0.8)
    plt.xlabel("F1-score"); plt.ylabel("Count")
    plt.title("Distribution of F1 across samples")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "f1_distribution.png"), dpi=200)
    plt.close()

    # ====== Precision–Recall scatter ======
    plt.figure(figsize=(5,5))
    plt.scatter(recalls, precisions, s=40, alpha=0.7, color="tomato")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Scatter")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "precision_recall_scatter.png"), dpi=200)
    plt.close()

    # ====== TopK distribution ======
    if args.save_topk and np.isfinite(topks).any():
        plt.figure(figsize=(6,4))
        plt.hist([t for t in topks if np.isfinite(t)], bins=20, color='gray', alpha=0.7)
        plt.xlabel("TopK (peptide + protein length)")
        plt.ylabel("Count")
        plt.title("Distribution of TopK contact length")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "topk_distribution.png"), dpi=200)
        plt.close()

    print(f"[done] Plots saved to: {out_dir}")

if __name__ == "__main__":
    main()
