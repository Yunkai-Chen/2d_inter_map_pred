#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_fullmap_regions.py
============================
Visualization suite for eval_fullmap_contact.py outputs.

Generates:
  - Region-wise metric distributions
  - Region comparison boxplots
  - Inter vs Intra difference histograms
  - Scatter correlations between regions
  - Global AUC summary plots (from global_summary_per_region.csv)
  - Top-K best/worst samples tables
  - Overview.txt with summary stats

Usage:
  python visualize_fullmap_regions.py \
      --csv summary_fullmap_regions.csv \
      --global-csv global_summary_per_region.csv \
      --out viz_fullmap_out \
      --topk 50
"""

import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# Utility
# ---------------------------------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def finite(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]

def safe_mean(x):
    x = finite(x)
    return float(np.nanmean(x)) if len(x) else np.nan

def to_num(series):
    return pd.to_numeric(series, errors="coerce")

# ---- alias helpers ----
# 兼容列名后缀：full↔all, intra_pep↔pep, intra_pro↔pro
REGION_ALIASES = {
    "full":       ["full", "all"],
    "inter":      ["inter"],
    "intra_pep":  ["intra_pep", "pep"],
    "intra_pro":  ["intra_pro", "pro"],
    "intra_both": ["intra_both"],
}

def resolve_region_col(df: pd.DataFrame, metric: str, region_key: str):
    """
    Return the first existing column name for (metric, region) considering aliases.
    e.g. metric='f1', region_key='full' -> try 'f1_full', then 'f1_all'
    """
    for rname in REGION_ALIASES.get(region_key, [region_key]):
        col = f"{metric}_{rname}"
        if col in df.columns:
            return col
    return None

# ---------------------------------------------------
# Plot functions
# ---------------------------------------------------
def plot_distributions(df, out, regions):
    """Plot per-region metric histograms."""
    out_dir = os.path.join(out, "distributions")
    ensure_dir(out_dir)
    metrics = [
        ("f1", "F1-score"),
        ("mcc", "MCC"),
        ("auprc", "AUPRC"),
        ("auroc", "AUROC"),
        ("cdd", "CDD"),
        ("topL", "Top-L Precision"),
        ("range_short", "Short-range P"),
        ("range_med", "Medium-range P"),
        ("range_long", "Long-range P"),
    ]

    for region in regions:
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        for ax, (metric, label) in zip(axes.ravel(), metrics):
            col = resolve_region_col(df, metric, region)
            if not col:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=10)
                ax.set_title(f"{label} (no data)")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            vals = finite(to_num(df[col]).values)
            if len(vals) == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=10)
                ax.set_title(f"{label} (no data)")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            ax.hist(vals, bins=30, alpha=0.8, color="steelblue", edgecolor="black")
            ax.set_title(label); ax.set_xlabel(metric); ax.set_ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{region}_metrics_distributions.png"), dpi=200)
        plt.close()


def plot_region_boxplots(df, out, regions):
    """Compare different regions for each key metric."""
    out_dir = os.path.join(out, "region_comparison")
    ensure_dir(out_dir)
    metrics = ["f1", "mcc", "auprc", "auroc", "cdd"]

    for metric in metrics:
        data = []
        for region in regions:
            col = resolve_region_col(df, metric, region)
            if not col:
                continue
            vals = finite(to_num(df[col]).values)
            data += [{"region": region, metric: v} for v in vals]
        if not data:
            continue

        dframe = pd.DataFrame(data)
        plt.figure(figsize=(8, 5))
        # 修复 seaborn FutureWarning：指定 hue 并关闭 legend
        sns.boxplot(data=dframe, x="region", y=metric, hue="region", palette="Set2", legend=False)
        plt.title(f"{metric.upper()} by Region")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}_by_region.png"), dpi=200)
        plt.close()


def plot_inter_intra_diff(df, out):
    """Plot histograms of inter - intra_both metric differences (列名已匹配你CSV)."""
    out_dir = os.path.join(out, "region_comparison")
    ensure_dir(out_dir)
    metrics = ["f1", "mcc", "auprc", "auroc", "cdd"]

    for metric in metrics:
        inter_col = f"{metric}_inter"
        intra_col = f"{metric}_intra_both"
        if inter_col not in df.columns or intra_col not in df.columns:
            continue
        diff = to_num(df[inter_col]) - to_num(df[intra_col])
        diff = finite(diff.values)
        if len(diff) == 0:
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(diff, bins=30, color="salmon", edgecolor="black", alpha=0.85)
        mu = float(np.nanmean(diff))
        plt.axvline(mu, color="k", linestyle="--", label=f"mean={mu:.3f}")
        plt.title(f"Inter − IntraBoth Δ{metric.upper()} Distribution")
        plt.xlabel(f"{metric}_inter - {metric}_intra_both")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"diff_inter_intra_{metric}.png"), dpi=200)
        plt.close()


def plot_scatter_relations(df, out):
    """Plot correlations between selected metric pairs."""
    out_dir = os.path.join(out, "scatters")
    ensure_dir(out_dir)

    # 用你CSV里的实际列名组合（all/inter/intra_both皆存在）
    pairs = [
        ("f1_all", "f1_inter"),
        ("f1_all", "f1_intra_both"),
        ("mcc_inter", "f1_inter"),
        ("auroc_inter", "auroc_intra_both"),
        ("auprc_inter", "auprc_intra_both"),
    ]

    for xcol, ycol in pairs:
        if xcol not in df.columns or ycol not in df.columns:
            continue
        x = to_num(df[xcol]).values
        y = to_num(df[ycol]).values
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            continue
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        plt.figure(figsize=(5, 4))
        plt.scatter(x[mask], y[mask], s=15, alpha=0.6, color="teal")
        plt.xlabel(xcol); plt.ylabel(ycol)
        plt.title(f"{ycol} vs {xcol}\nPearson r={r:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{ycol}_vs_{xcol}.png"), dpi=200)
        plt.close()


def plot_global_auc(df_global, out):
    """Barplots of global micro/macro AUC per region."""
    out_dir = os.path.join(out, "region_comparison")
    ensure_dir(out_dir)
    dfg = df_global.copy()
    if "region" not in dfg.columns:
        return
    regions = dfg["region"].tolist()

    plt.figure(figsize=(8, 5))
    barw = 0.35
    x = np.arange(len(regions))
    plt.bar(x - barw/2, to_num(dfg["AUPRC_micro"]), barw, label="AUPRC_micro")
    plt.bar(x + barw/2, to_num(dfg["AUPRC_macro"]), barw, label="AUPRC_macro")
    plt.xticks(x, regions, rotation=30)
    plt.ylabel("AUPRC")
    plt.title("Global AUPRC (Micro vs Macro)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_AUPRC_summary.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(x - barw/2, to_num(dfg["AUROC_micro"]), barw, label="AUROC_micro")
    plt.bar(x + barw/2, to_num(dfg["AUROC_macro"]), barw, label="AUROC_macro")
    plt.xticks(x, regions, rotation=30)
    plt.ylabel("AUROC")
    plt.title("Global AUROC (Micro vs Macro)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "global_AUROC_summary.png"), dpi=200)
    plt.close()


def export_topk_tables(df, out, k):
    """Save Top-K best/worst samples for selected metrics."""
    out_dir = os.path.join(out, "tables")
    ensure_dir(out_dir)
    metrics = ["f1_inter", "mcc_inter", "auprc_all", "cdd_inter"]
    for metric in metrics:
        if metric not in df.columns:
            continue
        vals = to_num(df[metric])
        df_sorted = df.assign(score=vals).dropna(subset=["score"])
        if df_sorted.empty:
            continue
        best = df_sorted.sort_values("score", ascending=False).head(k)
        worst = df_sorted.sort_values("score", ascending=True).head(k)
        best.to_csv(os.path.join(out_dir, f"top{k}_best_{metric}.csv"), index=False)
        worst.to_csv(os.path.join(out_dir, f"top{k}_worst_{metric}.csv"), index=False)


def write_overview(df, df_global, out, regions):
    """Save region-level mean stats to overview.txt."""
    txt_path = os.path.join(out, "overview.txt")
    with open(txt_path, "w") as f:
        f.write(f"Total samples: {len(df)}\n\n")
        f.write("[Per-region means]\n")
        for r in regions:
            sub = {}
            for m in ["f1", "mcc", "auprc", "auroc", "cdd"]:
                col = resolve_region_col(df, m, r)
                if col:
                    sub[m] = safe_mean(to_num(df[col]).values)
            if not sub:
                continue
            f.write(f"{r:12s}: " + "  ".join([f"{k.upper()}={v:.4f}" for k, v in sub.items()]) + "\n")

        f.write("\n[Global summary per region]\n")
        if not df_global.empty and "region" in df_global.columns:
            for _, row in df_global.iterrows():
                f.write(
                    f"{str(row['region']):12s}: "
                    f"AUPRC_micro={float(row['AUPRC_micro']):.4f} "
                    f"AUPRC_macro={float(row['AUPRC_macro']):.4f} "
                    f"AUROC_micro={float(row['AUROC_micro']):.4f} "
                    f"AUROC_macro={float(row['AUROC_macro']):.4f}\n"
                )

# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="summary_fullmap_regions.csv path")
    ap.add_argument("--global-csv", dest="global_csv", required=True, help="global_summary_per_region.csv path")
    ap.add_argument("--out", required=True, help="output dir for plots/tables")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--regions", nargs="+", default=["full","inter","intra_pep","intra_pro","intra_both"])
    args = ap.parse_args()

    ensure_dir(args.out)
    df = pd.read_csv(args.csv)
    df_global = pd.read_csv(args.global_csv)

    # A. per-region distributions
    plot_distributions(df, args.out, args.regions)

    # B. region comparison boxplots
    plot_region_boxplots(df, args.out, args.regions)

    # C. inter vs intra difference histograms
    plot_inter_intra_diff(df, args.out)

    # D. scatter correlations
    plot_scatter_relations(df, args.out)

    # E. global AUC summary
    plot_global_auc(df_global, args.out)

    # F. top-K tables
    export_topk_tables(df, args.out, args.topk)

    # G. overview text summary
    write_overview(df, df_global, args.out, args.regions)

    print(f"[done] all visualizations saved to {args.out}")

if __name__ == "__main__":
    main()
