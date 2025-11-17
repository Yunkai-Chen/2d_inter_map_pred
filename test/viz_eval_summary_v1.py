#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended visualization script for contact-map evaluation summaries.
Supports new metrics:
- MCC_at_main_th / MCC_at_0p8
- Short/Medium/Long_range_P_at_main_th / _0p8
- CDD_at_main_th / _0p8
"""
import os, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# utils
# ---------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def parse_shape_col(s):
    if isinstance(s, str) and s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            return int(arr[0]), int(arr[1])
        except Exception:
            return np.nan, np.nan
    return np.nan, np.nan

def add_shape_bins(df):
    Tl, Tp = [], []
    for s in df.get("shape_Tl_Tp", [np.nan]*len(df)):
        tl, tp = parse_shape_col(s)
        Tl.append(tl); Tp.append(tp)
    df = df.copy()
    df["Tl"], df["Tp"] = Tl, Tp
    df["Tl_bin"] = pd.cut(df["Tl"], bins=[-1, 6, 10, 20, 9999],
                          labels=["<=6","7-10","11-20",">20"])
    df["Tp_bin"] = pd.cut(df["Tp"], bins=[-1, 200, 400, 800, 9999],
                          labels=["<=200","201-400","401-800",">800"])
    return df

def hist(ax, x, bins, title, xlabel):
    arr = np.asarray(x, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        ax.axis("off"); ax.set_title(f"{title} (no data)"); return
    ax.hist(arr, bins=bins, edgecolor="black", alpha=0.85)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("count")

def scatter(ax, x, y, title, xlabel, ylabel):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        ax.axis("off"); ax.set_title(f"{title} (no data)"); return
    ax.scatter(x[m], y[m], s=10, alpha=0.6)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    try:
        r = np.corrcoef(x[m], y[m])[0,1]
        ax.text(0.02, 0.95, f"Pearson r={r:.3f}", transform=ax.transAxes, va="top")
    except Exception:
        pass

def heat_by_bins(df, value_col, agg="mean"):
    g = df.groupby(["Tl_bin","Tp_bin"])[value_col]
    tab = g.mean() if agg == "mean" else g.median()
    tab = tab.unstack("Tp_bin")
    return tab

def topk_table(df, score_col, k=50, mode="min"):
    d = df[["id", "shape_Tl_Tp", "Tl", "Tp", score_col]].copy()
    d = d.rename(columns={score_col: "score"})
    d = d.sort_values("score", ascending=(mode=="min"))
    return d.head(k)

# ---------------------------
# column alias mapping
# ---------------------------
ALIASES = {
    "contact_f1": ["contact_f1","f1"],
    "contact_precision": ["contact_precision","precision"],
    "contact_recall": ["contact_recall","recall"],
    "MAE_A": ["MAE_A","MAE"],
    "Pearson_r": ["Pearson_r","pearson"],
    "Spearman_rho": ["Spearman_rho","spearman"],
    "TopL_precision": ["TopL_precision","TopL"],
    "Top1_per_row_precision": ["Top1_per_row_precision","Top1_per_row"],
    "AUROC": ["AUROC","ROC_AUC"],
    "AUPRC": ["AUPRC","PR_AUC"],
    "prob_threshold": ["prob_threshold","prob_th"],
    # new metrics
    "MCC_at_main_th": ["MCC_at_main_th","MCC_main"],
    "MCC_at_0p8": ["MCC_at_0p8","MCC_0p8"],
    "Short_range_P_at_main_th": ["Short_range_P_at_main_th"],
    "Medium_range_P_at_main_th": ["Medium_range_P_at_main_th"],
    "Long_range_P_at_main_th": ["Long_range_P_at_main_th"],
    "Short_range_P_at_0p8": ["Short_range_P_at_0p8"],
    "Medium_range_P_at_0p8": ["Medium_range_P_at_0p8"],
    "Long_range_P_at_0p8": ["Long_range_P_at_0p8"],
    "CDD_at_main_th": ["CDD_at_main_th","ContactDensityDev_main"],
    "CDD_at_0p8": ["CDD_at_0p8","ContactDensityDev_0p8"],
}

def map_columns(df):
    df = df.copy()
    for canon, alts in ALIASES.items():
        for a in alts:
            if a in df.columns and canon not in df:
                df[canon] = df[a]; break
    if "id" not in df.columns:
        df["id"] = [f"row_{i}" for i in range(len(df))]
    if "shape_Tl_Tp" not in df.columns:
        df["shape_Tl_Tp"] = [np.nan]*len(df)
    return df

# ---------------------------
# main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="summary CSV path")
    ap.add_argument("--out", required=True, help="output dir for plots")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    ensure_dir(args.out)
    df_raw = pd.read_csv(args.csv)
    df = add_shape_bins(map_columns(df_raw))

    # ---------- overview ----------
    summary_txt = os.path.join(args.out, "overview.txt")
    with open(summary_txt, "w") as f:
        f.write(f"Total samples: {len(df)}\n")
        metrics_to_show = [
            "MAE_A","Pearson_r","Spearman_rho","contact_f1",
            "TopL_precision","AUROC","AUPRC",
            "MCC_at_main_th","MCC_at_0p8",
            "Long_range_P_at_main_th","Long_range_P_at_0p8",
            "CDD_at_main_th","CDD_at_0p8"
        ]
        for m in metrics_to_show:
            if m in df.columns:
                arr = pd.to_numeric(df[m], errors="coerce")
                arr = arr[np.isfinite(arr)]
                if len(arr):
                    f.write(f"{m}: mean={arr.mean():.4f}, median={np.median(arr):.4f}, P90={np.quantile(arr,0.9):.4f}\n")

    # ---------- distributions ----------
    dist_dir = os.path.join(args.out, "distributions")
    ensure_dir(dist_dir)
    metrics_plot = [
        ("MCC_at_main_th","MCC@main"),
        ("MCC_at_0p8","MCC@0.8"),
        ("Long_range_P_at_main_th","Long-Range P@main"),
        ("Long_range_P_at_0p8","Long-Range P@0.8"),
        ("CDD_at_main_th","CDD@main"),
        ("CDD_at_0p8","CDD@0.8"),
        ("contact_f1","F1@main"),
        ("AUPRC","AUPRC"),
        ("AUROC","AUROC")
    ]
    fig, axes = plt.subplots(3,3,figsize=(12,10))
    for ax,(col,title) in zip(axes.ravel(), metrics_plot):
        if col in df:
            hist(ax, df[col], bins=30, title=title, xlabel=col)
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(dist_dir,"metrics_distributions.png"),dpi=200)
    plt.close()

    # ---------- scatter ----------
    sc_dir = os.path.join(args.out,"scatters"); ensure_dir(sc_dir)
    if "contact_f1" in df:
        if "MCC_at_main_th" in df:
            plt.figure(); scatter(plt.gca(),df["MCC_at_main_th"],df["contact_f1"],"MCC vs F1","MCC","F1"); plt.tight_layout()
            plt.savefig(os.path.join(sc_dir,"mcc_vs_f1.png"),dpi=200); plt.close()
        if "Long_range_P_at_main_th" in df:
            plt.figure(); scatter(plt.gca(),df["Long_range_P_at_main_th"],df["contact_f1"],"Long-range P vs F1","Long-range Precision","F1")
            plt.tight_layout(); plt.savefig(os.path.join(sc_dir,"longP_vs_f1.png"),dpi=200); plt.close()
        if "CDD_at_main_th" in df:
            plt.figure(); scatter(plt.gca(),df["CDD_at_main_th"],df["contact_f1"],"CDD vs F1","CDD","F1")
            plt.tight_layout(); plt.savefig(os.path.join(sc_dir,"cdd_vs_f1.png"),dpi=200); plt.close()

    # ---------- heatmaps ----------
    heat_dir = os.path.join(args.out,"heatmaps"); ensure_dir(heat_dir)
    def plot_heat(tab,title,fname,vmin=None,vmax=None):
        if tab is None or tab.size==0: return
        plt.figure(figsize=(6,4))
        im=plt.imshow(tab.values.astype(float),aspect="auto",vmin=vmin,vmax=vmax)
        plt.xticks(range(tab.shape[1]),list(tab.columns))
        plt.yticks(range(tab.shape[0]),list(tab.index))
        plt.colorbar(im,fraction=0.046,pad=0.04)
        plt.title(title); plt.xlabel("Tp_bin"); plt.ylabel("Tl_bin")
        plt.tight_layout(); plt.savefig(os.path.join(heat_dir,fname),dpi=200); plt.close()

    if df["Tl_bin"].notna().any() and df["Tp_bin"].notna().any():
        for col, title in [
            ("MCC_at_main_th","Mean MCC by (Tl_bin,Tp_bin)"),
            ("Long_range_P_at_main_th","Mean Long-Range P by (Tl_bin,Tp_bin)"),
            ("CDD_at_main_th","Mean CDD by (Tl_bin,Tp_bin)")
        ]:
            if col in df:
                tab=heat_by_bins(df,col,"mean")
                plot_heat(tab,title,f"heat_{col}.png",vmin=None,vmax=None)

    # ---------- Top-K tables ----------
    tbl_dir = os.path.join(args.out,"tables"); ensure_dir(tbl_dir)
    for metric,mode in [("MCC_at_main_th","max"),("Long_range_P_at_main_th","max"),("CDD_at_main_th","max")]:
        if metric in df:
            best=topk_table(df,metric,k=args.topk,mode=mode)
            worst=topk_table(df,metric,k=args.topk,mode=("min" if mode=="max" else "max"))
            best.to_csv(os.path.join(tbl_dir,f"top{args.topk}_best_{metric}.csv"),index=False)
            worst.to_csv(os.path.join(tbl_dir,f"top{args.topk}_worst_{metric}.csv"),index=False)

    print(f"[done] all visualizations saved to {args.out}")

if __name__ == "__main__":
    main()
