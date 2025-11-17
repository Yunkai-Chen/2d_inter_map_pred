# contrasive_learning/utils/vis.py
import math
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def _to_2d(x):
    """Accept (Tl,Tp) or (B,Tl,Tp); return a 2D numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    arr = x
    if arr.ndim == 3:
        arr = arr[0]
    return arr.numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)

def plot_pred_vs_true(
    scores,
    y_true=None,
    pair_mask=None,
    apply_sigmoid=True,
    bin_threshold=0.5,
    title=None,
    save_path=None,
    mode="confusion"  # "confusion" or "error"
):
    """
    scores: (Tl,Tp) or (B,Tl,Tp) logits or probabilities
    y_true: (Tl,Tp) or (B,Tl,Tp), binary {0,1} (for contact). Optional.
    pair_mask: (Tl,Tp) or (B,Tl,Tp), bool. Optional (regions not trained).
    apply_sigmoid: if True, treat scores as logits; else assume already probs.
    bin_threshold: threshold for confusion map.
    mode: "confusion" (TP/FP/FN/TN) or "error" (abs diff heatmap).
    """
    S = _to_2d(scores)  # [Tl,Tp]
    if apply_sigmoid:
        S = 1.0 / (1.0 + np.exp(-S))
    Tl, Tp = S.shape

    Y = _to_2d(y_true) if y_true is not None else None
    M = _to_2d(pair_mask).astype(bool) if pair_mask is not None else None

    # Mask outside training/eval region as NaN (so they’ll fade in plots)
    if M is not None:
        Sm = np.where(M, S, np.nan)
        Ym = np.where(M, Y, np.nan) if Y is not None else None
    else:
        Sm = S
        Ym = Y

    # Figure layout
    ncols = 3 if Y is not None else 1
    fig_w = 12 if ncols == 3 else 5
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 4), constrained_layout=True)

    def imshow(ax, A, vmin=0, vmax=1, cmap="magma", title=""):
        im = ax.imshow(A, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
        ax.set_xlabel("protein (Tp)")
        ax.set_ylabel("peptide (Tl)")
        ax.set_title(title)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return im

    # Panel 1: predicted probability
    ax0 = axes if ncols == 1 else axes[0]
    imshow(ax0, Sm, vmin=0, vmax=1, cmap="magma", title="Predicted prob")

    if Y is not None:
        # Panel 2: ground truth
        ax1 = axes[1]
        imshow(ax1, Ym, vmin=0, vmax=1, cmap="magma", title="Ground truth (contact)")

        # Panel 3: confusion or error
        ax2 = axes[2]
        if mode == "confusion":
            Pbin = (Sm >= bin_threshold)
            Ybin = (Ym >= 0.5)
            # Codes: 3=TP, 2=FN, 1=FP, 0=TN; NaN stays NaN
            conf = np.full_like(Sm, np.nan, dtype=float)
            valid = ~np.isnan(Sm) & ~np.isnan(Ym)
            TP = valid & Pbin & Ybin
            FP = valid & Pbin & ~Ybin
            FN = valid & ~Pbin & Ybin
            TN = valid & ~Pbin & ~Ybin
            conf[TP] = 3.0
            conf[FN] = 2.0
            conf[FP] = 1.0
            conf[TN] = 0.0
            # custom colormap: TN gray, FP orange, FN blue, TP green
            cmap = ListedColormap(["#9E9E9E", "#FB8C00", "#1E88E5", "#43A047"])
            im = ax2.imshow(conf, vmin=0, vmax=3, cmap=cmap, interpolation="nearest")
            ax2.set_title(f"Confusion (th={bin_threshold})\nTN/FP/FN/TP")
            ax2.set_xlabel("protein (Tp)")
            ax2.set_ylabel("peptide (Tl)")
            plt.colorbar(im, ax=ax2, ticks=[0,1,2,3], fraction=0.046, pad=0.04)
        else:
            err = np.abs(Sm - Ym)
            imshow(ax2, err, vmin=0, vmax=1, cmap="viridis", title="|Pred - True|")

    if title:
        fig.suptitle(title, y=1.02)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
