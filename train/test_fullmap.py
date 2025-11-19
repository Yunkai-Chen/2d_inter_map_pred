#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_fullmap.py

Test script for evaluating trained full-map contact prediction model.

Usage:
  python test_fullmap.py \
    --checkpoint ../runs/contact_TA_fullmap_V1/best.pt \
    --index ../esm_npz/batch_index.json \
    --splits-dir ../data/splits \
    --test-split test \
    --gt-index ../../generate_phase1_db/full_distance_maps_v2/full_map_cb.json \
    --gt-threshold 8.0 \
    --gt-cap-pep 80 --gt-cap-pro 600 \
    --output-dir ../results/test_V1 \
    --save-predictions
"""

import sys
import os
import json
import argparse
from pathlib import Path
from dataclasses import asdict
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import data loading utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.data_loader_fullmap import (
    read_db_index, read_gt_index, read_split_list,
    ProtPepFullMapDataset, collate_fullmap, strip_suffix_key
)
from model.models_fullmap import PairwiseModelFullMap


def compute_metrics(scores, gt_map, gt_mask, threshold=0.5):
    """
    Compute classification metrics for contact prediction.

    Args:
        scores: [L, L] predicted logits
        gt_map: [Lg, Lg] ground truth contact map (0/1)
        gt_mask: [L, L] mask indicating valid pairs
        threshold: decision threshold for predictions

    Returns:
        dict with metrics: accuracy, precision, recall, f1, auc (if possible)
    """
    # Get valid region
    valid_mask = gt_mask > 0
    row_valid = valid_mask.any(dim=1)
    col_valid = valid_mask.any(dim=0)
    if not torch.equal(row_valid, col_valid):
        row_valid = row_valid & col_valid
    idx = row_valid.nonzero(as_tuple=False).squeeze(-1)

    if idx.numel() == 0:
        return None

    # Crop to valid submatrix
    s_sub = scores.index_select(0, idx).index_select(1, idx)
    g_sub = gt_map
    n = min(s_sub.shape[0], g_sub.shape[0])
    s_sub = s_sub[:n, :n]
    g_sub = g_sub[:n, :n]

    # Convert to numpy for sklearn metrics
    probs = torch.sigmoid(s_sub).cpu().numpy().flatten()
    preds = (probs > threshold).astype(int)
    labels = g_sub.cpu().numpy().flatten()

    # Basic metrics
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Try to compute AUC
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(labels)) > 1:  # Need both classes
            auc = roc_auc_score(labels, probs)
        else:
            auc = None
    except:
        auc = None

    # Top-L/5, L/2, L predictions (common in contact prediction)
    L = len(probs)
    top_metrics = {}
    for top_k_name, top_k in [("L/5", L//5), ("L/2", L//2), ("L", L)]:
        if top_k > 0:
            top_idx = np.argsort(probs)[-top_k:]
            top_precision = labels[top_idx].sum() / top_k if top_k > 0 else 0
            top_metrics[f"precision_top_{top_k_name}"] = float(top_precision)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc) if auc is not None else None,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_pairs": int(tp + tn + fp + fn),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        **top_metrics
    }


@torch.no_grad()
def test_model(model, loader, device, output_dir=None, save_predictions=False):
    """
    Test the model on a dataset.

    Args:
        model: trained model
        loader: DataLoader
        device: torch device
        output_dir: directory to save results (optional)
        save_predictions: whether to save prediction matrices

    Returns:
        dict with aggregated metrics
    """
    model.eval()

    all_metrics = []
    predictions = {}  # Store predictions if needed

    for batch in tqdm(loader, desc="Testing"):
        # Move to device
        batch["full_emb"] = batch["full_emb"].to(device, non_blocking=True)
        batch["full_mask"] = batch["full_mask"].to(device, non_blocking=True)
        batch["gt_map_full"] = batch["gt_map_full"].to(device, non_blocking=True)
        batch["gt_mask_full"] = batch["gt_mask_full"].to(device, non_blocking=True)

        # Forward pass
        model_batch = {
            "full_emb": batch["full_emb"],
            "full_mask": batch["full_mask"],
        }
        out = model(model_batch)
        scores = out["scores"]  # [B, L, L]

        # Compute metrics for each sample
        B = scores.shape[0]
        for b in range(B):
            key = batch["keys"][b]
            s = scores[b]
            g = batch["gt_map_full"][b]
            m = batch["gt_mask_full"][b]

            metrics = compute_metrics(s, g, m)
            if metrics is not None:
                metrics["key"] = key
                all_metrics.append(metrics)

                # Save predictions if requested
                if save_predictions:
                    predictions[key] = {
                        "scores": s.cpu().numpy(),
                        "probs": torch.sigmoid(s).cpu().numpy(),
                        "gt_map": g.cpu().numpy(),
                        "gt_mask": m.cpu().numpy()
                    }

    # Aggregate metrics
    if len(all_metrics) == 0:
        print("[ERROR] No valid samples to evaluate!")
        return None

    # Average metrics
    avg_metrics = {}
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        values = [m[key] for m in all_metrics if m.get(key) is not None]
        if values:
            avg_metrics[f"avg_{key}"] = np.mean(values)
            avg_metrics[f"std_{key}"] = np.std(values)

    # Sum metrics
    for key in ["tp", "fp", "tn", "fn", "n_pairs", "n_positive", "n_negative"]:
        avg_metrics[f"total_{key}"] = sum(m[key] for m in all_metrics)

    # Top-k metrics
    for top_k in ["L/5", "L/2", "L"]:
        key = f"precision_top_{top_k}"
        values = [m[key] for m in all_metrics if key in m]
        if values:
            avg_metrics[f"avg_{key}"] = np.mean(values)

    # Overall precision/recall from totals
    tp = avg_metrics["total_tp"]
    fp = avg_metrics["total_fp"]
    fn = avg_metrics["total_fn"]
    avg_metrics["overall_precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
    avg_metrics["overall_recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    avg_metrics["overall_f1"] = (
        2 * avg_metrics["overall_precision"] * avg_metrics["overall_recall"] /
        (avg_metrics["overall_precision"] + avg_metrics["overall_recall"])
        if (avg_metrics["overall_precision"] + avg_metrics["overall_recall"]) > 0 else 0
    )

    avg_metrics["n_samples"] = len(all_metrics)

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save aggregated metrics
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(avg_metrics, f, indent=2)

        # Save per-sample metrics
        with open(output_dir / "per_sample_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

        # Save predictions
        if save_predictions:
            np.savez_compressed(
                output_dir / "predictions.npz",
                **{k: v for k, v in predictions.items()}
            )

        print(f"\n[INFO] Results saved to {output_dir}")

    return avg_metrics, all_metrics


def main():
    ap = argparse.ArgumentParser()

    # Checkpoint
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt file)")

    # Data
    ap.add_argument("--index", required=True, help="Path to batch_index.json")
    ap.add_argument("--splits-dir", required=True, help="Directory containing split files")
    ap.add_argument("--test-split", default="test", help="Test split name")

    # GT
    ap.add_argument("--gt-index", required=True, help="Path to GT index JSON")
    ap.add_argument("--gt-threshold", type=float, default=8.0)
    ap.add_argument("--gt-cap-pep", type=int, default=None)
    ap.add_argument("--gt-cap-pro", type=int, default=None)

    # Testing
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    # Output
    ap.add_argument("--output-dir", type=str, default=None, help="Directory to save results")
    ap.add_argument("--save-predictions", action="store_true", help="Save prediction matrices")

    args = ap.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load checkpoint
    print(f"[INFO] Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # Load model config
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        print("[ERROR] Checkpoint does not contain model config!")
        print("[INFO] Available keys:", ckpt.keys())
        return

    # Reconstruct model
    from model.models_fullmap import PairModelConfig
    cfg = PairModelConfig(**cfg_dict)
    print(f"[INFO] Model config: {cfg}")

    model = PairwiseModelFullMap(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    print(f"[INFO] Model loaded successfully")
    print(f"[INFO] Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"[INFO] Best val loss: {ckpt.get('best_val', 'unknown')}")

    # Load data
    print(f"[INFO] Loading data...")
    key2npz = read_db_index(args.index)
    gt_key2npz = read_gt_index(args.gt_index)
    test_all = read_split_list(args.splits_dir, args.test_split)

    test_keys = [k for k in test_all if (k in key2npz and strip_suffix_key(k) in gt_key2npz)]
    gt_test_map = {k: gt_key2npz[strip_suffix_key(k)] for k in test_keys}

    print(f"[INFO] Test samples: {len(test_keys)} / {len(test_all)}")

    if len(test_keys) == 0:
        print("[ERROR] No valid test samples found!")
        return

    # Create dataset
    ds_test = ProtPepFullMapDataset(
        key_to_npz=key2npz,
        keys=test_keys,
        gt_key_to_npz=gt_test_map,
        gt_cap_pep=args.gt_cap_pep,
        gt_cap_pro=args.gt_cap_pro,
        gt_threshold=args.gt_threshold,
        gt_as_contact=True  # Always test as contact prediction
    )

    loader_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fullmap,
        drop_last=False
    )

    # Test
    print(f"\n{'='*60}")
    print("Starting evaluation...")
    print(f"{'='*60}\n")

    avg_metrics, per_sample = test_model(
        model, loader_test, device,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions
    )

    # Print results
    print(f"\n{'='*60}")
    print("Test Results")
    print(f"{'='*60}")
    print(f"Number of samples: {avg_metrics['n_samples']}")
    print(f"\nOverall Metrics (from totals):")
    print(f"  Precision: {avg_metrics['overall_precision']:.4f}")
    print(f"  Recall:    {avg_metrics['overall_recall']:.4f}")
    print(f"  F1:        {avg_metrics['overall_f1']:.4f}")

    print(f"\nAverage Metrics (per-sample mean):")
    print(f"  Accuracy:  {avg_metrics.get('avg_accuracy', 0):.4f} ± {avg_metrics.get('std_accuracy', 0):.4f}")
    print(f"  Precision: {avg_metrics.get('avg_precision', 0):.4f} ± {avg_metrics.get('std_precision', 0):.4f}")
    print(f"  Recall:    {avg_metrics.get('avg_recall', 0):.4f} ± {avg_metrics.get('std_recall', 0):.4f}")
    print(f"  F1:        {avg_metrics.get('avg_f1', 0):.4f} ± {avg_metrics.get('std_f1', 0):.4f}")
    if avg_metrics.get('avg_auc'):
        print(f"  AUC:       {avg_metrics.get('avg_auc', 0):.4f} ± {avg_metrics.get('std_auc', 0):.4f}")

    print(f"\nTop-k Precision:")
    for k in ["L/5", "L/2", "L"]:
        key = f"avg_precision_top_{k}"
        if key in avg_metrics:
            print(f"  Top-{k}: {avg_metrics[key]:.4f}")

    print(f"\nConfusion Matrix (totals):")
    print(f"  TP: {avg_metrics['total_tp']:,}")
    print(f"  FP: {avg_metrics['total_fp']:,}")
    print(f"  TN: {avg_metrics['total_tn']:,}")
    print(f"  FN: {avg_metrics['total_fn']:,}")
    print(f"  Total pairs: {avg_metrics['total_n_pairs']:,}")
    print(f"  Positive: {avg_metrics['total_n_positive']:,}")
    print(f"  Negative: {avg_metrics['total_n_negative']:,}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
