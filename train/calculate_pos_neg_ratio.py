#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate_pos_neg_ratio.py

统计整个数据集的正负样本比例，用于确定合适的 pos_weight。

Usage:
  python calculate_pos_neg_ratio.py \
    --index ../esm_npz/batch_index.json \
    --splits-dir ../data/splits \
    --train-split train \
    --val-split val \
    --gt-index ../../generate_phase1_db/full_distance_maps_v2/full_map_cb.json \
    --gt-threshold 8.0 \
    --gt-cap-pep 80 \
    --gt-cap-pro 600
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

# 导入数据加载工具
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.data_loader_fullmap import (
    read_db_index, read_gt_index, read_split_list,
    load_gt_full, strip_suffix_key
)


def calculate_ratio_for_samples(
    keys: List[str],
    key_to_npz: Dict[str, str],
    gt_key_to_npz: Dict[str, str],
    gt_threshold: float,
    gt_cap_pep: int,
    gt_cap_pro: int,
    split_name: str = "dataset"
) -> Dict[str, any]:
    """
    统计一组样本的正负样本比例

    Returns:
        dict with keys: total_positive, total_negative, ratio, samples_processed
    """
    total_positive = 0
    total_negative = 0
    samples_processed = 0
    sample_stats = []  # 记录每个样本的统计信息

    print(f"\n{'='*60}")
    print(f"Processing {split_name}: {len(keys)} samples")
    print(f"{'='*60}")

    for key in tqdm(keys, desc=f"Analyzing {split_name}"):
        # 找到对应的 GT 文件
        if key in gt_key_to_npz:
            gt_path = gt_key_to_npz[key]
        else:
            key_base = strip_suffix_key(key)
            if key_base in gt_key_to_npz:
                gt_path = gt_key_to_npz[key_base]
            else:
                print(f"[WARN] No GT found for {key}, skipping")
                continue

        try:
            # 加载 GT map
            gt_full = load_gt_full(
                gt_path,
                as_contact=True,
                threshold=gt_threshold,
                gt_cap_pep=gt_cap_pep,
                gt_cap_pro=gt_cap_pro
            )

            gt_map = gt_full["map"]
            gt_mask = gt_full["mask"]

            # 只统计有效区域（mask=True 的部分）
            valid_region = gt_map[gt_mask]

            # 统计正负样本
            n_positive = int((valid_region > 0.5).sum())
            n_negative = int((valid_region <= 0.5).sum())

            total_positive += n_positive
            total_negative += n_negative
            samples_processed += 1

            # 记录单个样本的信息
            sample_ratio = n_negative / n_positive if n_positive > 0 else float('inf')
            sample_stats.append({
                'key': key,
                'positive': n_positive,
                'negative': n_negative,
                'ratio': sample_ratio,
                'total_pairs': n_positive + n_negative
            })

        except Exception as e:
            print(f"[ERROR] Failed to process {key}: {e}")
            continue

    # 计算总体比例
    if total_positive > 0:
        overall_ratio = total_negative / total_positive
    else:
        overall_ratio = float('inf')

    return {
        'total_positive': total_positive,
        'total_negative': total_negative,
        'ratio': overall_ratio,
        'samples_processed': samples_processed,
        'sample_stats': sample_stats
    }


def print_statistics(results: Dict, split_name: str):
    """打印统计结果"""
    print(f"\n{'='*60}")
    print(f"Results for {split_name}")
    print(f"{'='*60}")
    print(f"Samples processed:    {results['samples_processed']}")
    print(f"Total positive pairs: {results['total_positive']:,}")
    print(f"Total negative pairs: {results['total_negative']:,}")
    print(f"Total pairs:          {results['total_positive'] + results['total_negative']:,}")
    print(f"\nPositive ratio:       {results['total_positive'] / (results['total_positive'] + results['total_negative']) * 100:.2f}%")
    print(f"Negative ratio:       {results['total_negative'] / (results['total_positive'] + results['total_negative']) * 100:.2f}%")
    print(f"\n{'*'*60}")
    print(f"NEGATIVE / POSITIVE RATIO: {results['ratio']:.2f}")
    print(f"{'*'*60}")
    print(f"\nRecommended pos_weight:")
    print(f"  - Exact balance:     {results['ratio']:.1f}")
    print(f"  - Conservative (×0.5): {results['ratio']*0.5:.1f}")
    print(f"  - Aggressive (×1.5):   {results['ratio']*1.5:.1f}")
    print(f"  - Very aggressive (×2): {results['ratio']*2:.1f}")

    # 打印样本分布统计
    sample_stats = results['sample_stats']
    if sample_stats:
        ratios = [s['ratio'] for s in sample_stats if s['ratio'] != float('inf')]
        if ratios:
            print(f"\nPer-sample ratio distribution:")
            print(f"  Min ratio:  {min(ratios):.2f}")
            print(f"  Max ratio:  {max(ratios):.2f}")
            print(f"  Mean ratio: {np.mean(ratios):.2f}")
            print(f"  Median ratio: {np.median(ratios):.2f}")
            print(f"  Std ratio:  {np.std(ratios):.2f}")

        # 打印前10个样本的详细信息
        print(f"\nTop 10 samples with most pairs:")
        sorted_samples = sorted(sample_stats, key=lambda x: x['total_pairs'], reverse=True)
        for i, s in enumerate(sorted_samples[:10], 1):
            print(f"  {i:2d}. {s['key']:40s} | pos={s['positive']:6d} neg={s['negative']:6d} ratio={s['ratio']:6.2f}")


def main():
    ap = argparse.ArgumentParser(description="Calculate positive/negative ratio for the entire dataset")

    # Data paths
    ap.add_argument("--index", required=True, help="Path to batch_index.json")
    ap.add_argument("--splits-dir", required=True, help="Directory containing train/val/test.txt")
    ap.add_argument("--train-split", default="train", help="Train split name")
    ap.add_argument("--val-split", default="val", help="Validation split name")

    # GT settings
    ap.add_argument("--gt-index", required=True, help="Path to GT index JSON")
    ap.add_argument("--gt-threshold", type=float, default=8.0, help="Distance threshold for contact")
    ap.add_argument("--gt-cap-pep", type=int, default=None, help="Max peptide length")
    ap.add_argument("--gt-cap-pro", type=int, default=None, help="Max protein length")

    # Options
    ap.add_argument("--analyze-train", action="store_true", default=True, help="Analyze train split")
    ap.add_argument("--analyze-val", action="store_true", help="Analyze validation split")
    ap.add_argument("--save-stats", type=str, default=None, help="Save detailed stats to file")

    args = ap.parse_args()

    # Load indices
    print("Loading indices...")
    key2npz = read_db_index(args.index)
    gt_key2npz = read_gt_index(args.gt_index)

    # Read splits
    train_all = read_split_list(args.splits_dir, args.train_split)
    val_all = read_split_list(args.splits_dir, args.val_split)

    # Filter usable keys
    train_keys = [k for k in train_all if (k in key2npz and strip_suffix_key(k) in gt_key2npz)]
    val_keys = [k for k in val_all if (k in key2npz and strip_suffix_key(k) in gt_key2npz)]

    print(f"\nFound {len(train_keys)} usable train samples (out of {len(train_all)})")
    print(f"Found {len(val_keys)} usable val samples (out of {len(val_all)})")

    # Analyze train split
    if args.analyze_train and train_keys:
        train_results = calculate_ratio_for_samples(
            train_keys, key2npz, gt_key2npz,
            args.gt_threshold, args.gt_cap_pep, args.gt_cap_pro,
            split_name="TRAIN"
        )
        print_statistics(train_results, "TRAIN")

    # Analyze validation split
    if args.analyze_val and val_keys:
        val_results = calculate_ratio_for_samples(
            val_keys, key2npz, gt_key2npz,
            args.gt_threshold, args.gt_cap_pep, args.gt_cap_pro,
            split_name="VALIDATION"
        )
        print_statistics(val_results, "VALIDATION")

    # Save detailed statistics if requested
    if args.save_stats:
        import json
        output = {
            'train': train_results if args.analyze_train else None,
            'val': val_results if args.analyze_val else None,
            'settings': {
                'gt_threshold': args.gt_threshold,
                'gt_cap_pep': args.gt_cap_pep,
                'gt_cap_pro': args.gt_cap_pro,
            }
        }
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj

        output = convert_types(output)

        with open(args.save_stats, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nDetailed statistics saved to {args.save_stats}")

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
