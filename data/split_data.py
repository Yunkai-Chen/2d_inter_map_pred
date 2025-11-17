#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_stratified_lists.py

从 batch_index.json（你批量生成 ESM3 npz 后的索引）读样本，
按三项特征做分层并切分 train/val/test，落盘成固定清单。

特征（来自每个 npz 的 metadata）：
- receptor_len = metadata["lengths"]["protein"]["raw"]    （若缺失则从 protein_seq_raw 长度回退）
- peptide_len  = metadata["lengths"]["peptide"]["raw"]    （同理回退）
- interface_len = len(metadata["source_meta"]["interface_globals"].split())  （缺失则 0）

分桶策略：
- 对每个特征使用分位点（quantile）生成离散 bins（默认：6/5/5 桶）
- 形成联合标签 "R{rb}-P{pb}-I{ib}" 做分层采样
- 对于极小桶（样本数过少无法严格按比例分配），优先保证有样本进入 train；其余尽量按比例入 val/test

输出：
- outdir/train.txt, val.txt, test.txt   （每行一个 key）
- outdir/summary.csv                    （每个 key 的统计和分桶标签、split）
- outdir/split_summary.json             （总体统计、分桶边界、每桶分布）

用法：
  python split_stratified_lists.py \
    --index esm_npz/batch_index.json \
    --outdir splits/ \
    --train 0.8 --val 0.1 --test 0.1 \
    --bins-receptor 6 --bins-peptide 5 --bins-interface 5 \
    --seed 42
"""

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import csv
from collections import defaultdict

# ------------------ 路径解析（兼容相对路径） ------------------ #

def resolve_npz_path(raw_path: str, index_file: Path) -> Path | None:
    p = Path(raw_path)
    if p.is_absolute():
        return p if p.exists() else None
    idx_dir = index_file.parent
    candidates = [
        idx_dir / p,           # relative to index dir
        idx_dir.parent / p,    # index's parent
        Path.cwd() / p,        # CWD
    ]
    if p.parts and p.parts[0] == idx_dir.name:
        p2 = Path(*p.parts[1:])
        candidates += [idx_dir / p2, idx_dir.parent / p2, Path.cwd() / p2]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None

def load_db(index_path: str, recheck_npz: bool = True) -> List[Dict[str, Any]]:
    index_file = Path(index_path).resolve()
    idx = json.loads(index_file.read_text(encoding="utf-8"))
    processed = idx.get("processed", {})
    out = []
    for key, info in processed.items():
        raw = info.get("npz_path") or f"{key}_esm3_alltokens.npz"
        resolved = resolve_npz_path(raw, index_file)
        if not recheck_npz or (resolved and resolved.exists()):
            out.append({"key": key, "npz_path": str(resolved)})
    return out

# ------------------ 只读 metadata（避免加载大矩阵） ------------------ #

def read_meta_from_npz(npz_path: str) -> Dict[str, Any] | None:
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            # metadata 是 json 字符串
            meta_raw = z.get("metadata", None)
            if meta_raw is None:
                # 回退：尽量从序列推长度
                m = {}
            else:
                # npz 中字符串以 0-D object ndarray 存，取 item() 再 loads
                if hasattr(meta_raw, "shape") and meta_raw.shape == ():
                    m = json.loads(str(meta_raw.item()))
                else:
                    m = json.loads(str(meta_raw))
            # 回退获取长度
            def _len_from_seq(field_name):
                arr = z.get(field_name, None)
                if arr is None:
                    return None
                try:
                    s = arr.item() if hasattr(arr, "shape") and arr.shape == () else arr[0]
                    return len(str(s))
                except Exception:
                    return None
            # 确保 lengths 存在
            lengths = m.setdefault("lengths", {"protein": {}, "peptide": {}})

            # receptor_len
            rec_len = lengths["protein"].get("raw")
            if rec_len is None:
                rec_len = _len_from_seq("protein_seq_raw")
                if rec_len is None:
                    rec_len = _len_from_seq("protein_seq_used")
            lengths["protein"]["raw"] = int(rec_len or 0)

            # peptide_len
            pep_len = lengths["peptide"].get("raw")
            if pep_len is None:
                pep_len = _len_from_seq("peptide_seq_raw")
                if pep_len is None:
                    pep_len = _len_from_seq("peptide_seq_used")
            lengths["peptide"]["raw"] = int(pep_len or 0)

            # interface_len
            src = m.get("source_meta", {})
            ig = src.get("interface_globals", "")
            if isinstance(ig, str) and ig.strip():
                iface_len = len(ig.split())
            else:
                iface_len = 0
            m["interface_len"] = int(iface_len)

            return m
    except Exception:
        return None

# ------------------ 分位分桶 ------------------ #

def compute_quantile_edges(values: List[int], nbins: int) -> List[float]:
    """
    基于分位点生成边界（不含最小值，含最大值），返回单调递增列表。
    避免重复边界，必要时回退到等宽 bins。
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return [0.0]
    if nbins <= 1:
        return [float(arr.max())]

    qs = np.linspace(0, 1, nbins + 1)  # 0..1 共 nbins+1 个点
    edges = np.quantile(arr, qs)
    # 去重
    edges = np.unique(edges)
    if len(edges) <= 2:
        # 退回等宽
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax == vmin:
            return [vmax]
        edges = np.linspace(vmin, vmax, nbins + 1)
    return list(edges)

def assign_bins(values: List[int], edges: List[float]) -> List[int]:
    """
    将每个值分配到 bin 索引（0..nbins-1）。
    使用 numpy.digitize，右开区间策略。
    """
    arr = np.asarray(values, dtype=float)
    # digitize 返回 1..len(edges)-1，转成 0-based
    idx = np.digitize(arr, edges[1:-1], right=False)  # 用中间边界切分
    return idx.tolist()

# ------------------ 分层切分 ------------------ #

def stratified_split(
    items: List[Dict[str, Any]],
    strat_keys: List[str],
    ratios: Tuple[float, float, float],
    seed: int = 42
) -> Tuple[List[str], List[str], List[str], Dict[str, Dict[str, int]]]:
    """
    基于联合分层标签 strat_keys（与 items 对齐）进行切分。
    返回 (train_keys, val_keys, test_keys, per_stratum_counts)
    """
    assert len(items) == len(strat_keys)
    rng = np.random.default_rng(seed)

    # 收集每个桶里的索引
    strata_to_idx: Dict[str, List[int]] = defaultdict(list)
    for i, sk in enumerate(strat_keys):
        strata_to_idx[sk].append(i)

    train, val, test = [], [], []
    counts_summary: Dict[str, Dict[str, int]] = {}

    tr_ratio, va_ratio, te_ratio = ratios
    total_ratio = tr_ratio + va_ratio + te_ratio
    tr_ratio /= total_ratio
    va_ratio /= total_ratio
    te_ratio /= total_ratio

    for sk, idxs in strata_to_idx.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n = len(idxs)
        # 计算目标数量（四舍五入后再纠偏）
        n_tr = int(round(n * tr_ratio))
        n_va = int(round(n * va_ratio))
        n_te = n - n_tr - n_va
        # 小桶纠偏保证非负
        if n_tr < 0: n_tr = 0
        if n_va < 0: n_va = 0
        if n_te < 0: n_te = 0
        # 处理极端情况：把尽可能多留给 train
        taken = 0
        tr = idxs[taken:taken+n_tr]; taken += len(tr)
        va = idxs[taken:taken+n_va]; taken += len(va)
        te = idxs[taken:]
        # 记录
        train.extend([items[i]["key"] for i in tr])
        val.extend([items[i]["key"] for i in va])
        test.extend([items[i]["key"] for i in te])
        counts_summary[sk] = {"total": n, "train": len(tr), "val": len(va), "test": len(te)}

    return train, val, test, counts_summary

# ------------------ 主流程 ------------------ #

def main():
    ap = argparse.ArgumentParser(description="Stratified split on receptor/peptide length & interface size; save fixed key lists.")
    ap.add_argument("--index", required=True, help="Path to batch_index.json")
    ap.add_argument("--outdir", required=True, help="Output dir for train/val/test lists and summaries")
    ap.add_argument("--train", type=float, default=0.8, help="Train ratio")
    ap.add_argument("--val",   type=float, default=0.1, help="Val ratio")
    ap.add_argument("--test",  type=float, default=0.1, help="Test ratio")
    ap.add_argument("--bins-receptor", type=int, default=6, help="Number of quantile bins for receptor length")
    ap.add_argument("--bins-peptide",  type=int, default=5, help="Number of quantile bins for peptide length")
    ap.add_argument("--bins-interface",type=int, default=5, help="Number of quantile bins for interface length")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="Only print stats, do not write files")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 读取 DB
    items = load_db(args.index, recheck_npz=True)
    if not items:
        print(f"Loaded 0 items from {args.index}. Check your path.")
        return
    print(f"Loaded {len(items)} items from {args.index}")

    # 扫描 meta（只取小字段）
    rec_lens, pep_lens, iface_lens = [], [], []
    metas: Dict[str, Dict[str, Any]] = {}
    miss_meta = 0

    for it in items:
        m = read_meta_from_npz(it["npz_path"])
        if m is None:
            miss_meta += 1
            continue
        key = it["key"]
        metas[key] = m
        rec_lens.append(int(m["lengths"]["protein"]["raw"]))
        pep_lens.append(int(m["lengths"]["peptide"]["raw"]))
        iface_lens.append(int(m.get("interface_len", 0)))

    print(f"Meta OK: {len(metas)} | missing: {miss_meta}")

    # 过滤掉没有 meta 的样本
    items = [it for it in items if it["key"] in metas]
    if not items:
        print("No items with metadata. Exit.")
        return

    # 分位边界
    rec_edges = compute_quantile_edges(rec_lens, args.bins_receptor)
    pep_edges = compute_quantile_edges(pep_lens, args.bins_peptide)
    ifc_edges = compute_quantile_edges(iface_lens, args.bins_interface)

    # 分配 bin
    rec_bins = assign_bins(rec_lens, rec_edges)
    pep_bins = assign_bins(pep_lens, pep_edges)
    ifc_bins = assign_bins(iface_lens, ifc_edges)

    # 联合分层标签
    keys_order = [it["key"] for it in items]
    strat_keys = []
    for i in range(len(items)):
        strat_keys.append(f"R{rec_bins[i]}-P{pep_bins[i]}-I{ifc_bins[i]}")

    # 切分
    train_keys, val_keys, test_keys, per_bucket = stratified_split(
        items=items,
        strat_keys=strat_keys,
        ratios=(args.train, args.val, args.test),
        seed=args.seed
    )

    print(f"Split sizes -> train: {len(train_keys)}, val: {len(val_keys)}, test: {len(test_keys)}")

    if args.dry_run:
        # 打印一些预览
        print("Train samples (first 5):", train_keys[:5])
        print("Val   samples (first 5):", val_keys[:5])
        print("Test  samples (first 5):", test_keys[:5])
        return

    # 固定清单落盘
    (outdir / "train.txt").write_text("\n".join(train_keys) + "\n", encoding="utf-8")
    (outdir / "val.txt").write_text("\n".join(val_keys) + "\n", encoding="utf-8")
    (outdir / "test.txt").write_text("\n".join(test_keys) + "\n", encoding="utf-8")

    # 保存 summary.csv（逐样本）
    csv_path = outdir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "receptor_len", "peptide_len", "interface_len", "stratum", "split"])
        # 重建 i->key 映射
        for i, it in enumerate(items):
            key = it["key"]
            m = metas[key]
            split = "train" if key in train_keys else ("val" if key in val_keys else "test")
            w.writerow([
                key,
                int(m["lengths"]["protein"]["raw"]),
                int(m["lengths"]["peptide"]["raw"]),
                int(m.get("interface_len", 0)),
                strat_keys[i],
                split
            ])

    # 保存 split_summary.json（总体信息、桶边界、每桶分布）
    summary = {
        "counts": {"train": len(train_keys), "val": len(val_keys), "test": len(test_keys)},
        "bin_edges": {
            "receptor": rec_edges,
            "peptide": pep_edges,
            "interface": ifc_edges
        },
        "per_bucket": per_bucket,
        "seed": args.seed,
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "bins": {
            "receptor": args.bins_receptor,
            "peptide": args.bins_peptide,
            "interface": args.bins_interface
        }
    }
    (outdir / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✔️  Saved lists to: {outdir}")
    print(f"   - train.txt / val.txt / test.txt")
    print(f"   - summary.csv")
    print(f"   - split_summary.json")

if __name__ == "__main__":
    main()
