#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_loader.py (GT-only capping; embeddings untouched)

- 读取 batch_index.json + splits/{train,val,test}.txt
- 返回 *全量 token* 的 embedding 和掩码（不裁剪）
- 【GT 侧】可选：读取 GT 索引 (--gt-index)，为每个样本加载 cross-domain 距离/接触图：
    batch["gt_map"] : (B, Tl_max, Tp_max)  # 行=肽，列=蛋白（每样本内左上角有效）
    batch["gt_mask"]: (B, Tl_max, Tp_max)  # 有效区域 True
- 【新增】只对 GT 做上限裁剪（pep/pro 分别可设 cap），embedding 不改动

示例：
  python data_loader.py \
    --index ../esm_npz/batch_index.json \
    --splits-dir splits \
    --split train \
    --gt-index ../cb_maps.json \
    --gt-as-contact \
    --gt-threshold 8.0 \
    --gt-cap-pep 80 \
    --gt-cap-pro 600 \
    --dry-run --preview 1 --print-values
"""
from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# 可选的序列读取工具（保留以兼容其他脚本调用，但本文件不对 embedding 侧做过滤）
# ---------------------------
# 放在文件头部合适位置
AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

def _labels_to_seq(labels):
    """labels 元素形如 'ALA:A23' 或 'X:B45'；转一字母序列（未知→'X'）。"""
    seq = []
    for lab in labels:
        lab = str(lab)
        aa3 = lab.split(':', 1)[0].strip().upper()
        seq.append(AA3_TO_1.get(aa3, 'X'))
    return ''.join(seq)

def read_seq_index(seq_index_path: str) -> Dict[str, str]:
    """
    读取 ligand_seq：
      支持两种结构：
        { stem: {... "Sequences": {"ligand_seq": "..."} ...}, ... }
      或  { "processed": { stem: {... 同上 ...} } }
    返回: { stem: ligand_seq_str }
    """
    idx_file = Path(seq_index_path).resolve()
    obj = json.loads(idx_file.read_text(encoding="utf-8"))
    items = obj.get("processed", obj)

    out: Dict[str, str] = {}
    for k, rec in items.items():
        if not isinstance(rec, dict):
            continue
        seq = rec.get("ligand_seq")
        if seq is None:
            seq = (rec.get("Sequences") or {}).get("ligand_seq")
        if isinstance(seq, str) and len(seq) > 0:
            out[k] = seq
    return out

def read_seq_index_receptor(seq_index_path: str) -> Dict[str, str]:
    """
    读取 receptor_seq（支持两种结构）：
      { stem: {... "Sequences": {"receptor_seq": "..."} ...}, ... }
      或  { "processed": { stem: {... 同上 ...} } }
    返回: { stem: receptor_seq_str }
    """
    idx_file = Path(seq_index_path).resolve()
    obj = json.loads(idx_file.read_text(encoding="utf-8"))
    items = obj.get("processed", obj)

    out: Dict[str, str] = {}
    for k, rec in items.items():
        if not isinstance(rec, dict):
            continue
        seq = rec.get("receptor_seq")
        if seq is None:
            seq = (rec.get("Sequences") or {}).get("receptor_seq")
        if isinstance(seq, str) and len(seq) > 0:
            out[k] = seq
    return out

# ---------------------------
# 工具：路径解析 / split 文件
# ---------------------------

def strip_suffix_key(k: str) -> str:
    """去掉 split 键里的 '_nomutation' 后缀，用于与 GT/seq 索引对齐。"""
    return k[:-11] if k.endswith("_nomutation") else k

def resolve_npz_path(raw_path: str, index_file: Path) -> Optional[Path]:
    """解析相对/绝对路径；尝试去掉重复的首段目录名（如 distance_maps/distance_maps）。"""
    p = Path(raw_path)
    if p.is_absolute():
        return p if p.exists() else None
    base = index_file.parent
    cands = [base / p, base.parent / p, Path.cwd() / p]

    # 若首段等于 base 目录名，丢弃首段再试
    if p.parts and p.parts[0] == base.name:
        p2 = Path(*p.parts[1:])
        cands += [base / p2, base.parent / p2, Path.cwd() / p2]

    # 只用文件名在 base 下找
    cands.append(base / p.name)

    for c in cands:
        if c.exists():
            return c.resolve()
    return None

def read_db_index(index_path: str) -> Dict[str, str]:
    """读取 embedding 索引（结构：{"processed": {key: {"npz_path": ...}}}）"""
    idx_file = Path(index_path).resolve()
    idx = json.loads(idx_file.read_text(encoding="utf-8"))
    processed = idx.get("processed", {})
    out: Dict[str, str] = {}
    for k, info in processed.items():
        raw = info.get("npz_path") or f"{k}_esm3_alltokens.npz"
        resolved = resolve_npz_path(raw, idx_file)
        if resolved:
            out[k] = str(resolved)
    return out

def read_gt_index(gt_index_path: str) -> Dict[str, str]:
    """
    读取 GT 索引（results.json / cb_maps.json 风格）：
      { stem: { "npz_path": "xxx.npz", ... }, ... }
    也兼容 {"processed": {...}} 结构。
    """
    idx_file = Path(gt_index_path).resolve()
    obj = json.loads(idx_file.read_text(encoding="utf-8"))
    items = obj.get("processed", obj)
    out: Dict[str, str] = {}
    for k, rec in items.items():
        raw = rec.get("npz_path") or rec.get("path") if isinstance(rec, dict) else None
        if not raw:
            continue
        resolved = resolve_npz_path(raw, idx_file)
        if resolved:
            out[k] = str(resolved)
    return out

def read_split_list(splits_dir: str, split_name: str) -> List[str]:
    """
    从 splits/{split}.txt 读取 key 列表（保留原样，不删后缀）。
    - 忽略空行与以 '#' 开头的注释行
    - 保序去重
    """
    path = Path(splits_dir) / f"{split_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    seen = set()
    for ln in raw_lines:
        k = ln.strip()
        if not k or k.startswith("#"):
            continue
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out

# ---------------------------
# GT 工具：从 .npz 载入并截出 cross-domain
# ---------------------------

def _load_npz_meta(npz_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    d = np.load(npz_path, allow_pickle=True)
    D = d["distance_map"].astype(np.float32, copy=False)
    labels = d["residue_labels"].astype(str)
    meta_raw = d["metadata"]
    if hasattr(meta_raw, "item"):
        meta_raw = meta_raw.item()
    if isinstance(meta_raw, (bytes, bytearray)):
        meta_raw = meta_raw.decode("utf-8")
    meta = json.loads(meta_raw)
    return D, labels, meta

def _extract_cross_block(
    D_full: np.ndarray,
    pep_len: int,
    pro_len: int,
    order: str = "peptide_protein",
    labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[List[str]], Optional[List[str]]]:
    D_full = np.asarray(D_full)
    p, q = int(pep_len), int(pro_len)
    if D_full.shape != (p + q, p + q):
        raise ValueError(f"full 矩阵尺寸不符: got {D_full.shape}, expected {(p+q, p+q)}")
    ord_norm = (order or "").lower().strip()
    if ord_norm == "peptide_protein":
        D_cross = D_full[:p, p:]
        pep_labels = labels[:p].tolist() if labels is not None else None
        pro_labels = labels[p:].tolist() if labels is not None else None
    elif ord_norm == "protein_peptide":
        D_cross = D_full[p:, :p]
        pep_labels = labels[p:].tolist() if labels is not None else None
        pro_labels = labels[:p].tolist() if labels is not None else None
    else:
        raise ValueError(f"未知 order: {order!r}")
    return D_cross.astype(np.float32, copy=False), pep_labels, pro_labels

def load_gt_cross(
    gt_npz_path: str,
    expect_order: str = "peptide_protein",
    as_contact: bool = True,
    threshold: float = 8.0,
) -> Dict[str, Any]:
    """
    载入一个 GT npz，返回：
      - map: np.ndarray  (Tl, Tp)
      - kind: "contact" | "distance"
      - pep_len, pro_len
      - pep_seq, pro_seq  （若 GT npz 含 residue_labels）
    """
    D, labels, meta = _load_npz_meta(gt_npz_path)
    p_len = int(meta["peptide_length"])
    pro_len = int(meta["protein_length"])
    order = (meta.get("order") or expect_order).lower().strip()

    pep_labels = None
    pro_labels = None

    if D.shape == (p_len, pro_len):
        # 已是 cross 形状；labels 通常仍是 (p_len+pro_len) 的串
        D_cross = D
        if labels is not None and labels.size == (p_len + pro_len):
            if order == "peptide_protein":
                pep_labels = labels[:p_len].tolist()
                pro_labels = labels[p_len:].tolist()
            elif order == "protein_peptide":
                pro_labels = labels[:pro_len].tolist()
                pep_labels = labels[pro_len:].tolist()
    elif D.shape == (p_len + pro_len, p_len + pro_len):
        # full 形状，走统一切块并同步切 labels
        D_cross, pep_labels, pro_labels = _extract_cross_block(D, p_len, pro_len, order, labels=labels)
    else:
        raise ValueError(f"GT distance_map 形状 {D.shape} 与 meta(p={p_len}, pro={pro_len}) 不匹配。")

    # contact or distance
    if as_contact:
        finite = np.isfinite(D_cross)
        M = (finite & (D_cross < threshold)).astype(np.uint8)
        kind = "contact"
        map_arr = M
    else:
        map_arr = D_cross.astype(np.float32, copy=False)
        kind = "distance"

    # 一字母序列（若有 labels）
    pep_seq = _labels_to_seq(pep_labels) if pep_labels is not None else None
    pro_seq = _labels_to_seq(pro_labels) if pro_labels is not None else None

    return {
        "map": map_arr,
        "kind": kind,
        "pep_len": p_len,
        "pro_len": pro_len,
        "distance_type": meta.get("distance_type"),
        "order": order,
        "pep_seq": pep_seq,
        "pro_seq": pro_seq,
    }

# ---------------------------
# Dataset：返回全量 token + 掩码 + （可选）GT（只在 GT 侧做上限裁剪）
# ---------------------------

class ProtPepFullTokenDataset(Dataset):
    def __init__(
        self,
        key_to_npz: Dict[str, str],
        keys: List[str],
        strict_fixed: bool = False,
        gt_key_to_npz: Optional[Dict[str, str]] = None,
        gt_as_contact: bool = True,
        gt_threshold: float = 8.0,
        # --- 新增：仅 GT 侧上限（embedding 侧不改动）---
        gt_cap_pep: Optional[int] = None,
        gt_cap_pro: Optional[int] = None,
    ):
        self.key_to_npz = key_to_npz
        self.keys = keys
        self.strict_fixed = strict_fixed

        self.gt_key_to_npz = gt_key_to_npz or {}
        self.use_gt = len(self.gt_key_to_npz) > 0
        self.gt_as_contact = gt_as_contact
        self.gt_threshold = gt_threshold

        # GT capping
        self.gt_cap_pep = int(gt_cap_pep) if gt_cap_pep is not None else None
        self.gt_cap_pro = int(gt_cap_pro) if gt_cap_pro is not None else None

    def __len__(self):
        return len(self.keys)

    def _load_one(self, key: str) -> Dict[str, Any]:
        f = self.key_to_npz[key]
        d = np.load(f, allow_pickle=True)

        # 全量 token 的 embedding（含 <cls>/<eos>/padding）
        p_emb = d["protein_embedding"]        # (Tp, D)
        l_emb = d["peptide_embedding"]        # (Tl, D)

        # 掩码
        p_is_special     = d["protein_is_special"].astype(bool)            # (Tp,)
        p_is_padding     = d["protein_is_padding"].astype(bool)            # (Tp,)
        p_residue_window = d["protein_residue_mask_real"].astype(bool)     # (Tp,)
        p_valid_real     = p_residue_window & (~p_is_padding)              # 真实残基

        l_is_special     = d["peptide_is_special"].astype(bool)            # (Tl,)
        l_is_padding     = d["peptide_is_padding"].astype(bool)            # (Tl,)
        l_residue_window = d["peptide_residue_mask_real"].astype(bool)     # (Tl,)
        l_valid_real     = l_residue_window & (~l_is_padding)

        out = {
            "key": key,
            "protein_emb": torch.from_numpy(p_emb).float(),
            "peptide_emb": torch.from_numpy(l_emb).float(),
            "protein_masks": {
                "is_special": torch.from_numpy(p_is_special),
                "is_residue_window": torch.from_numpy(p_residue_window),
                "is_padding": torch.from_numpy(p_is_padding),
                "valid_real_residue": torch.from_numpy(p_valid_real),
            },
            "peptide_masks": {
                "is_special": torch.from_numpy(l_is_special),
                "is_residue_window": torch.from_numpy(l_residue_window),
                "is_padding": torch.from_numpy(l_is_padding),
                "valid_real_residue": torch.from_numpy(l_valid_real),
            },
        }

        # 真实残基计数（供日志/对齐信息打印）
        out["prot_len_real"] = int(p_valid_real.sum())
        out["pep_len_real"]  = int(l_valid_real.sum())



        # （可选）加载 GT —— 只在 GT 侧做上限裁剪
        if self.use_gt:
            gt_path = self.gt_key_to_npz.get(key)
            if gt_path is None:
                raise FileNotFoundError(f"[GT] key={key} 无 GT npz 记录")

            gt = load_gt_cross(
                gt_path,
                expect_order="peptide_protein",
                as_contact=self.gt_as_contact,
                threshold=self.gt_threshold,
            )

            # 仅当 *原始 GT 长度* 超过 cap 时才裁剪；embedding 侧不改动
            pep_cap = gt["pep_len"]
            pro_cap = gt["pro_len"]
            clipped = False

            if (self.gt_cap_pep is not None) and (gt["pep_len"] > self.gt_cap_pep):
                pep_cap = self.gt_cap_pep
                clipped = True
            if (self.gt_cap_pro is not None) and (gt["pro_len"] > self.gt_cap_pro):
                pro_cap = self.gt_cap_pro
                clipped = True

            gt_map_np = gt["map"][:pep_cap, :pro_cap]

            if clipped:
                # 打印一次说明（不打断训练）
                pep_ok = (pep_cap <= out["pep_len_real"])
                pro_ok = (pro_cap <= out["prot_len_real"])
                print(
                    f"[GT-CLIP] key={key} "
                    f"raw_gt=(Tl={gt['pep_len']},Tp={gt['pro_len']}) "
                    f"-> clip_to=(Tl={pep_cap},Tp={pro_cap}); "
                    f"embed_real=(Tl={out['pep_len_real']},Tp={out['prot_len_real']}); "
                    f"cover=(pep:{pep_ok}, prot:{pro_ok})"
                )

            # 保存（保持原 dtype；trainer 会在需要处 to(float)）
            out["gt_kind"] = gt["kind"]                # "contact" | "distance"
            out["gt_map"]  = torch.from_numpy(gt_map_np.copy())
            out["gt_mask"] = torch.ones_like(out["gt_map"], dtype=torch.bool)
            # 新增：保存序列（若 GT 提供），并按裁剪后的长度截断
            if gt.get("pep_seq") is not None:
                out["gt_pep_seq"] = gt["pep_seq"][:pep_cap]
            if gt.get("pro_seq") is not None:
                out["gt_pro_seq"] = gt["pro_seq"][:pro_cap]

        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.keys[idx]
        sample = self._load_one(key)

        if self.strict_fixed:
            Tp = sample["protein_emb"].shape[0]
            Tl = sample["peptide_emb"].shape[0]
            if not ((Tp in (600,601,602,603)) and (Tl in (80,81,82,83))):
                raise ValueError(f"[strict_fixed] unexpected lengths for {key}: protein={Tp}, peptide={Tl}")

        return sample

# ---------------------------
# Collate：embeddings 按批填充；GT 2D 也按批填充
# ---------------------------

def _stack_or_pad_3d(items: List[torch.Tensor]) -> torch.Tensor:
    lengths = [x.shape[0] for x in items]
    if len(set(lengths)) == 1:
        return torch.stack(items, dim=0)
    T_max = max(lengths)
    D = items[0].shape[1]
    B = len(items)
    out = items[0].new_zeros((B, T_max, D))
    for i, x in enumerate(items):
        T = x.shape[0]
        out[i, :T] = x
    return out

def _stack_or_pad_2d_bool(items: List[torch.Tensor]) -> torch.BoolTensor:
    lengths = [x.shape[0] for x in items]
    if len(set(lengths)) == 1:
        return torch.stack(items, dim=0).bool()
    T_max = max(lengths)
    B = len(items)
    out = torch.zeros((B, T_max), dtype=torch.bool)
    for i, x in enumerate(items):
        T = x.shape[0]
        out[i, :T] = x.bool()
    return out

def _stack_or_pad_gt2d(items: List[torch.Tensor]) -> torch.Tensor:
    """将若干 (Tl_i, Tp_i) 2D 张量右下角零填充到 (B, Tl_max, Tp_max)。"""
    Tl_max = max(x.shape[0] for x in items)
    Tp_max = max(x.shape[1] for x in items)
    B = len(items)
    dtype = items[0].dtype
    out = items[0].new_zeros((B, Tl_max, Tp_max)).to(dtype)
    for i, x in enumerate(items):
        tl, tp = x.shape
        out[i, :tl, :tp] = x
    return out

def _stack_or_pad_gtmask(items: List[torch.Tensor]) -> torch.BoolTensor:
    Tl_max = max(x.shape[0] for x in items)
    Tp_max = max(x.shape[1] for x in items)
    B = len(items)
    out = torch.zeros((B, Tl_max, Tp_max), dtype=torch.bool)
    for i, x in enumerate(items):
        tl, tp = x.shape
        out[i, :tl, :tp] = True
    return out

def collate_full_tokens(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    prot = _stack_or_pad_3d([b["protein_emb"] for b in batch])   # (B, Tp*, D)
    pep  = _stack_or_pad_3d([b["peptide_emb"] for b in batch])   # (B, Tl*, D)

    def pack_masks(name: str) -> Dict[str, torch.Tensor]:
        return {
            "is_special":         _stack_or_pad_2d_bool([b[f"{name}_masks"]["is_special"] for b in batch]),
            "is_residue_window":  _stack_or_pad_2d_bool([b[f"{name}_masks"]["is_residue_window"] for b in batch]),
            "is_padding":         _stack_or_pad_2d_bool([b[f"{name}_masks"]["is_padding"] for b in batch]),
            "valid_real_residue": _stack_or_pad_2d_bool([b[f"{name}_masks"]["valid_real_residue"] for b in batch]),
        }

    out = {
        "protein_emb": prot,
        "peptide_emb": pep,
        "protein_masks": pack_masks("protein"),
        "peptide_masks": pack_masks("peptide"),
        "keys": [b["key"] for b in batch],
    }

    # （可选）GT 聚合
    if "gt_map" in batch[0]:
        gt_maps  = [b["gt_map"]  for b in batch]
        gt_masks = [b["gt_mask"] for b in batch]
        out["gt_map"]  = _stack_or_pad_gt2d(gt_maps)     # (B, Tl_max, Tp_max)
        out["gt_mask"] = _stack_or_pad_gtmask(gt_masks)  # (B, Tl_max, Tp_max)
        out["gt_kind"] = batch[0].get("gt_kind", "contact")  # 统一类型

        # 新增：把序列以 list[str|None] 的形式带出去（与 batch 对齐）
        out["gt_pep_seq"] = [b.get("gt_pep_seq", None) for b in batch]
        out["gt_pro_seq"] = [b.get("gt_pro_seq", None) for b in batch]


    return out

# ---------------------------
# CLI / main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Path to batch_index.json")
    ap.add_argument("--splits-dir", required=True, help="Dir that contains train.txt / val.txt / test.txt")
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="Load at most N samples from split")
    ap.add_argument("--strict-fixed", action="store_true", help="Require fixed token lengths; otherwise pad per-batch")

    # 预览控制
    ap.add_argument("--dry-run", action="store_true", help="build loader & print batches then exit")
    ap.add_argument("--preview", type=int, default=2, help="how many batches to preview in dry-run")
    ap.add_argument("--print-values", action="store_true", help="print real embedding & mask values (and GT if present)")
    ap.add_argument("--print-tokens", type=int, default=10, help="how many tokens to print per sample (head)")
    ap.add_argument("--print-dims", type=int, default=8, help="how many dims to print per token (head)")

    # ===== GT 配置 =====
    ap.add_argument("--gt-index", type=str, default=None, help="GT index JSON (results.json / cb_maps.json)")
    ap.add_argument("--gt-as-contact", action="store_true", default=False, help="Return contact (thresholded)")
    ap.add_argument("--gt-threshold", type=float, default=8.0, help="Contact threshold in Å (when --gt-as-contact)")
    ap.add_argument("--gt-cap-pep", type=int, default=None, help="Cap for GT peptide length (e.g., 80)")
    ap.add_argument("--gt-cap-pro", type=int, default=None, help="Cap for GT protein length (e.g., 600)")

    args = ap.parse_args()

    key2npz = read_db_index(args.index)
    print(f"Index loaded: {len(key2npz)} items from {args.index}")

    split_keys = read_split_list(args.splits_dir, args.split)
    print(f"Split '{args.split}': {len(split_keys)} keys from {Path(args.splits_dir)/ (args.split + '.txt')}")

    # 先加载 GT 索引（若提供）
    gt_key2npz = {}
    if args.gt_index:
        gt_key2npz = read_gt_index(args.gt_index)
        print(f"GT index loaded: {len(gt_key2npz)} items from {args.gt_index}")

    def strip_suffix_for_gt_from_split(k: str) -> str:
        return strip_suffix_key(k)

    usable: List[str] = []
    gt_map_for_ds: Dict[str, str] = {}
    matched = 0

    for k in split_keys:  # 保持 split 原始顺序
        if k not in key2npz:
            continue

        if not args.gt_index:
            usable.append(k)
            matched += 1
        else:
            k_gt = strip_suffix_for_gt_from_split(k)
            if k_gt in gt_key2npz:
                usable.append(k)
                gt_map_for_ds[k] = gt_key2npz[k_gt]
                matched += 1
            else:
                continue

        if args.limit and matched >= args.limit:
            break

    print(f"Usable keys intersect (suffix-aware): {len(usable)}")
    if len(usable) == 0:
        for k in split_keys[:5]:
            k_gt = strip_suffix_for_gt_from_split(k)
            print(f" sample '{k}': in_emb={k in key2npz}, in_gt_with_split_stripped={k_gt in gt_key2npz}")
        return

    ds = ProtPepFullTokenDataset(
        key_to_npz=key2npz,
        keys=usable,
        strict_fixed=args.strict_fixed,
        gt_key_to_npz=gt_map_for_ds if args.gt_index else None,
        gt_as_contact=args.gt_as_contact,
        gt_threshold=args.gt_threshold,
        gt_cap_pep=args.gt_cap_pep,
        gt_cap_pro=args.gt_cap_pro,
    )

    print(f"Usable keys in this split: {len(usable)}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(args.split == "train"),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_full_tokens,
        drop_last=False,
    )

    if args.dry_run:
        print(f"Dataset size: {len(ds)} | Batch size: {args.batch_size}")
        it = iter(loader)
        n_show = min(args.preview, max(1, len(loader)))
        for i in range(n_show):
            batch = next(it)
            if args.print_values:
                _print_batch_preview(batch, i+1, tokens=args.print_tokens, dims=args.print_dims)
            else:
                print(f"[batch {i+1}] protein_emb={tuple(batch['protein_emb'].shape)} peptide_emb={tuple(batch['peptide_emb'].shape)}")
        return

    # 非 dry-run 场景：把 loader 交给你的 Trainer 使用
    pass

# ---------------------------
# 打印预览：embedding/掩码 + （可选）GT 真实数值
# ---------------------------

def _mask_prefix_to_str(mask: torch.Tensor, n: int) -> str:
    n = min(n, mask.shape[-1])
    arr = mask[:n].cpu().numpy().astype(int).tolist()
    return "".join(str(x) for x in arr)

def _print_batch_preview(batch: Dict[str, Any], batch_idx: int, tokens: int, dims: int):
    prot = batch["protein_emb"]       # (B, Tp, D)
    pep  = batch["peptide_emb"]       # (B, Tl, D)
    B, Tp, Dp = prot.shape
    _, Tl, Dl = pep.shape
    print(f"[batch {batch_idx}]")
    print(f"  prot_emb: {tuple(prot.shape)} prot_mask: {tuple(batch['protein_masks']['valid_real_residue'].shape)}")
    print(f"  pep_emb : {tuple(pep.shape)}  pep_mask : {tuple(batch['peptide_masks']['valid_real_residue'].shape)}")
    print(f"  keys[0]: {batch['keys'][0]}")

    i = 0
    tP = min(tokens, Tp);  dP = min(dims, Dp)
    tL = min(tokens, Tl);  dL = min(dims, Dl)

    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print("\n  [protein] embedding slice [0, :{0}, :{1}] ->".format(tP, dP))
    print(prot[i, :tP, :dP].cpu().numpy())
    print("   mean={:.5f} std={:.5f} min={:.5f} max={:.5f}".format(
        prot[i].mean().item(), prot[i].std().item(), prot[i].min().item(), prot[i].max().item()
    ))

    print("\n  [peptide] embedding slice [0, :{0}, :{1}] ->".format(tL, dL))
    print(pep[i, :tL, :dL].cpu().numpy())
    print("   mean={:.5f} std={:.5f} min={:.5f} max={:.5f}".format(
        pep[i].mean().item(), pep[i].std().item(), pep[i].min().item(), pep[i].max().item()
    ))

    pm = batch["protein_masks"]
    lm = batch["peptide_masks"]
    def cnt(t: torch.Tensor) -> int:
        return int(t[i].sum().item())

    print("\n  [protein] masks (prefix of first {} tokens)".format(tokens))
    print("    is_special:        ", _mask_prefix_to_str(pm["is_special"][i], tokens), " | sum=", cnt(pm["is_special"]))
    print("    is_residue_window: ", _mask_prefix_to_str(pm["is_residue_window"][i], tokens), " | sum=", cnt(pm["is_residue_window"]))
    print("    is_padding:        ", _mask_prefix_to_str(pm["is_padding"][i], tokens), " | sum=", cnt(pm["is_padding"]))
    print("    valid_real_residue:", _mask_prefix_to_str(pm["valid_real_residue"][i], tokens), " | sum=", cnt(pm["valid_real_residue"]))

    print("\n  [peptide] masks (prefix of first {} tokens)".format(tokens))
    print("    is_special:        ", _mask_prefix_to_str(lm["is_special"][i], tokens), " | sum=", cnt(lm["is_special"]))
    print("    is_residue_window: ", _mask_prefix_to_str(lm["is_residue_window"][i], tokens), " | sum=", cnt(lm["is_residue_window"]))
    print("    is_padding:        ", _mask_prefix_to_str(lm["is_padding"][i], tokens), " | sum=", cnt(lm["is_padding"]))
    print("    valid_real_residue:", _mask_prefix_to_str(lm["valid_real_residue"][i], tokens), " | sum=", cnt(lm["valid_real_residue"]))

    if "gt_map" in batch:
        GT = batch["gt_map"]   # (B, Tl_max, Tp_max)
        GM = batch["gt_mask"]  # (B, Tl_max, Tp_max)
        tl = GT.shape[1]; tp = GT.shape[2]
        vr = GM[i]
        rr = min(tokens, tl); cc = min(tokens, tp)
        print("\n  [GT] gt_map slice [:,{0},{1}] ->".format(rr, cc))
        print(GT[i, :rr, :cc].cpu().numpy())
        if batch.get("gt_kind", "contact") == "contact":
            valid_sum = int(GT[i][vr].sum().item())
            total = int(vr.sum().item())
            print(f"   contact sum={valid_sum} / valid={total}  (ratio={valid_sum/max(1,total):.3f})")
        else:
            valid_vals = GT[i][vr].float()
            print("   distance  min/mean/max (valid): {:.3f} / {:.3f} / {:.3f}".format(
                valid_vals.min().item(), valid_vals.mean().item(), valid_vals.max().item()
            ))
    print()

if __name__ == "__main__":
    main()
