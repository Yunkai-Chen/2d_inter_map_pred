#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_loader_fullmap.py

专用于 full map 版本：
  - 从 batch_index.json + splits/{train,val,test}.txt 读取 embedding
  - 加载 GT full map（distance/contact）
  - 输出 full_emb / full_mask / gt_map_full / gt_mask_full
"""

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# 基础工具
# ---------------------------
def _stack_pad_long(items):
    maxlen = max(x.shape[0] for x in items)
    out = torch.zeros((len(items), maxlen), dtype=torch.long)
    for i, x in enumerate(items):
        out[i, :x.shape[0]] = x.long()
    return out

def resolve_npz_path(raw_path: str, index_file: Path) -> Optional[Path]:
    p = Path(raw_path)
    if p.is_absolute():
        return p if p.exists() else None
    base = index_file.parent
    cands = [base / p, base.parent / p, Path.cwd() / p, base / p.name]
    for c in cands:
        if c.exists():
            return c.resolve()
    return None

def strip_suffix_key(k: str) -> str:
    return k[:-11] if k.endswith("_nomutation") else k


def read_db_index(index_path: str) -> Dict[str, str]:
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
    path = Path(splits_dir) / f"{split_name}.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for ln in lines:
        k = ln.strip()
        if k and not k.startswith("#"):
            out.append(k)
    return out

# ---------------------------
# GT full map loader
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

AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G',
    'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}
def _labels_to_seq(labels): return ''.join(AA3_TO_1.get(l.split(':',1)[0].upper(),'X') for l in labels)

def load_gt_full(
    gt_npz_path: str,
    expect_order: str = "peptide_protein",
    as_contact: bool = True,
    threshold: float = 8.0,
    gt_cap_pep: Optional[int] = None,
    gt_cap_pro: Optional[int] = None,
) -> Dict[str, Any]:
    D_full, labels, meta = _load_npz_meta(gt_npz_path)
    order = (meta.get("order") or expect_order).lower().strip()
    p_len = int(meta["peptide_length"])
    pro_len = int(meta["protein_length"])
    if order == "protein_peptide":
        pep_block   = D_full[pro_len:, pro_len:]
        prot_block  = D_full[:pro_len, :pro_len]
        inter_block = D_full[pro_len:, :pro_len]
        cross_T     = D_full[:pro_len, pro_len:]
        D_full = np.block([[pep_block, inter_block],[cross_T, prot_block]])

    Tl_cap = min(p_len, gt_cap_pep) if gt_cap_pep else p_len
    Tp_cap = min(pro_len, gt_cap_pro) if gt_cap_pro else pro_len
    T_cap = Tl_cap + Tp_cap

    pep_intra  = D_full[:Tl_cap, :Tl_cap]
    prot_intra = D_full[p_len:p_len+Tp_cap, p_len:p_len+Tp_cap]
    inter      = D_full[:Tl_cap, p_len:p_len+Tp_cap]

    full_map = np.full((T_cap, T_cap), np.nan, dtype=np.float32)
    mask_map = np.zeros((T_cap, T_cap), dtype=bool)
    full_map[:Tl_cap, :Tl_cap] = pep_intra
    full_map[p_len:p_len+Tp_cap, p_len:p_len+Tp_cap] = prot_intra
    full_map[:Tl_cap, p_len:p_len+Tp_cap] = inter
    full_map[p_len:p_len+Tp_cap, :Tl_cap] = inter.T
    mask_map[:Tl_cap, :Tl_cap] = True
    mask_map[p_len:p_len+Tp_cap, p_len:p_len+Tp_cap] = True
    mask_map[:Tl_cap, p_len:p_len+Tp_cap] = True
    mask_map[p_len:p_len+Tp_cap, :Tl_cap] = True
    # 关键的两步：
    np.fill_diagonal(mask_map, False)                  # 1) 去掉对角线
    mask_map &= np.isfinite(D_full[:T_cap, :T_cap])    # 2) 去掉 NaN 位置

    if as_contact:
        finite = np.isfinite(full_map)
        full_map = (finite & (full_map < threshold)).astype(np.float32)

    return {"map": full_map, "mask": mask_map, "pep_len": p_len, "pro_len": pro_len}

# ---------------------------
# Dataset (full-map only)
# ---------------------------
class ProtPepFullMapDataset(Dataset):
    def __init__(self, key_to_npz, keys, gt_key_to_npz, gt_cap_pep=None, gt_cap_pro=None,
                 gt_threshold=8.0, gt_as_contact=True):
        self.keys = keys
        self.key_to_npz = key_to_npz
        self.gt_key_to_npz = gt_key_to_npz
        self.gt_cap_pep = gt_cap_pep
        self.gt_cap_pro = gt_cap_pro
        self.gt_threshold = gt_threshold
        self.gt_as_contact = gt_as_contact

    def __len__(self): return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        d = np.load(self.key_to_npz[key], allow_pickle=True)
        p_emb = torch.from_numpy(d["protein_embedding"]).float()
        l_emb = torch.from_numpy(d["peptide_embedding"]).float()
        pep_mask = torch.from_numpy(~d["peptide_is_padding"].astype(bool))
        prot_mask = torch.from_numpy(~d["protein_is_padding"].astype(bool))
        if pep_mask.shape[0] > 2:
            pep_mask[0] = False
            pep_mask[-1] = False
        if prot_mask.shape[0] > 2:
            prot_mask[0] = False
            prot_mask[-1] = False

        # ---- 新增：生成 chain_id ----
        pep_len = l_emb.shape[0]
        prot_len = p_emb.shape[0]
        chain_id = torch.cat([
            torch.zeros(pep_len, dtype=torch.long),   # 0 for peptide
            torch.ones(prot_len, dtype=torch.long)    # 1 for protein
        ])

        # ---- 查找对应的 GT npz 文件（容忍 _nomutation 后缀差异） ----
        if key in self.gt_key_to_npz:
            gt_path = self.gt_key_to_npz[key]
        else:
            key_base = key[:-11] if key.endswith("_nomutation") else key
            if key_base in self.gt_key_to_npz:
                gt_path = self.gt_key_to_npz[key_base]
            else:
                raise KeyError(f"[GT missing] Cannot find GT map for key {key} or {key_base}")

        gt_full = load_gt_full(gt_path, as_contact=self.gt_as_contact,
                            threshold=self.gt_threshold,
                            gt_cap_pep=self.gt_cap_pep,
                            gt_cap_pro=self.gt_cap_pro)

        return {
            "key": key,
            "full_emb": torch.cat([l_emb, p_emb], dim=0),
            "full_mask": torch.cat([pep_mask, prot_mask], dim=0),
            "chain_id": chain_id,   # ✅ 新增
            "gt_map_full": torch.from_numpy(gt_full["map"]),
            "gt_mask_full": torch.from_numpy(gt_full["mask"])
        }


# ---------------------------
# Collate (full-map only)
# ---------------------------
def _stack_pad(items, dim=0):
    maxlen = max(x.shape[0] for x in items)
    out = items[0].new_zeros((len(items), maxlen, items[0].shape[1]))
    for i,x in enumerate(items): out[i,:x.shape[0]] = x
    return out

def _stack_pad_mask(items):
    maxlen = max(x.shape[0] for x in items)
    out = torch.zeros((len(items), maxlen), dtype=torch.bool)
    for i,x in enumerate(items): out[i,:x.shape[0]] = x.bool()
    return out

def _stack_pad_map(items):
    maxL = max(x.shape[0] for x in items)
    out = items[0].new_zeros((len(items), maxL, maxL))
    for i,x in enumerate(items):
        L = x.shape[0]
        out[i,:L,:L] = x
    return out

def collate_fullmap(batch):
    return {
        "full_emb": _stack_pad([b["full_emb"] for b in batch]),
        "full_mask": _stack_pad_mask([b["full_mask"] for b in batch]),
        "chain_id": _stack_pad_long([b["chain_id"] for b in batch]),   # ✅ 新增
        "gt_map_full": _stack_pad_map([b["gt_map_full"] for b in batch]),
        "gt_mask_full": _stack_pad_map([b["gt_mask_full"] for b in batch]),
        "keys": [b["key"] for b in batch]
    }


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--split", choices=["train","val","test"], required=True)
    ap.add_argument("--gt-index", required=True)
    ap.add_argument("--gt-cap-pep", type=int, default=None)
    ap.add_argument("--gt-cap-pro", type=int, default=None)
    ap.add_argument("--gt-threshold", type=float, default=8.0)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    key2npz = read_db_index(args.index)
    gt_key2npz = read_gt_index(args.gt_index)
    split_keys = read_split_list(args.splits_dir, args.split)
    # 允许 _nomutation 等后缀的 split key 与 GT 对齐
    usable = []
    missing_emb, missing_gt = [], []
    for k in split_keys:
        has_emb = (k in key2npz)
        k_base = strip_suffix_key(k)
        has_gt = (k in gt_key2npz) or (k_base in gt_key2npz)
        if has_emb and has_gt:
            usable.append(k)
        else:
            if not has_emb:
                missing_emb.append(k)
            if not has_gt:
                missing_gt.append(k)


    ds = ProtPepFullMapDataset(key2npz, usable, gt_key2npz,
                               gt_cap_pep=args.gt_cap_pep, gt_cap_pro=args.gt_cap_pro,
                               gt_threshold=args.gt_threshold)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        collate_fn=collate_fullmap, shuffle=False)
    print(f"Loaded {len(ds)} samples; showing 1 batch")
    it = iter(loader)
    try:
        batch = next(it)
    except StopIteration:
        print("[warn] no batch")
        return

    def _shape(x):
        return tuple(x.shape) if torch.is_tensor(x) else len(x)

    def _head1d(x, n=100):
        n = min(n, x.shape[-1])
        return x[..., :n]

    def _stats_tensor(name, t: torch.Tensor):
        """
        兼容旧版 PyTorch（无 torch.nanmin/nanmean 等）：
        - 将张量搬到 CPU，float 类型过滤 NaN 后再统计
        - uint8 输出 0/1 计数
        - bool 输出 True 比例
        """
        # detach + to cpu，避免在 GPU 上做统计/打印
        t = t.detach().to("cpu")

        if t.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            # 转成 float32 以简化
            tf = t.to(torch.float32)
            if torch.isfinite(tf).any():
                valid = torch.isfinite(tf)
                vals = tf[valid]
                vmin = float(vals.min())
                vmax = float(vals.max())
                mean = float(vals.mean())
                print(f"[{name}] stats: min={vmin:.4f} mean={mean:.4f} max={vmax:.4f}  (valid {vals.numel()}/{tf.numel()})")
            else:
                print(f"[{name}] stats: all values are non-finite (NaN/Inf)")
        elif t.dtype == torch.uint8:
            # 常用于 contact map（0/1）
            uniq = torch.unique(t)
            counts = {int(u): int((t == u).sum()) for u in uniq}
            print(f"[{name}] unique counts: {counts}")
        elif t.dtype == torch.bool:
            ttrue = int(t.sum())
            total = int(t.numel())
            ratio = (ttrue / total) if total > 0 else 0.0
            print(f"[{name}] true={ttrue} / {total} ({ratio:.2%})")
        else:
            # 其他整型等
            print(f"[{name}] dtype={t.dtype} shape={tuple(t.shape)}")


    # 1) keys
    print(f"[keys] batch size={len(batch['keys'])}  example={batch['keys'][0]}")

    # 2) full_emb
    F = batch["full_emb"]
    print(f"[full_emb] dtype={F.dtype} shape={_shape(F)}")
    _stats_tensor("full_emb", F)
    # 打印第一条样本前2个残基的前8维，便于确认数值
    print("[full_emb] sample[0, :2, :8]:")
    print(F[0, :2, :8])

    # 3) full_mask
    M = batch["full_mask"]
    print(f"[full_mask] dtype={M.dtype} shape={_shape(M)}")
    _stats_tensor("full_mask", M)
    print("[full_mask] sample[0, :100]:")
    print(_head1d(M[0], 100))

    # 4) chain_id
    C = batch["chain_id"]
    print(f"[chain_id] dtype={C.dtype} shape={_shape(C)}")
    # 打印前100个 chain id
    print("[chain_id] sample[0, :100]:")
    print(_head1d(C[0], 100))

    # 5) gt_map_full
    G = batch["gt_map_full"]
    print(f"[gt_map_full] dtype={G.dtype} shape={_shape(G)}")
    _stats_tensor("gt_map_full", G)
    # 打印左上角 5x5（避免巨量输出）
    sz = min(5, G.shape[-1])
    print("[gt_map_full] sample[0, :5, :5]:")
    print(G[0, :sz, :sz])

    # 6) gt_mask_full
    GM = batch["gt_mask_full"]
    print(f"[gt_mask_full] dtype={GM.dtype} shape={_shape(GM)}")
    _stats_tensor("gt_mask_full", GM)
    print("[gt_mask_full] sample[0, :5, :5]:")
    print(GM[0, :sz, :sz])

    # 7) L 与 T 的对齐情况（便于你核对 L != T 的原因）
    L = F.shape[1]
    T = G.shape[1]
    print(f"[align] L(full_emb)={L}, T(gt_map_full)={T}, L_common={min(L, T)}")

if __name__ == "__main__":
    main()
