#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_npz.py
----------------
可视化 & 自检：由 json_to_esm_embeddings_* 生成的 <key>_esm3_alltokens.npz

功能：
- 打印概要（模型/维度/设备、三块 embedding 形状、长度 raw/used、mask 计数、前几个 token）
- 生成 token 预览表（CSV）：protein / peptide
- 画简单诊断图：
  1) Protein 每个 token 的 L2 范数折线
  2) Peptide 每个 token 的 L2 范数折线
  3) Complex 的 1×T L2 范数条带图（热力图）

用法：
  python visualize_npz.py --npz your_file.npz --outdir viz_out
  python visualize_npz.py --npz your_file.npz --outdir viz_out --preview 80
  python visualize_npz.py --npz your_file.npz --no-plots  # 只打印与导出CSV，不画图
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False


# ------------------------- 读写辅助 -------------------------

def load_npz(path: str) -> dict:
    """加载 .npz 并转成普通 dict。"""
    arr = np.load(path, allow_pickle=True)
    data = {k: arr[k] for k in arr.files}
    return data


def _numpy_scalar_to_py(x):
    """把 0 维 ndarray / numpy scalar 转成 Python 标量。"""
    try:
        if isinstance(x, np.ndarray) and x.shape == ():
            return x.item()
        if isinstance(x, (np.generic,)):
            return x.item()
    except Exception:
        pass
    return x


def _to_text(x):
    """尽量把对象变成 Python str。"""
    x = _numpy_scalar_to_py(x)
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        try:
            return x.decode("utf-8")
        except Exception:
            return x.decode("latin-1", errors="ignore")
    if isinstance(x, str):
        return x
    # 对于长度为1的 object array，取 item
    if isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        return str(x.reshape(()).item())
    return str(x)


def robust_json_load(x, default=None):
    """
    兼容性很强的 JSON 解析：
    - 0维/标量 ndarray：.item() 后解析
    - bytes/bytearray：解码后解析
    - str：直接解析
    - 失败：返回 default
    """
    try:
        s = _to_text(x)
        if isinstance(s, str):
            return json.loads(s)
        return default
    except Exception:
        return default


# ------------------------- 表格预览 -------------------------

def peek_tokens_table(prefix: str, data: dict, preview: int = 40):
    """
    生成前 N 个 token 的预览表（DataFrame 或 list[dict]）。
    prefix: "protein" 或 "peptide"
    """
    toks = data.get(f"{prefix}_tokens_str", None)
    if toks is None:
        return None
    is_special = data[f"{prefix}_is_special"].astype(bool)
    is_residue = data[f"{prefix}_is_residue"].astype(bool)
    is_padding = data[f"{prefix}_is_padding"].astype(bool)
    real_mask  = data[f"{prefix}_residue_mask_real"].astype(bool)

    T = len(toks)
    n = min(preview, T)
    rows = []
    for i in range(n):
        rows.append({
            "idx": i,
            "token": _to_text(toks[i]),
            "is_special": bool(is_special[i]),
            "is_residue": bool(is_residue[i]),
            "is_padding": bool(is_padding[i]),
            "in_residue_window": bool(real_mask[i]),
        })
    if HAS_PANDAS:
        return pd.DataFrame(rows)
    return rows


# ------------------------- 概要打印 -------------------------

def print_summary(npz_path: str, data: dict, show_offsets: bool = True):
    print("="*80)
    print(f"NPZ: {npz_path}")
    print("="*80)

    # 解析 metadata（里面是 JSON 字符串）
    meta = robust_json_load(data.get("metadata", None), default={}) or {}
    dim    = meta.get("dim")
    order  = meta.get("order")
    device = meta.get("device")
    model  = meta.get("model")
    lengths = meta.get("lengths", {})

    chain_ids = data.get("chain_ids", None)
    try:
        chain_ids_list = list(chain_ids) if chain_ids is not None else None
    except Exception:
        chain_ids_list = None

    print(f"Model: {model} | Dim: {dim} | Device: {device} | Order: {order}")
    if chain_ids_list is not None:
        print(f"Chain IDs (order): {chain_ids_list}")

    # 形状
    def shp(name):
        return tuple(data[name].shape) if name in data else None
    print(f"protein_embedding shape: {shp('protein_embedding')}")
    print(f"peptide_embedding shape: {shp('peptide_embedding')}")
    print(f"complex_embedding shape: {shp('complex_embedding')}")

    # 原始/投喂长度
    print("Lengths (raw/used):")
    print(f"  protein: {lengths.get('protein', {})}")
    print(f"  peptide: {lengths.get('peptide', {})}")

    # 掩码计数
    for prefix in ["protein", "peptide"]:
        if f"{prefix}_embedding" in data:
            is_special = data[f"{prefix}_is_special"].astype(bool)
            is_residue = data[f"{prefix}_is_residue"].astype(bool)
            is_padding = data[f"{prefix}_is_padding"].astype(bool)
            T = len(is_special)
            print(f"[{prefix}] T={T} | specials={is_special.sum()} | residues={is_residue.sum()} | padding={is_padding.sum()}")

    # token 预览
    for prefix in ["protein", "peptide"]:
        toks = data.get(f"{prefix}_tokens_str", None)
        if toks is None:
            continue
        print(f"\n{prefix} first 12 tokens:")
        try:
            to_show = [ _to_text(x) for x in toks[:12] ]
        except Exception:
            to_show = list(toks[:12])
        print(to_show)

    # offsets（如存在）
    if show_offsets:
        cot = robust_json_load(data.get("chain_offsets_tokens", None))
        cor = robust_json_load(data.get("chain_offsets_residues", None))
        if cot:
            print("\nchain_offsets_tokens:")
            for seg in cot:
                print(f"  - {seg}")
        if cor:
            print("chain_offsets_residues:")
            for seg in cor:
                print(f"  - {seg}")

    print("="*80)


# ------------------------- 绘图 -------------------------

def plot_token_norms_line(embedding: np.ndarray, title: str, save_path: str | None = None):
    """
    每个 token 的向量 L2 范数折线。
    注意：不指定任何颜色样式（保持默认）。
    """
    norms = np.linalg.norm(embedding, axis=1)
    plt.figure(figsize=(10, 3))
    plt.plot(norms)
    plt.title(title)
    plt.xlabel("token index")
    plt.ylabel("L2 norm")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_token_norms_strip(embedding: np.ndarray, title: str, save_path: str | None = None):
    """
    1 × T 的热力条带图（每个 token 的 L2 范数）。
    """
    norms = np.linalg.norm(embedding, axis=1)[None, :]  # (1, T)
    plt.figure(figsize=(12, 1.8))
    plt.imshow(norms, aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel("token index")
    plt.yticks([])
    cbar = plt.colorbar()
    cbar.set_label("L2 norm")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ------------------------- 主流程 -------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize & sanity-check an ESM3 NPZ (all tokens kept).")
    parser.add_argument("--npz", required=True, help="Path to the NPZ file (e.g., <key>_esm3_alltokens.npz)")
    parser.add_argument("--outdir", default="viz_out", help="Directory to save plots and CSV previews")
    parser.add_argument("--preview", type=int, default=40, help="How many tokens to show in preview tables")
    parser.add_argument("--no-plots", action="store_true", help="Skip plotting (print summary + tables only)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = load_npz(args.npz)

    # 打印概要
    print_summary(args.npz, data, show_offsets=True)

    # 生成 token 预览表
    for prefix in ["protein", "peptide"]:
        tbl = peek_tokens_table(prefix, data, preview=args.preview)
        if tbl is None:
            continue
        print(f"\n== {prefix.upper()} token preview (first {args.preview}) ==")
        if HAS_PANDAS:
            # 控制台打印前10行
            try:
                print(tbl.head(min(10, len(tbl))))
            except Exception:
                print(tbl)
            # 保存完整预览到 CSV
            csv_path = os.path.join(args.outdir, f"{prefix}_token_preview.csv")
            tbl.to_csv(csv_path, index=False)
            print(f"Saved {prefix} preview CSV -> {csv_path}")
        else:
            # 纯文本回退
            for row in tbl:
                print(row)

    # 画图（可关闭）
    if not args.no_plots:
        if "protein_embedding" in data:
            save_p = os.path.join(args.outdir, "protein_token_norms.png")
            plot_token_norms_line(data["protein_embedding"], "Protein per-token L2 norms", save_p)

        if "peptide_embedding" in data:
            save_l = os.path.join(args.outdir, "peptide_token_norms.png")
            plot_token_norms_line(data["peptide_embedding"], "Peptide per-token L2 norms", save_l)

        if "complex_embedding" in data:
            save_c = os.path.join(args.outdir, "complex_token_norms_strip.png")
            plot_token_norms_strip(data["complex_embedding"], "Complex per-token L2 norm strip", save_c)


if __name__ == "__main__":
    main()
