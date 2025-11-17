#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdb_interface_to_json_singlepair.py

从目录中批量读取形如 <pdbid>_<peptideid>_<proteinid>.pdb 的文件（两个链ID均为单字符），
基于重原子最短距离 ≤ cutoff 判定界面残基，构建三套编号体系（原始PDB/分链重编号/全局重编号），
标注未知残基（非常规三字母→X），并输出与既有流水线兼容的 JSON（含 Encoding 元信息）。

依赖：
  pip install biopython numpy pandas scipy tqdm

用法示例：
  python pdb_interface_to_json_singlepair.py \
      --pdb_dir path/to/pdbs \
      --output_dir out_json \
      --cutoff 6.0
"""

import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from Bio.PDB import PDBParser, is_aa

# ----------------------------- 配置常量 -----------------------------

# ESM3 / 下游编码的约定长度
PROTEIN_MAX_LEN = 600   # receptor
PEPTIDE_MAX_LEN = 80    # ligand
CONCAT_TOTAL_MAX = PROTEIN_MAX_LEN + PEPTIDE_MAX_LEN
PAD_CHAR = 'X'

# 三字母到一字母映射（非常规残基默认→'X'）
AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

# ----------------------------- 工具函数 -----------------------------

def three_to_one(res3: str) -> str:
    """三字母转一字母；未知→'X'"""
    return AA3_TO_1.get(res3.upper(), 'X')

def get_heavy_atom_coords(residue) -> np.ndarray:
    """返回残基的重原子坐标 (N,3)，无重原子返回 (0,3)"""
    coords = []
    for atom in residue:
        # BioPython atom.element 可能为 ''，稳妥起见按名称判断是否氢
        name = atom.get_name()
        # 通常 H, 1H, 2H, HD1 等以 H 开头
        if name and name[0] == 'H':
            continue
        # 一些结构的 atom.element 可靠，也可并用
        if hasattr(atom, 'element') and atom.element == 'H':
            continue
        coords.append(atom.coord)
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)

def parse_filename(fname: str):
    """
    从文件名解析 pdbid, peptide_chain, protein_chain.
    预期：<pdbid>_<peptideid>_<proteinid>.pdb 且两个链ID均为单字符。
    """
    base = os.path.basename(fname)
    if not base.lower().endswith('.pdb'):
        return None
    stem = base[:-4]
    parts = stem.split('_')
    if len(parts) != 3:
        return None
    pdbid, pep, prot = parts
    if len(pep) != 1 or len(prot) != 1:
        return None
    return pdbid, pep, prot

def build_numbering_for_two_chains(structure, protein_chain_id: str, peptide_chain_id: str):
    """
    仅对给定的两条链（protein_chain_id, peptide_chain_id）构建：
      1) 原始PDB编号串： RES3:ChainID<原始号>
      2) 分链重编号串：  RES3:ChainID<链内顺序号，从1开始>
      3) 全局重编号串：  RES3:ChainID<全局顺序号，跨链连续>
    同时返回：
      - 映射： (chain_id, 原始号) -> 全局顺序号
      - 残基按顺序列表（包含三字母、一字母、链ID、原始号、全局号等）
    """
    model = list(structure.get_models())[0]

    # 只收集目标链的氨基酸残基（is_aa）
    residues_by_chain = {protein_chain_id: [], peptide_chain_id: []}
    for chain in model:
        cid = chain.id
        if cid not in residues_by_chain:
            continue
        for res in chain:
            if is_aa(res, standard=True) or is_aa(res, standard=False):
                residues_by_chain[cid].append(res)

    # 分链重编号：每条链从1开始
    chain_sequential = {protein_chain_id: 1, peptide_chain_id: 1}

    # 结果容器
    pdb_parts = []
    chain_renum_parts = []
    global_renum_parts = []

    residues_linear = []  # 线性表（用于全局递增）
    global_index = 1

    # 顺序：先 protein，再 peptide（保证全局编号稳定）
    for cid in [protein_chain_id, peptide_chain_id]:
        for res in residues_by_chain[cid]:
            resname3 = res.resname
            one = three_to_one(resname3)
            pdb_num = res.id[1]  # 原始 PDB 号（忽略插入码）
            # 构造三串
            pdb_parts.append(f"{resname3}:{cid}{pdb_num}")
            chain_renum_parts.append(f"{resname3}:{cid}{chain_sequential[cid]}")
            global_renum_parts.append(f"{resname3}:{cid}{global_index}")
            # 收集线性项
            residues_linear.append({
                "chain": cid,
                "resname3": resname3,
                "one": one,
                "pdb_num": pdb_num,
                "global_idx": global_index,
                "biopy_res": res
            })
            chain_sequential[cid] += 1
            global_index += 1

    # 建立 (chain, pdb_num) -> global_idx 的映射
    pdb_to_global = {}
    for item in residues_linear:
        pdb_to_global[(item["chain"], item["pdb_num"])] = item["global_idx"]

    numbering = {
        "pdb_number": " ".join(pdb_parts),
        "chain_renumber": " ".join(chain_renum_parts),
        "renumber": " ".join(global_renum_parts)
    }

    return numbering, pdb_to_global, residues_linear, residues_by_chain

def compute_interface(residues_by_chain, protein_chain_id: str, peptide_chain_id: str, cutoff: float):
    """
    计算界面残基集合（两侧），返回：
      - interface_globals_prot: set 的 (chain, pdb_num) 键（仅 protein 链）
      - interface_globals_pep: 同上（仅 peptide 链）
      - pairs_within_cutoff: 命中对的计数
    """
    prot_residues = residues_by_chain.get(protein_chain_id, [])
    pep_residues = residues_by_chain.get(peptide_chain_id, [])

    interface_prot = set()
    interface_pep = set()
    pairs_within = 0

    for r in prot_residues:
        r_coords = get_heavy_atom_coords(r)
        if r_coords.shape[0] == 0:
            continue
        for l in pep_residues:
            l_coords = get_heavy_atom_coords(l)
            if l_coords.shape[0] == 0:
                continue
            dmin = np.min(cdist(r_coords, l_coords))
            if dmin <= cutoff:
                interface_prot.add((r.parent.id, r.id[1]))
                interface_pep.add((l.parent.id, l.id[1]))
                pairs_within += 1

    return interface_prot, interface_pep, pairs_within

def collect_unknowns(residues_linear):
    """
    统计未知残基（映射为 'X'）的位置（全局重编号）、数量和详情。
    """
    positions = []
    details = []
    for item in residues_linear:
        if item["one"] == 'X' and item["resname3"].upper() not in AA3_TO_1:
            positions.append(item["global_idx"])
            details.append({
                "chain": item["chain"],
                "pdb_resnum": item["pdb_num"],
                "res3": item["resname3"],
                "mapped": "X"
            })
    return positions, len(positions), details

def build_sequences(residues_linear, protein_chain_id: str, peptide_chain_id: str):
    """
    构建 receptor_seq（protein）与 ligand_seq（peptide）的单字母序列。
    """
    prot_seq = []
    pep_seq = []
    for item in residues_linear:
        if item["chain"] == protein_chain_id:
            prot_seq.append(item["one"])
        elif item["chain"] == peptide_chain_id:
            pep_seq.append(item["one"])
    return "".join(prot_seq), "".join(pep_seq)

def encoding_metadata(receptor_len: int, ligand_len: int):
    """
    生成 Encoding 元信息（不做实际前向，仅声明策略）。
    """
    sep_would_pad_rec = receptor_len < PROTEIN_MAX_LEN
    sep_would_pad_lig = ligand_len < PEPTIDE_MAX_LEN
    sep_final_rec = min(receptor_len, PROTEIN_MAX_LEN) if not sep_would_pad_rec else PROTEIN_MAX_LEN
    sep_final_lig = min(ligand_len, PEPTIDE_MAX_LEN) if not sep_would_pad_lig else PEPTIDE_MAX_LEN

    concat_len = receptor_len + ligand_len
    concat_would_trunc = concat_len > CONCAT_TOTAL_MAX
    concat_final = min(concat_len, CONCAT_TOTAL_MAX)

    return {
        "separate": {
            "receptor_max_len": PROTEIN_MAX_LEN,
            "ligand_max_len": PEPTIDE_MAX_LEN,
            "pad_char": PAD_CHAR,
            "would_pad": {"receptor": sep_would_pad_rec, "ligand": sep_would_pad_lig},
            "final_lengths": {"receptor": sep_final_rec, "ligand": sep_final_lig}
        },
        "concatenated": {
            "total_max_len": CONCAT_TOTAL_MAX,
            "pad_char": PAD_CHAR,
            "would_truncate": concat_would_trunc,
            "final_length": concat_final
        },
        "interface_only": {
            "length": 0  # 由实际界面残基数填充
        }
    }

# ----------------------------- 主流程 -----------------------------

def process_one_pdb(pdb_path: str, cutoff: float):
    """
    处理单个 PDB 文件，返回 (key, json_obj) 或 (None, None) 若失败/跳过。
    """
    parsed = parse_filename(pdb_path)
    if parsed is None:
        return None, None
    pdbid, pep_chain, prot_chain = parsed

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("struct", pdb_path)
    except Exception:
        return None, None

    # 构建编号、线性残基列表
    numbering, pdb_to_global, residues_linear, residues_by_chain = build_numbering_for_two_chains(
        structure, prot_chain, pep_chain
    )

    # 无残基则跳过
    if len(residues_linear) == 0:
        return None, None

    # 界面判定
    interface_prot, interface_pep, pairs_within = compute_interface(
        residues_by_chain, prot_chain, pep_chain, cutoff
    )

    # 转成全局编号并排序
    interface_globals = []
    for (cid, pdbnum) in sorted(interface_prot):
        if (cid, pdbnum) in pdb_to_global:
            interface_globals.append(pdb_to_global[(cid, pdbnum)])
    for (cid, pdbnum) in sorted(interface_pep):
        if (cid, pdbnum) in pdb_to_global:
            interface_globals.append(pdb_to_global[(cid, pdbnum)])
    interface_globals = sorted(set(interface_globals))

    # 未知残基统计
    unk_positions, unk_count, unk_details = collect_unknowns(residues_linear)

    # 序列
    receptor_seq, ligand_seq = build_sequences(residues_linear, prot_chain, pep_chain)

    # 编码元信息
    enc = encoding_metadata(len(receptor_seq), len(ligand_seq))
    enc["interface_only"]["length"] = len(interface_globals)

    # 汇总统计
    stats = {
        "receptor_residues": len(receptor_seq),
        "ligand_residues": len(ligand_seq),
        "pairs_within_cutoff": pairs_within,
        "cutoff_A": float(cutoff)
    }

    # 组织 JSON
    key = f"{pdbid.lower()}_{pep_chain}_{prot_chain}_nomutation"
    json_obj = {
        key: {
            "Input": {
                "pdb_number": numbering["pdb_number"],
                "chain_renumber": numbering["chain_renumber"],
                "renumber": numbering["renumber"]
            },
            "Sequences": {
                "receptor_seq": receptor_seq,
                "ligand_seq": ligand_seq,
                "unknown": {
                    "positions_global": sorted(unk_positions),
                    "count": int(unk_count),
                    "detail": unk_details
                }
            },
            "Output": {
                "Interface": " ".join(map(str, interface_globals)),
                "Stats": stats
            },
            "Encoding": enc,
            "Meta": {
                "pdb_file": os.path.abspath(pdb_path),
                "chains": {
                    "receptor": prot_chain,
                    "ligand": pep_chain
                },
                "model_used": 0
            }
        }
    }
    return key, json_obj[key]

def main():
    ap = argparse.ArgumentParser(description="批量提取 <pdbid>_<peptide>_<protein>.pdb 的界面并导出 JSON")
    ap.add_argument("--pdb_dir", required=True, help="包含 PDB 文件的目录")
    ap.add_argument("--output_dir", required=True, help="输出目录")
    ap.add_argument("--cutoff", type=float, default=6.0, help="界面判定距离阈值(Å)，默认6.0")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "interface_data.json")

    files = [os.path.join(args.pdb_dir, f) for f in os.listdir(args.pdb_dir)
             if f.lower().endswith(".pdb")]

    results = {}
    skipped = 0
    with tqdm(total=len(files), desc="Processing PDBs") as pbar:
        for fp in files:
            key, obj = process_one_pdb(fp, args.cutoff)
            if key is not None and obj is not None:
                results[key] = obj
            else:
                skipped += 1
            pbar.update(1)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n" + "="*60)
    print(f"完成。共输入 {len(files)} 个文件，成功 {len(results)} 个，跳过 {skipped} 个。")
    print(f"输出 JSON：{out_json}")
    print("="*60)

if __name__ == "__main__":
    main()
