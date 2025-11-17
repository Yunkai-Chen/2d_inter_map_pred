#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_eval_pairformer_batch.py

Batch evaluation runner for MSA_Pairformer.
Mirrors run_eval.py output structure, but:
  - Input: --test-file test.txt + --interface-json interface_data.json
  - Model: MSAPairformer.from_pretrained(device=device)
  - Added switch: --map-mode {cross, full}

Minimal patches added:
  - --resume: skip keys whose pred npz already exists
  - per-sample CUDA cleanup: torch.cuda.empty_cache() + gc.collect()
"""

import os
import io
import re
import json
import time
import tarfile
import shutil
import hashlib
import warnings
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/data/yunkai/apps/MSA_Pairformer')
from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.dataset import MSA, aa2tok_d, prepare_msa_masks

# ===== minimal-add: cleanup utils =====
import gc
def _cleanup_cuda(tag: str = ""):
    """Minimal CUDA cleanup after each sample; no functional changes elsewhere."""
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[MEM] after {tag}: alloc={alloc:.2f} GB, reserved={reserved:.2f} GB")
    gc.collect()

# ---------------------------
# Utility Functions
# ---------------------------
def seed_all(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    """NumPy sigmoid function."""
    return 1 / (1 + np.exp(-x))

def _binarize_from_distance(dist_map: np.ndarray, threshold: float) -> np.ndarray:
    """Convert distance map to binary contact map."""
    return (dist_map < threshold).astype(np.float32)

def _metrics_from_binary(pred: np.ndarray, gt: np.ndarray, threshold: float) -> Dict[str, float]:
    """Calculate metrics from predicted and ground truth binary maps."""
    pred_binary = (pred > threshold).astype(np.float32)
    
    # Calculate metrics
    tp = np.sum(pred_binary * gt)
    fp = np.sum(pred_binary * (1 - gt))
    fn = np.sum((1 - pred_binary) * gt)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Calculate AUPRC using sklearn if available
    try:
        from sklearn.metrics import average_precision_score
        auprc = average_precision_score(gt.flatten(), pred.flatten())
    except ImportError:
        # Simple approximation if sklearn is not available
        auprc = precision * recall
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auprc': float(auprc)
    }

def _format_head(arr: np.ndarray, H: int = 10) -> str:
    """Format array head for printing."""
    flat = arr.flatten()
    head = flat[:min(H, len(flat))]
    return f"Array shape: {arr.shape}, First {len(head)} values: {head}"

# ---------------------------
# ColabFoldPairedMSA (drop-in, fixed & improved)  —— 未改动你的逻辑
# ---------------------------
import os, io, re, json, time, tarfile, shutil, hashlib, warnings, tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import requests

class ColabFoldPairedMSA:
    """Improved ColabFoldPairedMSA client (safe, cacheable, robust)."""
    def __init__(self, host_url: str = "https://api.colabfold.com", cache_dir: Optional[str] = None):
        self.host_url = host_url
        self.job_id = None
        self.parsed_entries = None
        self.cache_dir = Path(cache_dir or (Path.home() / ".colabfold_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # internal tables for UniProt numeric encoding
        from string import ascii_uppercase
        self.pa = {a: 0 for a in ascii_uppercase}
        for a in ["O", "P", "Q"]: self.pa[a] = 1
        self.ma = [[{} for _ in range(6)], [{} for _ in range(6)]]
        for n, t in enumerate(range(10)):
            for i in [0, 1]:
                for j in [0, 4]: self.ma[i][j][str(t)] = n
        for n, t in enumerate(list(ascii_uppercase) + list(range(10))):
            for i in [0, 1]:
                for j in [1, 2]: self.ma[i][j][str(t)] = n
            self.ma[1][3][str(t)] = n
        for n, t in enumerate(ascii_uppercase):
            self.ma[0][3][str(t)] = n
            for i in [0, 1]: self.ma[i][5][str(t)] = n
        self.upi_encoding = {str(c): i for i, c in enumerate(list(range(10)) + ['A', 'B', 'C', 'D', 'E', 'F'])}

    def _extract_uniprot_id(self, header: str) -> str:
        pos = header.find("UniRef")
        if pos == -1: return ""
        start = header.find('_', pos)
        if start == -1: return ""
        start += 1
        end = start
        while end < len(header) and header[end] not in ' _\t': end += 1
        uid = header[start:end]
        if len(uid) >= 3 and uid[:3] == "UPI": return uid
        if len(uid) not in [6, 10]: return ""
        if not uid[0].isalpha(): return ""
        return uid

    def _uniprot_to_number(self, uniprot_ids: List[str]) -> List[int]:
        numbers = []
        for uni in uniprot_ids:
            if not uni or not uni[0].isalpha():
                numbers.append(0)
                continue
            if uni.startswith("UPI") and len(uni) == 13:
                hex_part = uni[3:]
                num = 0
                tot = 1
                for u in reversed(hex_part):
                    if str(u) in self.upi_encoding:
                        num += self.upi_encoding[str(u)] * tot
                        tot *= 16
                    else:
                        num = 0
                        break
                numbers.append(num + 10**15)
                continue
            p = self.pa.get(uni[0], 0)
            tot, num = 1, 0
            if len(uni) == 10:
                for n, u in enumerate(reversed(uni[-4:])):
                    if str(u) in self.ma[p][n]:
                        num += self.ma[p][n][str(u)] * tot
                        tot *= len(self.ma[p][n])
            for n, u in enumerate(reversed(uni[:6])):
                if n < len(self.ma[p]) and str(u) in self.ma[p][n]:
                    num += self.ma[p][n][str(u)] * tot
                    tot *= len(self.ma[p][n])
            numbers.append(num)
        return numbers

    def _calculate_genomic_distances(self, entry: Dict) -> List[int]:
        distances = []
        nums = entry['uniprot_nums']
        for i in range(1, len(nums)):
            if nums[i - 1] and nums[i]:
                distances.append(abs(nums[i] - nums[i - 1]))
            else:
                distances.append(-1)
        return distances

    def _parse_msa_lines(self, lines: List[str]) -> List[Dict]:
        entries = []
        i = 0
        is_first = True
        while i < len(lines):
            line = lines[i].rstrip()
            if line.startswith('>'):
                header = line
                seq_lines = []
                i += 1
                while i < len(lines) and not lines[i].startswith('>'):
                    if lines[i].strip():
                        seq_lines.append(lines[i].rstrip())
                    i += 1
                sequence = ''.join(seq_lines)
                header_parts = header.split('\t')
                header_clean = header_parts[0].lstrip('>').replace('UniRef100_', '')
                uid = self._extract_uniprot_id(header)
                has_uniref = "UniRef" in header
                uniprot_num = 0
                if uid:
                    n = self._uniprot_to_number([uid])
                    uniprot_num = n[0] if n else 0
                if is_first:
                    coverage = 1.0
                    identity = 1.0
                    evalue = 0.0
                    alnscore = float('inf')
                    is_first = False
                else:
                    coverage = identity = evalue = alnscore = None
                    if len(header_parts) >= 10:
                        try:
                            alnscore = float(header_parts[1])
                            identity = float(header_parts[2])
                            evalue = float(header_parts[3])
                            q_start = int(header_parts[4])
                            q_end = int(header_parts[5])
                            q_len = int(header_parts[6])
                            coverage = (q_end - q_start + 1) / q_len
                        except:
                            pass
                    coverage = coverage or 0.0
                    identity = identity or 0.0
                    evalue = evalue or float('inf')
                    alnscore = alnscore or 0.0
                entries.append({
                    'header': header_clean,
                    'sequence': sequence,
                    'coverage': coverage,
                    'identity': identity,
                    'evalue': evalue,
                    'alnscore': alnscore,
                    'uid': uid,
                    'uniprot_num': uniprot_num,
                    'has_uniref': has_uniref
                })
            else:
                i += 1
        return entries

    def _parse_paired_a3m(self, a3m_path: str) -> List[Dict]:
        raw_msas = {}
        update_M = True
        M = None
        with open(a3m_path, 'r', errors='ignore') as f:
            for line in f:
                if "\x00" in line:
                    line = line.replace("\x00", "")
                    update_M = True
                if line.startswith(">") and update_M:
                    M = int(line[1:].rstrip().split('_')[-1])
                    update_M = False
                    if M not in raw_msas:
                        raw_msas[M] = []
                if M is not None:
                    raw_msas[M].append(line.rstrip())
        parsed_msas = {sid: self._parse_msa_lines(lines) for sid, lines in raw_msas.items()}
        seq_ids = sorted(parsed_msas.keys())
        min_entries = min(len(parsed_msas[s]) for s in seq_ids)
        stitched = []
        for i in range(min_entries):
            headers = []
            sequences = []
            coverages = []
            identities = []
            evalues = []
            alnscores = []
            uids = []
            uniprot_nums = []
            has_uniref = True
            for sid in seq_ids:
                e = parsed_msas[sid][i]
                headers.append(e['header'])
                sequences.append(e['sequence'])
                coverages.append(e['coverage'])
                identities.append(e['identity'])
                evalues.append(e['evalue'])
                alnscores.append(e['alnscore'])
                uids.append(e['uid'])
                uniprot_nums.append(e['uniprot_num'])
                has_uniref = has_uniref and e['has_uniref']
            stitched.append({
                'headers': headers,
                'sequences': sequences,
                'coverages': coverages,
                'identities': identities,
                'evalues': evalues,
                'alnscores': alnscores,
                'uids': uids,
                'uniprot_nums': uniprot_nums,
                'has_uniref': has_uniref,
                'is_query': (i == 0)
            })
        return stitched

    def submit(self, sequences: List[str], genomic_distance: Optional[int] = 20, prefix: Optional[str] = None) -> str:
        query = ""
        for i, seq in enumerate(sequences, start=101):
            query += f">{prefix + '_' if prefix else ''}{i}\n{seq}\n"
        if len(sequences) == 1:
            endpoint = "ticket/msa"
            mode = "env"
        else:
            endpoint = "ticket/pair"
            mode = "paircomplete" if genomic_distance is None else f"paircomplete-pairfilterprox_{genomic_distance}"
        r = requests.post(f"{self.host_url}/{endpoint}", data={'q': query, 'mode': mode}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Submit failed: {r.text}")
        self.job_id = r.json()['id']
        return self.job_id

    def wait(self, poll: int = 5):
        while True:
            s = requests.get(f"{self.host_url}/ticket/{self.job_id}", timeout=30).json().get('status', 'UNKNOWN')
            print(f"[STATUS] {s}")
            if s == "COMPLETE":
                break
            if s == "ERROR":
                raise RuntimeError("Job failed")
            time.sleep(poll)

    def download_and_parse(self, output_dir: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        tar_path = Path(output_dir) / f"{self.job_id}.tar.gz"
        r = requests.get(f"{self.host_url}/result/download/{self.job_id}", timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Download failed: {r.text}")
        tar_path.write_bytes(r.content)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(output_dir)
        pair_a3m = Path(output_dir) / "pair.a3m"
        if pair_a3m.exists():
            self.parsed_entries = self._parse_paired_a3m(str(pair_a3m))
        else:
            self.parsed_entries = []

    def save_msa(self, output_file: str,
                 min_coverage: Optional[float] = None,
                 min_identity: Optional[float] = None,
                 max_evalue: Optional[float] = None,
                 min_alnscore: Optional[float] = None,
                 max_genomic_distance: Optional[int] = None) -> Tuple[int, List[Dict]]:
        assert self.parsed_entries, "No MSA parsed yet."
        if min_coverage and min_coverage > 1:
            min_coverage /= 100.0
        if min_identity and min_identity > 1:
            min_identity /= 100.0
        num_written = 0
        kept = []
        with open(output_file, 'w') as f:
            for entry in self.parsed_entries:
                if not entry['is_query']:
                    reasons = []
                    if min_coverage and any(c < min_coverage for c in entry['coverages']): reasons.append("cov")
                    if min_identity and any(i < min_identity for i in entry['identities']): reasons.append("id")
                    if max_evalue is not None and any(e is not None and e > max_evalue for e in entry['evalues']): reasons.append("e")
                    if min_alnscore is not None and any(a is not None and a < min_alnscore for a in entry['alnscores']): reasons.append("aln")
                    if reasons: continue
                header = "query" if entry['is_query'] else "_".join([u or h for u, h in zip(entry['uids'], entry['headers'])])
                seq = ''.join(entry['sequences']).replace('\x00', '')
                f.write(f">{header}\n{seq}\n")
                num_written += 1
                kept.append(entry)
        return num_written, kept

def get_paired_msa(sequences: Union[str, List[str]],
                   output_file: str,
                   genomic_distance: Optional[int] = 20,
                   min_coverage: Optional[float] = 75,
                   min_identity: Optional[float] = 15,
                   prefix: Optional[str] = None,
                   host_url: str = "https://api.colabfold.com",
                   cache_dir: Optional[str] = None) -> str:
    """Full automated paired MSA pipeline."""
    if isinstance(sequences, str):
        sequences = [s.strip().upper() for s in sequences.split(':') if s.strip()]
    else:
        sequences = [s.strip().upper() for s in sequences if s.strip()]
    assert sequences, "Empty sequences"
    msa = ColabFoldPairedMSA(host_url, cache_dir)
    msa.submit(sequences, genomic_distance, prefix)
    msa.wait()
    tmp_dir = tempfile.mkdtemp(prefix="colabfold_api_")
    try:
        msa.download_and_parse(tmp_dir)
        _, _ = msa.save_msa(output_file,
                            min_coverage=min_coverage,
                            min_identity=min_identity,
                            max_genomic_distance=genomic_distance)
    finally:
        p = Path(output_file)
        if p.exists():
            p.write_bytes(p.read_bytes().replace(b"\x00", b""))
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return output_file
# --------------------------- END ColabFoldPairedMSA ---------------------------

# ---------------------------
# Visualization
# ---------------------------
def _save_heatmap(mat: np.ndarray, path: Path, title: str):
    plt.figure(figsize=(6, 3))
    plt.imshow(mat, origin="upper", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.title(title); plt.colorbar(fraction=0.03, pad=0.02)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200); plt.close()

# ---------------------------
# Data loading
# ---------------------------
def load_pairs_from_test_and_json(test_file: str, interface_json: str) -> Dict[str, Tuple[str, str]]:
    with open(test_file) as f:
        keys = [line.strip() for line in f if line.strip()]
    with open(interface_json) as f:
        data = json.load(f)
    pairs = {}
    for k in keys:
        if k not in data:
            print(f"[WARN] {k} not found in interface_json, skip"); continue
        seqs = data[k]["Sequences"]
        pairs[k] = (seqs["ligand_seq"], seqs["receptor_seq"])
    print(f"[data] loaded {len(pairs)}/{len(keys)} pairs")
    return pairs

def _load_gt_npz(path: str) -> Tuple[np.ndarray, str]:
    d = np.load(path)
    # 新增兼容字段名
    for k in ["labels_contact", "contact", "gt_map"]:
        if k in d:
            return d[k].astype(np.float32), "contact"
    for k in ["distance", "distance_map", "distances"]:
        if k in d:
            return d[k].astype(np.float32), "distance"
    raise RuntimeError(f"Unknown GT format: {list(d.keys())}")

# ---------------------------
# Core
# ---------------------------
def run_single_pair(
    key: str, pep_seq: str, prot_seq: str, model, device: torch.device,
    out_dir: Path, gt_index: Optional[str], gt_as_contact: bool, gt_th: float,
    bin_th: float, save_text: bool, save_plots: bool, map_mode: str, use_amp: bool,
    interface_data: Optional[dict] = None
):
    """Run prediction for a single protein-peptide pair."""
    out_npz, out_txt, out_fig = [out_dir / x for x in ("pred_npz", "texts", "figs")]
    for p in (out_npz, out_txt, out_fig): p.mkdir(parents=True, exist_ok=True)

    # 临时工作目录
    with tempfile.TemporaryDirectory(prefix=f"pair_{key}_") as tmpdir:
        tmp = Path(tmpdir)
        a3m = tmp / f"{key}.a3m"

        # --- 1️⃣ 提前裁剪序列 ---
        MAX_PEP_LEN = 80
        MAX_PROT_LEN = 600
        if len(pep_seq) > MAX_PEP_LEN:
            print(f"[{key}] Peptide too long ({len(pep_seq)}), cropping to {MAX_PEP_LEN}")
            pep_seq = pep_seq[:MAX_PEP_LEN]
        if len(prot_seq) > MAX_PROT_LEN:
            print(f"[{key}] Protein too long ({len(prot_seq)}), cropping to {MAX_PROT_LEN}")
            prot_seq = prot_seq[:MAX_PROT_LEN]

        # --- 2️⃣ 生成或复用 paired MSA (使用裁剪后的序列) ---
        msa_output_dir = out_dir / "msa_files"
        msa_output_dir.mkdir(parents=True, exist_ok=True)
        msa_cached_path = msa_output_dir / f"{key}.a3m"

        if msa_cached_path.exists() and msa_cached_path.stat().st_size > 0:
            print(f"[{key}] MSA already exists → using cached file: {msa_cached_path}")
            shutil.copy(msa_cached_path, a3m)
        else:
            print(f"[{key}] fetching MSA via ColabFold API...")
            try:
                get_paired_msa([pep_seq, prot_seq], str(a3m),
                               genomic_distance=20, min_coverage=75, min_identity=15)
                shutil.copy(str(a3m), msa_cached_path)
                print(f"[{key}] MSA saved to cache: {msa_cached_path}")
            except Exception as e:
                print(f"[ERROR] Failed to fetch MSA for {key}: {e}")
                return {"key": key, "metrics": None, "error": str(e)}

        # --- 3️⃣ hhfilter 检查 ---
        bin_hhfilter = shutil.which("hhfilter")
        diverse_method = "hhfilter" if bin_hhfilter else "none"
        hh_args = {"binary": "hhfilter"} if bin_hhfilter else None

        # --- 4️⃣ 加载 MSA 对象 ---
        try:
            msa = MSA(
                msa_file_path=str(a3m),
                max_seqs=512,
                max_length=MAX_PEP_LEN + MAX_PROT_LEN,
                max_tokens=1e12,
                diverse_select_method=diverse_method,
                hhfilter_kwargs=hh_args,
            )
        except Exception as e:
            print(f"[ERROR] Failed to process MSA for {key}: {e}")
            return {"key": key, "metrics": None, "error": str(e)}

        # --- 5️⃣ 准备模型输入 ---
        msa_onehot = F.one_hot(msa.diverse_tokenized_msa, num_classes=len(aa2tok_d)).unsqueeze(0).float().to(device)
        masks = [x.to(device) for x in prepare_msa_masks(msa.diverse_tokenized_msa.unsqueeze(0))]
        breaks = [len(pep_seq)]

        msa_output_dir = out_dir / "msa_files"
        msa_output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(a3m), msa_output_dir / f"{key}.a3m")
        print(f"[{key}] MSA saved to {msa_output_dir / f'{key}.a3m'}")

        # --- 6️⃣ 推理 ---
        try:
            with torch.no_grad():
                if device.type == "cuda" and use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = model(
                            msa=msa_onehot.to(torch.bfloat16),
                            mask=masks[0], msa_mask=masks[1], full_mask=masks[2],
                            pairwise_mask=masks[3], complex_chain_break_indices=[breaks]
                        )
                else:
                    out = model(
                        msa=msa_onehot,
                        mask=masks[0], msa_mask=masks[1], full_mask=masks[2],
                        pairwise_mask=masks[3], complex_chain_break_indices=[breaks]
                    )
        except Exception as e:
            print(f"[ERROR] Model inference failed for {key}: {e}")
            return {"key": key, "metrics": None, "error": str(e)}

        # --- 7️⃣ 概率矩阵裁剪同步 ---
        contacts = out["contacts"].cpu().numpy()[0]
        prob = contacts
        Lp, Lprot = len(pep_seq), len(prot_seq)
        cross_prob = prob[:Lp, Lp:Lp + Lprot]

        # --- 8️⃣ 保存预测 ---
        # --- 8️⃣ 保存预测（修改版，与 test.py 对齐） ---
        npz_name = f"{key}_full.npz" if map_mode == "full" else f"{key}.npz"
        pred_map = prob if map_mode == "full" else cross_prob

        # 构建输出记录
        out_rec = {
            "scores": pred_map.astype(np.float32),
            "prob": pred_map.astype(np.float32),
            "pair_mask": np.ones_like(pred_map, dtype=bool),
            "key": key
        }

        # 二值图
        bin_th = float(bin_th)
        out_rec["binary"] = (pred_map >= bin_th).astype(np.uint8)

        # 若稍后加载到 GT，则在第 9️⃣ 步追加 gt_map
        npz_path = out_npz / npz_name
        np.savez_compressed(npz_path, **out_rec)

        if save_text:
            stats = f"prob: min={pred_map.min():.3f} mean={pred_map.mean():.3f} max={pred_map.max():.3f}"
            (out_txt / (f"{key}_full.txt" if map_mode == "full" else f"{key}.txt")).write_text(stats)
        if save_plots:
            _save_heatmap(pred_map, out_fig / f"{key}_prob.png", f"{key} prob map")



        # --- 9️⃣ GT 裁剪评估 ---
        gt_metrics = None
        if gt_index and Path(gt_index).exists():
            try:
                gtdb = json.load(open(gt_index))
                if key in gtdb or key.replace("_nomutation", "") in gtdb:
                    gtpath = gtdb.get(key, gtdb.get(key.replace("_nomutation", "")))
                    if isinstance(gtpath, dict):
                        gtpath = gtpath.get("npz_path", gtpath)
                    if not os.path.isabs(gtpath):
                        gt_dir = os.path.dirname(gt_index)
                        gt_base = os.path.basename(gt_dir)

                        # --- 情况 1: gtpath 本身已经包含目录 (e.g. full_distance_maps_v2/xxx.npz)
                        if gtpath.startswith(gt_base + "/") or gt_base in gtpath.split("/"):
                            gtpath = os.path.join(os.path.dirname(gt_dir), gtpath)
                        # --- 情况 2: gtpath 只是文件名 (e.g. 3kti_N_G.npz)
                        else:
                            gtpath = os.path.join(gt_dir, gtpath)

                    # Debug log
                    print(f"[GT-DEBUG] resolved path for {key}: {gtpath}")


                    # ✅ 打印调试信息
                    if not os.path.exists(gtpath):
                        print(f"[GT-WARN] Missing GT file for {key}: {gtpath}")

                    gt, kind = _load_gt_npz(gtpath)

                    pep_len_orig = len(interface_data[key]["Sequences"]["ligand_seq"])
                    prot_len_orig = len(interface_data[key]["Sequences"]["receptor_seq"])
                    pep_len_trim, prot_len_trim = min(pep_len_orig, 80), min(prot_len_orig, 600)

                    start_col = pep_len_orig
                    end_col = min(gt.shape[1], start_col + prot_len_trim)
                    gt_cross = gt[:pep_len_trim, start_col:end_col]

                    # align shape
                    min_rows = min(gt_cross.shape[0], cross_prob.shape[0])
                    min_cols = min(gt_cross.shape[1], cross_prob.shape[1])
                    gt_cross, cross_prob = gt_cross[:min_rows, :min_cols], cross_prob[:min_rows, :min_cols]

                    print(f"[{key}] GT cropped: full={gt.shape}, cross={gt_cross.shape} (pep={pep_len_trim}, prot={prot_len_trim})")

                    if kind == "distance":
                        gt_bin = _binarize_from_distance(gt_cross, gt_th)
                    else:
                        gt_bin = (gt_cross > 0.5).astype(np.float32)

                    gt_metrics = _metrics_from_binary(cross_prob, gt_bin, bin_th)
                    if save_text:
                        with open(out_txt / f"{key}.txt", "a") as f:
                            f.write(f"\n[GT] F1={gt_metrics['f1']:.3f} AUPRC={gt_metrics['auprc']:.3f}\n")
                    if save_plots:
                        _save_heatmap(gt_bin, out_fig / f"{key}_gt.png", f"{key} GT")

                    # --- 附加: 保存 GT 到同一个 npz 中，与 test.py 格式完全一致 ---
                    try:
                        if os.path.exists(npz_path):
                            old = dict(np.load(npz_path))
                            old["gt_map"] = gt_bin.astype(np.float32)
                            np.savez_compressed(npz_path, **old)
                            print(f"[{key}] gt_map appended to {npz_path.name}")
                    except Exception as e:
                        print(f"[WARN] Failed to append gt_map for {key}: {e}")


            except Exception as e:
                print(f"[WARNING] Failed to evaluate against GT for {key}: {e}")

        return {"key": key, "metrics": gt_metrics}

# ---------------------------
# Main
# ---------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", required=True, help="File containing test sample keys")
    ap.add_argument("--interface-json", required=True, help="JSON file with interface data")
    ap.add_argument("--gt-index", type=str, help="Ground truth index JSON file")
    ap.add_argument("--gt-as-contact", action="store_true", default=True)
    ap.add_argument("--gt-threshold", type=float, default=8.0, help="Distance threshold for GT binarization")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--bin-th", type=float, default=0.8, help="Binary threshold for predictions")
    ap.add_argument("--save-text", action="store_true", help="Save text statistics")
    ap.add_argument("--save-plots", action="store_true", help="Save visualization plots")
    ap.add_argument("--map-mode", choices=["cross","full"], default="cross", help="Output map mode")
    ap.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--print-first", type=int, default=3, help="Print first N predictions")
    # ===== minimal-add: resume =====
    ap.add_argument("--resume", action="store_true", help="Skip samples whose output npz already exists")
    args = ap.parse_args()

    # Check if required modules are available
    if MSAPairformer is None or MSA is None or prepare_msa_masks is None:
        print("[ERROR] Required MSAPairformer modules not found. Please install them first.")
        return
    
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    
    # Load data
    pairs = load_pairs_from_test_and_json(args.test_file, args.interface_json)
    # 新增：加载完整的 interface_json 数据
    with open(args.interface_json) as f:
        interface_data = json.load(f)
    
    # Load model
    try:
        model = MSAPairformer.from_pretrained(device=device).to(device).eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    
    # Process each pair
    for i, (k, (pep, prot)) in enumerate(pairs.items(), 1):
        print(f"\n[{i}/{len(pairs)}] {k}")

        # ===== minimal-add: resume check (仅依据 pred_npz 是否存在) =====
        npz_path = out_dir / "pred_npz" / (f"{k}.npz" if args.map_mode == "cross" else f"{k}_full.npz")
        if args.resume and npz_path.exists():
            print(f"[{k}] → skipped (resume mode)")
            _cleanup_cuda(tag=f"{k} (resume-skip)")
            continue

        # 执行单样本推理
        try:
            res = run_single_pair(
                k, pep, prot, model, device, out_dir,
                args.gt_index, args.gt_as_contact, args.gt_threshold,
                args.bin_th, args.save_text, args.save_plots,
                args.map_mode, args.amp,
                interface_data=interface_data 
            )
        finally:
            # ===== minimal-add: 每样本后清理 CUDA 显存 =====
            _cleanup_cuda(tag=k)

        if res and res.get("metrics"):
            summary.append({k: res["metrics"]})
        
        # Print first few predictions（保持你的原逻辑）
        if i <= args.print_first:
            if npz_path.exists():
                try:
                    z = np.load(npz_path)
                    if "prob" in z:
                        print(_format_head(z["prob"], H=10))
                except Exception as e:
                    print(f"[WARN] failed to read {npz_path.name}: {e}")
    
    # Save summary
    with open(out_dir/"summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[done] {len(pairs)} samples processed.")

if __name__ == "__main__":
    main()
