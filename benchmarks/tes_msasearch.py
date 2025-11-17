#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_colabfold_pair_msa.py
---------------------------------
Quick test to verify ColabFold paired MSA depth & coverage.

Usage:
    python test_colabfold_pair_msa.py --pep MFLK... --prot MGDV... --out test_pair_out
"""

import os, json, argparse
from pathlib import Path
from statistics import mean
from pprint import pprint
from textwrap import shorten

from MSA_Pairformer.dataset import MSA
from pairformer import get_paired_msa  # ✅ uses your fixed new version

# ---------------------------
# CLI
# ---------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--pep", required=True, help="Peptide sequence (A chain)")
ap.add_argument("--prot", required=True, help="Protein sequence (B chain)")
ap.add_argument("--out", default="msa_test_out", help="Output directory")
ap.add_argument("--mincov", type=float, default=75)
ap.add_argument("--minid", type=float, default=15)
ap.add_argument("--dist", type=int, default=20)
args = ap.parse_args()

out_dir = Path(args.out)
out_dir.mkdir(exist_ok=True)
a3m_path = out_dir / "pair.a3m"

# ---------------------------
# Run paired MSA search
# ---------------------------
print(f"\n[INFO] Running ColabFold paired MSA search ...")
print(f"Peptide len={len(args.pep)}, Protein len={len(args.prot)}")
print(f"→ Output: {a3m_path}")

try:
    get_paired_msa(
        [args.pep, args.prot],
        output_file=str(a3m_path),
        genomic_distance=args.dist,
        min_coverage=args.mincov,
        min_identity=args.minid,
    )
except Exception as e:
    print(f"[ERROR] MSA retrieval failed: {e}")
    exit(1)

if not a3m_path.exists() or a3m_path.stat().st_size == 0:
    print("[ERROR] No MSA file generated.")
    exit(1)

print(f"[OK] MSA file saved: {a3m_path.resolve()}")

# ---------------------------
# Parse and summarize MSA content
# ---------------------------
print("\n[INFO] Parsing MSA content ...")
msa = MSA(
    msa_file_path=str(a3m_path),
    max_seqs=2048,
    max_length=5000,
    diverse_select_method="none",
)

num_seqs = msa.tokenized_msa.shape[0]
seq_lens = [len(seq) for seq in msa.sequences] if hasattr(msa, "sequences") else []
print(f"[SUMMARY]")
print(f"  • total sequences: {num_seqs}")
print(f"  • query length: {len(args.pep) + len(args.prot)}")
print(f"  • MSA file size: {a3m_path.stat().st_size / 1024:.1f} KB")

# peek first few headers (raw)
lines = [l.strip() for l in open(a3m_path) if l.startswith(">")]
print(f"  • first 5 headers:")
for l in lines[:5]:
    print("    ", shorten(l, width=120))

# Optional: coverage / identity stats (approximate)
covs, ids = [], []
try:
    for l in open(a3m_path):
        if "\t" in l and not l.startswith(">query"):
            parts = l.strip().split("\t")
            if len(parts) >= 7:
                try:
                    alnscore = float(parts[1])
                    identity = float(parts[2])
                    evalue = float(parts[3])
                    qstart = int(parts[4])
                    qend = int(parts[5])
                    qlen = int(parts[6])
                    coverage = (qend - qstart + 1) / qlen
                    covs.append(coverage)
                    ids.append(identity)
                except:
                    pass
except Exception:
    pass

if len(covs) > 0:
    print(f"  • avg coverage={mean(covs)*100:.1f}%  avg identity={mean(ids)*100:.1f}%")
else:
    print("  • coverage/identity info not available (no hits with annotation)")

# Check if only query
if num_seqs <= 1:
    print("[WARN] MSA only contains query sequence! Try adjusting min_coverage / min_identity or use single-chain mode.")
elif num_seqs < 10:
    print(f"[INFO] MSA shallow ({num_seqs} seqs). You may lower min_coverage/min_identity thresholds.")
else:
    print(f"[INFO] MSA depth OK ({num_seqs} seqs).")

print("\n[DONE] Summary complete.\n")
