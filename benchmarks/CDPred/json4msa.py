#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_msa_from_json_concat.py

严格按要求：拼接 ligand_seq + receptor_seq 为单条复合序列后，用 colabfold_batch 搜索。
保留原参数、日志、--num-recycle、--msa-mode 等所有设置。
"""

import os
import sys
import json
import subprocess
import argparse

def generate_msa_for_test(test_file, json_file, output_base_dir, use_env=True, num_recycles=0):
    # 读取 test keys
    with open(test_file) as f:
        keys = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Loaded {len(keys)} test entries from {test_file}")

    # 读取 JSON 数据
    with open(json_file) as f:
        data = json.load(f)

    os.makedirs(output_base_dir, exist_ok=True)

    # 参数配置
    msa_mode = "mmseqs2_uniref_env" if use_env else "mmseqs2_uniref"
    print(f"[INFO] Using MSA mode: {msa_mode}")

    for key in keys:
        if key not in data:
            print(f"[WARN] {key} not found in JSON, skipping")
            continue

        seqs = data[key]["Sequences"]
        pep_seq = seqs["ligand_seq"].strip().upper()
        prot_seq = seqs["receptor_seq"].strip().upper()

        # 限制长度
        if len(pep_seq) > 80:
            print(f"[{key}] Peptide too long ({len(pep_seq)}), cropping to 80")
            pep_seq = pep_seq[:80]
        if len(prot_seq) > 600:
            print(f"[{key}] Protein too long ({len(prot_seq)}), cropping to 600")
            prot_seq = prot_seq[:600]

        # ======= 仅修改这一部分：拼接复合序列 =======
        concat_seq = pep_seq + prot_seq
        # =========================================

        # 生成 FASTA 文件
        fasta_dir = os.path.join(output_base_dir, key)
        os.makedirs(fasta_dir, exist_ok=True)
        fasta_path = os.path.join(fasta_dir, f"{key}.fasta")

        with open(fasta_path, "w") as f:
            f.write(f">{key}\n{concat_seq}\n")

        print(f"[{key}] FASTA written → {fasta_path}")

        # 使用 colabfold_batch 生成 MSA（保留你原参数）
        cmd = [
            "colabfold_batch",
            "--msa-mode", msa_mode,
            "--num-models", "1",
            "--num-recycle", str(num_recycles),
            fasta_path,
            fasta_dir
        ]

        print(f"[RUN] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"[DONE] MSA complete for {key}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed on {key}: {e}")
        except Exception as e:
            print(f"[ERROR] Unknown error for {key}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate concatenated (ligand+receptor) MSA for test complexes using ColabFold")
    parser.add_argument("--test-file", required=True, help="Path to test.txt containing complex keys")
    parser.add_argument("--json-file", required=True, help="Path to interface_data.json containing sequences")
    parser.add_argument("--output-dir", required=True, help="Base directory for MSA outputs")
    parser.add_argument("--no-env", action="store_false", dest="use_env", help="Do not use environmental database")
    parser.add_argument("--num-recycles", type=int, default=0, help="Number of recycles for ColabFold")
    args = parser.parse_args()

    if not os.path.isfile(args.test_file):
        print(f"错误: test 文件 {args.test_file} 不存在")
        sys.exit(1)
    if not os.path.isfile(args.json_file):
        print(f"错误: JSON 文件 {args.json_file} 不存在")
        sys.exit(1)

    generate_msa_for_test(args.test_file, args.json_file, args.output_dir, args.use_env, args.num_recycles)

if __name__ == "__main__":
    main()
