#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
json_to_esm_embeddings_onekey.py

Process ONE entry (by --key) from interface_data.json (the output of
pdb_interface_to_json_singlepair.py) and generate ESM3 embeddings with
ALL tokens kept (<cls>, residues incl. padding-X, <eos>).

Requirements:
  pip install esm torch numpy tqdm

Usage:
  python json_to_esm_embeddings_onekey.py \
      --input_json out_json/interface_data.json \
      --key 2rkz_O_C_nomutation \
      --output_dir esm_npz/ \
      --order protein_peptide \
      --pad-protein 600 --pad-peptide 80 \
      --dtype float32 --device auto
"""

import os
import json
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# --------------------------- helpers ---------------------------

STD = set("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS_TO_X = set("BJOUZ")  # map to 'X'

def clean_seq(seq: str) -> str:
    """Map any non-20AA / ambiguous letters to 'X'."""
    s = (seq or "").upper().replace(" ", "").replace("\n", "")
    out = []
    for ch in s:
        if ch in STD:
            out.append(ch)
        elif ch in AMBIGUOUS_TO_X:
            out.append("X")
        else:
            out.append("X")
    return "".join(out)

def pad_seq(seq: str, max_len: int | None) -> Tuple[str, np.ndarray]:
    """
    Right-pad sequence with 'X' to max_len (if provided).
    Returns (seq_used, is_real_mask) where is_real_mask marks real (non-padding) residues.
    (special tokens are NOT counted here)
    """
    if max_len is None:
        return seq, np.ones(len(seq), dtype=bool)
    if len(seq) >= max_len:
        return seq[:max_len], np.ones(max_len, dtype=bool)
    pad_len = max_len - len(seq)
    return seq + "X" * pad_len, np.r_[np.ones(len(seq), bool), np.zeros(pad_len, bool)]

def load_esm3_model(device: str):
    """
    Load ESM3 open small via official 'esm' package.
    Returns (model, tokenizers).
    """
    try:
        from esm.pretrained import ESM3_sm_open_v0, get_esm3_model_tokenizers
        model = ESM3_sm_open_v0(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        toks = get_esm3_model_tokenizers()
        return model, toks
    except Exception as e:
        raise RuntimeError(
            "Failed to load ESM3 model. Install with: pip install esm\n"
            f"Original error: {e}"
        )

@torch.no_grad()
def forward_chain_all_tokens(model, tokenizers, seq_used: str, device: str):
    """
    Forward a single chain sequence and RETURN ALL TOKEN EMBEDDINGS (keep <cls>, <eos>, padding-X).
    Outputs:
      tokens (np.int64 [T]),
      tokens_str (object [T]),
      embedding (float32 [T, D]),
      is_special (bool [T]), is_residue (bool [T]), is_padding (bool [T] - init zeros),
      residue_mask_real (bool [T] - True on residue window [1:1+L_used]),
      cls_index (int), eos_index (int)
    """
    tokens = tokenizers.sequence.encode(seq_used)  # list[int], tokenizer adds <cls>/<eos>
    T = len(tokens)

    vocab = tokenizers.sequence.get_vocab()  # token->id
    id2tok = {v: k for k, v in vocab.items()}
    tokens_str = np.array([id2tok.get(tid, f"<UNK_ID_{tid}>") for tid in tokens], dtype=object)

    is_special = np.array([tok.startswith("<") and tok.endswith(">") for tok in tokens_str], dtype=bool)
    is_residue = ~is_special

    L_used = len(seq_used)
    cls_index = 0
    eos_index = T - 1

    residue_mask_real = np.zeros(T, dtype=bool)
    if L_used > 0:
        residue_mask_real[1:1+L_used] = True  # residues occupy [1 .. 1+L_used-1]

    is_padding = np.zeros(T, dtype=bool)  # will be filled in outer scope knowing raw vs used

    t = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0)
    if device == "cuda" and torch.cuda.is_available():
        t = t.cuda()
    out = model(sequence_tokens=t)
    emb = out.embeddings[0]  # [T, D]
    if emb.is_cuda:
        emb = emb.cpu()
    embedding = emb.float().numpy()

    return (
        np.array(tokens, dtype=np.int64),
        tokens_str,
        embedding.astype(np.float32),
        is_special,
        is_residue,
        is_padding,
        residue_mask_real,
        int(cls_index),
        int(eos_index),
    )

def mark_padding_on_token_axis(is_padding_token: np.ndarray,
                               residue_mask_real: np.ndarray,
                               raw_len: int,
                               used_len: int):
    """
    Mark padding positions on TOKEN axis: inside residue window, the last
    (used_len - raw_len) positions correspond to padded-X.
    """
    if used_len <= raw_len:
        return
    if residue_mask_real.sum() == 0:
        return
    start_idx = np.argmax(residue_mask_real)  # first True (should be 1)
    pad_count = used_len - raw_len
    is_padding_token[start_idx + raw_len : start_idx + used_len] = True

# --------------------------- main for a single key ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate ESM3 embeddings (ALL tokens kept) for ONE key from interface_data.json")
    ap.add_argument("--input_json", required=True, help="Path to interface_data.json")
    ap.add_argument("--key", required=True, help="Key to process (e.g., 2rkz_O_C_nomutation)")
    ap.add_argument("--output_dir", required=True, help="Directory to save the .npz")
    ap.add_argument("--order", choices=["protein_peptide", "peptide_protein"], default="protein_peptide",
                    help="Concatenation order at complex level")
    ap.add_argument("--pad-protein", type=int, default=None, help="Pad/crop protein to this length (e.g., 600)")
    ap.add_argument("--pad-peptide", type=int, default=None, help="Pad/crop peptide to this length (e.g., 80)")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32", help="Embedding dtype on disk")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args()

    # device
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else \
             (args.device if args.device != "auto" else "cpu")

    # load input json & select key
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.key not in data:
        raise SystemExit(f"Key '{args.key}' not found in {args.input_json}")

    item = data[args.key]

    # sequences
    receptor_seq = item["Sequences"]["receptor_seq"]  # protein
    ligand_seq   = item["Sequences"]["ligand_seq"]    # peptide

    chains = item.get("Meta", {}).get("chains", {})
    prot_id = chains.get("receptor", "PROTEIN")
    pep_id  = chains.get("ligand",   "PEPTIDE")

    # clean & pad
    prot_raw = clean_seq(receptor_seq)
    pep_raw  = clean_seq(ligand_seq)
    prot_used, prot_real_mask = pad_seq(prot_raw, args.pad_protein)
    pep_used,  pep_real_mask  = pad_seq(pep_raw,  args.pad_peptide)

    # load model/tokenizer
    model, tokenizers = load_esm3_model(device=device)

    # forward (ALL tokens kept)
    (
        p_tokens, p_tokens_str, p_emb, p_is_special, p_is_residue, p_is_padding, p_residue_mask_real, p_cls_idx, p_eos_idx
    ) = forward_chain_all_tokens(model, tokenizers, prot_used, device)

    (
        l_tokens, l_tokens_str, l_emb, l_is_special, l_is_residue, l_is_padding, l_residue_mask_real, l_cls_idx, l_eos_idx
    ) = forward_chain_all_tokens(model, tokenizers, pep_used, device)

    # mark padding token positions using raw vs used lengths
    mark_padding_on_token_axis(p_is_padding, p_residue_mask_real, raw_len=len(prot_raw), used_len=len(prot_used))
    mark_padding_on_token_axis(l_is_padding, l_residue_mask_real, raw_len=len(pep_raw),  used_len=len(pep_used))

    # complex stacking
    if args.order == "protein_peptide":
        complex_emb = np.vstack([p_emb, l_emb])
        complex_is_special = np.r_[p_is_special, p_is_residue*False]  # placeholder to ensure same length? (not needed)
        complex_is_special = np.r_[p_is_special, l_is_special]
        complex_is_residue = np.r_[p_is_residue, l_is_residue]
        complex_is_padding = np.r_[p_is_padding, l_is_padding]
        chain_offsets_tokens = [
            {"chain_id": prot_id, "start": 0,           "end": len(p_emb)},
            {"chain_id": pep_id,  "start": len(p_emb),  "end": len(p_emb) + len(l_emb)},
        ]
        chain_offsets_residues = [
            {"chain_id": prot_id, "start": 1, "end": len(p_emb)-1},  # [start, end) excludes <cls>/<eos>
            {"chain_id": pep_id,  "start": len(p_emb)+1, "end": len(p_emb)+len(l_emb)-1},
        ]
        chain_ids = np.array([prot_id, pep_id], dtype=object)
    else:
        complex_emb = np.vstack([l_emb, p_emb])
        complex_is_special = np.r_[l_is_special, p_is_special]
        complex_is_residue = np.r_[l_is_residue, p_is_residue]
        complex_is_padding = np.r_[l_is_padding, p_is_padding]
        chain_offsets_tokens = [
            {"chain_id": pep_id,  "start": 0,           "end": len(l_emb)},
            {"chain_id": prot_id, "start": len(l_emb),  "end": len(l_emb) + len(p_emb)},
        ]
        chain_offsets_residues = [
            {"chain_id": pep_id,  "start": 1, "end": len(l_emb)-1},
            {"chain_id": prot_id, "start": len(l_emb)+1, "end": len(l_emb)+len(p_emb)-1},
        ]
        chain_ids = np.array([pep_id, prot_id], dtype=object)

    # dtype cast
    if args.dtype == "float16":
        complex_emb = complex_emb.astype(np.float16)
        p_emb = p_emb.astype(np.float16)
        l_emb = l_emb.astype(np.float16)
    else:
        complex_emb = complex_emb.astype(np.float32)
        p_emb = p_emb.astype(np.float32)
        l_emb = l_emb.astype(np.float32)

    # token ids meta
    vocab = tokenizers.sequence.get_vocab()
    tok2id = {k: int(v) for k, v in vocab.items()}
    token_ids_meta = {
        "cls": tok2id.get("<cls>", None),
        "eos": tok2id.get("<eos>", None),
        "unk": tok2id.get("<unk>", None),
        "x": tok2id.get("X", None),
    }

    # metadata
    meta = {
        "model": "ESM3_sm_open_v0",
        "library": "esm",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dim": int(p_emb.shape[1]),
        "order": args.order,
        "source_key": args.key,
        "source_meta": {
            "pdb_file": item.get("Meta", {}).get("pdb_file"),
            "chains": item.get("Meta", {}).get("chains"),
            "stats": item.get("Output", {}).get("Stats"),
            "interface_globals": item.get("Output", {}).get("Interface"),
        },
        "lengths": {
            "protein": {"raw": len(prot_raw), "used": len(prot_used)},
            "peptide": {"raw": len(pep_raw),  "used": len(pep_used)},
        },
        "token_ids": token_ids_meta,
        "notes": {
            "kept_special_tokens": True,
            "kept_padding": True,
            "is_residue = ~is_special": True,
            "residue_window_per_chain": "[1, T-1)",
        },
    }

    # save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{args.key}_esm3_alltokens.npz"

    np.savez_compressed(
        npz_path,
        # per-chain protein
        protein_tokens=p_tokens.astype(np.int64),
        protein_tokens_str=p_tokens_str,
        protein_embedding=p_emb,
        protein_is_special=p_is_special.astype(np.bool_),
        protein_is_residue=p_is_residue.astype(np.bool_),
        protein_is_padding=p_is_padding.astype(np.bool_),
        protein_residue_mask_real=p_residue_mask_real.astype(np.bool_),
        protein_cls_index=np.array([0], dtype=np.int32),
        protein_eos_index=np.array([len(p_emb)-1], dtype=np.int32),
        protein_id=np.array([prot_id], dtype=object),
        protein_seq_raw=np.array([prot_raw], dtype=object),
        protein_seq_used=np.array([prot_used], dtype=object),

        # per-chain peptide
        peptide_tokens=l_tokens.astype(np.int64),
        peptide_tokens_str=l_tokens_str,
        peptide_embedding=l_emb,
        peptide_is_special=l_is_special.astype(np.bool_),
        peptide_is_residue=l_is_residue.astype(np.bool_),
        peptide_is_padding=l_is_padding.astype(np.bool_),
        peptide_residue_mask_real=l_residue_mask_real.astype(np.bool_),
        peptide_cls_index=np.array([0], dtype=np.int32),
        peptide_eos_index=np.array([len(l_emb)-1], dtype=np.int32),
        peptide_id=np.array([pep_id], dtype=object),
        peptide_seq_raw=np.array([pep_raw], dtype=object),
        peptide_seq_used=np.array([pep_used], dtype=object),

        # complex-level (stacked)
        complex_embedding=complex_emb,
        complex_is_special=complex_is_special.astype(np.bool_),
        complex_is_residue=complex_is_residue.astype(np.bool_),
        complex_is_padding=complex_is_padding.astype(np.bool_),
        chain_ids=chain_ids,
        chain_offsets_tokens=json.dumps(chain_offsets_tokens, ensure_ascii=False),
        chain_offsets_residues=json.dumps(chain_offsets_residues, ensure_ascii=False),
        order=np.array([args.order], dtype=object),

        # meta
        metadata=json.dumps(meta, ensure_ascii=False),
    )

    print(f"✅ Saved: {npz_path}")
    print(f"   protein tokens: {p_emb.shape[0]} | peptide tokens: {l_emb.shape[0]} | complex tokens: {complex_emb.shape[0]}")
    print(f"   dim: {p_emb.shape[1]} | order: {args.order} | device: {meta['device']}")

if __name__ == "__main__":
    main()
