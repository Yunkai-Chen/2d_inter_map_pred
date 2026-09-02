#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
json_to_embeddings_batch.py

批量读取 interface_data.json（来自 pdb_interface_to_json_singlepair.py），
对每个 key（peptide-protein）分别用 ESM3-open 或 ESMC 计算 embedding，
保留 ALL tokens（<cls> + residues(含padding X) + <eos>），并保存为 .npz。

修复点：
- 新增 ensure_TD：若上游返回 (D, T) 形状，自动转置为 (T, D)，避免 vstack 报维度不一致。
- ESM3 与 ESMC 的前向分别实现；ESMC 无 token ids，使用 -1 占位。
- 保留各种 mask / 映射，方便下游按需筛选。

依赖：
  pip install esm torch numpy tqdm

示例：
  python json_to_embeddings_batch.py \
      --input_json out_json/interface_data.json \
      --output_dir esm_npz/ \
      --model esm3 \
      --order protein_peptide \
      --pad-protein 600 --pad-peptide 80 \
      --dtype float32 --device auto \
      --checkpoint-every 50 --resume

  # 切换到 ESMC（如 600M）并指定 GPU 1
  CUDA_VISIBLE_DEVICES=1 python json_to_embeddings_batch.py \
      --input_json out_json/interface_data.json \
      --output_dir esmc_npz/ \
      --model esmc --esmc-ckpt esmc_600m \
      --device cuda --cuda-id 0 \
      --pad-protein 600 --pad-peptide 80 \
      --dtype float16 --checkpoint-every 100 --resume
"""

import os
import re
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import Tuple, Dict, List, Any

import numpy as np
import torch
from tqdm import tqdm

# --------------------------- basic helpers ---------------------------

STD = set("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS_TO_X = set("BJOUZ")


def clean_seq(seq: str) -> str:
    s = (seq or "").upper().replace(" ", "").replace("\n", "")
    out = []
    for ch in s:
        if ch in STD:
            out.append(ch)
        else:
            out.append("X")
    return "".join(out)


def pad_seq(seq: str, max_len: int | None) -> Tuple[str, np.ndarray]:
    """右侧 pad 到 max_len；返回 (seq_used, is_real_mask)（只对残基窗口）"""
    if max_len is None:
        return seq, np.ones(len(seq), dtype=bool)
    if len(seq) >= max_len:
        return seq[:max_len], np.ones(max_len, dtype=bool)
    pad_len = max_len - len(seq)
    return seq + "X" * pad_len, np.r_[np.ones(len(seq), bool), np.zeros(pad_len, bool)]


def ensure_TD(emb: np.ndarray, T_expected: int) -> np.ndarray:
    """
    确保 embedding 形状为 (T, D)。若收到 (D, T) 则转置。
    T_expected：该链 tokens 数（含 specials）。
    """
    if emb.ndim != 2:
        raise ValueError(f"Embedding must be 2D, got {emb.ndim}D with shape {emb.shape}")
    T, D = emb.shape
    if T == T_expected:
        return emb
    if D == T_expected:
        return emb.T
    raise ValueError(f"Unexpected embedding shape {emb.shape}; none axis equals expected T={T_expected}")


def mark_padding_on_token_axis(is_padding_token: np.ndarray,
                               residue_mask_real: np.ndarray,
                               raw_len: int,
                               used_len: int,
                               has_cls: bool):
    """把残基窗口内尾部 pad 的 X 标成 True."""
    if used_len <= raw_len or residue_mask_real.sum() == 0:
        return
    start_idx = 1 if has_cls else 0
    is_padding_token[start_idx + raw_len: start_idx + used_len] = True


# --------------------------- ESM3 open ---------------------------

def load_esm3_model(device: str):
    """
    ESM3 open small v0
    返回 (model, tokenizers, device_used)
    """
    try:
        from esm.pretrained import ESM3_sm_open_v0, get_esm3_model_tokenizers
        if device in ("auto", "cuda"):
            device_used = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_used = "cpu"
        model = ESM3_sm_open_v0(device_used)
        model = model.float()  # ensure float32; newer ESM3 loads as bfloat16
        toks = get_esm3_model_tokenizers()
        return model, toks, device_used
    except Exception as e:
        raise RuntimeError(f"Failed to load ESM3 model. pip install esm. Error: {e}")


@torch.no_grad()
def forward_chain_all_tokens_esm3(model, tokenizers, seq_used: str, device_used: str):
    """
    返回：
      tokens_ids np.int64[T]
      tokens_str np.object_[T]
      embedding  np.float32[T,D]
      is_special/is_residue/is_padding/residue_mask_real/cls_idx/eos_idx
    """
    tokens = tokenizers.sequence.encode(seq_used)  # 含 <cls>/<eos>
    T = len(tokens)

    vocab = tokenizers.sequence.get_vocab()
    id2tok = {v: k for k, v in vocab.items()}
    tokens_str = np.array([id2tok.get(tid, f"<UNK_ID_{tid}>") for tid in tokens], dtype=object)

    is_special = np.array([tok.startswith("<") and tok.endswith(">") for tok in tokens_str], dtype=bool)
    is_residue = ~is_special

    L_used = len(seq_used)
    cls_index = 0
    eos_index = T - 1

    residue_mask_real = np.zeros(T, dtype=bool)
    if L_used > 0:
        residue_mask_real[1:1+L_used] = True

    is_padding = np.zeros(T, dtype=bool)

    t = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0)
    if device_used == "cuda" and torch.cuda.is_available():
        t = t.cuda()
    out = model(sequence_tokens=t)
    emb = out.embeddings[0]  # [T,D]（个别版本可能 [D,T]，后续 ensure_TD 统一）
    if emb.is_cuda:
        emb = emb.cpu()
    embedding = emb.float().numpy()

    # 统一 (T,D)
    embedding = ensure_TD(embedding, T_expected=T)

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


# --------------------------- ESMC ---------------------------

def _patch_esm_checkpoint_loader():
    """
    esm==3.3.0's ESMC_*_202412() builds its model via init_empty_weights() (meta device),
    then calls huggingface_hub.load_torch_model(model, data_root(...)) to fill it in. Two
    independent bugs make this always fail here, regardless of local setup:

    1. Path discovery: for esmc_600m/esmc_300m, the HF snapshot nests the real weights at
       data/weights/<ckpt>.pth, but load_torch_model's directory-glob only looks directly in
       the given dir for *.safetensors/*.bin (no recursion) -> "does not contain a valid
       checkpoint". esmc_6b instead ships a standard top-level sharded-safetensors layout
       (model-0000X-of-N.safetensors + model.safetensors.index.json), so this part works.
    2. assign=True: both huggingface_hub's single-file and its _load_sharded_checkpoint path
       call model.load_state_dict(state_dict, strict=...) WITHOUT assign=True, and neither
       exposes a way to pass it through. Since the model's params are on the meta device,
       this silently no-ops (params stay meta) instead of raising -- the failure only
       surfaces later as "Cannot copy out of meta tensor" when .to(device) is called.

    Fix: replace esm.pretrained's bound name with a loader that handles both the nested
    single-.pth case and the sharded-safetensors-index case directly, always with
    assign=True.
    """
    import esm.pretrained as _esm_pretrained

    if getattr(_esm_pretrained, "_patched_for_meta_assign", False):
        return

    def _patched_load_torch_model(model, checkpoint_path, **kwargs):
        strict = kwargs.get("strict", False)
        p = Path(checkpoint_path)

        if p.is_file():
            sd = torch.load(p, map_location="cpu", weights_only=True)
            return model.load_state_dict(sd, strict=strict, assign=True)

        # Directory: single nested .pth (esmc_600m/esmc_300m layout)
        pth_candidates = list(p.glob("**/*.pth"))
        if len(pth_candidates) == 1:
            sd = torch.load(pth_candidates[0], map_location="cpu", weights_only=True)
            return model.load_state_dict(sd, strict=strict, assign=True)

        # Directory: sharded safetensors with a top-level index (esmc_6b layout).
        # Checkpoint keys are prefixed (e.g. "esmc.embed.weight") and use an older module
        # name ("lm_head" -> the model's "sequence_head") vs. the live model's own names;
        # verified this two-rule remap gives a full 1:1 match against model param names.
        index_path = p / "model.safetensors.index.json"
        if index_path.is_file():
            from safetensors.torch import load_file

            def _remap_key(k: str) -> str:
                if k.startswith("esmc."):
                    k = k[len("esmc."):]
                if k.startswith("lm_head."):
                    k = "sequence_head." + k[len("lm_head."):]
                return k

            with open(index_path) as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            for shard_file in shard_files:
                shard_sd = load_file(str(p / shard_file))
                shard_sd = {_remap_key(k): v for k, v in shard_sd.items()}
                model.load_state_dict(shard_sd, strict=False, assign=True)
            remaining_meta = [n for n, prm in model.named_parameters() if prm.is_meta]
            if remaining_meta:
                raise RuntimeError(
                    f"_patched_load_torch_model: {len(remaining_meta)} params still on meta "
                    f"device after loading all shards, e.g. {remaining_meta[:5]}"
                )
            return None

        raise ValueError(
            f"_patched_load_torch_model: couldn't find a recognized checkpoint layout under {p} "
            f"(looked for a single nested .pth and for model.safetensors.index.json)"
        )

    _esm_pretrained.load_torch_model = _patched_load_torch_model
    _esm_pretrained._patched_for_meta_assign = True


def load_esmc_model(device: str, ckpt_name: str = "esmc_600m", cuda_id: int = 0):
    """
    ESMC: esmc_300m / esmc_600m / esmc_6b (视可用性)
    返回 (model, device_used)
    """
    try:
        _patch_esm_checkpoint_loader()
        from esm.models.esmc import ESMC
    except Exception as e:
        raise RuntimeError(f"Failed to import ESMC. pip install esm (>=3.x). Error: {e}")

    if device in ("auto", "cuda"):
        device_used = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_used = "cpu"

    if device_used == "cuda":
        torch.cuda.set_device(cuda_id)
        if "6b" in ckpt_name.lower():
            # ESMC.from_pretrained(device=None) auto-detects cuda and moves the model
            # there in fp32 *before* its own later bf16 cast, so the OOM (6B params,
            # ~24GB fp32, no headroom on a 24GB GPU) happens inside that call, before we
            # get control back. Force it to build on CPU, cast to bf16 there (halves
            # footprint), then move the already-halved tensors to GPU.
            model = ESMC.from_pretrained(ckpt_name, device=torch.device("cpu"))
            model = model.to(torch.bfloat16)
        else:
            model = ESMC.from_pretrained(ckpt_name)
        model = model.to(f"cuda:{cuda_id}")
        print(f"Using cuda:{cuda_id} -> {torch.cuda.get_device_name(cuda_id)}")
    else:
        model = ESMC.from_pretrained(ckpt_name).to("cpu")

    return model, device_used


@torch.no_grad()
def forward_chain_all_tokens_esmc(model, seq_used: str, device_used: str):
    """
    使用 ESMC SDK 前向，拿到 per-token embeddings。
    ESMC 端通常没有公开 tokenizer id，这里用 -1 占位。
    """
    try:
        from esm.sdk.api import ESMProtein, LogitsConfig
    except Exception as e:
        raise RuntimeError(f"Failed to import ESM SDK API. Error: {e}")

    protein = ESMProtein(sequence=seq_used)
    pt = model.encode(protein)
    out = model.logits(pt, LogitsConfig(sequence=True, return_embeddings=True))
    emb = out.embeddings  # torch.Tensor [T,D] or [1,T,D] (ESM >= 3.1.4 returns 3D)
    if emb.is_cuda:
        emb = emb.cpu()
    embedding = emb.float().numpy()
    # ESM 3.1.4+ returns (1, T, D) — squeeze batch dim
    if embedding.ndim == 3 and embedding.shape[0] == 1:
        embedding = embedding[0]
    L = len(seq_used)
    T = embedding.shape[0] if embedding.ndim == 2 else None  # ensure_TD 统一

    # 两种可能：有 specials（T=L+2）/无 specials（T=L）
    # 先归一化形状
    # 推断 T_expected：若有 specials，T_expected=L+2；否则 L
    # 经验：ESMC 多数返回含 specials（与 ESM3 对齐），但做自适应更稳
    def try_make(Texp):
        try:
            return ensure_TD(embedding, T_expected=Texp)
        except Exception:
            return None

    embedding_TD = try_make(L + 2)
    has_specials = True
    if embedding_TD is None:
        embedding_TD = try_make(L)
        has_specials = False
    if embedding_TD is None:
        # 兜底：直接报错
        raise ValueError(f"Unexpected ESMC embedding shape {embedding.shape} for sequence length {L}")

    embedding = embedding_TD
    T = embedding.shape[0]

    if has_specials:  # <cls> + residues + <eos>
        tokens_str = np.array(["<cls>"] + list(seq_used) + ["<eos>"], dtype=object)
        is_special = np.zeros(T, dtype=bool); is_special[0] = True; is_special[-1] = True
        is_residue = ~is_special
        residue_mask_real = np.zeros(T, dtype=bool); residue_mask_real[1:1+L] = True
        cls_index, eos_index = 0, T - 1
    else:  # 纯 residues
        tokens_str = np.array(list(seq_used), dtype=object)
        is_special = np.zeros(T, dtype=bool)
        is_residue = ~is_special
        residue_mask_real = np.zeros(T, dtype=bool); residue_mask_real[:L] = True
        cls_index, eos_index = -1, -1

    tokens_ids = np.full(T, -1, dtype=np.int64)  # 无法提供真实 token id
    is_padding = np.zeros(T, dtype=bool)

    return (
        tokens_ids,
        tokens_str,
        embedding.astype(np.float32),
        is_special,
        is_residue,
        is_padding,
        residue_mask_real,
        int(cls_index),
        int(eos_index),
    )


# --------------------------- index / checkpoint ---------------------------

def index_path(out_dir: Path) -> Path:
    return out_dir / "batch_index.json"


def load_index(out_dir: Path) -> Dict[str, Any]:
    p = index_path(out_dir)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"meta": {}, "params": {}, "processed": {}, "errors": {}, "skipped": []}


def save_index(out_dir: Path, idx: Dict[str, Any]):
    p = index_path(out_dir)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


def save_checkpoint(out_dir: Path, idx: Dict[str, Any], processed_count: int):
    base = index_path(out_dir).with_suffix("")
    ckpt = Path(str(base) + f".checkpoint_{processed_count}.json")
    with open(ckpt, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)


# --------------------------- per-key processing ---------------------------

def process_one_key(
    key: str,
    item: Dict[str, Any],
    out_dir: Path,
    model_type: str,
    model,
    tokenizers_or_None,
    device_used: str,
    order: str,
    pad_protein: int | None,
    pad_peptide: int | None,
    dtype: str,
    print_shapes: bool = False,
) -> Dict[str, Any]:
    receptor_seq = item["Sequences"]["receptor_seq"]
    ligand_seq   = item["Sequences"]["ligand_seq"]

    chains = item.get("Meta", {}).get("chains", {})
    prot_id = chains.get("receptor", "PROTEIN")
    pep_id  = chains.get("ligand",   "PEPTIDE")

    prot_raw = clean_seq(receptor_seq)
    pep_raw  = clean_seq(ligand_seq)
    prot_used, prot_real_mask = pad_seq(prot_raw, pad_protein)
    pep_used,  pep_real_mask  = pad_seq(pep_raw,  pad_peptide)

    # forward
    if model_type == "esm3":
        (
            p_tokens, p_tokens_str, p_emb, p_is_special, p_is_residue, p_is_padding, p_residue_mask_real, p_cls_idx, p_eos_idx
        ) = forward_chain_all_tokens_esm3(model, tokenizers_or_None, prot_used, device_used)
        (
            l_tokens, l_tokens_str, l_emb, l_is_special, l_is_residue, l_is_padding, l_residue_mask_real, l_cls_idx, l_eos_idx
        ) = forward_chain_all_tokens_esm3(model, tokenizers_or_None, pep_used, device_used)
        token_ids_meta = {
            "cls": tokenizers_or_None.sequence.get_vocab().get("<cls>"),
            "eos": tokenizers_or_None.sequence.get_vocab().get("<eos>"),
            "unk": tokenizers_or_None.sequence.get_vocab().get("<unk>"),
            "x":   tokenizers_or_None.sequence.get_vocab().get("X"),
        }
    else:  # esmc
        (
            p_tokens, p_tokens_str, p_emb, p_is_special, p_is_residue, p_is_padding, p_residue_mask_real, p_cls_idx, p_eos_idx
        ) = forward_chain_all_tokens_esmc(model, prot_used, device_used)
        (
            l_tokens, l_tokens_str, l_emb, l_is_special, l_is_residue, l_is_padding, l_residue_mask_real, l_cls_idx, l_eos_idx
        ) = forward_chain_all_tokens_esmc(model, pep_used, device_used)
        token_ids_meta = {"available": False, "note": "ESMC SDK 未暴露 tokenizer id；tokens_ids=-1"}

    # 形状自检（用于诊断）
    if print_shapes:
        print(f"[{key}] protein emb {p_emb.shape}, tokens {len(p_tokens_str)} | peptide emb {l_emb.shape}, tokens {len(l_tokens_str)}")

    # Fix residue_mask_real: ESM3 forward sets it to the full padded length;
    # override to mark only the true (non-X-padded) residue positions.
    p_has_cls = (p_cls_idx == 0)
    l_has_cls = (l_cls_idx == 0)
    p_residue_mask_real = np.zeros(len(p_emb), dtype=bool)
    p_residue_mask_real[1 if p_has_cls else 0: (1 if p_has_cls else 0) + len(prot_raw)] = True
    l_residue_mask_real = np.zeros(len(l_emb), dtype=bool)
    l_residue_mask_real[1 if l_has_cls else 0: (1 if l_has_cls else 0) + len(pep_raw)] = True

    # 标记 padding
    mark_padding_on_token_axis(p_is_padding, p_residue_mask_real, len(prot_raw), len(prot_used), p_has_cls)
    mark_padding_on_token_axis(l_is_padding, l_residue_mask_real, len(pep_raw),  len(pep_used),  l_has_cls)

    # complex stacking
    if order == "protein_peptide":
        complex_emb = np.vstack([p_emb, l_emb])
        complex_is_special = np.r_[p_is_special, l_is_special]
        complex_is_residue = np.r_[p_is_residue, l_is_residue]
        complex_is_padding = np.r_[p_is_padding, l_is_padding]
        chain_offsets_tokens = [
            {"chain_id": prot_id, "start": 0,          "end": len(p_emb)},
            {"chain_id": pep_id,  "start": len(p_emb), "end": len(p_emb) + len(l_emb)},
        ]
        chain_offsets_residues = [
            {"chain_id": prot_id, "start": 1, "end": len(p_emb)-1},
            {"chain_id": pep_id,  "start": len(p_emb)+1, "end": len(p_emb)+len(l_emb)-1},
        ]
        chain_ids = np.array([prot_id, pep_id], dtype=object)
    else:
        complex_emb = np.vstack([l_emb, p_emb])
        complex_is_special = np.r_[l_is_special, p_is_special]
        complex_is_residue = np.r_[l_is_residue, p_is_residue]
        complex_is_padding = np.r_[l_is_padding, p_is_padding]
        chain_offsets_tokens = [
            {"chain_id": pep_id,  "start": 0,          "end": len(l_emb)},
            {"chain_id": prot_id, "start": len(l_emb), "end": len(l_emb) + len(p_emb)},
        ]
        chain_offsets_residues = [
            {"chain_id": pep_id,  "start": 1, "end": len(l_emb)-1},
            {"chain_id": prot_id, "start": len(l_emb)+1, "end": len(l_emb)+len(p_emb)-1},
        ]
        chain_ids = np.array([pep_id, prot_id], dtype=object)

    # dtype cast
    if dtype == "float16":
        complex_emb = complex_emb.astype(np.float16)
        p_emb = p_emb.astype(np.float16)
        l_emb = l_emb.astype(np.float16)
    else:
        complex_emb = complex_emb.astype(np.float32)
        p_emb = p_emb.astype(np.float32)
        l_emb = l_emb.astype(np.float32)

    # metadata
    meta = {
        "family": "ESM3-open" if model_type == "esm3" else "ESMC",
        "model": "ESM3_sm_open_v0" if model_type == "esm3" else "custom_esmc_ckpt",
        "library": "esm",
        "device": device_used,
        "dim": int(p_emb.shape[1]),
        "order": order,
        "source_key": key,
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
            "residue_window_per_chain": "[1, T-1) if specials else [0, T)",
        },
    }

    # save
    npz_path = out_dir / f"{key}_{'esm3' if model_type=='esm3' else 'esmc'}_alltokens.npz"
    np.savez_compressed(
        npz_path,
        # protein
        protein_tokens=p_tokens.astype(np.int64),
        protein_tokens_str=p_tokens_str,
        protein_embedding=p_emb,
        protein_is_special=p_is_special.astype(np.bool_),
        protein_is_residue=p_is_residue.astype(np.bool_),
        protein_is_padding=p_is_padding.astype(np.bool_),
        protein_residue_mask_real=p_residue_mask_real.astype(np.bool_),
        protein_cls_index=np.array([0 if p_has_cls else -1], dtype=np.int32),
        protein_eos_index=np.array([len(p_emb)-1 if p_has_cls else -1], dtype=np.int32),
        protein_id=np.array([prot_id], dtype=object),
        protein_seq_raw=np.array([prot_raw], dtype=object),
        protein_seq_used=np.array([prot_used], dtype=object),

        # peptide
        peptide_tokens=l_tokens.astype(np.int64),
        peptide_tokens_str=l_tokens_str,
        peptide_embedding=l_emb,
        peptide_is_special=l_is_special.astype(np.bool_),
        peptide_is_residue=l_is_residue.astype(np.bool_),
        peptide_is_padding=l_is_padding.astype(np.bool_),
        peptide_residue_mask_real=l_residue_mask_real.astype(np.bool_),
        peptide_cls_index=np.array([0 if l_has_cls else -1], dtype=np.int32),
        peptide_eos_index=np.array([len(l_emb)-1 if l_has_cls else -1], dtype=np.int32),
        peptide_id=np.array([pep_id], dtype=object),
        peptide_seq_raw=np.array([pep_raw], dtype=object),
        peptide_seq_used=np.array([pep_used], dtype=object),

        # complex
        complex_embedding=complex_emb,
        complex_is_special=complex_is_special.astype(np.bool_),
        complex_is_residue=complex_is_residue.astype(np.bool_),
        complex_is_padding=complex_is_padding.astype(np.bool_),
        chain_ids=chain_ids,
        chain_offsets_tokens=json.dumps(chain_offsets_tokens, ensure_ascii=False),
        chain_offsets_residues=json.dumps(chain_offsets_residues, ensure_ascii=False),
        order=np.array([order], dtype=object),

        # meta
        metadata=json.dumps(meta, ensure_ascii=False),
    )

    return {
        "npz_path": str(npz_path),
        "protein_T": int(p_emb.shape[0]),
        "peptide_T": int(l_emb.shape[0]),
        "complex_T": int(complex_emb.shape[0]),
        "dim": int(p_emb.shape[1]),
        "order": order,
        "device": device_used
    }


# --------------------------- multi-GPU worker ---------------------------

def worker_process(
    worker_id: int,
    gpu_id: int,
    keys_shard: List[str],
    all_data: Dict[str, Any],
    args,
    out_dir: Path,
):
    """Single-GPU worker process. Handles its assigned shard of keys."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if args.model == "esm3":
        model, tokenizers, device_used = load_esm3_model("cuda")
        tok_or_none = tokenizers
        print(f"[GPU{gpu_id}] ESM3-open ready on {device_used}")
    else:
        model, device_used = load_esmc_model("cuda", ckpt_name=args.esmc_ckpt, cuda_id=0)
        tok_or_none = None
        print(f"[GPU{gpu_id}] ESMC ready on {device_used}")

    worker_idx_path = out_dir / f"batch_index_gpu{worker_id}.json"
    worker_idx: Dict[str, Any] = {"processed": {}, "errors": {}}

    suffix = "esm3" if args.model == "esm3" else "esmc"
    todo = [k for k in keys_shard if not (out_dir / f"{k}_{suffix}_alltokens.npz").exists()]

    with tqdm(total=len(todo), desc=f"GPU{gpu_id}", position=worker_id, leave=True) as pbar:
        for k in todo:
            try:
                res = process_one_key(
                    key=k,
                    item=all_data[k],
                    out_dir=out_dir,
                    model_type=args.model,
                    model=model,
                    tokenizers_or_None=tok_or_none,
                    device_used=device_used,
                    order=args.order,
                    pad_protein=args.pad_protein,
                    pad_peptide=args.pad_peptide,
                    dtype=args.dtype,
                )
                worker_idx["processed"][k] = res
            except Exception as e:
                worker_idx["errors"][k] = {"error": f"{type(e).__name__}: {e}",
                                           "traceback": traceback.format_exc(limit=3)}
            finally:
                pbar.update(1)
            n_done = len(worker_idx["processed"])
            if args.checkpoint_every and n_done > 0 and n_done % args.checkpoint_every == 0:
                with open(worker_idx_path, "w", encoding="utf-8") as f:
                    json.dump(worker_idx, f)

    with open(worker_idx_path, "w", encoding="utf-8") as f:
        json.dump(worker_idx, f)
    print(f"[GPU{gpu_id}] Done. processed={len(worker_idx['processed'])}, errors={len(worker_idx['errors'])}")


# --------------------------- main ---------------------------

def parse_only_keys(s: str) -> List[str]:
    p = Path(s)
    if p.exists() and p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Batch-generate embeddings (ESM3-open or ESMC) with ALL tokens kept")
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--output_dir", required=True)

    # 模型选择
    ap.add_argument("--model", choices=["esm3", "esmc"], default="esm3", help="Use ESM3-open or ESMC")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--cuda-id", type=int, default=0, help="CUDA device index when --device=cuda or auto & cuda is available")
    ap.add_argument("--gpu-ids", type=int, nargs="+", default=None,
                    help="Multi-GPU: specify GPU IDs e.g. --gpu-ids 0 1 2 3. Each GPU runs one process.")
    ap.add_argument("--esmc-ckpt", type=str, default="esmc_600m", help="ESMC checkpoint name")

    # 编码/前处理
    ap.add_argument("--order", choices=["protein_peptide", "peptide_protein"], default="peptide_protein")
    ap.add_argument("--pad-protein", type=int, default=None)
    ap.add_argument("--pad-peptide", type=int, default=None)
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")

    # 批处理控制
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--recheck-npz", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-keys", type=int, default=None)

    # 过滤
    ap.add_argument("--only-keys", type=str, default=None)
    ap.add_argument("--include-pattern", type=str, default=None)
    ap.add_argument("--exclude-pattern", type=str, default=None)

    # debug
    ap.add_argument("--print-shapes", action="store_true", help="Print per-key embedding shapes for debugging")

    args = ap.parse_args()
    if args.no_resume:
        args.resume = False

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input_json, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    keys = sorted(list(all_data.keys()))

    if args.only_keys:
        target = set(parse_only_keys(args.only_keys))
        keys = [k for k in keys if k in target]
    if args.include_pattern:
        inc = re.compile(args.include_pattern)
        keys = [k for k in keys if inc.search(k)]
    if args.exclude_pattern:
        exc = re.compile(args.exclude_pattern)
        keys = [k for k in keys if not exc.search(k)]
    if args.max_keys is not None:
        keys = keys[:args.max_keys]

    idx = load_index(out_dir) if args.resume else {"meta": {}, "params": {}, "processed": {}, "errors": {}, "skipped": []}
    if not idx.get("meta"):
        idx["meta"] = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    idx["params"] = {
        "model": args.model,
        "esmc_ckpt": args.esmc_ckpt,
        "order": args.order,
        "pad_protein": args.pad_protein,
        "pad_peptide": args.pad_peptide,
        "dtype": args.dtype,
        "device": args.device,
        "cuda_id": args.cuda_id,
        "input_json": os.path.abspath(args.input_json),
    }

    processed_dict = idx.get("processed", {})
    errors_dict = idx.get("errors", {})
    skipped_list = idx.get("skipped", [])

    todo: List[str] = []
    for k in keys:
        suffix = "esm3" if args.model == "esm3" else "esmc"
        npz_p = out_dir / f"{k}_{suffix}_alltokens.npz"
        if k in processed_dict:
            if args.recheck_npz and not npz_p.exists():
                todo.append(k)
            else:
                skipped_list.append(k)
        else:
            if npz_p.exists() and not args.recheck_npz:
                skipped_list.append(k)
            else:
                todo.append(k)

    if args.dry_run:
        print(f"[DRY RUN] will process {len(todo)} keys, skip {len(skipped_list)}, errors on record {len(errors_dict)}")
        for k in todo[:20]:
            print("  +", k)
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        return

    start = time.time()

    # ---- multi-GPU branch ----
    if args.gpu_ids and len(args.gpu_ids) >= 1:
        import multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # already set

        n = len(args.gpu_ids)
        shards = [todo[i::n] for i in range(n)]
        print(f"Multi-GPU mode: {n} GPUs {args.gpu_ids}, "
              f"{len(todo)} todo keys split ~{len(todo)//n} each")

        processes = []
        for i, (gpu_id, shard) in enumerate(zip(args.gpu_ids, shards)):
            p = mp.Process(
                target=worker_process,
                args=(i, gpu_id, shard, all_data, args, out_dir),
                daemon=False,
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # merge worker indexes back into main index
        for i in range(n):
            worker_idx_path = out_dir / f"batch_index_gpu{i}.json"
            if worker_idx_path.exists():
                with open(worker_idx_path, "r", encoding="utf-8") as f:
                    w = json.load(f)
                processed_dict.update(w.get("processed", {}))
                errors_dict.update(w.get("errors", {}))
                worker_idx_path.unlink()

        idx["processed"] = processed_dict
        idx["errors"] = errors_dict
        save_index(out_dir, idx)

        elapsed = time.time() - start
        print("=" * 80)
        print("BATCH COMPLETE (multi-GPU)")
        print(f"Elapsed: {elapsed/60:.1f} min | GPUs: {args.gpu_ids}")
        print(f"Total processed in index: {len(processed_dict)} | errors: {len(errors_dict)} | skipped: {len(skipped_list)}")
        print(f"Index: {index_path(out_dir)}")
        print("=" * 80)
        return

    # ---- single-GPU branch (original) ----
    if args.model == "esm3":
        model, tokenizers, device_used = load_esm3_model(args.device)
        tok_or_none = tokenizers
        print(f"✅ ESM3-open ready on {device_used}")
    else:
        model, device_used = load_esmc_model(args.device, ckpt_name=args.esmc_ckpt, cuda_id=args.cuda_id)
        tok_or_none = None
        print(f"✅ ESMC({args.esmc_ckpt}) ready on {device_used}")

    done = 0
    total = len(todo)

    with tqdm(total=total, desc="Processing keys") as pbar:
        for k in todo:
            try:
                res = process_one_key(
                    key=k,
                    item=all_data[k],
                    out_dir=out_dir,
                    model_type=args.model,
                    model=model,
                    tokenizers_or_None=tok_or_none,
                    device_used=device_used,
                    order=args.order,
                    pad_protein=args.pad_protein,
                    pad_peptide=args.pad_peptide,
                    dtype=args.dtype,
                    print_shapes=args.print_shapes,
                )
                processed_dict[k] = res
                save_index(out_dir, idx)
                done += 1
                if args.checkpoint_every and done % args.checkpoint_every == 0:
                    save_checkpoint(out_dir, idx, done)
            except KeyboardInterrupt:
                print("\nInterrupted by user, saving progress...")
                save_index(out_dir, idx)
                if args.checkpoint_every and done % args.checkpoint_every != 0:
                    save_checkpoint(out_dir, idx, done)
                raise
            except Exception as e:
                errors_dict[k] = {"error": f"{type(e).__name__}: {e}",
                                  "traceback": traceback.format_exc(limit=3)}
                save_index(out_dir, idx)
            finally:
                pbar.update(1)

    elapsed = time.time() - start
    print("=" * 80)
    print("🎉 BATCH COMPLETE")
    print(f"Processed: {done} / {total} keys in {elapsed/60:.1f} min")
    print(f"Total processed in index: {len(processed_dict)} | errors: {len(errors_dict)} | skipped: {len(skipped_list)}")
    print(f"Index: {index_path(out_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
