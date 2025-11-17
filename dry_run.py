#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, torch
from torch.utils.data import DataLoader
import numpy as np

# 你的工程结构：contrasive_learning/
from data.data_loader import (
    ProtPepFullTokenDataset,
    collate_full_tokens,
    read_db_index,
    read_split_list,
)
from model.models import PairwiseModel, PairModelConfig, PairwiseCriterion


# --------------------- 打印辅助 ---------------------

def _tensor_stats(x: torch.Tensor) -> str:
    x = x.detach()
    if x.numel() == 0:
        return "empty"
    return f"min={float(x.min()):.4f}, mean={float(x.mean()):.4f}, max={float(x.max()):.4f}"

def _first_last_indices(mask_1d: torch.Tensor, k: int = 5):
    """
    返回 mask=True 的前 k 个 & 后 k 个索引（cpu numpy）
    """
    idx = torch.nonzero(mask_1d, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    head = idx[:k].cpu().numpy()
    tail = idx[-k:].cpu().numpy()
    return head, tail

def _clip_preview_indices(valid_idx: torch.Tensor, preview: int = 5):
    """
    给定有效位置索引，返回 head/tail 的切片索引（torch 1D）。
    """
    n = valid_idx.numel()
    if n == 0:
        return valid_idx
    head = valid_idx[:min(preview, n)]
    if n > preview:
        tail = valid_idx[max(0, n - preview):]
        # 合并去重，保持顺序：head + (中间可能重合) + tail
        merged = torch.unique(torch.cat([head, tail]), sorted=True)
        return merged
    return head


# --------------------- 原始脚本函数 ---------------------

def adapt_for_model(batch, device):
    """把 data_loader 的键转成模型期望的键名，并搬到 device。"""
    return {
        "prot_emb": batch["protein_emb"].to(device),  # [B,Lp,D]
        "pep_emb":  batch["peptide_emb"].to(device),  # [B,Ll,D]
        "prot_mask": batch["protein_masks"]["valid_real_residue"].to(device),  # [B,Lp]
        "pep_mask":  batch["peptide_masks"]["valid_real_residue"].to(device),  # [B,Ll]
        # 如果你有监督标签，在这里塞： "labels_contact": ..., "labels_distance": ...
    }

def print_batch_shapes(batch, title="[DataLoader batch]"):
    """打印 DataLoader 原始 batch 的形状信息"""
    print(f"\n{title}")
    P, L = batch["protein_emb"], batch["peptide_emb"]
    pm, lm = batch["protein_masks"], batch["peptide_masks"]
    print("  protein_emb:", tuple(P.shape), "| dtype:", P.dtype, "| device:", P.device)
    print("  peptide_emb:", tuple(L.shape), "| dtype:", L.dtype, "| device:", L.device)

    def _m(name, m):
        print(f"  {name}.is_special:        {tuple(m['is_special'].shape)}")
        print(f"  {name}.is_residue_window: {tuple(m['is_residue_window'].shape)}")
        print(f"  {name}.is_padding:        {tuple(m['is_padding'].shape)}")
        print(f"  {name}.valid_real_residue:{tuple(m['valid_real_residue'].shape)}")

    _m("protein_masks", pm)
    _m("peptide_masks", lm)

def print_model_inputs(mb, title="[Model inputs]"):
    """打印喂给模型之前的张量形状"""
    print(f"\n{title}")
    print("  prot_emb:", tuple(mb["prot_emb"].shape), "| dtype:", mb["prot_emb"].dtype, "| device:", mb["prot_emb"].device)
    print("  pep_emb: ", tuple(mb["pep_emb"].shape),  "| dtype:", mb["pep_emb"].dtype,  "| device:", mb["pep_emb"].device)
    print("  prot_mask:", tuple(mb["prot_mask"].shape))
    print("  pep_mask: ", tuple(mb["pep_mask"].shape))

def print_head_internals(model, mb, max_chunk_for_mlp: int = 1):
    """
    可选：打印 head 内部中间张量的 shape（区分 bilinear 与 mlp）。
    不改变模型，只做一次 no_grad 前向的中间计算。
    """
    head = model.head
    P, L = mb["prot_emb"], mb["pep_emb"]           # [B,Lp,D] / [B,Ll,D]
    B, Lp, D = P.shape
    Ll = L.size(1)
    print("\n[Head internals]")
    if head.__class__.__name__.lower().startswith("pairbilinear"):
        with torch.no_grad():
            Pp = head.proj_p(P)                    # [B,Lp,d]
            Ll_ = head.proj_l(L)                   # [B,Ll,d]
            print("  proj_p(P):", tuple(Pp.shape))
            print("  proj_l(L):", tuple(Ll_.shape))

            bilinear = torch.bmm(Pp, Ll_.transpose(1, 2))    # [B,Lp,Ll]
            print("  bilinear(Pp @ Lp^T):", tuple(bilinear.shape))

            up = head.u(P).expand(-1, -1, Ll)                 # [B,Lp,Ll]
            vl = head.v(L).transpose(1, 2).expand(-1, Lp, -1) # [B,Lp,Ll]
            print("  u(P) expand:", tuple(up.shape))
            print("  v(L) expand:", tuple(vl.shape))
            if head.bias is not None:
                print("  bias: [1]")
    else:
        from model.models import PairMLP  # 仅为了类型判断
        if isinstance(head, PairMLP):
            j1 = min(max_chunk_for_mlp, Ll)
            with torch.no_grad():
                P4 = P[:, :, None, :]             # [B,Lp,1,D]
                L4 = L[:, :j1, :][:, None, :, :]  # [B,1,j1,D]
                PP = P4.expand(-1, -1, j1, -1)    # [B,Lp,j1,D]
                LL = L4.expand(-1, Lp, -1, -1)    # [B,Lp,j1,D]
                feats = torch.cat([PP, LL, torch.abs(PP - LL), PP * LL], dim=-1)  # [B,Lp,j1,4D]
                print("  concat pair feats (4D):", tuple(feats.shape))
                test_out = head.mlp(feats.reshape(B * Lp * j1, -1)).reshape(B, Lp, j1, 1).squeeze(-1)
                print("  MLP head out (small chunk):", tuple(test_out.shape))


# --------------------- 新增：打印“具体内容”的预览 ---------------------

def preview_masks_and_embeddings(batch, preview: int = 5, peek_dim: int = 8, print_values: bool = False):
    """
    针对 batch[0] 打印：
      - 有效残基位置（valid_real_residue）的前/后若干索引
      - 对应位置的 embedding（前 peek_dim 维）
    """
    print("\n[Preview: masks & embeddings on batch[0]]")
    P = batch["protein_emb"][0]                    # (Lp, D)
    L = batch["peptide_emb"][0]                    # (Ll, D)
    pm = batch["protein_masks"]["valid_real_residue"][0]  # (Lp,)
    lm = batch["peptide_masks"]["valid_real_residue"][0]  # (Ll,)

    # --- mask 概览 ---
    ph, pt = _first_last_indices(pm, preview)
    lh, lt = _first_last_indices(lm, preview)
    print(f"  protein valid count: {int(pm.sum())} / {pm.numel()} | head idx: {ph.tolist()} | tail idx: {pt.tolist()}")
    print(f"  peptide valid count: {int(lm.sum())} / {lm.numel()} | head idx: {lh.tolist()} | tail idx: {lt.tolist()}")

    # 取有效索引做内容预览（前N+后N）
    p_valid_idx = torch.nonzero(pm, as_tuple=False).squeeze(-1)
    l_valid_idx = torch.nonzero(lm, as_tuple=False).squeeze(-1)
    p_sel = _clip_preview_indices(p_valid_idx, preview)
    l_sel = _clip_preview_indices(l_valid_idx, preview)

    def _fmt_block(name, X, sel):
        block = X.index_select(0, sel)[:, :peek_dim].detach().cpu().float().numpy()  # (K, peek_dim)
        if print_values:
            np.set_printoptions(precision=4, suppress=True)
            print(f"\n  {name} emb[ first/last {len(sel)} valid tokens, first {peek_dim} dims ]:")
            print(block)
        else:
            # 只给统计
            print(f"\n  {name} emb stats on {len(sel)} tokens (first {peek_dim} dims slice):")
            print("   " + _tensor_stats(torch.from_numpy(block)))

    if p_sel.numel() > 0:
        _fmt_block("protein", P, p_sel)
    if l_sel.numel() > 0:
        _fmt_block("peptide", L, l_sel)


def preview_scores(outputs: dict, mb: dict, preview: int = 5):
    """
    打印 scores 的小块和 top-k（prot->pep）。
    """
    S = outputs["scores"].detach().cpu()  # (B, Lp, Ll)
    pm = mb["prot_mask"].detach().cpu()   # (B, Lp)
    lm = mb["pep_mask"].detach().cpu()    # (B, Ll)

    # 只看 batch[0]
    S0, pm0, lm0 = S[0], pm[0], lm[0]

    # 取有效窗口的左上角 preview×preview
    p_idx = torch.nonzero(pm0, as_tuple=False).squeeze(-1)
    l_idx = torch.nonzero(lm0, as_tuple=False).squeeze(-1)
    if p_idx.numel() == 0 or l_idx.numel() == 0:
        print("\n[Scores preview] no valid residues to preview.")
        return

    p_sel = p_idx[:min(preview, p_idx.numel())]
    l_sel = l_idx[:min(preview, l_idx.numel())]
    sub = S0.index_select(0, p_sel).index_select(1, l_sel).numpy()

    print("\n[Scores preview] (protein valid head × peptide valid head)")
    with np.printoptions(precision=3, suppress=True):
        print(sub)

    # 按行 top-3（prot->pep），在有效列范围
    topk = min(3, l_idx.numel())
    row_top = []
    S0_eff = S0[:, l_idx]  # 只在有效肽列
    vals, idxs = torch.topk(S0_eff, k=topk, dim=1)  # (Lp, k)
    # 只打印前 preview 行（也在有效 prot 行范围内）
    for r in p_sel.tolist():
        top_cols = l_idx[idxs[r]].tolist()
        row_top.append({"prot_row": r, "pep_cols": top_cols})
    print("\n[Top-k matches per protein row (first few)]")
    for item in row_top:
        print(f"  row {item['prot_row']}: cols {item['pep_cols']}")


# --------------------- 主程序 ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="data/esm_npz/batch_index.json")
    ap.add_argument("--splits-dir", required=True, help="data/splits")
    ap.add_argument("--split", choices=["train","val","test"], default="train")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    ap.add_argument("--print-internals", action="store_true", help="打印 head 内部中间 shape")
    # 新增：
    ap.add_argument("--preview", type=int, default=5, help="取多少个有效残基位置做数值预览（前N+后N合并）")
    ap.add_argument("--peek-dim", type=int, default=8, help="每个位置打印 embedding 的前 K 个维度")
    ap.add_argument("--print-values", action="store_true", help="打印具体浮点数（否则只统计）")
    args = ap.parse_args()

    device = "cuda" if (args.device=="auto" and torch.cuda.is_available()) else (args.device if args.device!="auto" else "cpu")

    # 1) DataLoader
    key2npz = read_db_index(args.index)
    keys = read_split_list(args.splits_dir, args.split)
    keys = [k for k in keys if k in key2npz]
    ds = ProtPepFullTokenDataset(key2npz, keys, strict_fixed=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True, collate_fn=collate_full_tokens)

    # 2) Model
    cfg = PairModelConfig(d_model=1536, head="bilinear", d_proj=256, score_activation=None)
    model = PairwiseModel(cfg).to(device)
    crit  = PairwiseCriterion(use_bce_for_contact=False, m_pos=1.0, m_neg=0.0, pos_weight=3.0)

    # 3) 取一个 batch，逐步打印
    batch = next(iter(loader))
    print_batch_shapes(batch, "[DataLoader batch]")

    # 新增：更详细的数值预览
    preview_masks_and_embeddings(
        batch,
        preview=args.preview,
        peek_dim=args.peek_dim,
        print_values=args.print_values,
    )

    mb = adapt_for_model(batch, device)
    print_model_inputs(mb, "[Model inputs]")

    if args.print_internals:
        print_head_internals(model, mb, max_chunk_for_mlp=1)

    # 4) 前向 & 打印输出 shape + 预览 score 小块
    out = model(mb)
    print("\n[Model outputs]")
    print("  scores:", tuple(out["scores"].shape))
    print("  pair_mask:", tuple(out["pair_mask"].shape))

    preview_scores(out, mb, preview=args.preview)

    # 5) 伪造标签 -> 打印 loss 的标量
    B, Lp, Ll = out["scores"].shape
    fake_labels = torch.zeros((B, Lp, Ll), dtype=torch.long, device=device)
    fake_labels[:, :min(5, Lp), :min(5, Ll)] = 1
    loss_dict = crit(out, {**mb, "labels_contact": fake_labels})
    loss_dict_printable = {k: (float(v) if torch.is_tensor(v) else v) for k, v in loss_dict.items()}
    print("\n[Loss]")
    print(loss_dict_printable)

if __name__ == "__main__":
    main()
