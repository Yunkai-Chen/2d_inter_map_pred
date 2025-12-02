#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # 用于热图

# 项目内导入
from contrasive_learning.data.data_loader_inter import (
    read_db_index, read_split_list, ProtPepFullTokenDataset, collate_full_tokens,
    read_gt_index, strip_suffix_key
)
from contrasive_learning.model.models_final_inter import PairwiseModel, PairModelConfig, PairwiseCriterion


# ---------------------------
# Utils
# ---------------------------

def seed_all(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def adapt_batch_for_model(batch):
    """把 batch 变成模型输入 + 构造监督标签与 pair_mask。"""
    device = batch["protein_emb"].device
    prot_mask = batch["protein_masks"]["valid_real_residue"]  # [B,Tp]
    pep_mask  = batch["peptide_masks"]["valid_real_residue"]  # [B,Tl]
    model_batch = {
        "prot_emb": batch["protein_emb"],
        "pep_emb":  batch["peptide_emb"],
        "prot_mask": prot_mask,
        "pep_mask":  pep_mask,
    }
    labels = {}
    pair_mask = pep_mask[:, :, None] & prot_mask[:, None, :]  # [B,Tl,Tp]

    if "gt_map" in batch:
        gt_map  = batch["gt_map"].to(device)
        gt_mask = batch["gt_mask"].to(device).bool()

        B, Tl_pad = pep_mask.shape
        _, Tp_pad = prot_mask.shape
        gt_full      = torch.zeros((B, Tl_pad, Tp_pad), dtype=torch.float32, device=device)
        gt_full_mask = torch.zeros((B, Tl_pad, Tp_pad), dtype=torch.bool,    device=device)

        for b in range(B):
            row_idx = torch.nonzero(pep_mask[b],  as_tuple=False).squeeze(1)
            col_idx = torch.nonzero(prot_mask[b], as_tuple=False).squeeze(1)
            tl_gt = int(gt_mask[b].any(dim=1).sum().item())
            tp_gt = int(gt_mask[b].any(dim=0).sum().item())
            tl = min(tl_gt, row_idx.numel()); tp = min(tp_gt, col_idx.numel())
            if tl == 0 or tp == 0:
                continue
            gt_slice = gt_map[b, :tl, :tp].to(dtype=gt_full.dtype)
            rr = row_idx[:tl].unsqueeze(1); cc = col_idx[:tp].unsqueeze(0)
            gt_full[b].index_put_((rr, cc), gt_slice)
            gt_full_mask[b].index_put_((rr, cc), torch.ones_like(gt_slice, dtype=torch.bool))

        if batch.get("gt_kind", "contact") == "contact":
            labels["labels_contact"] = gt_full
        else:
            labels["labels_distance"] = gt_full

        pair_mask = pair_mask & gt_full_mask

    labels["pair_mask_override"] = pair_mask
    return model_batch, labels


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _format_head(mat: np.ndarray, H: int = 10, fmt: str = "{:6.3f}") -> str:
    """把矩阵的左上角 H×H 切片格式化成文本方阵。"""
    if mat.ndim != 2:
        return f"<ndim={mat.ndim} not supported>"
    r = min(H, mat.shape[0]); c = min(H, mat.shape[1])
    lines = []
    for i in range(r):
        row = " ".join(fmt.format(float(v)) for v in mat[i, :c])
        lines.append(row)
    if mat.shape[0] > r or mat.shape[1] > c:
        lines.append(f"... ({mat.shape[0]}x{mat.shape[1]} shown {r}x{c})")
    return "\n".join(lines)


def _topk_coords(prob: np.ndarray, mask: np.ndarray, k: int = 50):
    """在 mask=True 的区域里取 top-k 概率，返回 [(i,j,p)]（按 p 降序）。"""
    m = mask.astype(bool)
    if m.sum() == 0:
        return []
    flat = prob[m]
    if flat.size <= k:
        coords = list(zip(*np.where(m)))
        vals = prob[m]
        order = np.argsort(-vals)
        coords_sorted = [(int(coords[idx][0]), int(coords[idx][1]), float(vals[idx])) for idx in order]
        return coords_sorted
    else:
        idx_flat = np.argpartition(-flat, kth=min(k-1, flat.size-1))[:k]
        top_vals = flat[idx_flat]
        ii, jj = np.where(m)
        ii = ii[idx_flat]; jj = jj[idx_flat]
        order = np.argsort(-top_vals)
        return [(int(ii[o]), int(jj[o]), float(top_vals[o])) for o in order]


def _stats_txt(x: np.ndarray, m: np.ndarray | None = None, name: str = "pred"):
    if m is not None:
        xs = x[m.astype(bool)]
    else:
        xs = x.ravel()
    if xs.size == 0:
        return f"{name}: empty"
    return f"{name}: min/mean/max = {xs.min():.3f} / {xs.mean():.3f} / {xs.max():.3f}"


def _save_heatmap(mat: np.ndarray, path: Path, title: str = "", vmin: float | None = None, vmax: float | None = None):
    plt.figure()
    if (vmin is not None) or (vmax is not None):
        plt.imshow(mat, origin="upper", vmin=vmin, vmax=vmax)
    else:
        plt.imshow(mat, origin="upper")
    plt.title(title)
    plt.colorbar()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



# ---------------------------
# Eval
# ---------------------------

@torch.no_grad()
def run_eval(
    model, criterion, loader, device,
    out_dir: Path,
    save_prob: bool,
    save_binary: bool,
    bin_th: float,
    use_amp: bool,
    print_first: int,
    print_head: int,
    save_text: bool,
    topk: int,
    score_activation: str | None,
    save_plots: bool = False,
    plots_first: int = 10,
    vmax: float | None = None,
):
    model.eval()
    out_npz = out_dir / "pred_npz"
    out_txt = out_dir / "texts"
    out_fig = out_dir / "figs"
    out_npz.mkdir(parents=True, exist_ok=True)
    if save_text:
        out_txt.mkdir(parents=True, exist_ok=True)
    if save_plots:
        out_fig.mkdir(parents=True, exist_ok=True)

    have_gt = False
    meters = {"loss": 0.0, "n": 0}
    printed = 0
    plotted = 0

    amp_ctx = torch.amp.autocast(device_type=device.type, enabled=use_amp)

    for batch in loader:
        # to(device)
        for k in ["protein_emb", "peptide_emb"]:
            batch[k] = batch[k].to(device, non_blocking=True)
        for side in ["protein_masks", "peptide_masks"]:
            for kk in batch[side]:
                batch[side][kk] = batch[side][kk].to(device, non_blocking=True)
        if "gt_map" in batch:
            batch["gt_map"]  = batch["gt_map"].to(device, non_blocking=True)
            batch["gt_mask"] = batch["gt_mask"].to(device, non_blocking=True)
            have_gt = True

        model_batch, labels = adapt_batch_for_model(batch)

        with amp_ctx:
            out = model(model_batch)

        # 统一 scores 为 [B,Tl,Tp]
        s = out["scores"]
        Tl = model_batch["pep_emb"].size(1)
        Tp = model_batch["prot_emb"].size(1)
        if s.shape[-2:] == (Tp, Tl):
            s = s.transpose(1, 2).contiguous()
        elif s.shape[-2:] != (Tl, Tp):
            raise RuntimeError(f"Unexpected scores shape {s.shape}, expected (B,Tl,Tp) or (B,Tp,Tl)")
        out["scores"] = s

        if "pair_mask_override" in labels:
            out["pair_mask"] = labels["pair_mask_override"]

        # 评估 loss（若有 GT）
        if have_gt:
            loss_dict = criterion(out, labels)
            meters["loss"] += float(loss_dict["loss"]); meters["n"] += 1

        # 按样本落盘 & 打印
        scores = out["scores"]           # logits / prob / distance（取决于训练配置）
        pair_m = out["pair_mask"].bool()
        keys   = batch["keys"]

        # 判定本批是否 distance（优先用 batch['gt_kind']；否则回退看 score_activation==softplus）
        batch_is_distance = False
        if "gt_kind" in batch:
            # DataLoader collate 后，这里通常是字符串
            kind = batch["gt_kind"]
            batch_is_distance = (kind == "distance")
        if not batch_is_distance and (score_activation or "None").lower() == "softplus":
            batch_is_distance = True

        B = scores.size(0)
        for b in range(B):
            tl_real = int(model_batch["pep_mask"][b].sum().item())
            tp_real = int(model_batch["prot_mask"][b].sum().item())
            sb = scores[b, :tl_real, :tp_real].detach().cpu().float().numpy()
            mb = pair_m[b, :tl_real, :tp_real].detach().cpu().numpy().astype(np.bool_)

            out_rec = {"scores": sb, "pair_mask": mb, "key": keys[b]}

            # GT（若有）
            gt_map_np = None
            if have_gt:
                gt_map  = batch["gt_map"][b]
                gt_mask = batch["gt_mask"][b]
                tl_gt = int(gt_mask.any(dim=1).sum().item())
                tp_gt = int(gt_mask.any(dim=0).sum().item())
                tl_cut = min(tl_gt, tl_real); tp_cut = min(tp_gt, tp_real)
                if tl_cut > 0 and tp_cut > 0:
                    gt_map_np = gt_map[:tl_cut, :tp_cut].detach().cpu().to(dtype=torch.float32).numpy()
                    out_rec["gt_map"] = gt_map_np

            # —— contact vs distance 的后处理 —— #
            if not batch_is_distance:
                # contact：logits -> prob（若模型已 sigmoid，也可直接当 prob 处理）
                prob = sb if (score_activation or "").lower() == "sigmoid" else _sigmoid_np(sb)
                if save_prob:
                    out_rec["prob"] = prob
                if save_binary:
                    out_rec["binary"] = (prob >= float(bin_th)).astype(np.uint8)
            else:
                # distance：scores 即距离（若训练时用 softplus，这里 sb 已 ≥0）
                out_rec["distance"] = sb

            # npz 落盘
            np.savez_compressed(out_npz / f"{keys[b]}.npz", **out_rec)

            # —— 控制台打印 —— #
            if printed < print_first:
                print(f"\n=== [{printed+1}] key={keys[b]}  shape: Tl={tl_real}, Tp={tp_real}")
                if not batch_is_distance:
                    print(f"[prob head {print_head}x{print_head}]")
                    print(_format_head(prob, H=print_head))
                    if gt_map_np is not None:
                        print(f"[GT head {print_head}x{print_head}]")
                        print(_format_head(gt_map_np, H=print_head, fmt="{:6.0f}"))
                    # Top-K（仅 contact）
                    tk = _topk_coords(prob, mb, k=topk)
                    if len(tk) > 0:
                        print(f"[Top-{min(topk,len(tk))} predicted (i,j,prob){' with GT tag' if gt_map_np is not None else ''}]")
                        for (ii, jj, pp) in tk[:topk]:
                            if gt_map_np is not None and ii < gt_map_np.shape[0] and jj < gt_map_np.shape[1]:
                                print(f"  ({ii:4d},{jj:4d})  p={pp:.4f}  GT={int(gt_map_np[ii,jj])}")
                            else:
                                print(f"  ({ii:4d},{jj:4d})  p={pp:.4f}")
                else:
                    # distance 分支：打印预测/GT切片与统计
                    print(f"[pred distance head {print_head}x{print_head}]")
                    print(_format_head(sb, H=print_head, fmt="{:6.2f}"))
                    print(_stats_txt(sb, mb, name="pred-dist"))
                    if gt_map_np is not None:
                        print(f"[GT distance head {print_head}x{print_head}]")
                        print(_format_head(gt_map_np, H=print_head, fmt="{:6.2f}"))
                        print(_stats_txt(gt_map_np, None, name="gt-dist"))
                        if mb.sum() > 0:
                            err = np.abs(sb - gt_map_np)
                            mae = err[mb].mean()
                            rmse = np.sqrt(((sb - gt_map_np)[mb] ** 2).mean())
                            print(f"[error] MAE={mae:.3f} Å  RMSE={rmse:.3f} Å")
                printed += 1

            # —— 保存 txt 概要（可选） —— #
            if save_text:
                lines = [f"key: {keys[b]}", f"shape: Tl={tl_real}, Tp={tp_real}"]
                if not batch_is_distance:
                    if gt_map_np is not None:
                        pos_ratio = float(np.mean(gt_map_np)) if gt_map_np.size > 0 else 0.0
                        lines.append(f"GT positives: {int(np.sum(gt_map_np))} (ratio={pos_ratio:.4f})")
                    lines.append(f"\n[prob head {print_head}x{print_head}]")
                    lines.append(_format_head(prob, H=print_head))
                    if gt_map_np is not None:
                        lines.append(f"\n[GT head {print_head}x{print_head}]")
                        lines.append(_format_head(gt_map_np, H=print_head, fmt="{:6.0f}"))
                    tk = _topk_coords(prob, mb, k=topk)
                    if len(tk) > 0:
                        lines.append(f"\n[Top-{min(topk,len(tk))} predicted (i,j,prob){' with GT' if gt_map_np is not None else ''}]")
                        for (ii, jj, pp) in tk[:topk]:
                            if gt_map_np is not None and ii < gt_map_np.shape[0] and jj < gt_map_np.shape[1]:
                                lines.append(f"({ii},{jj})\t{pp:.6f}\tGT={int(gt_map_np[ii,jj])}")
                            else:
                                lines.append(f"({ii},{jj})\t{pp:.6f}")
                else:
                    lines.append(_stats_txt(sb, mb, name="pred-dist"))
                    if gt_map_np is not None:
                        lines.append(_stats_txt(gt_map_np, None, name="gt-dist"))
                        if mb.sum() > 0:
                            mae = np.abs(sb - gt_map_np)[mb].mean()
                            rmse = np.sqrt(((sb - gt_map_np)[mb] ** 2).mean())
                            lines.append(f"MAE={mae:.4f} Å  RMSE={rmse:.4f} Å")
                (out_txt / f"{keys[b]}.txt").write_text("\n".join(lines), encoding="utf-8")

            # —— 保存热图（两种模式都支持） —— #
            if save_plots and plotted < plots_first:
                if batch_is_distance:
                    # === distance: pred/GT/abs_err ===
                    vmax_use = vmax
                    if vmax_use is None:
                        vals = [sb]
                        if gt_map_np is not None:
                            vals.append(gt_map_np)
                        cat = np.concatenate([v.ravel() for v in vals]) if len(vals) else np.array([])
                        vmax_use = float(np.quantile(cat, 0.95)) if cat.size > 0 else None
                    _save_heatmap(sb, out_fig / f"{keys[b]}_pred.png", title=f"{keys[b]} pred dist (Å)", vmin=None, vmax=vmax_use)
                    if gt_map_np is not None:
                        _save_heatmap(gt_map_np, out_fig / f"{keys[b]}_gt.png", title=f"{keys[b]} GT dist (Å)", vmin=None, vmax=vmax_use)
                        diff = np.abs(sb - gt_map_np)
                        _save_heatmap(diff, out_fig / f"{keys[b]}_abs_err.png", title=f"{keys[b]} |pred-GT| (Å)", vmin=None, vmax=None)
                else:
                    # === contact: prob/GT/abs_err（以及可选二值） ===
                    prob = sb if (score_activation or "").lower() == "sigmoid" else _sigmoid_np(sb)
                    _save_heatmap(prob, out_fig / f"{keys[b]}_prob.png", title=f"{keys[b]} pred prob", vmin=0.0, vmax=1.0)
                    if gt_map_np is not None:
                        _save_heatmap(gt_map_np.astype(np.float32), out_fig / f"{keys[b]}_gt.png",
                                      title=f"{keys[b]} GT contact", vmin=0.0, vmax=1.0)
                        diff = np.abs(prob - gt_map_np.astype(np.float32))
                        _save_heatmap(diff, out_fig / f"{keys[b]}_abs_err.png", title=f"{keys[b]} |prob-GT|", vmin=0.0, vmax=1.0)
                    if save_binary:
                        binary = (prob >= float(bin_th)).astype(np.float32)
                        _save_heatmap(binary, out_fig / f"{keys[b]}_bin_th{bin_th}.png",
                                      title=f"{keys[b]} pred>= {bin_th}", vmin=0.0, vmax=1.0)
                plotted += 1

    if have_gt and meters["n"] > 0:
        print(f"\n[test] mean_loss={meters['loss']/meters['n']:.6f}")
    else:
        print("\n[test] saved predictions (no GT provided).")


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    # 数据
    ap.add_argument("--index", required=True)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)

    # GT 可选（互斥：contact vs distance）
    ap.add_argument("--gt-index", type=str, default=None)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--gt-as-contact",  dest="gt_as_contact", action="store_true",
                       help="Use thresholded contact labels (0/1)")
    group.add_argument("--gt-as-distance", dest="gt_as_contact", action="store_false",
                       help="Use raw distance labels (Å)")
    ap.set_defaults(gt_as_contact=True)
    ap.add_argument("--gt-threshold", type=float, default=8.0,
                    help="Contact threshold in Å when using --gt-as-contact")

    # 模型 / 权重
    ap.add_argument("--weights", type=str, required=True, help="path to best.pt / last.pt")
    ap.add_argument("--head", choices=["bilinear","mlp","axial"], default=None)
    ap.add_argument("--d-model", type=int, default=1536)
    ap.add_argument("--d-proj", type=int, default=256)
    ap.add_argument("--mlp-hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--score-activation", choices=[None, "softplus", "sigmoid"], default=None)

    # 损失（若评估用到）
    ap.add_argument("--use-bce", action="store_true")
    ap.add_argument("--m-pos", type=float, default=1.0)
    ap.add_argument("--m-neg", type=float, default=0.0)
    ap.add_argument("--pos-weight", type=float, default=None)
    ap.add_argument("--huber-delta", type=float, default=1.0)

    # 输出
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--save-prob", action="store_true")
    ap.add_argument("--save-binary", action="store_true")
    ap.add_argument("--bin-th", type=float, default=0.5)

    # 打印 / 文本增强
    ap.add_argument("--print-first", type=int, default=3, help="在控制台打印前 N 个样本的详情")
    ap.add_argument("--print-head", type=int, default=10, help="矩阵切片的边长")
    ap.add_argument("--save-text", action="store_true", help="为每个样本保存 .txt 人类可读摘要")
    ap.add_argument("--topk", type=int, default=50, help="Top-K 预测坐标数量（contact 模式使用）")

    # Distance 可视化
    ap.add_argument("--save-plots", action="store_true", help="保存热图（contact: prob/gt/abs_err；distance: pred/gt/abs_err）")
    ap.add_argument("--plots-first", type=int, default=10, help="至多为前 N 个样本保存热图")
    ap.add_argument("--vmax", type=float, default=None, help="距离热图的 vmax（Å），不设则自适应（95%分位）")

    # 其他
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")

    args = ap.parse_args()
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    # === 数据索引/keys ===
    key2npz = read_db_index(args.index)
    keys_all = read_split_list(args.splits_dir, args.split)

    gt_map = {}
    if args.gt_index:
        gt_key2npz = read_gt_index(args.gt_index)
        keys = [k for k in keys_all if (k in key2npz and strip_suffix_key(k) in gt_key2npz)]
        gt_map = {k: gt_key2npz[strip_suffix_key(k)] for k in keys}
    else:
        keys = [k for k in keys_all if k in key2npz]

    print(f"[data] {args.split} usable={len(keys)}")
    if args.gt_index:
        print(f"[data] GT matched: {len(gt_map)}/{len(keys)}")

    ds = ProtPepFullTokenDataset(
        key_to_npz=key2npz,
        keys=keys,
        strict_fixed=False,
        gt_key_to_npz=(gt_map or None),
        gt_as_contact=args.gt_as_contact,
        gt_threshold=args.gt_threshold,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_full_tokens, drop_last=False
    )

    # === 加载权重 / 模型 ===
    ckpt = torch.load(args.weights, map_location="cpu")

    # 简单检查权重是否含 NaN/Inf（可保留也可删）
    for name, tensor in ckpt["model"].items():
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print("Bad param:", name)

    ckpt_cfg = ckpt.get("cfg", None)
    if ckpt_cfg is not None:
        cfg = PairModelConfig(**ckpt_cfg)
        score_activation = (ckpt_cfg or {}).get("score_activation", None)
    else:
        cfg = PairModelConfig(
            d_model=args.d_model,
            head=(args.head or "bilinear"),
            hidden=args.mlp_hidden,
            d_proj=args.d_proj,
            dropout=args.dropout,
            score_activation=(None if args.score_activation in [None,"None","null"] else args.score_activation),
        )
        score_activation = cfg.score_activation

    model = PairwiseModel(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    # 准则（只有提供 GT 时才用）
    crit = PairwiseCriterion(
        use_bce_for_contact=args.use_bce,
        m_pos=args.m_pos, m_neg=args.m_neg,
        pos_weight=args.pos_weight,
        huber_delta=args.huber_delta,
    )

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    run_eval(
        model, crit, loader, device,
        out_dir=out_dir,
        save_prob=bool(args.save_prob),
        save_binary=bool(args.save_binary),
        bin_th=float(args.bin_th),
        use_amp=use_amp,
        print_first=int(args.print_first),
        print_head=int(args.print_head),
        save_text=bool(args.save_text),
        topk=int(args.topk),
        score_activation=score_activation,
        save_plots=bool(args.save_plots),
        plots_first=int(args.plots_first),
        vmax=(None if args.vmax is None else float(args.vmax)),
    )


if __name__ == "__main__":
    main()
