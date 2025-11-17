#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from trainer import *  # 直接复用原逻辑
import torch

# 覆盖 evaluate 与 train 部分以加 debug
@torch.no_grad()
def evaluate_debug(model, criterion, loader, device, is_bins=False):
    model.eval()
    meters = {"loss_sum": 0.0, "n": 0}
    print("\n=== [DEBUG] Start Evaluation ===")

    for i, batch in enumerate(loader):
        # ----- Move to device -----
        if "full_emb" in batch:
            print(f"[DEBUG] Batch {i}: detected full_emb mode!")
            batch["full_emb"] = batch["full_emb"].to(device, non_blocking=True)
            if "full_mask" in batch:
                batch["full_mask"] = batch["full_mask"].to(device, non_blocking=True)
        else:
            for k in ["protein_emb", "peptide_emb"]:
                batch[k] = batch[k].to(device, non_blocking=True)
            for side in ["protein_masks", "peptide_masks"]:
                for kk in batch[side]:
                    batch[side][kk] = batch[side][kk].to(device, non_blocking=True)
            if "gt_map" in batch:
                batch["gt_map"] = batch["gt_map"].to(device)
                batch["gt_mask"] = batch["gt_mask"].to(device)

        model_batch, labels = adapt_batch_for_model(batch)
        out = model(model_batch)

        # ----- Debug shapes -----
        if "full_emb" in model_batch:
            print(f"[DEBUG] full_emb: {tuple(model_batch['full_emb'].shape)} | full_mask: {tuple(model_batch['full_mask'].shape)}")
        else:
            print(f"[DEBUG] pep_emb: {tuple(model_batch['pep_emb'].shape)}, prot_emb: {tuple(model_batch['prot_emb'].shape)}")

        if "labels_contact" in labels:
            gt = labels["labels_contact"]
        elif "labels_distance" in labels:
            gt = labels["labels_distance"]
        else:
            gt = None
        if gt is not None:
            print(f"[DEBUG] GT map shape: {tuple(gt.shape)}, pair_mask sum={labels['pair_mask_override'].sum().item()}")

        # ----- Score ordering check -----
        s = out["scores"]
        print(f"[DEBUG] out['scores'] shape={tuple(s.shape)}")

        out["pair_mask"] = labels["pair_mask_override"]
        loss_dict = criterion(out, {**labels})
        loss = float(loss_dict["loss"])
        print(f"[DEBUG] loss={loss:.4f}")
        meters["loss_sum"] += loss
        meters["n"] += 1
        if i >= 2: break  # 只跑前几个batch就够

    print("=== [DEBUG] Eval Done ===")
    return {"loss": meters["loss_sum"] / max(1, meters["n"])}


# 替换原 evaluate
evaluate = evaluate_debug

if __name__ == "__main__":
    print("=== [DEBUG MODE] FullMap Sanity Trainer ===")
    main()
