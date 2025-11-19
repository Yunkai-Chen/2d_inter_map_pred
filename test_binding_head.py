#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_binding_head.py

Test the binding classification head (side task).
Demonstrates how to use the BindingClassificationHead with your model.
"""

import torch
import torch.nn as nn
from model.models_fullmap import PairwiseModelFullMap, PairModelConfig

def test_binding_head():
    """Test the binding classification head."""

    # Setup
    B, L_pep, L_prot, D = 2, 10, 50, 1536
    L = L_pep + L_prot

    print("="*60)
    print("Testing Binding Classification Head")
    print("="*60)

    # Create fake data
    full_emb = torch.randn(B, L, D)
    full_mask = torch.ones(B, L, dtype=torch.bool)

    # Create chain_id: 0 for peptide, 1 for protein
    chain_id = torch.zeros(B, L, dtype=torch.long)
    chain_id[:, L_pep:] = 1  # First L_pep tokens are peptide, rest are protein

    print(f"\nInput shapes:")
    print(f"  full_emb: {full_emb.shape}")
    print(f"  full_mask: {full_mask.shape}")
    print(f"  chain_id: {chain_id.shape}")
    print(f"  L_peptide: {L_pep}, L_protein: {L_prot}")

    # Test different interaction modes
    interaction_modes = ["concat", "sum", "bilinear", "attention"]

    for mode in interaction_modes:
        print(f"\n{'-'*60}")
        print(f"Testing interaction_mode: {mode}")
        print(f"{'-'*60}")

        # Create model with binding head
        cfg = PairModelConfig(
            d_model=D,
            head="ta",  # Main task head
            axial_layers=2,
            axial_heads=4,
            axial_c_pair=128,
            use_binding_head=True,  # Enable binding classification
            binding_interaction_mode=mode,
            binding_hidden=512
        )

        model = PairwiseModelFullMap(cfg)
        model.eval()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        binding_params = sum(p.numel() for p in model.binding_head.parameters())

        print(f"  Total parameters: {total_params:,}")
        print(f"  Binding head parameters: {binding_params:,}")
        print(f"  Binding head %: {100*binding_params/total_params:.2f}%")

        # Forward pass
        with torch.no_grad():
            batch = {
                "full_emb": full_emb,
                "full_mask": full_mask,
                "chain_id": chain_id
            }
            output = model(batch)

        print(f"\nOutput keys: {list(output.keys())}")
        print(f"  scores shape: {output['scores'].shape}")
        print(f"  binding_logits shape: {output['binding_logits'].shape}")
        print(f"  binding_logits: {output['binding_logits']}")

        # Convert to probabilities
        binding_probs = torch.sigmoid(output['binding_logits'])
        print(f"  binding_probs: {binding_probs}")


def test_multitask_training_example():
    """
    Example of how to train with both main task (contact) and side task (binding).
    """
    print("\n" + "="*60)
    print("Multi-Task Training Example")
    print("="*60)

    B, L, D = 2, 60, 1536

    # Create model
    cfg = PairModelConfig(
        d_model=D,
        head="ta",
        axial_layers=2,
        use_binding_head=True,
        binding_interaction_mode="concat"
    )
    model = PairwiseModelFullMap(cfg)

    # Fake data
    full_emb = torch.randn(B, L, D)
    full_mask = torch.ones(B, L, dtype=torch.bool)
    chain_id = torch.cat([
        torch.zeros(B, 10, dtype=torch.long),  # Peptide
        torch.ones(B, 50, dtype=torch.long)    # Protein
    ], dim=1)

    # Fake labels
    gt_contact_map = torch.randint(0, 2, (B, L, L), dtype=torch.float32)
    gt_binding_label = torch.tensor([1.0, 0.0])  # Sample 1 binds, sample 2 doesn't

    # Forward
    batch = {"full_emb": full_emb, "full_mask": full_mask, "chain_id": chain_id}
    output = model(batch)

    # Compute losses
    # Main task: contact prediction (BCE)
    contact_loss = nn.functional.binary_cross_entropy_with_logits(
        output["scores"], gt_contact_map, reduction="mean"
    )

    # Side task: binding classification (BCE)
    binding_loss = nn.functional.binary_cross_entropy_with_logits(
        output["binding_logits"], gt_binding_label, reduction="mean"
    )

    # Combined loss with weighting
    alpha = 1.0  # Weight for contact task
    beta = 0.1   # Weight for binding task (usually smaller)
    total_loss = alpha * contact_loss + beta * binding_loss

    print(f"\nLoss breakdown:")
    print(f"  Contact loss: {contact_loss.item():.4f}")
    print(f"  Binding loss: {binding_loss.item():.4f}")
    print(f"  Total loss (α={alpha}, β={beta}): {total_loss.item():.4f}")

    print(f"\nPredicted binding probabilities:")
    binding_probs = torch.sigmoid(output["binding_logits"])
    for i in range(B):
        print(f"  Sample {i}: prob={binding_probs[i].item():.4f}, "
              f"label={gt_binding_label[i].item():.0f}")


if __name__ == "__main__":
    # Test binding head
    test_binding_head()

    # Test multi-task training
    test_multitask_training_example()

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
