#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pair_model_fullmap.py

Full-map–only version of the pairwise model:
  - Expects batch["full_emb"], batch["full_mask"]
  - Outputs symmetric [B, L, L] logits (with the symmetric feature set)
  - Supports MLP / Bilinear / Axial heads (full mode only)
  - Plus an Axial+Triangular-Attention parallel head (head="axial_ta")
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F  # keep alias; use torch.nn.functional.normalize explicitly to avoid name clash
from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal


# -----------------------------
# Utilities: pairwise feature builder (4d)
# -----------------------------
def build_pair_features(x: torch.Tensor, d_use: int) -> torch.Tensor:
    """
    x: [B, L, d_use]
    return: feats [B, L, L, 4*d_use] with (Xi+Xj), |Xi-Xj|, cosine (L2-normed hadamard), Xi*Xj/√d
    """
    B, L, _ = x.shape
    Xi = x[:, :, None, :].expand(-1, -1, L, -1)   # [B, L, L, d]
    Xj = x[:, None, :, :].expand(-1, L, -1, -1)   # [B, L, L, d]

    # cosine interaction (elementwise) with L2-normalized vectors
    Xi_norm = torch.nn.functional.normalize(Xi, dim=-1, eps=1e-6)
    Xj_norm = torch.nn.functional.normalize(Xj, dim=-1, eps=1e-6)
    cosine = Xi_norm * Xj_norm

    # hadamard product with scale factor
    prod = (Xi * Xj) / math.sqrt(float(d_use))

    feats = torch.cat([
        (Xi + Xj),           # d
        (Xi - Xj).abs(),     # d
        cosine,              # d
        prod                 # d
    ], dim=-1)               # [B, L, L, 4d]
    return feats


# -----------------------------
# PairMLP (full-map only)
# -----------------------------
class PairMLP(nn.Module):
    def __init__(self, d_model: int, hidden: int = 512, dropout: float = 0.1,
                 out_channels: int = 1, d_proj: Optional[int] = None,
                 temperature: float = 1.0):
        super().__init__()
        self.temperature = float(max(1e-6, temperature))
        self.out_channels = int(out_channels)
        d_in = int(d_model)
        self.ln = nn.LayerNorm(d_in)

        if d_proj and d_proj != d_in:
            self.proj = nn.Linear(d_in, d_proj, bias=False)
            d_use = int(d_proj)
        else:
            self.proj = None
            d_use = d_in
        self._d_use = d_use

        # feature dimension: 4 × d_use
        feat_dim = 4 * d_use
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.act = nn.GELU()
        self.ln_h = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, out_channels)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: [B, L, D] -> logits [B, L, L]"""
        B, L, D = emb.shape
        x = self.ln(emb)
        if self.proj is not None:
            x = self.proj(x)

        feats = build_pair_features(x, self._d_use)  # [B, L, L, 4d]

        # MLP head
        h = self.fc1(feats.reshape(B * L * L, -1))
        h = self.act(h)
        h = self.ln_h(h)
        h = self.drop(h)
        out = self.fc2(h).view(B, L, L, self.out_channels).squeeze(-1)
        return out / self.temperature


# -----------------------------
# PairBilinear (full-map only)
# -----------------------------
class PairBilinear(nn.Module):
    def __init__(self, d_model: int, d_proj: int = 256):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        d_use = d_proj or d_model
        self.proj = nn.Linear(d_model, d_use, bias=False) if d_proj else None
        self.W = nn.Parameter(torch.empty(d_use, d_use))
        self.u = nn.Linear(d_use, 1, bias=False)
        self.v = nn.Linear(d_use, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        self._d_use = d_use
        nn.init.xavier_uniform_(self.W)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Full-map bilinear scores"""
        B, L, D = emb.shape
        x = self.ln(emb)
        if self.proj is not None:
            x = self.proj(x)
        PW = torch.matmul(x, self.W)                              # [B, L, d]
        bilinear = torch.bmm(PW, x.transpose(1, 2)) / math.sqrt(float(self._d_use))  # [B, L, L]
        up = self.u(x).expand(-1, -1, L)                          # [B, L, L]
        vl = self.v(x).transpose(1, 2).expand(-1, L, -1)          # [B, L, L]
        return bilinear + up + vl + self.bias


# -----------------------------
# Axial Head (full-map only)
# -----------------------------
class AxialSelfAttn(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1, axis="row"):
        super().__init__()
        self.axis = axis
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _, C = x.shape
        h = self.ln(x)
        if self.axis == "row":
            h = h.permute(0, 2, 1, 3).reshape(B * L, L, C)  # attend along rows (fix i, attend over j)
        else:
            h = h.reshape(B * L, L, C)                      # attend along cols (fix j, attend over i)
        y, _ = self.mha(h, h, h, need_weights=False)
        y = y.reshape(B, L, L, C) if self.axis == "col" else y.view(B, L, L, C).permute(0, 2, 1, 3)
        return x + self.drop(y)


class AxialFFN(nn.Module):
    def __init__(self, dim, hidden=512, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.fc1(self.ln(x)))
        h = self.drop(h)
        h = self.fc2(h)
        return x + self.drop(h)


class PairAxialHead(nn.Module):
    def __init__(self, d_model: int, c_pair=128, layers=2, heads=4,
                 dropout=0.1, ffn_hidden=512, out_channels=1,
                 d_proj=256, temperature=1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.out_channels = out_channels
        self.ln = nn.LayerNorm(d_model)
        d_use = d_proj or d_model
        self.proj = nn.Linear(d_model, d_use, bias=False) if d_proj else None
        # pair_in expects 4d features, consistent with PairMLP
        self.pair_in = nn.Linear(4 * d_use, c_pair)
        nn.init.xavier_uniform_(self.pair_in.weight)
        self.blocks = nn.ModuleList([
            AxialSelfAttn(c_pair, heads, dropout, "row"),
            AxialSelfAttn(c_pair, heads, dropout, "col"),
            AxialFFN(c_pair, ffn_hidden, dropout),
        ] * layers)
        self.out = nn.Linear(c_pair, out_channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self._d_use = d_use

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        B, L, D = emb.shape
        x = self.ln(emb)
        if self.proj is not None:
            x = self.proj(x)

        feats = build_pair_features(x, self._d_use)  # [B, L, L, 4d]

        x = self.pair_in(feats)
        for m in self.blocks:
            x = m(x)
        logits = self.out(x).squeeze(-1)
        return logits / self.temperature


# -----------------------------
# Triangular Attention (simplified AF-style)
# -----------------------------
class TriangularAttention(nn.Module):
    """
    Simplified triangular attention over k:
      - direction="outgoing": i–k–j, fix i; attend over k for each j
      - direction="incoming": j–k–i, fix j; attend over k for each i
    Operates on pair embeddings: [B, L, L, C]
    """
    def __init__(self, dim, heads=4, dropout=0.1, direction: Literal["outgoing", "incoming"] = "outgoing"):
        super().__init__()
        self.direction = direction
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, pair_emb: torch.Tensor) -> torch.Tensor:
        B, L, _, C = pair_emb.shape
        x = self.ln(pair_emb)
        if self.direction == "outgoing":
            # shape to [B*L, L, C]: for each i, attend over k to update all j
            seq = x.view(B * L, L, C)
            y, _ = self.mha(seq, seq, seq, need_weights=False)
            y = y.view(B, L, L, C)
        else:
            # incoming: swap i<->j, attend, then swap back
            seq = x.permute(0, 2, 1, 3).reshape(B * L, L, C)
            y, _ = self.mha(seq, seq, seq, need_weights=False)
            y = y.view(B, L, L, C).permute(0, 2, 1, 3)
        return pair_emb + self.drop(y)


class TriangularFFN(nn.Module):
    def __init__(self, dim, hidden=512, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.fc1(self.ln(x)))
        h = self.drop(h)
        h = self.fc2(h)
        return x + self.drop(h)


# -----------------------------
# Axial + Triangular Attention (parallel fusion head)
# -----------------------------
class PairAxialTAHead(nn.Module):
    """
    Parallel Axial path and Triangular-Attention path, fuse with a learnable gate.
    """
    def __init__(self, d_model: int, c_pair=128, layers=2, heads=4,
                 dropout=0.1, ffn_hidden=512, out_channels=1,
                 d_proj=256, temperature=1.0, fuse_mode: Literal["gate", "avg", "concat"] = "gate"):
        super().__init__()
        self.temperature = float(temperature)
        self.out_channels = out_channels
        self.fuse_mode = fuse_mode

        self.ln = nn.LayerNorm(d_model)
        d_use = d_proj or d_model
        self.proj = nn.Linear(d_model, d_use, bias=False) if d_proj else None
        self._d_use = d_use

        # shared pair feature in -> channel
        self.pair_in = nn.Linear(4 * d_use, c_pair)
        nn.init.xavier_uniform_(self.pair_in.weight)

        # Axial stack
        self.axial_blocks = nn.ModuleList([
            AxialSelfAttn(c_pair, heads, dropout, "row"),
            AxialSelfAttn(c_pair, heads, dropout, "col"),
            AxialFFN(c_pair, ffn_hidden, dropout),
        ] * layers)

        # Triangular stack (outgoing + incoming per layer)
        ta_blocks = []
        for _ in range(layers):
            ta_blocks += [
                TriangularAttention(c_pair, heads, dropout, "outgoing"),
                TriangularAttention(c_pair, heads, dropout, "incoming"),
                TriangularFFN(c_pair, ffn_hidden, dropout),
            ]
        self.tri_blocks = nn.ModuleList(ta_blocks)

        # fusion
        if fuse_mode == "concat":
            self.mix = nn.Linear(2 * c_pair, c_pair)
            nn.init.xavier_uniform_(self.mix.weight)
        elif fuse_mode == "gate":
            self.alpha = nn.Parameter(torch.tensor(0.5))  # scalar gate in (0,1) after sigmoid

        # output
        self.out = nn.Linear(c_pair, out_channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        B, L, D = emb.shape
        x = self.ln(emb)
        if self.proj is not None:
            x = self.proj(x)

        feats = build_pair_features(x, self._d_use)   # [B, L, L, 4d]
        z0 = self.pair_in(feats)                     # [B, L, L, C]

        # Axial path
        xa = z0
        for m in self.axial_blocks:
            xa = m(xa)

        # Triangular path
        xt = z0
        for m in self.tri_blocks:
            xt = m(xt)

        # Fuse
        if self.fuse_mode == "avg":
            z = 0.5 * (xa + xt)
        elif self.fuse_mode == "concat":
            z = torch.cat([xa, xt], dim=-1)   # [B, L, L, 2C]
            z = self.mix(z)                  # [B, L, L, C]
        else:  # "gate"
            alpha = torch.sigmoid(self.alpha)
            z = alpha * xa + (1.0 - alpha) * xt

        logits = self.out(z).squeeze(-1)     # [B, L, L]
        return logits / self.temperature
# -----------------------------
# Triangular Attention only head
# -----------------------------
class PairTriangularHead(nn.Module):
    """
    Triangular-Attention-only version (no Axial path).
    Uses both outgoing and incoming triangular attention per layer,
    followed by FFN, similar to AF-style pair update block.
    """
    def __init__(self, d_model: int, c_pair=128, layers=2, heads=4,
                 dropout=0.1, ffn_hidden=512, out_channels=1,
                 d_proj=256, temperature=1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.out_channels = out_channels
        self.ln = nn.LayerNorm(d_model)

        d_use = d_proj or d_model
        self.proj = nn.Linear(d_model, d_use, bias=False) if d_proj else None
        self._d_use = d_use

        # shared pair feature input
        self.pair_in = nn.Linear(4 * d_use, c_pair)
        nn.init.xavier_uniform_(self.pair_in.weight)

        # build Triangular Attention stack
        blocks = []
        for _ in range(layers):
            blocks += [
                TriangularAttention(c_pair, heads, dropout, "outgoing"),
                TriangularAttention(c_pair, heads, dropout, "incoming"),
                TriangularFFN(c_pair, ffn_hidden, dropout),
            ]
        self.blocks = nn.ModuleList(blocks)

        # output
        self.out = nn.Linear(c_pair, out_channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        B, L, D = emb.shape
        x = self.ln(emb)
        if self.proj is not None:
            x = self.proj(x)

        feats = build_pair_features(x, self._d_use)   # [B, L, L, 4d]
        z = self.pair_in(feats)                      # [B, L, L, C]

        for m in self.blocks:
            z = m(z)

        logits = self.out(z).squeeze(-1)
        return logits / self.temperature


# -----------------------------
# Config & Wrapper
# -----------------------------
HeadType = Literal["mlp", "bilinear", "axial", "axial_ta","ta"]

@dataclass
class PairModelConfig:
    d_model: int = 1536
    head: HeadType = "mlp"
    hidden: int = 512
    d_proj: int = 256
    dropout: float = 0.1
    out_channels: int = 1
    temperature: float = 1.0
    axial_c_pair: int = 128
    axial_layers: int = 2
    axial_heads: int = 4
    axial_ffn_hidden: int = 512
    # ---- optional: ESM-3 style chain embedding ----
    use_chain_embed: bool = False   # if True, require batch["chain_id"]
    num_chains: int = 2             # typically 2 for peptide/protein
    # ---- for axial+ta ----
    fuse_mode: Literal["gate", "avg", "concat"] = "gate"

class PairwiseModelFullMap(nn.Module):
    """Simplified full-map-only version with selectable head."""
    def __init__(self, cfg: PairModelConfig):
        super().__init__()
        self.cfg = cfg

        # optional chain embedding
        self.chain_embed = None
        if cfg.use_chain_embed:
            self.chain_embed = nn.Embedding(cfg.num_chains, cfg.d_model)
            nn.init.normal_(self.chain_embed.weight, mean=0.0, std=cfg.d_model ** -0.5)

        # select head
        if cfg.head == "mlp":
            self.head = PairMLP(cfg.d_model, cfg.hidden, cfg.dropout,
                                cfg.out_channels, cfg.d_proj, cfg.temperature)
        elif cfg.head == "bilinear":
            self.head = PairBilinear(cfg.d_model, cfg.d_proj)
        elif cfg.head == "axial":
            self.head = PairAxialHead(cfg.d_model, cfg.axial_c_pair,
                                      cfg.axial_layers, cfg.axial_heads,
                                      cfg.dropout, cfg.axial_ffn_hidden,
                                      cfg.out_channels, cfg.d_proj,
                                      cfg.temperature)
        elif cfg.head == "axial_ta":
            self.head = PairAxialTAHead(cfg.d_model, cfg.axial_c_pair,
                                        cfg.axial_layers, cfg.axial_heads,
                                        cfg.dropout, cfg.axial_ffn_hidden,
                                        cfg.out_channels, cfg.d_proj,
                                        cfg.temperature, cfg.fuse_mode)
        elif cfg.head == "ta":
            self.head = PairTriangularHead(cfg.d_model, cfg.axial_c_pair,
                                        cfg.axial_layers, cfg.axial_heads,
                                        cfg.dropout, cfg.axial_ffn_hidden,
                                        cfg.out_channels, cfg.d_proj,
                                        cfg.temperature)

        else:
            raise ValueError(f"Unknown head: {cfg.head}")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        emb = batch["full_emb"]          # [B, L, D]
        mask = batch.get("full_mask", None)

        # add chain embedding if enabled
        if self.chain_embed is not None:
            chain_id = batch.get("chain_id", None)   # expect [B, L] in [0..num_chains-1]
            if chain_id is None:
                raise KeyError("cfg.use_chain_embed=True requires batch['chain_id'] (LongTensor[B,L]).")
            chain_id = chain_id.to(emb.device)
            emb = emb + self.chain_embed(chain_id)   # [B, L, D]

        scores = self.head(emb)          # [B, L, L] or [B, L, L, C]
        if scores.ndim == 3:
            sT = scores.transpose(-1, -2)
        elif scores.ndim == 4:
            sT = scores.transpose(1, 2)  # swap the two spatial dims, keep channel
        else:
            raise RuntimeError(f"Unexpected score shape {scores.shape}")
        scores = 0.5 * (scores + sT)
        pair_mask = mask[:, :, None] & mask[:, None, :] if mask is not None else None
        return {"scores": scores, "pair_mask": pair_mask}


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    B, L, D = 2, 32, 1536
    F_test = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.bool)

    # Try different heads:
    for head_name in ["mlp", "bilinear", "axial", "axial_ta","ta"]:
        print(f"== Testing head={head_name} ==")
        cfg = PairModelConfig(d_model=D, head=head_name)
        model = PairwiseModelFullMap(cfg)
        out = model({"full_emb": F_test, "full_mask": mask})
        print({k: v.shape if torch.is_tensor(v) else v for k, v in out.items()})
