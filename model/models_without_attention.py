#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Pair feature heads (stable)
# -----------------------------

class PairMLP(nn.Module):
    """
    稳定版 Pair-MLP 头：
      - 输入先 LN（可选再降维到 d_proj）
      - 特征: [P, L, |P-L|, (P*L)/√d]
      - MLP: Linear→GELU→LayerNorm→Dropout→Linear
      - 最后一层零初始化（避免一开始押单类）
      - 支持 chunk_ll 沿 Ll 分块以省显存
    """
    def __init__(
        self,
        d_model: int,
        hidden: int = 512,
        dropout: float = 0.1,
        out_channels: int = 1,
        d_proj: Optional[int] = None,
        temperature: float = 1.0,   # logits / τ
    ):
        super().__init__()
        self.out_channels = int(out_channels)
        self.temperature = float(max(1e-6, temperature))

        d_in = int(d_model)
        self.ln_p = nn.LayerNorm(d_in)
        self.ln_l = nn.LayerNorm(d_in)

        # 可选降维
        if d_proj is not None and d_proj > 0 and d_proj != d_in:
            self.proj_p = nn.Linear(d_in, d_proj, bias=False)
            self.proj_l = nn.Linear(d_in, d_proj, bias=False)
            d_use = int(d_proj)
        else:
            self.proj_p = None
            self.proj_l = None
            d_use = d_in

        feat_dim = 4 * d_use  # [P, L, |P-L|, P*L/√d]
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.act = nn.GELU()
        self.ln_h = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, self.out_channels)

        # 温和初始化：最后层置零，前面用xavier
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        self._d_use = d_use  # for scaling

    def _prep(self, prot_emb: torch.Tensor, pep_emb: torch.Tensor):
        # LN (fp32) + 可选降维
        P = self.ln_p(prot_emb)
        L = self.ln_l(pep_emb)
        if self.proj_p is not None:
            P = self.proj_p(P)
            L = self.proj_l(L)
        return P, L  # [B,Lp,d], [B,Ll,d]

    def _make_pair_feats(self, P: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        # P: [B,Lp,d], L: [B,Ll,d]  ->  [B,Lp,Ll,d]
        B, Lp, d = P.shape
        Ll = L.size(1)
        PP = P[:, :, None, :].expand(-1, -1, Ll, -1)
        LL = L[:, None, :, :].expand(-1, Lp, -1, -1)

        prod = (PP * LL) / math.sqrt(float(self._d_use))
        feats = torch.cat([PP, LL, (PP - LL).abs(), prod], dim=-1)  # [B,Lp,Ll,4d]
        return feats

    def forward(self, prot_emb: torch.Tensor, pep_emb: torch.Tensor, chunk_ll: int = 0) -> torch.Tensor:
        P, L = self._prep(prot_emb, pep_emb)
        B, Lp, d = P.shape
        Ll = L.size(1)

        if chunk_ll and chunk_ll < Ll:
            if self.out_channels == 1:
                out_all = P.new_zeros(B, Lp, Ll)
            else:
                out_all = P.new_zeros(B, Lp, Ll, self.out_channels)
            for j0 in range(0, Ll, chunk_ll):
                j1 = min(j0 + chunk_ll, Ll)
                feats = self._make_pair_feats(P, L[:, j0:j1, :])  # [B,Lp,chunk,4d]
                x = self.fc1(feats.reshape(B * Lp * (j1 - j0), -1))
                x = self.act(x)
                x = self.ln_h(x)
                x = self.drop(x)
                x = self.fc2(x) / self.temperature
                if self.out_channels == 1:
                    x = x.view(B, Lp, j1 - j0, 1).squeeze(-1)
                    out_all[:, :, j0:j1] = x
                else:
                    out_all[:, :, j0:j1, :] = x.view(B, Lp, j1 - j0, self.out_channels)
            return out_all

        feats = self._make_pair_feats(P, L)             # [B,Lp,Ll,4d]
        x = self.fc1(feats.reshape(B * Lp * Ll, -1))
        x = self.act(x)
        x = self.ln_h(x)
        x = self.drop(x)
        x = self.fc2(x) / self.temperature
        if self.out_channels == 1:
            return x.view(B, Lp, Ll, 1).squeeze(-1)     # [B,Lp,Ll]
        else:
            return x.view(B, Lp, Ll, self.out_channels) # [B,Lp,Ll,C]


class PairBilinear(nn.Module):
    """
    稳定化的双线性头（仅标量）：
      scores = (P W L^T)/√d + u^T P + v^T L + b
      P/L 先 LN + 可选降维。
    """
    def __init__(self, d_model: int, d_proj: int = 256, bias: bool = True):
        super().__init__()
        self.ln_p = nn.LayerNorm(d_model)
        self.ln_l = nn.LayerNorm(d_model)

        d_in = int(d_model)
        d_use = int(d_proj) if d_proj is not None and d_proj > 0 else d_in
        self.proj_p = nn.Linear(d_in, d_use, bias=False) if d_use != d_in else None
        self.proj_l = nn.Linear(d_in, d_use, bias=False) if d_use != d_in else None

        self.W = nn.Parameter(torch.empty(d_use, d_use))
        nn.init.xavier_uniform_(self.W, gain=1.0)

        self.u = nn.Linear(d_use, 1, bias=False)
        self.v = nn.Linear(d_use, 1, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("bias", None)

        self._d_use = d_use

    def _prep(self, prot_emb: torch.Tensor, pep_emb: torch.Tensor):
        P = self.ln_p(prot_emb)
        L = self.ln_l(pep_emb)
        if self.proj_p is not None:
            P = self.proj_p(P)
            L = self.proj_l(L)
        return P, L

    def forward(self, prot_emb: torch.Tensor, pep_emb: torch.Tensor) -> torch.Tensor:
        P, L = self._prep(prot_emb, pep_emb)            # [B,Lp,d], [B,Ll,d]
        B, Lp, d = P.shape
        Ll = L.size(1)

        # 双线性项 (P @ W) @ L^T / √d
        PW = torch.matmul(P, self.W)                    # [B,Lp,d]
        bilinear = torch.bmm(PW, L.transpose(1, 2))     # [B,Lp,Ll]
        bilinear = bilinear / math.sqrt(float(self._d_use))

        up = self.u(P).expand(-1, -1, Ll)               # [B,Lp,Ll]
        vl = self.v(L).transpose(1, 2).expand(-1, Lp, -1)
        scores = bilinear + up + vl
        if self.bias is not None:
            scores = scores + self.bias
        return scores


# -----------------------------
# Model wrapper
# -----------------------------

HeadType = Literal["mlp", "bilinear"]

@dataclass
class PairModelConfig:
    d_model: int = 1536
    head: HeadType = "bilinear"         # "mlp" or "bilinear"
    hidden: int = 512                   # for MLP head
    d_proj: int = 256                   # proj dim for bilinear; also用于 MLP 降维
    dropout: float = 0.1                # for MLP head
    chunk_ll: int = 0                   # chunk size for MLP (0=off)
    score_activation: Optional[Literal["softplus", "sigmoid"]] = None  # 仅 out_channels==1 时生效
    out_channels: int = 1               # =1: 标量；>1: 多类
    temperature: float = 1.0            # logits 温度（MLP head）

class PairwiseModel(nn.Module):
    """
    输入：prot_emb [B,Lp,D], pep_emb [B,Ll,D], prot_mask [B,Lp], pep_mask [B,Ll]
    输出：
      - out_channels==1:  scores [B,Lp,Ll]
      - out_channels>1 :  logits [B,Lp,Ll,C]
    以及 pair_mask [B,Lp,Ll]
    """
    def __init__(self, cfg: PairModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.head == "mlp":
            self.head = PairMLP(
                d_model=cfg.d_model,
                hidden=cfg.hidden,
                dropout=cfg.dropout,
                out_channels=cfg.out_channels,
                d_proj=cfg.d_proj,                 # 用同一 d_proj 降维
                temperature=max(1e-6, cfg.temperature),
            )
        elif cfg.head == "bilinear":
            if cfg.out_channels != 1:
                raise ValueError("Bilinear head only supports out_channels=1. Use --head mlp for distance bins.")
            self.head = PairBilinear(cfg.d_model, cfg.d_proj)
        else:
            raise ValueError(f"Unknown head: {cfg.head}")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        P = batch["prot_emb"]    # [B,Lp,D]
        L = batch["pep_emb"]     # [B,Ll,D]
        pm = batch.get("prot_mask")
        lm = batch.get("pep_mask")

        if pm is None:
            pm = torch.ones(P.size(0), P.size(1), dtype=torch.bool, device=P.device)
        if lm is None:
            lm = torch.ones(L.size(0), L.size(1), dtype=torch.bool, device=L.device)

        if isinstance(self.head, PairMLP):
            out = self.head(P, L, chunk_ll=self.cfg.chunk_ll)  # [B,Lp,Ll] or [B,Lp,Ll,C]
        else:
            out = self.head(P, L)                               # [B,Lp,Ll]

        # 可选分数激活（仅标量输出时启用）
        if self.cfg.out_channels == 1:
            if self.cfg.score_activation == "softplus":
                out = F.softplus(out)
            elif self.cfg.score_activation == "sigmoid":
                out = torch.sigmoid(out)

        pair_mask = pm[:, :, None] & lm[:, None, :]
        return {"scores": out, "pair_mask": pair_mask}


# -----------------------------
# Losses (unchanged APIs)
# -----------------------------

def positive_margin_loss(scores: torch.Tensor,
                         labels: torch.Tensor,
                         mask: Optional[torch.Tensor] = None,
                         m_pos: float = 1.0,
                         m_neg: float = 0.0,
                         pos_weight: Optional[float] = None) -> torch.Tensor:
    assert set(torch.unique(labels).tolist()) <= {0, 1}, "labels must be 0/1"
    y = labels.float()
    s = scores
    pos_term = F.relu(m_pos - s) * y
    neg_term = F.relu(s - m_neg) * (1.0 - y)
    loss = pos_term + neg_term
    if pos_weight is not None:
        loss = loss * (1.0 + (pos_weight - 1.0) * y)
    if mask is not None:
        loss = loss * mask.float()
        denom = mask.float().sum().clamp_min(1.0)
    else:
        denom = torch.tensor(loss.numel(), device=loss.device, dtype=loss.dtype).clamp_min(1.0)
    return loss.sum() / denom


def bce_contact_loss(scores: torch.Tensor,
                     labels: torch.Tensor,
                     mask: Optional[torch.Tensor] = None,
                     pos_weight: Optional[float] = None) -> torch.Tensor:
    if pos_weight is not None:
        pw = torch.tensor(pos_weight, device=scores.device, dtype=scores.dtype)
        loss = F.binary_cross_entropy_with_logits(scores, labels.float(), pos_weight=pw, reduction="none")
    else:
        loss = F.binary_cross_entropy_with_logits(scores, labels.float(), reduction="none")
    if mask is not None:
        loss = loss * mask.float()
        return loss.sum() / mask.float().sum().clamp_min(1.0)
    return loss.mean()


def huber_distance_loss(pred_dist: torch.Tensor,
                        true_dist: torch.Tensor,
                        mask: Optional[torch.Tensor] = None,
                        delta: float = 1.0) -> torch.Tensor:
    loss = F.huber_loss(pred_dist, true_dist, delta=delta, reduction="none")
    if mask is not None:
        loss = loss * mask.float()
        return loss.sum() / mask.float().sum().clamp_min(1.0)
    return loss.mean()


# -----------------------------
# Criterion 封装（与你现有 trainer 兼容）
# -----------------------------
class PairwiseCriterion(nn.Module):
    def __init__(self,
                 use_bce_for_contact: bool = True,
                 m_pos: float = 1.0, m_neg: float = 0.0,
                 pos_weight: Optional[float] = None,
                 max_pos_weight: float = 50.0,
                 focal_gamma: float = 0.0,
                 dice_lambda: float = 0.0,
                 neg_pos_ratio: Optional[float] = None,
                 huber_delta: float = 1.0):
        super().__init__()
        self.use_bce_for_contact = use_bce_for_contact
        self.m_pos = m_pos; self.m_neg = m_neg
        self.pos_weight = pos_weight
        self.max_pos_weight = max_pos_weight
        self.focal_gamma = focal_gamma
        self.dice_lambda = dice_lambda
        self.neg_pos_ratio = neg_pos_ratio
        self.huber_delta = huber_delta

    def _make_effective_mask(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.neg_pos_ratio is None:
            return mask
        yb = (y > 0.5)
        pos_m = (mask & yb)
        neg_m = (mask & (~yb))

        n_pos = int(pos_m.sum().item())
        if n_pos == 0:
            return mask
        target_neg = int(min(neg_m.sum().item(), max(1, n_pos * self.neg_pos_ratio)))

        idx_neg = torch.nonzero(neg_m, as_tuple=False)
        if idx_neg.numel() == 0:
            return pos_m
        choice = torch.randperm(idx_neg.shape[0], device=idx_neg.device)[:target_neg]
        keep_neg = torch.zeros_like(neg_m)
        keep_neg[idx_neg[choice][:,0], idx_neg[choice][:,1], idx_neg[choice][:,2]] = True
        return pos_m | keep_neg

    def _dynamic_pos_weight(self, y: torch.Tensor, mask: torch.Tensor) -> float:
        pos = (y > 0.5) & mask
        n_pos = float(pos.sum().item())
        n_all = float(mask.sum().item())
        n_neg = max(0.0, n_all - n_pos)
        if n_pos < 1.0:
            return 1.0
        w = n_neg / n_pos
        return float(min(self.max_pos_weight, max(1.0, w)))

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        scores = outputs["scores"]
        pair_mask = outputs["pair_mask"].bool()

        losses = {}
        # contact
        if "labels_contact" in batch:
            y = batch["labels_contact"].to(scores.dtype)
            eff_mask = self._make_effective_mask(y, pair_mask)

            if self.use_bce_for_contact:
                pw = self.pos_weight
                if pw is None:
                    pw = self._dynamic_pos_weight(y, eff_mask)
                pw_t = torch.tensor(pw, device=scores.device, dtype=scores.dtype)

                loss_bce = F.binary_cross_entropy_with_logits(
                    scores, y, pos_weight=pw_t, reduction="none"
                )
                if self.focal_gamma and self.focal_gamma > 0.0:
                    p = torch.sigmoid(scores)
                    pt = torch.where(y > 0.5, p, 1.0 - p)
                    loss_bce = loss_bce * ((1.0 - pt).clamp_min(1e-6) ** self.focal_gamma)

                loss = (loss_bce * eff_mask.float()).sum() / eff_mask.float().sum().clamp_min(1.0)

                if self.dice_lambda and self.dice_lambda > 0.0:
                    prob = torch.sigmoid(scores)
                    inter = (prob * y * eff_mask).sum()
                    denom = (prob * eff_mask).sum() + (y * eff_mask).sum() + 1e-8
                    loss_dice = 1.0 - (2.0 * inter / denom)
                    loss = loss + self.dice_lambda * loss_dice

                losses["loss"] = loss

                with torch.no_grad():
                    prob = torch.sigmoid(scores)
                    m = eff_mask
                    pos_m = m & (y > 0.5)
                    neg_m = m & (~(y > 0.5))
                    pos_prob = prob[pos_m].mean().item() if pos_m.any() else 0.0
                    neg_prob = prob[neg_m].mean().item() if neg_m.any() else 0.0
                    pos_ratio = float((y[m] > 0.5).float().mean().item()) if m.any() else 0.0
                    logit_std = scores[m].float().std().item() if m.any() else 0.0
                losses.update({
                    "pos_prob": pos_prob,
                    "neg_prob": neg_prob,
                    "pos_ratio": pos_ratio,
                    "logit_std": logit_std,
                    "pos_weight_eff": float(pw),
                })
            else:
                yb = (y > 0.5).float()
                pos_term = F.relu(self.m_pos - scores) * yb
                neg_term = F.relu(scores - self.m_neg) * (1.0 - yb)
                loss_h = (pos_term + neg_term)
                loss = (loss_h * eff_mask.float()).sum() / eff_mask.float().sum().clamp_min(1.0)
                losses["loss"] = loss

        # distance（保持与原逻辑兼容）
        if "labels_distance" in batch:
            dist_t = batch["labels_distance"].to(scores.dtype)
            loss_d = F.huber_loss(scores, dist_t, delta=self.huber_delta, reduction="none")
            loss_d = (loss_d * pair_mask.float()).sum() / pair_mask.float().sum().clamp_min(1.0)
            losses["loss"] = losses.get("loss", 0.0) + loss_d
            with torch.no_grad():
                mae = (torch.abs(scores - dist_t)[pair_mask]).mean().item() if pair_mask.any() else 0.0
            losses["dist_mae"] = mae

        if "loss" not in losses:
            losses["loss"] = scores.new_tensor(0.0)
        return losses


# -----------------------------
# Quick self-test
# -----------------------------
if __name__ == "__main__":
    B, Lp, Ll, D = 2, 20, 12, 1536
    prot_emb = torch.randn(B, Lp, D)
    pep_emb  = torch.randn(B, Ll, D)
    prot_mask = torch.ones(B, Lp, dtype=torch.bool)
    pep_mask  = torch.ones(B, Ll, dtype=torch.bool)

    # 多类（例如距离分桶）
    cfg = PairModelConfig(d_model=D, head="mlp", hidden=512, d_proj=256, dropout=0.1,
                          out_channels=6, temperature=2.0, chunk_ll=0)
    model = PairwiseModel(cfg)
    batch = {"prot_emb": prot_emb, "pep_emb": pep_emb, "prot_mask": prot_mask, "pep_mask": pep_mask}
    out = model(batch)
    print("MLP logits:", out["scores"].shape, "| pair_mask:", out["pair_mask"].shape)

    # 标量（双线性）
    cfg2 = PairModelConfig(d_model=D, head="bilinear", d_proj=256, out_channels=1)
    model2 = PairwiseModel(cfg2)
    out2 = model2(batch)
    print("Bilinear scores:", out2["scores"].shape)
