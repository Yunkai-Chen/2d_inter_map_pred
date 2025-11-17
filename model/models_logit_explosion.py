# contrasive_learning/model/models.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Pair feature heads
# -----------------------------

class PairMLP(nn.Module):
    """
    使用对偶特征拼接 [p, l, |p-l|, p*l] -> MLP。
    支持输出 out_channels（=1 时为标量；>1 时为多类别 logits）。
    """
    def __init__(self, d_model: int, hidden: int = 512, dropout: float = 0.1, out_channels: int = 1):
        super().__init__()
        self.out_channels = int(out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.out_channels)
        )

    @staticmethod
    def _make_pair_feats(P: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        # P: [B, Lp, 1, D], L: [B, 1, Ll, D]  -> broadcast 到 [B, Lp, Ll, D]
        PP = P.expand(-1, -1, L.size(2), -1)
        LL = L.expand(-1, P.size(1), -1, -1)
        feats = torch.cat([PP, LL, torch.abs(PP - LL), PP * LL], dim=-1)  # [B,Lp,Ll,4D]
        return feats

    def forward(self, prot_emb: torch.Tensor, pep_emb: torch.Tensor,
                chunk_ll: int = 0) -> torch.Tensor:
        """
        返回：
          - out_channels == 1: scores:[B, Lp, Ll]
          - out_channels > 1 : logits:[B, Lp, Ll, C]
        可通过 chunk_ll 沿 Ll 方向分块降低显存。
        """
        B, Lp, D = prot_emb.shape
        Ll = pep_emb.size(1)
        P = prot_emb[:, :, None, :]    # [B,Lp,1,D]

        if chunk_ll and chunk_ll < Ll:
            if self.out_channels == 1:
                scores = prot_emb.new_zeros(B, Lp, Ll)
            else:
                scores = prot_emb.new_zeros(B, Lp, Ll, self.out_channels)
            for j0 in range(0, Ll, chunk_ll):
                j1 = min(j0 + chunk_ll, Ll)
                L = pep_emb[:, j0:j1, :]      # [B,chunk,D]
                feats = self._make_pair_feats(P, L[:, None, :, :])    # [B,Lp,chunk,4D]
                out = self.mlp(feats.reshape(B * Lp * (j1 - j0), -1))
                if self.out_channels == 1:
                    out = out.reshape(B, Lp, j1 - j0, 1).squeeze(-1)
                    scores[:, :, j0:j1] = out
                else:
                    out = out.reshape(B, Lp, j1 - j0, self.out_channels)
                    scores[:, :, j0:j1, :] = out
            return scores
        else:
            feats = self._make_pair_feats(P, pep_emb[:, None, :, :])  # [B,Lp,Ll,4D]
            out = self.mlp(feats.reshape(B * Lp * Ll, -1))
            if self.out_channels == 1:
                return out.reshape(B, Lp, Ll, 1).squeeze(-1)  # [B,Lp,Ll]
            else:
                return out.reshape(B, Lp, Ll, self.out_channels)  # [B,Lp,Ll,C]


class PairBilinear(nn.Module):
    """
    仍然只输出标量打分：p_i^T W l_j + u^T p_i + v^T l_j + b
    （分桶多类请使用 MLP 头）
    """
    def __init__(self, d_model: int, d_proj: int = 256, bias: bool = True):
        super().__init__()
        self.proj_p = nn.Linear(d_model, d_proj, bias=False)
        self.proj_l = nn.Linear(d_model, d_proj, bias=False)
        self.u = nn.Linear(d_model, 1, bias=False)
        self.v = nn.Linear(d_model, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1)) if bias else None

    def forward(self, prot_emb: torch.Tensor, pep_emb: torch.Tensor) -> torch.Tensor:
        B, Lp, D = prot_emb.shape
        Ll = pep_emb.size(1)
        P = self.proj_p(prot_emb)              # [B,Lp,d]
        L = self.proj_l(pep_emb)               # [B,Ll,d]
        bilinear = torch.bmm(P, L.transpose(1, 2))     # [B,Lp,Ll]
        up = self.u(prot_emb).expand(-1, -1, Ll)       # [B,Lp,Ll]
        vl = self.v(pep_emb).transpose(1, 2).expand(-1, Lp, -1)  # [B,Lp,Ll]
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
    head: HeadType = "bilinear"     # "mlp" or "bilinear"
    hidden: int = 512               # for MLP head
    d_proj: int = 256               # for bilinear head
    dropout: float = 0.1            # for MLP head
    chunk_ll: int = 0               # chunk size for MLP (0=off)
    score_activation: Optional[Literal["softplus", "sigmoid"]] = None  # 仅 out_channels==1 时生效
    out_channels: int = 1           # =1: 标量（contact/distance）；>1: 分桶多类（C 类）

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
            self.head = PairMLP(cfg.d_model, cfg.hidden, cfg.dropout, out_channels=cfg.out_channels)
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

        out = self.head(P, L)  # [B,Lp,Ll] or [B,Lp,Ll,C]

        # 可选分数激活（仅标量输出时启用）
        if self.cfg.out_channels == 1:
            if self.cfg.score_activation == "softplus":
                out = F.softplus(out)
            elif self.cfg.score_activation == "sigmoid":
                out = torch.sigmoid(out)

        pair_mask = pm[:, :, None] & lm[:, None, :]
        return {"scores": out, "pair_mask": pair_mask}


# -----------------------------
# Losses
# -----------------------------

def positive_margin_loss(scores: torch.Tensor,
                         labels: torch.Tensor,
                         mask: Optional[torch.Tensor] = None,
                         m_pos: float = 1.0,
                         m_neg: float = 0.0,
                         pos_weight: Optional[float] = None) -> torch.Tensor:
    """
    Positive-Margin hinge:
      y=1: max(0, m_pos - s)
      y=0: max(0, s - m_neg)
    支持 mask & 类别不均衡权重。
    """
    assert set(torch.unique(labels).tolist()) <= {0, 1}, "labels must be 0/1"
    y = labels.float()
    s = scores

    pos_term = F.relu(m_pos - s) * y
    neg_term = F.relu(s - m_neg) * (1.0 - y)

    loss = pos_term + neg_term
    if pos_weight is not None:
        # 放大正样本的损失
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
    """
    BCE with logits（建议 scores 是原始分数，不做 sigmoid；本函数内部做）。
    """
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
    """
    距离回归的 Huber 损失（pred_dist/true_dist 以 Å 为单位）。
    """
    loss = F.huber_loss(pred_dist, true_dist, delta=delta, reduction="none")
    if mask is not None:
        loss = loss * mask.float()
        return loss.sum() / mask.float().sum().clamp_min(1.0)
    return loss.mean()


# -----------------------------
# Criterion 封装（示例）
# -----------------------------
class PairwiseCriterion(nn.Module):
    def __init__(self,
                 use_bce_for_contact: bool = True,
                 # hinge 仍保留可选
                 m_pos: float = 1.0, m_neg: float = 0.0,
                 # BCE 相关
                 pos_weight: Optional[float] = None,
                 max_pos_weight: float = 50.0,  # 动态权重上限
                 focal_gamma: float = 0.0,      # >0 开启 Focal，常用 1~2
                 dice_lambda: float = 0.0,      # >0 叠加 Dice，常用 0.5
                 # 负样本采样
                 neg_pos_ratio: Optional[float] = None,  # 例如 5.0 表示每个 batch 负:正=5:1
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
        """按需要对子采样负样本：保留全部正样本 + 采样部分负样本。"""
        if self.neg_pos_ratio is None:
            return mask
        yb = (y > 0.5)
        pos_m = (mask & yb)
        neg_m = (mask & (~yb))

        n_pos = int(pos_m.sum().item())
        if n_pos == 0:
            return mask  # 没正样本就不采样，避免全空

        target_neg = int(min(neg_m.sum().item(), max(1, n_pos * self.neg_pos_ratio)))

        # 随机采样指定数量的负样本
        idx_neg = torch.nonzero(neg_m, as_tuple=False)
        if idx_neg.numel() == 0:
            return pos_m  # 极端情况：没有负样本
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
            return 1.0  # 避免除零，一般这个 batch 就跳过动态放大
        w = n_neg / n_pos
        return float(min(self.max_pos_weight, max(1.0, w)))

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        scores = outputs["scores"]              # [B,Lp,Ll] 或 [B,Ll,Lp]（外层已转成 [B,Tl,Tp]）
        pair_mask = outputs["pair_mask"].bool()

        losses = {}
        # ---- contact 分支（默认）----
        if "labels_contact" in batch:
            y = batch["labels_contact"].to(scores.dtype)

            eff_mask = self._make_effective_mask(y, pair_mask)

            if self.use_bce_for_contact:
                # 动态/静态 pos_weight
                pw = self.pos_weight
                if pw is None:
                    pw = self._dynamic_pos_weight(y, eff_mask)
                pw_t = torch.tensor(pw, device=scores.device, dtype=scores.dtype)

                # BCE-with-logits
                loss_bce = F.binary_cross_entropy_with_logits(
                    scores, y, pos_weight=pw_t, reduction="none"
                )

                # Focal（可选）
                if self.focal_gamma and self.focal_gamma > 0.0:
                    p = torch.sigmoid(scores)
                    pt = torch.where(y > 0.5, p, 1.0 - p)
                    loss_bce = loss_bce * ((1.0 - pt).clamp_min(1e-6) ** self.focal_gamma)

                # 掩码平均
                loss = (loss_bce * eff_mask.float()).sum() / eff_mask.float().sum().clamp_min(1.0)

                # Dice（可选）
                if self.dice_lambda and self.dice_lambda > 0.0:
                    prob = torch.sigmoid(scores)
                    inter = (prob * y * eff_mask).sum()
                    denom = (prob * eff_mask).sum() + (y * eff_mask).sum() + 1e-8
                    loss_dice = 1.0 - (2.0 * inter / denom)
                    loss = loss + self.dice_lambda * loss_dice

                losses["loss"] = loss

                # 监控
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
                # 需要用 hinge 的话也加采样和权重（不如 BCE 稳）
                yb = (y > 0.5).float()
                pos_term = F.relu(self.m_pos - scores) * yb
                neg_term = F.relu(scores - self.m_neg) * (1.0 - yb)
                loss_h = (pos_term + neg_term)
                loss = (loss_h * eff_mask.float()).sum() / eff_mask.float().sum().clamp_min(1.0)
                losses["loss"] = loss

        # ---- distance 分支：保留（你回到 contact 就不会走这里）----
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
    # 假数据自检
    B, Lp, Ll, D = 2, 20, 12, 1536
    prot_emb = torch.randn(B, Lp, D)
    pep_emb  = torch.randn(B, Ll, D)
    prot_mask = torch.ones(B, Lp, dtype=torch.bool)
    pep_mask  = torch.ones(B, Ll, dtype=torch.bool)

    cfg = PairModelConfig(d_model=D, head="bilinear", d_proj=256, score_activation=None)
    model = PairwiseModel(cfg)

    batch = {"prot_emb": prot_emb, "pep_emb": pep_emb, "prot_mask": prot_mask, "pep_mask": pep_mask}
    out = model(batch)
    print("scores:", out["scores"].shape, "| pair_mask:", out["pair_mask"].shape)

    # 伪造 contact 标签
    labels = (torch.rand(B, Lp, Ll) < 0.1).long()
    batch["labels_contact"] = labels

    crit = PairwiseCriterion(use_bce_for_contact=False, m_pos=0.8, m_neg=0.5, pos_weight=3.0)#customed
    loss_dict = crit(out, batch)
    print({k: (float(v) if isinstance(v, torch.Tensor) else v) for k, v in loss_dict.items()})
