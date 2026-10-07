"""
Loss functions. All disease losses are MASK-AWARE: label entries that are
unknown (e.g. 'cataract' in RFMiD, which has no cataract column) contribute 0.

  bce   - plain BCEWithLogitsLoss
  wbce  - BCEWithLogitsLoss with pos_weight = (#neg / #pos) per class (clipped)
  focal - multi-label focal loss (Lin et al., 2017) on top of BCE
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pos_weight(train_df, codes, clip=(1.0, 30.0)) -> torch.Tensor:
    w = []
    for c in codes:
        col = train_df[c].dropna()
        pos, neg = (col == 1).sum(), (col == 0).sum()
        w.append(np.clip(neg / max(pos, 1), *clip))
    return torch.tensor(w, dtype=torch.float32)


class MaskedBCE(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits, target, mask):
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none",
                                                  pos_weight=self.pos_weight)
        return (loss * mask).sum() / mask.sum().clamp(min=1)


class MaskedFocal(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits, target, mask):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none",
                                                 pos_weight=self.pos_weight)
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        loss = (1 - p_t) ** self.gamma * bce
        return (loss * mask).sum() / mask.sum().clamp(min=1)


class RefractiveLoss(nn.Module):
    """SE loss (+ optional sphere/cyl/axis losses when multihead and labels exist)."""

    def __init__(self, kind="smoothl1", multihead=False, aux_weight=0.3):
        super().__init__()
        self.kind, self.multihead, self.aux_weight = kind, multihead, aux_weight

    def _base(self, a, b):
        return F.smooth_l1_loss(a, b, reduction="none") if self.kind == "smoothl1" else (a - b) ** 2

    def forward(self, out, target, mask):
        l = (self._base(out[:, 0], target[:, 0]) * mask[:, 0]).sum() / mask[:, 0].sum().clamp(min=1)
        if self.multihead:
            aux = (self._base(out[:, 1:], target[:, 1:]) * mask[:, 1:]).sum() / mask[:, 1:].sum().clamp(min=1)
            l = l + self.aux_weight * aux
        return l


def build_loss(cfg, train_df, codes):
    if cfg.task == "refractive":
        return RefractiveLoss("mse" if cfg.loss == "mse" else "smoothl1", cfg.multihead)
    pw = compute_pos_weight(train_df, codes) if cfg.loss in ("wbce", "focal") else None
    if cfg.loss == "focal":
        return MaskedFocal(cfg.focal_gamma, pw)
    return MaskedBCE(pw)
