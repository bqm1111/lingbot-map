"""Binary occupancy loss for the visible-voxel corrector.

Weighted BCE plus a soft Dice term, both restricted to the permitted supervision region
``S = correction_region AND evaluation-valid``. Occupied voxels are ~2 % of ``S``, so the
BCE carries a ``pos_weight`` resolved from the **source-training** class balance only.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def masked_bce(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               pos_weight: float) -> torch.Tensor:
    w = torch.as_tensor(pos_weight, dtype=logit.dtype, device=logit.device)
    per = F.binary_cross_entropy_with_logits(logit, target, pos_weight=w, reduction="none")
    n = mask.sum()
    return (per * mask).sum() / n.clamp_min(1.0)


def soft_dice(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
              eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logit) * mask
    t = target * mask
    inter = (p * t).sum()
    return 1.0 - (2.0 * inter + eps) / (p.sum() + t.sum() + eps)


def compute_loss(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 pos_weight: float, dice_weight: float) -> Dict[str, torch.Tensor]:
    bce = masked_bce(logit, target, mask, pos_weight)
    dice = soft_dice(logit, target, mask)
    return {"loss": bce + dice_weight * dice, "bce": bce, "dice": dice}
