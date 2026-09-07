# extracted from gate8/losses.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
"""Imbalance-aware occupancy losses and teacher distillation, all masked explicitly."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_bce(logits, target, weight, gamma: float = 2.0, pos_weight: float = 1.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none",
        pos_weight=torch.as_tensor(pos_weight, device=logits.device))
    pt = torch.where(target > 0.5, p, 1 - p)
    loss = ((1 - pt) ** gamma) * ce * weight
    return loss.sum() / weight.sum().clamp_min(1.0)


def soft_dice(logits, target, mask, eps: float = 1.0):
    """IoU-oriented: soft Dice over the supervised voxels."""
    p = torch.sigmoid(logits) * mask
    t = target * mask
    inter = (p * t).sum()
    return 1.0 - (2 * inter + eps) / (p.sum() + t.sum() + eps)


def semantic_kl(sem_logits, target_p, mask):
    """KL(teacher || student) on voxels with a valid frozen-teacher target."""
    logq = F.log_softmax(sem_logits, dim=1)                       # [B, U, X, Y, Z]
    t = target_p.clamp_min(1e-6)
    kl = (t * (t.log() - logq)).sum(dim=1)                        # [B, X, Y, Z]
    return (kl * mask).sum() / mask.sum().clamp_min(1.0)


def balanced_weights(gt_occ, gt_valid, observed, w_unknown: float = 2.0):
    """Per-voxel weights: valid supervision only; unknown/frontier voxels up-weighted."""
    w = gt_valid.float()
    w = w * torch.where(observed, torch.ones_like(w), torch.full_like(w, w_unknown))
    return w


__all__ = ["focal_bce", "soft_dice", "semantic_kl", "balanced_weights"]
