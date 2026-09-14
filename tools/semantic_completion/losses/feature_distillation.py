"""Cosine distillation onto the causal teacher evidence -- the only semantic loss of Gate 15.

Supervised voxels: exactly the evidence voxels of the crop (direct causal surface observations).
Every other voxel -- free, completed, unobserved, future -- contributes nothing. Among the supervised
voxels, those whose evidence was hidden from the input are reported separately because only they can
tell copying from propagation.
"""
from __future__ import annotations

from typing import Dict

import torch


def cosine_distillation(pred: torch.Tensor, teacher: torch.Tensor, supervised: torch.Tensor, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
    """``pred``/``teacher``: ``[N, 1024]`` unit norm at voxels selected by ``supervised`` (already gathered);
    ``hidden``: ``[N]`` bool, evidence removed from the input. Returns the mean loss and its two parts."""
    assert pred.shape == teacher.shape and supervised.dtype == torch.bool
    loss_per = 1.0 - (pred * teacher).sum(-1)
    out = {"loss": loss_per.mean() if loss_per.numel() else pred.sum() * 0.0}
    with torch.no_grad():
        out["loss_visible"] = loss_per[~hidden].mean() if (~hidden).any() else torch.tensor(float("nan"))
        out["loss_hidden"] = loss_per[hidden].mean() if hidden.any() else torch.tensor(float("nan"))
    return out


__all__ = ["cosine_distillation"]
