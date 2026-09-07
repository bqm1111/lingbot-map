"""Losses for the depth-refinement head. Sparse LiDAR supervision only."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def log_smooth_l1(refined: torch.Tensor, lidar: torch.Tensor, valid: torch.Tensor,
                  weight: Optional[torch.Tensor] = None, beta: float = 0.1) -> torch.Tensor:
    """SmoothL1 between log depths at valid LiDAR pixels -- the primary objective."""
    if valid.sum() == 0:
        return refined.sum() * 0.0
    a = torch.log(refined[valid].clamp_min(1e-6))
    b = torch.log(lidar[valid].clamp_min(1e-6))
    l = F.smooth_l1_loss(a, b, beta=beta, reduction="none")
    if weight is None:
        return l.mean()
    w = weight[valid].clamp_min(0)
    return (l * w).sum() / w.sum().clamp_min(1e-6)


def relative_loss(refined: torch.Tensor, lidar: torch.Tensor, valid: torch.Tensor,
                  weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean absolute relative error -- ablated separately, never mixed with the primary."""
    if valid.sum() == 0:
        return refined.sum() * 0.0
    l = (refined[valid] - lidar[valid]).abs() / lidar[valid].clamp_min(1e-6)
    if weight is None:
        return l.mean()
    w = weight[valid].clamp_min(0)
    return (l * w).sum() / w.sum().clamp_min(1e-6)


def residual_regularizer(r: torch.Tensor) -> torch.Tensor:
    """Keep the correction small unless the data demands otherwise."""
    return r.abs().mean()


def sparse_gradient_loss(refined: torch.Tensor, lidar: torch.Tensor,
                         valid: torch.Tensor) -> torch.Tensor:
    """Optional: match log-depth differences where *both* neighbours have LiDAR."""
    lr = torch.log(refined.clamp_min(1e-6))
    lg = torch.log(lidar.clamp_min(1e-6))
    total, n = refined.sum() * 0.0, 0
    for d in (1, 2):                                    # right and down neighbours
        a = [slice(None)] * 4
        b = [slice(None)] * 4
        a[d] = slice(None, -1); b[d] = slice(1, None)
        m = valid[tuple(a)] & valid[tuple(b)]
        if m.any():
            total = total + ((lr[tuple(a)] - lr[tuple(b)])[m]
                             - (lg[tuple(a)] - lg[tuple(b)])[m]).abs().mean()
            n += 1
    return total / max(n, 1)


def compute_loss(cfg, refined: torch.Tensor, r: torch.Tensor, batch: Dict,
                 ) -> Dict[str, torch.Tensor]:
    lidar = batch["projected_lidar_depth"]
    valid = batch["projected_lidar_valid_mask"]
    w = batch["lingbot_confidence"] if cfg.loss.confidence_weighted else None
    if cfg.loss.primary == "relative":
        depth_term = relative_loss(refined, lidar, valid, w)
    else:
        depth_term = log_smooth_l1(refined, lidar, valid, w, cfg.loss.smooth_l1_beta)
    reg = residual_regularizer(r)
    return {"loss": depth_term + cfg.loss.lambda_residual * reg,
            "depth": depth_term, "residual": reg}
