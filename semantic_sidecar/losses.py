"""Training objectives for the semantic sidecar.

Four configurable terms, each logged independently:

* ``L_pixel``  = 1 − cos(ŷ, y_t(u))                       — plain 2D distillation
* ``L_cons``   = 1 − cos(ŷ, ȳ_x)                          — 3D-consolidated target
* ``L_mv``     = 1 − cos(ŷ, stopgrad(track mean of ŷ))    — cross-view consistency
* ``L_rel``    = ‖ŶŶᵀ − ȲȲᵀ‖₁ / n²                        — relational preservation

``L_mv`` compares each prediction to its track's predicted mean instead of
enumerating pairs, so cost is linear in the number of observations even for tracks
with hundreds of views.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from semantic_sidecar.config import LossConfig


def cosine_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``1 - cos`` averaged over the masked entries; 0 when nothing is valid."""
    cos = (F.normalize(pred, dim=-1) * F.normalize(target, dim=-1)).sum(-1)
    loss = 1.0 - cos
    if mask is None:
        return loss.mean()
    mask = mask.to(loss.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def centered_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mean_direction: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``1 - cos`` after projecting out a fixed mean direction.

    CLIP-space features are strongly anisotropic: on the training targets used here
    the mean cosine to the corpus mean direction is ~0.87, so a model that predicts
    only that mean direction already scores ~0.9 on the plain cosine while carrying
    almost no class information.  Removing the mean makes the *residual* structure —
    the part that decides which class a voxel is — the thing being optimised.

    Args:
        mean_direction: ``[d]`` unit vector, estimated on training targets only.
    """
    mu = F.normalize(mean_direction.to(pred.device, pred.dtype), dim=-1)
    def _residual(x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        return F.normalize(x - (x @ mu).unsqueeze(-1) * mu, dim=-1)

    cos = (_residual(pred) * _residual(target)).sum(-1)
    loss = 1.0 - cos
    if mask is None:
        return loss.mean()
    mask = mask.to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def multiview_consistency_loss(
    pred: torch.Tensor, track_id: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Pull every observation of a track towards that track's own predicted mean.

    Args:
        pred: ``[N, d]`` predictions (already normalised or not).
        track_id: ``[N]`` non-negative track ids; ``-1`` marks "no track".
        mask: Optional ``[N]`` bool of usable rows.

    The mean is detached, so the loss shrinks the spread without a degenerate
    incentive to collapse every track to the same vector.
    """
    valid = track_id >= 0
    if mask is not None:
        valid = valid & mask.bool()
    if not bool(valid.any()):
        return pred.sum() * 0.0

    p = F.normalize(pred, dim=-1)
    ids = track_id[valid]
    uniq, inverse = torch.unique(ids, return_inverse=True)
    sums = torch.zeros(uniq.numel(), p.shape[-1], device=p.device, dtype=p.dtype)
    sums.index_add_(0, inverse, p[valid])
    counts = torch.zeros(uniq.numel(), device=p.device, dtype=p.dtype)
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=p.dtype))

    # Only tracks with >= 2 observations in the batch carry any signal.
    multi = counts >= 2
    if not bool(multi.any()):
        return pred.sum() * 0.0
    means = F.normalize(sums / counts.unsqueeze(1), dim=-1).detach()
    keep = multi[inverse]
    cos = (p[valid][keep] * means[inverse][keep]).sum(-1)
    return (1.0 - cos).mean()


def relational_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    group_size: int = 256,
    num_groups: int = 4,
    generator: Optional[torch.Generator] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L1 between predicted and target Gram matrices on randomly sampled groups.

    Computing the full ``N x N`` Gram is quadratic, so a few small groups are sampled
    per batch; over many steps this covers the same structure at linear cost.
    """
    if mask is not None:
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idx.numel() < 2:
            return pred.sum() * 0.0
        pred, target = pred[idx], target[idx]
    n = pred.shape[0]
    if n < 2:
        return pred.sum() * 0.0

    p = F.normalize(pred, dim=-1)
    t = F.normalize(target, dim=-1)
    total = pred.sum() * 0.0
    for _ in range(max(num_groups, 1)):
        k = min(group_size, n)
        sel = torch.randperm(n, device=pred.device, generator=generator)[:k]
        gp = p[sel] @ p[sel].T
        gt = t[sel] @ t[sel].T
        total = total + (gp - gt).abs().mean()
    return total / max(num_groups, 1)


class SidecarLoss:
    """Weighted sum of the configured terms, returning a per-term breakdown."""

    def __init__(self, cfg: LossConfig, mean_direction: Optional[torch.Tensor] = None) -> None:
        self.cfg = cfg
        self.mean_direction = mean_direction
        if cfg.lambda_center > 0 and mean_direction is None:
            raise ValueError("lambda_center > 0 requires a mean_direction estimated on training targets")

    def __call__(
        self,
        pred: torch.Tensor,
        pixel_target: Optional[torch.Tensor] = None,
        pixel_mask: Optional[torch.Tensor] = None,
        consensus_target: Optional[torch.Tensor] = None,
        consensus_mask: Optional[torch.Tensor] = None,
        track_id: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Args:
            pred: ``[N, d]`` sidecar output.
            pixel_target / pixel_mask: per-observation teacher feature and validity.
            consensus_target / consensus_mask: 3D-consolidated target and validity.
            track_id: ``[N]`` track ids for the cross-view term.

        Returns:
            Dict with ``total`` plus one entry per active term.
        """
        cfg = self.cfg
        out: Dict[str, torch.Tensor] = {}
        total = pred.sum() * 0.0

        if cfg.lambda_pixel > 0:
            if pixel_target is None:
                raise ValueError("lambda_pixel > 0 but no pixel_target supplied")
            term = cosine_loss(pred, pixel_target, pixel_mask)
            out["pixel"] = term
            total = total + cfg.lambda_pixel * term

        if cfg.lambda_consensus > 0:
            if consensus_target is None:
                raise ValueError("lambda_consensus > 0 but no consensus_target supplied")
            term = cosine_loss(pred, consensus_target, consensus_mask)
            out["consensus"] = term
            total = total + cfg.lambda_consensus * term

        if cfg.lambda_center > 0:
            target = consensus_target if consensus_target is not None else pixel_target
            mask = consensus_mask if consensus_target is not None else pixel_mask
            if target is None:
                raise ValueError("lambda_center > 0 but no target supplied")
            term = centered_cosine_loss(pred, target, self.mean_direction, mask)
            out["center"] = term
            total = total + cfg.lambda_center * term

        if cfg.lambda_mv > 0:
            if track_id is None:
                raise ValueError("lambda_mv > 0 but no track_id supplied")
            term = multiview_consistency_loss(pred, track_id)
            out["mv"] = term
            total = total + cfg.lambda_mv * term

        if cfg.lambda_rel > 0:
            target = consensus_target if consensus_target is not None else pixel_target
            mask = consensus_mask if consensus_target is not None else pixel_mask
            if target is None:
                raise ValueError("lambda_rel > 0 but no target supplied")
            term = relational_loss(
                pred, target, cfg.rel_group_size, cfg.rel_groups_per_batch, generator, mask
            )
            out["rel"] = term
            total = total + cfg.lambda_rel * term

        out["total"] = total
        return out
