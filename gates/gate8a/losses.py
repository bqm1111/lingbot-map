"""The unweighted-BCE arm of the Gate 8A ablation.

Gate 8's occupancy objective was focal BCE (gamma 2) with a per-batch positive weight of
up to 8x, a 2x up-weight on unobserved voxels and a soft-Dice term, applied over every
valid voxel. Each of those pushes the decision boundary toward predicting occupied, and
Gate 8 then read the prediction off at a fixed threshold of 0. This arm removes all four
and keeps nothing but a plain per-voxel BCE, restricted to the voxels the residual can
actually move -- so the network is asked for a *calibrated* posterior over exactly its own
action space, and nothing else.

The teacher-KL term is untouched: it is not part of the ablation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_editable(final_logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                 editable: torch.Tensor) -> torch.Tensor:
    """Mean ``BCEWithLogitsLoss`` over ``valid & editable``. No weighting of any kind."""
    m = (valid & editable).to(final_logits.dtype)
    ce = F.binary_cross_entropy_with_logits(final_logits, target, reduction="none")
    return (ce * m).sum() / m.sum().clamp_min(1.0)


__all__ = ["bce_editable"]
