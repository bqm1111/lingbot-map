"""OccAny's 3x3x3 "separate" majority pooling, reproduced from the official code.

Source of truth: ``/home/minh/workspace/OccAny/occany/utils/helpers.py::apply_majority_pooling``
as called by ``compute_metrics_from_saved_voxels.py`` with ``--pooling_mode separate``.
Two facts about that function matter for an honest comparison and are easy to get wrong:

1. In **geometry-only** mode its default is ``use_dilation=True``, which is *not* a
   majority vote: it is ``max_pool3d(kernel 3, stride 1, padding 1)`` on the binary
   occupancy -- a one-voxel (0.2 m) Chebyshev dilation. The true majority vote
   (``use_dilation=False``: occupied if more than 13.5 of the 27 neighbours are occupied,
   never removing occupancy) is a different operator and is reproduced separately.
2. In **semantic** mode (``is_geometry_only=False``) the operator relabels occupied voxels
   by neighbourhood vote and **never changes occupancy**, so SC IoU is untouched by it.

Both are reproduced here exactly, without tuning, and ``tests/gate8b`` checks them against
the official function running in the OccAny environment.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def geometry_dilation(occ: torch.Tensor) -> torch.Tensor:
    """OccAny ``is_geometry_only=True, use_dilation=True`` (the default): 3x3x3 max-pool."""
    x = occ.to(torch.float32)[None, None]
    return (F.max_pool3d(x, kernel_size=3, stride=1, padding=1)[0, 0] > 0.0)


def geometry_majority(occ: torch.Tensor) -> torch.Tensor:
    """OccAny ``is_geometry_only=True, use_dilation=False``: occupied if > 13.5 of the 27
    neighbours (self included) are occupied; existing occupancy is never removed."""
    x = occ.to(torch.float32)[None, None]
    count = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)[0, 0] * 27.0
    return occ.bool() | (count > 13.5)


def semantic_separate(labels: torch.Tensor, n_classes: int, other_class: int,
                      empty_class: int, update_other_only: bool = False) -> torch.Tensor:
    """OccAny semantic-mode separate pooling: relabel occupied voxels by the argmax of a
    3x3x3 vote in which only real classes (not empty, not other) vote. Occupancy is
    unchanged by construction."""
    lab = labels.long()
    if update_other_only:
        to_update = lab == other_class
    else:
        to_update = lab != empty_class
    onehot = torch.zeros((n_classes,) + tuple(lab.shape), dtype=torch.float32,
                         device=lab.device)
    for c in range(n_classes):
        if c == other_class or c == empty_class:
            continue
        onehot[c] = (lab == c).float()
    pooled = F.avg_pool3d(onehot[None], kernel_size=3, stride=1, padding=1)[0]
    has = pooled.sum(dim=0) > 0
    new = pooled.argmax(dim=0)
    out = lab.clone()
    m = to_update & has
    out[m] = new[m]
    return out


__all__ = ["geometry_dilation", "geometry_majority", "semantic_separate"]
