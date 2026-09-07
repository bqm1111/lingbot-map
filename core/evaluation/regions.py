# extracted from gate8a/regions.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
"""The two evaluation regions, and the reduction that carries them to the eval grid.

Region 1 is the complete valid evaluation grid (the benchmark's own ``keep`` mask).
Region 2 is the **editable completion region**: the voxels the occupancy residual is
allowed to move. That is not a new definition -- it is read straight off the rule the
network is trained and evaluated under, :func:`gate8.net.apply_residual`, whose gate is
``abs(base_logodds) < LOCK_LOGODDS``. :func:`editable_native` is that predicate and
nothing else, so a change to the residual rule changes the region automatically.

Occ3D predicts at 0.2 m and is scored at 0.4 m, and the frozen reduction is
*any sub-voxel occupied* (``gate6.grids.reduce_occ3d_probs``). Scores therefore reduce by
**max** over the eight children, a coarse voxel is editable if **any** child is editable,
and it is *forced occupied* if any child is locked above the occupancy threshold -- the
residual cannot take that occupancy away, so no baseline may either.
"""

from __future__ import annotations

from typing import Tuple

import torch

from core.datasets import grids as G6G
from core.model.net import LOCK_LOGODDS


def editable_native(base_logodds: torch.Tensor, lock: float = LOCK_LOGODDS) -> torch.Tensor:
    """Exactly the gate of :func:`gate8.net.apply_residual`, as a boolean mask."""
    return base_logodds.abs() < lock


def locked_occupied_native(base_logodds: torch.Tensor, lock: float = LOCK_LOGODDS,
                           occupied_at: float = 0.0) -> torch.Tensor:
    """Voxels whose occupancy the residual cannot remove."""
    return (base_logodds >= lock) & (base_logodds > occupied_at)


def _blocks(x: torch.Tensor, dims) -> torch.Tensor:
    """[X,Y,Z] -> [X/r, Y/r, Z/r, r**3] using the frozen Occ3D reduction ratio."""
    r = G6G.RATIO
    X, Y, Z = (int(d) for d in dims)
    return (x.reshape(X // r, r, Y // r, r, Z // r, r)
             .permute(0, 2, 4, 1, 3, 5).reshape(X // r, Y // r, Z // r, r ** 3))


def to_eval_grid(score: torch.Tensor, base_logodds: torch.Tensor, dataset: str,
                 lock: float = LOCK_LOGODDS) -> Tuple[torch.Tensor, torch.Tensor,
                                                      torch.Tensor]:
    """``(score, editable, forced_occupied)`` flattened on the *evaluation* grid.

    For SemanticKITTI and KITTI-360 the prediction and evaluation grids coincide and this
    is the identity. For Occ3D it applies the frozen 2x reduction.
    """
    pred_dims = G6G.PREDICTION_GRID[dataset].dims
    edit = editable_native(base_logodds, lock)
    forced = locked_occupied_native(base_logodds, lock)
    if not G6G.NEEDS_REDUCTION[dataset]:
        return score.reshape(-1), edit.reshape(-1), forced.reshape(-1)
    s = _blocks(score.reshape(pred_dims), pred_dims).amax(-1)
    e = _blocks(edit.reshape(pred_dims), pred_dims).any(-1)
    f = _blocks(forced.reshape(pred_dims), pred_dims).any(-1)
    return s.reshape(-1), e.reshape(-1), f.reshape(-1)


def occ_to_eval_grid(occ: torch.Tensor, dataset: str) -> torch.Tensor:
    """Boolean occupancy reduced with the frozen any-sub-voxel rule."""
    if not G6G.NEEDS_REDUCTION[dataset]:
        return occ.reshape(-1)
    dims = G6G.PREDICTION_GRID[dataset].dims
    return _blocks(occ.reshape(dims), dims).any(-1).reshape(-1)


__all__ = ["editable_native", "locked_occupied_native", "to_eval_grid", "occ_to_eval_grid",
           "LOCK_LOGODDS"]
