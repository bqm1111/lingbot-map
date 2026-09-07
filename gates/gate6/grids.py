"""One place where each benchmark's prediction grid, frame transform and target live.

The prediction grid is always the *frozen* one of the reproduction gate: SemanticKITTI and
SSCBench-KITTI-360 predict directly on their native 0.2 m volume; Occ3D-nuScenes predicts
on the frozen canonical 0.2 m grid and is reduced to the official 0.4 m grid afterwards by
the frozen "any subvoxel occupied" rule.

Target loaders live here too, but they are only ever called by the evaluator -- the
prediction stage runs inside :class:`gate6.audit.Gate6Audit`, which raises if any of them
is reached.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

from prompted_lingbot.occupancy import SEMANTICKITTI_GRID
from occ3d_zeroshot.grid import CANONICAL as OCC3D_CANONICAL, NATIVE as OCC3D_NATIVE, RATIO
from sscbench_kitti360.adapter import SSCBENCH_KITTI360_GRID

PREDICTION_GRID = {
    "semantickitti": SEMANTICKITTI_GRID,
    "kitti360": SSCBENCH_KITTI360_GRID,
    "occ3d": OCC3D_CANONICAL,
}
EVAL_GRID = {
    "semantickitti": SEMANTICKITTI_GRID,
    "kitti360": SSCBENCH_KITTI360_GRID,
    "occ3d": OCC3D_NATIVE,
}
NEEDS_REDUCTION = {"semantickitti": False, "kitti360": False, "occ3d": True}


def voxelize(points: np.ndarray, grid) -> Tuple[np.ndarray, np.ndarray]:
    """``floor((p - origin) / voxel_size)`` with an in-grid mask.

    Deliberately the same three lines as ``prompted_lingbot.occupancy.voxelize_points`` and
    ``occ3d_zeroshot.grid.points_to_canonical``; ``tests/gate6`` asserts equality with both
    rather than trusting that they stayed in step.
    """
    p = np.asarray(points, float)
    idx = np.floor((p - np.asarray(grid.origin, float)) / grid.voxel_size).astype(np.int64)
    keep = np.ones(len(idx), bool)
    for a in range(3):
        keep &= (idx[:, a] >= 0) & (idx[:, a] < grid.dims[a])
    return idx[keep], keep


def flat_of(idx: np.ndarray, grid) -> np.ndarray:
    return np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), grid.dims)


def unflat(flat: np.ndarray, grid) -> np.ndarray:
    return np.stack(np.unravel_index(flat, grid.dims), axis=-1)


def reduce_occ3d_semantics(flat_canon: np.ndarray, labels: np.ndarray):
    """Canonical 0.2 m labelled voxels -> native 0.4 m labelled voxels.

    Occupancy follows the frozen rule (a native voxel is occupied iff **any** of its eight
    subvoxels is). The label is decided by the accompanying probability reduction in
    :func:`reduce_occ3d_probs`; this helper exists for the occupancy side alone.
    """
    idx = unflat(flat_canon, OCC3D_CANONICAL) // RATIO
    return np.unique(np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]),
                                          OCC3D_NATIVE.dims))


def reduce_occ3d_probs(flat_canon: np.ndarray, probs):
    """Mean probability vector of a native voxel's **occupied** subvoxels.

    The natural extension of the frozen any-subvoxel occupancy rule and of the uniform
    voxel-fusion rule: an occupied native voxel averages exactly the subvoxels that made
    it occupied, and empty subvoxels contribute nothing.
    """
    import torch
    idx = unflat(flat_canon, OCC3D_CANONICAL) // RATIO
    nat = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), OCC3D_NATIVE.dims)
    uniq, inv = np.unique(nat, return_inverse=True)
    ii = torch.from_numpy(inv.astype(np.int64)).to(probs.device)
    acc = torch.zeros(len(uniq), probs.shape[1], dtype=torch.float32, device=probs.device)
    acc.index_add_(0, ii, probs.float())
    cnt = torch.zeros(len(uniq), dtype=torch.float32, device=probs.device)
    cnt.index_add_(0, ii, torch.ones_like(ii, dtype=torch.float32))
    return uniq.astype(np.int64), acc / cnt.unsqueeze(1)


__all__ = ["PREDICTION_GRID", "EVAL_GRID", "NEEDS_REDUCTION", "voxelize", "flat_of",
           "unflat", "reduce_occ3d_semantics", "reduce_occ3d_probs", "RATIO"]
