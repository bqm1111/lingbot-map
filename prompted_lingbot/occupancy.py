"""Benchmark-compatible binary occupancy: voxel grids, voxelization, metrics.

Every constant and rule here is copied from the OccAny reference implementation
at ``/home/minh/workspace/OccAny`` -- the source this experiment must reproduce --
and cross-checked against ``semantic/eval_ssc.py`` where both define the same
thing.  Nothing is guessed.  See ``docs/lingbot_occupancy_feasibility.md`` §2 for
the file/line citations.

Two corrections to common assumptions, both verified:

* **There is no trilinear splatting in OccAny.**  ``grep -rn "trilinear\\|splat"``
  returns nothing outside ``third_party``.  The only point->voxel routine is
  ``occany/utils/ray.py:217``, which floor-bins.  That is what is implemented
  here; calling anything else "OccAny-compatible" would be a fabrication.
* **Occ3D's empty class is 17**, not 0, and 255 is the ignore label.  SemanticKITTI
  uses 0 for empty and 255 for invalid.  They are not interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Grids
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VoxelGrid:
    """An axis-aligned voxel grid in a named reference frame."""

    dims: Tuple[int, int, int]
    voxel_size: float
    origin: Tuple[float, float, float]
    frame: str
    empty_class: int
    ignore_label: int = 255
    name: str = ""

    @property
    def scene_size(self) -> Tuple[float, float, float]:
        return tuple(float(d * self.voxel_size) for d in self.dims)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(self.origin, float) + np.asarray(self.scene_size, float)

    def voxel_centres(self, index: np.ndarray) -> np.ndarray:
        """``(N, 3)`` integer indices -> world-frame centres."""
        return (np.asarray(index, float) + 0.5) * self.voxel_size + np.asarray(self.origin, float)


# OccAny occany/datasets/kitti.py:301-310; identical in semantic/eval_ssc.py:51-53.
SEMANTICKITTI_GRID = VoxelGrid(
    dims=(256, 256, 32), voxel_size=0.2, origin=(0.0, -25.6, -2.0),
    frame="velodyne_of_anchor_frame", empty_class=0, ignore_label=255,
    name="semantickitti",
)

# OccAny occany/datasets/nuscenes.py:284-294.
OCC3D_NUSCENES_GRID = VoxelGrid(
    dims=(200, 200, 16), voxel_size=0.4, origin=(-40.0, -40.0, -1.0),
    frame="ego_of_reference_sample", empty_class=17, ignore_label=255,
    name="occ3d_nuscenes",
)

GRIDS = {g.name: g for g in (SEMANTICKITTI_GRID, OCC3D_NUSCENES_GRID)}

# SemanticKITTI SSC is evaluated on every 5th frame (OccAny frame_interval=5).
SEMANTICKITTI_EVAL_STRIDE = 5

# Occ3D single-camera (CAM_FRONT) setting zeroes the rear half of the grid:
# OccAny occany/datasets/nuscenes.py:903-905, `voxel_label[:100, :, :] = 255`.
OCC3D_SINGLE_CAMERA_X_CUT = 100


# --------------------------------------------------------------------------- #
# Voxelization
# --------------------------------------------------------------------------- #
def voxelize_points(points: np.ndarray, grid: VoxelGrid):
    """Floor-bin ``(N, 3)`` points into voxel indices.

    Reproduces ``OccAny/occany/utils/ray.py:217`` exactly::

        idx = floor((p - origin) / voxel_size)

    Returns ``(index [M, 3] int32, keep [N] bool)`` where ``keep`` marks the
    points that landed inside the grid.
    """
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {p.shape}")
    idx = np.floor((p - np.asarray(grid.origin, float)) / grid.voxel_size).astype(np.int32)
    keep = np.ones(len(idx), bool)
    for a in range(3):
        keep &= (idx[:, a] >= 0) & (idx[:, a] < grid.dims[a])
    return idx[keep], keep


def occupancy_from_points(points: np.ndarray, grid: VoxelGrid,
                          min_points_per_voxel: int = 1) -> np.ndarray:
    """``(N, 3)`` points -> boolean occupancy volume of shape ``grid.dims``.

    ``min_points_per_voxel`` raises the bar for calling a voxel occupied; with the
    default of 1 a single point suffices, which is the OccAny behaviour.
    Duplicate points in one voxel are aggregated, never double-counted.
    """
    vol = np.zeros(grid.dims, dtype=bool)
    idx, _ = voxelize_points(points, grid)
    if idx.size == 0:
        return vol
    if min_points_per_voxel <= 1:
        vol[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return vol
    flat = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), grid.dims)
    counts = np.bincount(flat, minlength=int(np.prod(grid.dims)))
    vol.reshape(-1)[counts >= min_points_per_voxel] = True
    return vol


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def binary_occupancy_scores(pred_occupied: np.ndarray, target: np.ndarray,
                            grid: VoxelGrid,
                            valid: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Binary occupancy precision / recall / IoU.

    Reproduces ``OccAny/occany/metrics/ssc.py:153-190`` (``get_score_completion``):
    voxels whose target is ``ignore_label`` are removed from both sides, the rest
    are binarised as ``!= empty_class``, and tp/fp/fn are counted over what is left.

    Args:
        pred_occupied: bool volume, ``grid.dims``.
        target: integer label volume, ``grid.dims``.
        valid: optional extra mask of voxels to evaluate (e.g. Occ3D's camera mask).
    """
    if pred_occupied.shape != tuple(grid.dims) or target.shape != tuple(grid.dims):
        raise ValueError(f"shape mismatch: pred {pred_occupied.shape}, "
                         f"target {target.shape}, grid {grid.dims}")
    t = np.asarray(target)
    keep = t != grid.ignore_label
    if valid is not None:
        keep &= np.asarray(valid, bool)

    y_true = (t != grid.empty_class) & keep
    y_pred = np.asarray(pred_occupied, bool) & keep

    tp = int(np.count_nonzero(y_true & y_pred))
    fp = int(np.count_nonzero(~y_true & y_pred))
    fn = int(np.count_nonzero(y_true & ~y_pred))
    denom = tp + fp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "iou": float(tp / denom) if denom else 0.0,
        "n_pred_occupied": int(np.count_nonzero(y_pred)),
        "n_gt_occupied": int(np.count_nonzero(y_true)),
        "n_valid_voxels": int(np.count_nonzero(keep)),
    }


def accumulate_scores(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Pool tp/fp/fn across frames, the way the benchmark aggregates."""
    tp = sum(int(r["tp"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)
    denom = tp + fp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "iou": float(tp / denom) if denom else 0.0,
        "n_frames": len(rows),
        "n_pred_occupied": sum(int(r["n_pred_occupied"]) for r in rows),
        "n_gt_occupied": sum(int(r["n_gt_occupied"]) for r in rows),
        "n_valid_voxels": sum(int(r["n_valid_voxels"]) for r in rows),
    }
