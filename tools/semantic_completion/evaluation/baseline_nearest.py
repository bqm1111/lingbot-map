"""B0 -- the fixed trivial propagation baseline, frozen before any learned result exists.

Rule (one, predeclared, no parameter): every voxel takes the class of its **nearest evidence voxel** in
3D Euclidean distance on the 0.2 m grid (no radius cap, ties broken by scipy's EDT index order). An
evidence voxel's class is the open-vocabulary readout of its own lifted teacher feature, so at evidence
voxels B0 *is* the teacher (copying), and elsewhere it is pure nearest-neighbour copying.

The same EDT gives the propagation distance (metres) of every voxel used for the distance bins.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy import ndimage

from tools.semantic_completion import config as C


def nearest_evidence(evidence_mask: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """``(nearest evidence flat index [N_VOXELS] int64, distance_m [N_VOXELS] float32)``; all -1 / inf if no evidence."""
    m = evidence_mask.reshape(C.GRID_DIMS).cpu().numpy()
    if not m.any():
        return np.full(C.N_VOXELS, -1, np.int64), np.full(C.N_VOXELS, np.inf, np.float32)
    dist, idx = ndimage.distance_transform_edt(~m, return_indices=True)
    flat = np.ravel_multi_index((idx[0], idx[1], idx[2]), C.GRID_DIMS).reshape(-1)
    return flat.astype(np.int64), (dist.reshape(-1) * C.VOXEL_SIZE_M).astype(np.float32)


def nearest_label(evidence_label_dense: torch.Tensor, nearest_flat: np.ndarray) -> torch.Tensor:
    """B0 label per voxel: the label at the nearest evidence voxel (0 if the anchor has no evidence)."""
    if (nearest_flat < 0).all():
        return torch.zeros(C.N_VOXELS, dtype=torch.long, device=evidence_label_dense.device)
    return evidence_label_dense[torch.from_numpy(nearest_flat).to(evidence_label_dense.device)]


__all__ = ["nearest_evidence", "nearest_label"]
