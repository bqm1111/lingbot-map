"""Canonical 0.2 m grid over the official Occ3D extent, and the frozen native conversion.

Gate 3.1 was trained at 0.2 m with a 3-voxel (0.6 m) correction radius and a 7-voxel
(1.4 m) receptive field. Occ3D-nuScenes evaluates on 0.4 m voxels. Applying "radius 3"
on the native grid would silently double the physical radius to 1.2 m, so the whole
frozen stack runs on a **canonical 0.2 m grid covering the identical official extent**
and predictions are converted to the native grid afterwards.

    native      200 x 200 x 16 @ 0.4 m, origin (-40, -40, -1)   -> upper (40, 40, 5.4)
    canonical   400 x 400 x 32 @ 0.2 m, origin (-40, -40, -1)   -> upper (40, 40, 5.4)

Conversion rule, frozen before any label was read and applied identically to every
configuration, learned and deterministic:

    a native voxel is occupied iff ANY of its eight 0.2 m subvoxels is occupied
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from prompted_lingbot.occupancy import OCC3D_NUSCENES_GRID, VoxelGrid

NATIVE = OCC3D_NUSCENES_GRID
RATIO = 2                                    # 0.4 / 0.2

CANONICAL = VoxelGrid(
    dims=(NATIVE.dims[0] * RATIO, NATIVE.dims[1] * RATIO, NATIVE.dims[2] * RATIO),
    voxel_size=NATIVE.voxel_size / RATIO,
    origin=NATIVE.origin,
    frame="ego_of_anchor_keyframe",
    empty_class=NATIVE.empty_class,
    ignore_label=NATIVE.ignore_label,
    name="occ3d_nuscenes_canonical_0p2m",
)

assert np.allclose(CANONICAL.upper, NATIVE.upper), "canonical grid must cover the official extent"


def canonical_to_native(vol: torch.Tensor) -> torch.Tensor:
    """``[400, 400, 32]`` bool -> ``[200, 200, 16]`` bool by the any-subvoxel rule."""
    if tuple(vol.shape) != tuple(CANONICAL.dims):
        raise ValueError(f"expected {CANONICAL.dims}, got {tuple(vol.shape)}")
    x, y, z = NATIVE.dims
    return vol.reshape(x, RATIO, y, RATIO, z, RATIO).amax(dim=(1, 3, 5)).bool()


def points_to_canonical(points_ego: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ego-frame points -> canonical voxel indices, plus the in-grid keep mask."""
    idx = np.floor((points_ego - np.asarray(CANONICAL.origin)) / CANONICAL.voxel_size)
    idx = idx.astype(np.int64)
    keep = np.ones(len(idx), bool)
    for a in range(3):
        keep &= (idx[:, a] >= 0) & (idx[:, a] < CANONICAL.dims[a])
    return idx[keep], keep


def occupancy_canonical(points_ego: np.ndarray, device) -> torch.Tensor:
    vol = torch.zeros(int(np.prod(CANONICAL.dims)), dtype=torch.bool, device=device)
    if len(points_ego):
        idx, _ = points_to_canonical(points_ego)
        if len(idx):
            flat = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), CANONICAL.dims)
            vol[torch.from_numpy(flat).to(device)] = True
    return vol.view(CANONICAL.dims)


def native_binary_target(labels_npz: Dict[str, np.ndarray], apply_camera_mask: bool,
                         apply_lidar_mask: bool, single_camera_x_cut: int):
    """Official Occ3D labels -> ``(occupied, keep)`` on the native grid.

    Reproduces OccAny ``occany/datasets/nuscenes.py:894-904`` and the binary reduction in
    ``occany/metrics/ssc.py:get_score_completion``:

        voxel_label[mask_camera == 0] = 255                 (apply_camera_mask)
        voxel_label[mask_lidar  == 0] = 255                 (apply_lidar_mask, off by default)
        voxel_label[:100, :, :] = 255                       (single-camera setting)
        ignore 255; occupied := label != free(17)

    Called by the evaluator only.
    """
    label = np.asarray(labels_npz["semantics"]).astype(np.int32)
    mask_camera = np.asarray(labels_npz["mask_camera"]).astype(bool)
    mask_lidar = np.asarray(labels_npz["mask_lidar"]).astype(bool)
    if label.shape != tuple(NATIVE.dims):
        raise ValueError(f"label shape {label.shape} != {NATIVE.dims}")
    if apply_camera_mask:
        label[~mask_camera] = NATIVE.ignore_label
    if apply_lidar_mask:
        label[~mask_lidar] = NATIVE.ignore_label
    if single_camera_x_cut:
        label[:single_camera_x_cut, :, :] = NATIVE.ignore_label
    keep = label != NATIVE.ignore_label
    occupied = (label != NATIVE.empty_class) & keep
    return occupied, keep


def native_distance_bands(bands) -> Dict[str, np.ndarray]:
    """Horizontal-range masks on the native grid, measured from the ego origin."""
    ix, iy, iz = np.meshgrid(*[np.arange(d) for d in NATIVE.dims], indexing="ij")
    c = (np.stack([ix, iy, iz], -1).astype(np.float64) + 0.5) * NATIVE.voxel_size \
        + np.asarray(NATIVE.origin)
    rng = np.linalg.norm(c[..., :2], axis=-1)
    return {f"{int(lo)}-{int(hi)}m": (rng >= lo) & (rng < hi) for lo, hi in bands}
