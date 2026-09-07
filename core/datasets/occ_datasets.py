# extracted from prompted_lingbot/occ_datasets.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
"""Benchmark adapters: ground-truth volumes and the frames they live in.

SemanticKITTI SSC and Occ3D-nuScenes place their evaluation volume in different
frames and use different empty/ignore conventions.  Both are transcribed from the
OccAny reference implementation; see ``docs/lingbot_occupancy_feasibility.md`` §2.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .occupancy import (
    OCC3D_NUSCENES_GRID, OCC3D_SINGLE_CAMERA_X_CUT, SEMANTICKITTI_GRID, VoxelGrid,
)


# --------------------------------------------------------------------------- #
# SemanticKITTI
# --------------------------------------------------------------------------- #
def read_kitti_calib(path: str) -> Dict[str, np.ndarray]:
    """Parse a SemanticKITTI/KITTI-odometry ``calib.txt``.

    ``Tr`` is **velodyne-to-camera**; the camera-to-velodyne transform used to put
    predictions into the SSC grid is its inverse (OccAny kitti.py:331-332).
    """
    out: Dict[str, np.ndarray] = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, vals = line.split(":", 1)
            arr = np.fromstring(vals, sep=" ")
            if arr.size == 12:
                m = np.eye(4)
                m[:3, :4] = arr.reshape(3, 4)
                out[key.strip()] = m
    if "Tr" not in out:
        raise ValueError(f"no Tr in {path}")
    return out


def T_cam_to_velo(calib: Dict[str, np.ndarray]) -> np.ndarray:
    return np.linalg.inv(calib["Tr"])


_SEMANTIC_KITTI_YAML = "/home/minh/workspace/OccAny/occany/datasets/semantic_kitti.yaml"
_LEARNING_LUT: Optional[np.ndarray] = None


def semantickitti_learning_lut(path: str = _SEMANTIC_KITTI_YAML) -> Optional[np.ndarray]:
    """Official ``learning_map`` as a lookup table, or None if unavailable.

    This matters for *binary* occupancy, which is not obvious: raw ids 1
    ("outlier") and 99 ("other-object") map to 0, so OccAny treats them as EMPTY.
    Deciding occupancy on the raw ids instead counts them as occupied and inflates
    the ground-truth occupied set by ~0.4%.
    """
    global _LEARNING_LUT
    if _LEARNING_LUT is not None:
        return _LEARNING_LUT
    if not os.path.isfile(path):
        return None
    import yaml
    lm = yaml.safe_load(open(path))["learning_map"]
    lut = np.zeros(max(lm) + 1, dtype=np.int32)
    for raw, learned in lm.items():
        lut[int(raw)] = int(learned)
    _LEARNING_LUT = lut
    return lut


def load_semantickitti_target(root: str, sequence: str, frame: int,
                              grid: VoxelGrid = SEMANTICKITTI_GRID,
                              remap: bool = True):
    """Return ``(target, valid)`` for one SSC anchor frame.

    Reproduces ``OccAny/occany/datasets/kitti.py:read_voxel_label``: the raw ids
    are passed through the official ``learning_map`` **before** anything is called
    occupied, then ``.invalid`` voxels are set to ``255``.
    """
    base = os.path.join(root, "sequences", sequence, "voxels", f"{frame:06d}")
    label_path, invalid_path = base + ".label", base + ".invalid"
    if not (os.path.isfile(label_path) and os.path.isfile(invalid_path)):
        return None, None
    label = np.fromfile(label_path, dtype=np.uint16).astype(np.int32).reshape(grid.dims)
    if remap:
        lut = semantickitti_learning_lut()
        if lut is not None:
            label = lut[np.clip(label, 0, len(lut) - 1)]
    invalid = np.unpackbits(np.fromfile(invalid_path, dtype=np.uint8)).reshape(grid.dims)
    valid = invalid == 0
    target = label.copy()
    target[~valid] = grid.ignore_label
    return target, valid


@dataclass
class SemanticKittiOccSpec:
    """Everything needed to evaluate one SemanticKITTI sequence."""

    root: str
    sequence: str
    calib: Dict[str, np.ndarray]
    grid: VoxelGrid = SEMANTICKITTI_GRID

    @staticmethod
    def build(root: str, sequence: str) -> "SemanticKittiOccSpec":
        calib = read_kitti_calib(os.path.join(root, "sequences", sequence, "calib.txt"))
        return SemanticKittiOccSpec(root=root, sequence=sequence, calib=calib)

    @property
    def cam_to_velo(self) -> np.ndarray:
        return T_cam_to_velo(self.calib)

    def target(self, frame: int):
        return load_semantickitti_target(self.root, self.sequence, frame, self.grid)


# --------------------------------------------------------------------------- #
# Occ3D-nuScenes
# --------------------------------------------------------------------------- #
def load_occ3d_target(gt_dir: str, grid: VoxelGrid = OCC3D_NUSCENES_GRID,
                      apply_camera_mask: bool = True,
                      apply_lidar_mask: bool = False,
                      single_camera: bool = True):
    """Return ``(target, valid)`` for one Occ3D-nuScenes sample.

    Reproduces ``OccAny/occany/datasets/nuscenes.py:889-905``:

    * ``labels.npz`` provides ``semantics``, ``mask_camera``, ``mask_lidar``;
    * with ``apply_camera_mask`` the voxels a camera cannot see become ``255``;
    * the lidar mask is **off** in the reference setting;
    * in the single-camera (CAM_FRONT) setting the rear half of the grid --
      ``[:100, :, :]``, i.e. ego ``x < 0`` -- is set to ``255`` and excluded.
    """
    path = os.path.join(gt_dir, "labels.npz")
    if not os.path.isfile(path):
        return None, None
    d = np.load(path)
    target = d["semantics"].astype(np.int32)
    mask_camera = d["mask_camera"].astype(bool)
    mask_lidar = d["mask_lidar"].astype(bool)
    if target.shape != tuple(grid.dims):
        raise ValueError(f"{path}: expected {grid.dims}, got {target.shape}")

    valid = np.ones(grid.dims, bool)
    if apply_camera_mask:
        target[~mask_camera] = grid.ignore_label
        valid &= mask_camera
    if apply_lidar_mask:
        target[~mask_lidar] = grid.ignore_label
        valid &= mask_lidar
    if single_camera:
        target[:OCC3D_SINGLE_CAMERA_X_CUT, :, :] = grid.ignore_label
        valid[:OCC3D_SINGLE_CAMERA_X_CUT, :, :] = False
    return target, valid


# --------------------------------------------------------------------------- #
# Geometry: predictions -> the benchmark's evaluation frame
# --------------------------------------------------------------------------- #
def camera_points_from_depth(depth: np.ndarray, K: np.ndarray, conf: Optional[np.ndarray],
                             conf_threshold: float, min_depth: float, max_depth: float):
    """``(H, W)`` Z-depth -> ``(M, 3)`` camera-frame points, filtered.

    Uses the verified Z-depth convention: ``x = (u - cx) * d / fx``.
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    d = depth.astype(np.float64)
    keep = np.isfinite(d) & (d > min_depth) & (d < max_depth)
    if conf is not None and conf_threshold > 0:
        keep &= conf.astype(np.float64) >= conf_threshold
    if not keep.any():
        return np.zeros((0, 3))
    du = d[keep]
    return np.stack([(u[keep] - K[0, 2]) * du / K[0, 0],
                     (v[keep] - K[1, 2]) * du / K[1, 1],
                     du], axis=-1)


def relative_c2w(pose_c2w_from: np.ndarray, pose_c2w_to: np.ndarray) -> np.ndarray:
    """Rigid transform taking points in camera ``from`` into camera ``to``.

    Both arguments are camera-to-world (the verified convention), so the result is
    ``inv(pose_to) @ pose_from``.
    """
    R_f, t_f = pose_c2w_from[:3, :3], pose_c2w_from[:3, 3]
    R_t, t_t = pose_c2w_to[:3, :3], pose_c2w_to[:3, 3]
    R = R_t.T @ R_f
    t = R_t.T @ (t_f - t_t)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def apply_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


# --------------------------------------------------------------------------- #
# Sparse metric depth prompts from the LiDAR scan
# --------------------------------------------------------------------------- #
def sparse_depth_from_velodyne(
    root: str, sequence: str, frame: int, calib: Dict[str, np.ndarray],
    source_hw: Tuple[int, int], target_hw: Tuple[int, int],
    image_size: int = 518, patch_size: int = 14, depth_stride: int = 2,
    min_depth: float = 1.0, max_depth: float = 80.0,
):
    """Project a velodyne scan into the cached depth lattice.

    This is what a sparse metric depth prompt *is* on this platform: a real LiDAR
    return, not a synthetic sample of a dense ground-truth map.  Returns
    ``(depth, valid)`` on the cached lattice, where ``depth`` is metric Z-depth in
    the **camera** frame -- the same quantity LingbotMap predicts.

    The projection chain is ``p_cam = Tr @ p_velo`` then ``P2 @ p_cam``; the pixel
    coordinates are then mapped onto the preprocessed grid the cache uses.
    """
    from .preprocess import target_grid

    path = os.path.join(root, "sequences", sequence, "velodyne", f"{frame:06d}.bin")
    if not os.path.isfile(path):
        return None, None
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)

    Tr, P2 = calib["Tr"], calib["P2"]
    p_cam = pts @ Tr[:3, :3].T + Tr[:3, 3]
    z = p_cam[:, 2]
    front = z > min_depth
    p_cam, z = p_cam[front], z[front]

    proj = p_cam @ P2[:3, :3].T + P2[:3, 3]
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]

    H_src, W_src = source_hw
    out_h, new_w, crop_top, new_h = target_grid((H_src, W_src), image_size, patch_size)
    u = u * (new_w / W_src) / depth_stride
    v = (v * (new_h / H_src) - crop_top) / depth_stride

    Ht, Wt = target_hw
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok = (ui >= 0) & (ui < Wt) & (vi >= 0) & (vi < Ht) & (z > min_depth) & (z < max_depth)
    ui, vi, zz = ui[ok], vi[ok], z[ok]

    depth = np.zeros((Ht, Wt), np.float32)
    valid = np.zeros((Ht, Wt), bool)
    if zz.size:
        # Nearest surface wins where several returns share a pixel.
        order = np.argsort(-zz)
        depth[vi[order], ui[order]] = zz[order]
        valid[vi[order], ui[order]] = True
    return depth, valid
