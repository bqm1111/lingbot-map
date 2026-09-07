# extracted from gate8c0/transforms.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
"""The complete KITTI-360 transform chain, written down once and then tested.

Every convention below was read out of the code that actually runs, not assumed. The
chain has two halves that must agree, and Gate 8C-0 exists partly to check that they do:

**The prediction half** (what the mapper builds, ``gate8.mapper.IncrementalMapper``)::

    pixel (u,v) --K^-1--> ray in rectified cam0 of frame f
      --depth d_m = s * d_canonical--> point in rectified cam0 of frame f
      --T_c2w[f], translation scaled by the SAME s--> LingBot canonical world
      --floor(p / 0.2)--> integer world lattice key           (the map's own frame)

**The evaluation half** (what ``query`` reads back)::

    grid index (i,j,k) --(ijk + 0.5) * 0.2 + origin--> point in velodyne of the ANCHOR
      --inv(rect_cam_to_velo)--> rectified cam0 of the anchor
      --T_c2w[anchor], translation scaled by s--> LingBot canonical world
      --floor(p / 0.2)--> the same integer world lattice key

**The oracle half** (ground truth only, used by Stages 2-4)::

    velodyne point at frame f --velo_to_world[f] = cam0_to_world[f] @ inv(cam0_to_velo)-->
      KITTI-360 world  --inv(velo_to_world[anchor])--> velodyne of the anchor
      --floor((p - origin) / 0.2)--> grid index

Conventions, all verified by the round-trip tests in ``tests/gate8c0``:

* **Matrix convention**: 4x4 homogeneous, **column-vector** semantics
  (``p_dst = R @ p_src + t``). Code that stores points row-wise applies it as
  ``p @ R.T + t``; ``gate8.mapper._integrate`` does exactly that.
* **Pose direction**: ``pred_pose_c2w`` and ``cam0_to_world.txt`` are both
  **source-to-world** (camera-to-world). The inverse is taken explicitly wherever a
  world-to-local transform is needed; nothing relies on a transpose as an inverse.
* **Axis order / handedness**: right-handed throughout. Rectified cam0 is
  x-right, y-down, z-forward. Velodyne is x-forward, y-left, z-up. The grid inherits the
  velodyne axes, so grid ``i`` runs forward, ``j`` left, ``k`` up.
* **Grid**: ``dims = (256, 256, 32)``, ``voxel_size = 0.2 m``,
  ``origin = (0.0, -25.6, -2.0)`` in the velodyne frame of the anchor, i.e. the box spans
  x in [0, 51.2), y in [-25.6, 25.6), z in [-2, 4.4) metres.
* **Binning**: ``floor``, never ``round`` -- ``floor((p - origin) / voxel_size)`` at
  evaluation and ``floor(p / voxel_size)`` on the map's own world lattice (whose origin is
  the world origin, so no offset term).
* **Rectification**: ``rect_cam_to_velo = cam0_to_velo @ inv(R_rect_00)``. The shipped
  ``calib_cam_to_velo.txt`` is *unrectified* cam0 to velodyne, and the images are
  rectified, so omitting ``R_rect_00`` would introduce a ~0.4 deg rotation.
* **Intrinsics**: ``P_rect_00[:3, :3]`` on the native 1408x376 lattice; the mapper uses
  LingBot's ``pred_K`` on the processed 518x140 lattice instead, which is the same camera
  scaled to the network's input.
* **Target index**: the file is ``<anchor:06d>_1_1.npy`` where ``anchor`` is the
  **SSCBench index**, and the coordinate frame is the velodyne frame of the native frame
  ``pose_frames[anchor + 1]``. The target is anchored at ``t`` -- the last frame of the
  clip and the frame the causal map is queried at -- not at ``t+1``.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

from core.datasets import kitti360 as K3

GRID = K3.SSCBENCH_KITTI360_GRID
VOXEL = float(GRID.voxel_size)
ORIGIN = np.asarray(GRID.origin, np.float64)
DIMS = tuple(int(d) for d in GRID.dims)


def inv(T: np.ndarray) -> np.ndarray:
    """True matrix inverse, matching the production path.

    ``tools/gate8/evaluate._grid_to_world`` uses ``np.linalg.inv``, and so does this, on
    purpose: KITTI-360's shipped matrices are only *approximately* rigid. ``R_rect_00``
    has determinant 0.999999548 and ``cam0_to_world`` 1.000000611 because the published
    files carry six decimals, so substituting ``R.T`` for the inverse leaves a round-trip
    residual of ~7e-5 m at 40 m range. That is 3.5e-4 of a voxel and harmless either way,
    but the true inverse makes the round-trip tests exact and cannot mask a real
    convention error behind a data-rounding one. :func:`inv_rigid` is kept so Stage 1 can
    report the difference instead of assuming it.
    """
    return np.linalg.inv(np.asarray(T, np.float64))


def inv_rigid(T: np.ndarray) -> np.ndarray:
    """``(R.T, -R.T @ t)``: the inverse *if* ``T`` were exactly rigid."""
    T = np.asarray(T, np.float64)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Column-vector transform applied to row-stored points: ``p @ R.T + t``."""
    T = np.asarray(T, np.float64)
    p = np.asarray(pts, np.float64)
    return p @ T[:3, :3].T + T[:3, 3]


class DriveGeometry:
    """Ground-truth geometry of one KITTI-360 drive: calibration, poses, frame indexing."""

    def __init__(self, drive: str, sscbench_root: str, kitti360_root: str):
        self.drive = drive
        self.sscbench_root = sscbench_root
        self.kitti360_root = kitti360_root
        self.calib = K3.parse_calibration(os.path.join(sscbench_root, "calibration"))
        self.pose_frames = K3.pose_frames(os.path.join(sscbench_root, "data_poses", drive,
                                                       "poses.txt"))
        self.cam0_to_world = K3.load_cam0_to_world(
            os.path.join(sscbench_root, "data_poses", drive, "cam0_to_world.txt"))
        self._ts: Dict[str, np.ndarray] = {}

    # -- frame indexing ----------------------------------------------------
    def native(self, sscbench_index: int) -> int:
        return int(K3.sscbench_to_native(sscbench_index, self.pose_frames))

    def timestamps(self, kind: str = "image") -> np.ndarray:
        if kind not in self._ts:
            sub = ("data_2d_raw", "image_00") if kind == "image" else \
                  ("data_3d_raw", "velodyne_points")
            self._ts[kind] = K3.load_timestamps(
                os.path.join(self.kitti360_root, sub[0], self.drive, sub[1], "timestamps.txt"))
        return self._ts[kind]

    # -- transforms --------------------------------------------------------
    @property
    def rect_cam_to_velo(self) -> np.ndarray:
        return np.asarray(self.calib.rect_cam_to_velo, np.float64)

    def velo_to_world(self, native_frame: int) -> np.ndarray:
        """velodyne of ``native_frame`` -> KITTI-360 world, via the *unrectified* cam0."""
        c2w = self.cam0_to_world[int(native_frame)]
        return np.asarray(c2w, np.float64) @ inv(np.asarray(self.calib.cam0_to_velo, np.float64))

    def rect_cam_to_world(self, native_frame: int) -> np.ndarray:
        return self.velo_to_world(native_frame) @ self.rect_cam_to_velo

    def velo_to_velo(self, src_frame: int, anchor_frame: int) -> np.ndarray:
        """velodyne of ``src_frame`` -> velodyne of ``anchor_frame`` (the grid frame)."""
        return inv(self.velo_to_world(anchor_frame)) @ self.velo_to_world(src_frame)

    # -- LiDAR -------------------------------------------------------------
    def velodyne_path(self, native_frame: int) -> str:
        return os.path.join(self.kitti360_root, "data_3d_raw", self.drive,
                            "velodyne_points", "data", f"{int(native_frame):010d}.bin")

    def read_velodyne(self, native_frame: int) -> np.ndarray:
        """``(N, 3)`` xyz in the velodyne frame of that frame; intensity dropped."""
        p = self.velodyne_path(native_frame)
        return np.fromfile(p, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)


# --------------------------------------------------------------------------- #
# grid <-> metric
# --------------------------------------------------------------------------- #
def voxelize(pts: np.ndarray, grid=GRID) -> Tuple[np.ndarray, np.ndarray]:
    """``floor((p - origin) / voxel_size)`` with an in-grid mask. The evaluator's rule."""
    from core.datasets.grids import voxelize as g6_voxelize
    return g6_voxelize(pts, grid)


def centres(idx: np.ndarray, grid=GRID) -> np.ndarray:
    """Integer grid indices -> metric centres in the grid frame."""
    return (np.asarray(idx, np.float64) + 0.5) * float(grid.voxel_size) \
        + np.asarray(grid.origin, np.float64)


def flat(idx: np.ndarray, grid=GRID) -> np.ndarray:
    from core.datasets.grids import flat_of
    return flat_of(np.asarray(idx, np.int64), grid)


def grid_to_world_prediction(pose_c2w_canonical: np.ndarray, scale: float,
                             T_cam_to_grid: np.ndarray) -> np.ndarray:
    """The evaluator's ``T_grid_to_world`` (``tools/gate8/evaluate._grid_to_world``)."""
    P = np.asarray(pose_c2w_canonical, np.float64).copy()
    P[:3, 3] *= float(scale)
    return P @ inv(np.asarray(T_cam_to_grid, np.float64))


__all__ = ["GRID", "VOXEL", "ORIGIN", "DIMS", "inv", "inv_rigid", "apply", "DriveGeometry", "voxelize",
           "centres", "flat", "grid_to_world_prediction"]
