"""SSCBench-KITTI-360 validation adapter (official validation sequence 0006).

Every convention below was **verified against the downloaded files**, not assumed from
SemanticKITTI. The three that differ from a naive port are called out here because each
would silently corrupt the evaluation:

1. **Frame indexing.** SSCBench renumbered KITTI-360. Its index ``i`` is *not* KITTI-360
   frame ``i``: KITTI-360 ships poses only for a subset of frames, and SSCBench indexes
   into that subset, skipping its first entry::

       sscbench index i  ->  kitti360 frame = pose_frames[i + 1]

   Verified by exact pixel equality between the archive's ``image_00/data_rect/{i}.png``
   and KITTI-360's ``data_rect/{frame}.png`` at i = 0, 1, 5, 4500, 9056.

2. **Grid frame.** The occupancy volume lives in the **velodyne frame of the anchor**,
   with the SemanticKITTI extent -- origin ``(0, -25.6, -2)``, 256 x 256 x 32 at 0.2 m.
   Verified by voxelising the raw KITTI-360 velodyne sweep of the anchor frame and
   cross-correlating with the official ``.bin``: the peak sits at **zero shift** with
   IoU ~0.92, whereas the camera frame gives IoU ~0.03.

3. **Target semantics.** The official ``preprocess/labels/<seq>/<anchor>_1_1.npy`` target
   is exactly::

       255  <=>  invalid == 1  AND  label == 0        (unobserved -> excluded)
       0    <=>  invalid == 0  AND  label == 0        (observed free)
       1..18 <=> learning_map(label), label > 0       (occupied, kept even where invalid)

   Verified elementwise on sampled anchors. Note the asymmetry: an occupied voxel is kept
   even when ``.invalid`` marks it, so ``.invalid`` alone is **not** the evaluation mask.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from prompted_lingbot.occupancy import SEMANTICKITTI_GRID, VoxelGrid

SEQUENCE = "2013_05_28_drive_0006_sync"
OFFICIAL_SPLIT = {"train": ["2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync",
                            "2013_05_28_drive_0003_sync", "2013_05_28_drive_0004_sync",
                            "2013_05_28_drive_0005_sync", "2013_05_28_drive_0007_sync",
                            "2013_05_28_drive_0010_sync"],
                  "val": [SEQUENCE],
                  "test": ["2013_05_28_drive_0009_sync"]}
ANCHOR_STRIDE = 5          # SSCBench places an occupancy anchor every 5 native frames

# Same geometry as SemanticKITTI's SSC volume; named separately so no code can conflate
# the two benchmarks by accident.
SSCBENCH_KITTI360_GRID = VoxelGrid(
    dims=(256, 256, 32), voxel_size=0.2, origin=(0.0, -25.6, -2.0),
    frame="velodyne_of_anchor_frame", empty_class=0, ignore_label=255,
    name="sscbench_kitti360",
)
assert (SSCBENCH_KITTI360_GRID.dims == SEMANTICKITTI_GRID.dims
        and SSCBENCH_KITTI360_GRID.voxel_size == SEMANTICKITTI_GRID.voxel_size
        and SSCBENCH_KITTI360_GRID.origin == SEMANTICKITTI_GRID.origin), \
    "SSCBench-KITTI-360 grid must match the frozen SemanticKITTI extent"

IGNORE = SSCBENCH_KITTI360_GRID.ignore_label
FREE = SSCBENCH_KITTI360_GRID.empty_class


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
@dataclass
class Kitti360Calibration:
    """Rectified perspective calibration for ``image_00`` plus the velodyne extrinsic."""

    K: np.ndarray              # (3, 3) native rectified intrinsics of image_00
    P_rect_00: np.ndarray      # (3, 4) raw projection matrix, kept for provenance
    R_rect_00: np.ndarray      # (4, 4) unrectified cam0 -> rectified cam0
    cam0_to_velo: np.ndarray   # (4, 4) UNrectified cam0 -> velodyne, as shipped
    native_hw: Tuple[int, int]

    @property
    def rect_cam_to_velo(self) -> np.ndarray:
        """Rectified cam0 -> velodyne. This is the transform into the SSC grid frame."""
        return self.cam0_to_velo @ np.linalg.inv(self.R_rect_00)

    @property
    def velo_to_rect_cam(self) -> np.ndarray:
        return np.linalg.inv(self.rect_cam_to_velo)

    def velo_to_cam(self, pts: np.ndarray) -> np.ndarray:
        T = self.velo_to_rect_cam
        return np.asarray(pts, np.float64) @ T[:3, :3].T + T[:3, 3]

    def to_dict(self) -> Dict[str, object]:
        return {"K": self.K.tolist(), "P_rect_00": self.P_rect_00.tolist(),
                "R_rect_00": self.R_rect_00.tolist(),
                "cam0_to_velo": self.cam0_to_velo.tolist(),
                "rect_cam_to_velo": self.rect_cam_to_velo.tolist(),
                "native_hw": list(self.native_hw)}


def parse_calibration(calib_dir: str, camera: str = "00") -> Kitti360Calibration:
    """Parse KITTI-360 ``perspective.txt`` + ``calib_cam_to_velo.txt``."""
    vals: Dict[str, np.ndarray] = {}
    with open(os.path.join(calib_dir, "perspective.txt")) as fh:
        for line in fh:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            vals[k.strip()] = np.fromstring(v, sep=" ")
    P = vals[f"P_rect_{camera}"].reshape(3, 4)
    R = np.eye(4)
    R[:3, :3] = vals[f"R_rect_{camera}"].reshape(3, 3)
    w, h = vals[f"S_rect_{camera}"]
    if abs(np.linalg.det(R[:3, :3]) - 1.0) > 1e-6:
        raise ValueError("R_rect is not a rotation")
    c2v = np.eye(4)
    c2v[:3, :4] = np.loadtxt(os.path.join(calib_dir, "calib_cam_to_velo.txt")).reshape(3, 4)
    if P[0, 0] <= 0 or P[1, 1] <= 0:
        raise ValueError("non-positive focal length in P_rect")
    # image_00 is the reference camera, so P_rect_00 carries no stereo baseline.
    if camera == "00" and np.abs(P[:, 3]).max() > 1e-9:
        raise ValueError(f"P_rect_00 has a non-zero 4th column: {P[:, 3]}")
    return Kitti360Calibration(K=P[:3, :3].copy(), P_rect_00=P, R_rect_00=R,
                               cam0_to_velo=c2v, native_hw=(int(h), int(w)))


# --------------------------------------------------------------------------- #
# Frame indexing and poses
# --------------------------------------------------------------------------- #
def pose_frames(poses_path: str) -> np.ndarray:
    """Native KITTI-360 frame indices that have a pose, ascending."""
    idx = np.loadtxt(poses_path, usecols=0).astype(np.int64)
    if np.any(np.diff(idx) <= 0):
        raise ValueError(f"{poses_path}: frame indices are not strictly increasing")
    return idx


def sscbench_to_native(index, frames: np.ndarray):
    """SSCBench index -> native KITTI-360 frame. See the module docstring, point 1."""
    i = np.asarray(index) + 1
    if np.any(i < 0) or np.any(i >= len(frames)):
        raise IndexError(f"sscbench index out of range for {len(frames)} pose frames")
    return frames[i] if i.ndim else int(frames[int(i)])


def load_cam0_to_world(path: str) -> Dict[int, np.ndarray]:
    """``cam0_to_world.txt`` -> {native frame: 4x4 unrectified-cam0-to-world}."""
    raw = np.loadtxt(path)
    return {int(r[0]): r[1:].reshape(4, 4).astype(np.float64) for r in raw}


def load_timestamps(path: str) -> np.ndarray:
    """``timestamps.txt`` -> seconds since the first entry (float64)."""
    import datetime
    secs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            secs.append(datetime.datetime.strptime(line[:26],
                                                   "%Y-%m-%d %H:%M:%S.%f").timestamp())
    t = np.asarray(secs, np.float64)
    return t - t[0]


# --------------------------------------------------------------------------- #
# Anchors and clips
# --------------------------------------------------------------------------- #
def anchor_indices(voxel_dir: str) -> List[int]:
    """Official occupancy anchors, as SSCBench indices, ascending."""
    out = sorted(int(f[: -len(".label")]) for f in os.listdir(voxel_dir)
                 if f.endswith(".label"))
    if not out:
        raise FileNotFoundError(f"no .label files under {voxel_dir}")
    return out


@dataclass
class ClipRecord:
    clip_id: str
    anchor: int                       # SSCBench index of the evaluated anchor
    sscbench_indices: List[int]       # five anchors, chronological, anchor last
    native_frames: List[int]          # the KITTI-360 frames they resolve to
    timestamps_s: List[float]
    span_s: float
    block: int                        # contiguous-block id for the paired bootstrap

    def to_dict(self) -> Dict[str, object]:
        return {"dataset": "sscbench_kitti360", "sequence": SEQUENCE,
                "clip_id": self.clip_id, "anchor": self.anchor,
                "sscbench_indices": self.sscbench_indices,
                "native_frames": self.native_frames,
                "timestamps_s": [round(t, 6) for t in self.timestamps_s],
                "span_s": round(self.span_s, 6), "block": self.block,
                "image_paths": [f"image_00/data_rect/{f:010d}.png"
                                for f in self.native_frames],
                "lidar_paths": [f"velodyne_points/data/{f:010d}.bin"
                                for f in self.native_frames],
                "target_path": f"preprocess/labels/{SEQUENCE}/{self.anchor:06d}_1_1.npy"}


def build_clips(anchors: Sequence[int], frames: np.ndarray, timestamps: np.ndarray,
                clip_length: int = 5, block_size: int = 20,
                max_native_gap: int = ANCHOR_STRIDE) -> Tuple[List[ClipRecord],
                                                              List[Dict[str, object]]]:
    """Chronological ``[t-4, ..., t]`` anchor clips, plus the exclusion ledger.

    The eligibility rule is declared before any result is seen and is purely temporal:
    a clip is eligible iff it has four earlier anchors **and** consecutive anchors are
    exactly ``max_native_gap`` native frames apart. The second condition is what "do not
    cross sequence discontinuities" means here -- KITTI-360 has no pose for every frame,
    so consecutive SSCBench anchors are occasionally far apart in real time, and such a
    clip would silently span many seconds instead of two.
    """
    aset = set(int(a) for a in anchors)
    clips, excluded = [], []
    for a in sorted(aset):
        want = [a - (clip_length - 1 - k) * ANCHOR_STRIDE for k in range(clip_length)]
        if any(w not in aset for w in want):
            excluded.append({"anchor": a, "reason": "missing_history_anchor"})
            continue
        nat = [int(sscbench_to_native(w, frames)) for w in want]
        gaps = np.diff(nat)
        if np.any(gaps != max_native_gap):
            excluded.append({"anchor": a, "reason": "non_contiguous_native_frames",
                             "native_frames": nat, "gaps": gaps.tolist()})
            continue
        ts = [float(timestamps[f]) for f in nat]
        clips.append(ClipRecord(clip_id=f"{SEQUENCE}_{a:06d}", anchor=a,
                                sscbench_indices=want, native_frames=nat,
                                timestamps_s=ts, span_s=ts[-1] - ts[0], block=0))
    # Contiguous blocks of the chronological list. A trailing partial block is appended
    # to the last full one rather than standing alone, so no block is short.
    n_full = max(len(clips) // block_size, 1)
    for i, c in enumerate(clips):
        c.block = min(i // block_size, n_full - 1)
    return clips, excluded


# --------------------------------------------------------------------------- #
# Targets  (opened by the evaluator only)
# --------------------------------------------------------------------------- #
def load_target(root: str, anchor: int, grid: VoxelGrid = SSCBENCH_KITTI360_GRID,
                sequence: str = SEQUENCE):
    """Official preprocessed target -> ``(target, valid)`` on the SSC grid.

    ``sequence`` defaults to the validation drive; Gate 8B passes an official *train*
    drive when it uses KITTI-360 as a completion-training source.
    """
    p = os.path.join(root, "preprocess", "labels", sequence, f"{anchor:06d}_1_1.npy")
    t = np.load(p).astype(np.int32)
    if t.shape != tuple(grid.dims):
        raise ValueError(f"{p}: shape {t.shape} != {grid.dims}")
    return t, t != grid.ignore_label


def target_from_raw(root: str, anchor: int, grid: VoxelGrid = SSCBENCH_KITTI360_GRID,
                    learning_lut: Optional[np.ndarray] = None) -> np.ndarray:
    """Reconstruct the official target from ``.label`` + ``.invalid`` (parity check).

    Implements the rule stated in the module docstring. Without a learning map only the
    binary occupied/free/ignore structure is reproduced, which is all this gate scores.
    """
    base = os.path.join(root, "data_2d_raw", SEQUENCE, "voxels", f"{anchor:06d}")
    label = np.fromfile(base + ".label", dtype=np.uint16).astype(np.int32).reshape(grid.dims)
    invalid = np.fromfile(base + ".invalid", dtype=np.uint8).reshape(grid.dims)
    out = np.zeros(grid.dims, np.int32)
    out[(invalid == 1) & (label == 0)] = grid.ignore_label
    occ = label > 0
    out[occ] = learning_lut[np.clip(label[occ], 0, len(learning_lut) - 1)] \
        if learning_lut is not None else 1
    return out


def binary_target(target: np.ndarray, grid: VoxelGrid = SSCBENCH_KITTI360_GRID):
    """Official binary scene-completion target: ``(occupied, valid)``."""
    valid = target != grid.ignore_label
    return (target != grid.empty_class) & valid, valid


def voxelized_lidar_input(root: str, anchor: int,
                          grid: VoxelGrid = SSCBENCH_KITTI360_GRID) -> np.ndarray:
    """Official bit-packed ``.bin`` voxelised LiDAR **input** (never a target)."""
    p = os.path.join(root, "data_2d_raw", SEQUENCE, "voxels", f"{anchor:06d}.bin")
    return np.unpackbits(np.fromfile(p, np.uint8)).reshape(grid.dims).astype(bool)


def read_velodyne(kitti360_root: str, frame: int) -> np.ndarray:
    """Raw KITTI-360 velodyne sweep ``[N, 4]`` XYZI."""
    p = os.path.join(kitti360_root, "data_3d_raw", SEQUENCE, "velodyne_points", "data",
                     f"{frame:010d}.bin")
    raw = np.fromfile(p, np.float32)
    if raw.size % 4:
        raise ValueError(f"{p}: {raw.size} floats is not a multiple of 4")
    pts = raw.reshape(-1, 4)
    return pts[np.isfinite(pts).all(axis=1)]
