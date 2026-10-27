"""Single source of truth for the Waymo training arm: the frozen recipe, trained on Waymo instead of KITTI-360.

The question this arm answers: does replacing the three KITTI-360 drives with 798 Waymo
segments change what the completion head learns, with the method, grid, causal contract,
losses, optimiser and step budget held fixed?

Reused byte for byte, never redefined here:

* the causal input contract and grid of ``tools/waymo_pilot.waymo_geometry`` (five frames of
  one 50 deg camera at stride 5, anchor-vehicle grid with the z offset aligned to the
  measured road);
* the hard mapper and network input of ``tools/baseline_reconstruction.current.geometry``;
* the occupancy supervision RULE of ``core/supervision/rawtarget.py``, applied per
  (frame, laser) sweep by ``core/supervision/rawtarget_multisensor.py``, which is gated to
  reproduce the single-sensor function bit for bit;
* the training loop, losses and recipe of ``tools/baseline_reconstruction.current.train``.

What differs from the KITTI-360 arm, and is recorded rather than inherited silently:

* **camera field of view.** KITTI-360 ``image_00`` is 103.9 deg horizontal; every Waymo
  camera is 50.0 deg (MEASURED from the cached calibration: 1920x1280, fu ~2050). This is
  the largest single difference and it cannot be removed inside Waymo -- there is no wider
  camera in the set. It moves voxels out of the frustum rather than into occlusion, so the
  supervised volume is less observed at training time than KITTI-360's and less observed
  than SemanticKITTI's 82 deg / nuScenes' 64 deg at test time. :mod:`diagnose` measures the
  shift before anything is trained.
* **sensor count.** KITTI-360 carves from one Velodyne; Waymo has five lasers at different
  origins, so a sweep is a (frame, laser) pair. The per-point laser identity is not in the
  frame cache and is recovered by ``cli/extract_laser_sidecar.py``.
* **scene structure.** A 20 s Waymo segment yields ONE anchor whose privileged window does
  not overlap another's (``waymo_pilot.config``, measured). 798 segments therefore give 798
  independent anchors; sampling :data:`ANCHORS_PER_SEGMENT` per segment raises the training
  volume to match the KITTI-360 arm's but does NOT raise the independent count, and no
  table may report it as such.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from tools.baseline_reconstruction import config as BR
from tools.waymo_pilot import config as WP

VERSION = "waymo_arm.1"
REPO_ROOT = BR.REPO_ROOT
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "waymo_arm")
#: the derived arm lives beside the frame cache on the 7.3 TB volume; SSD1 has ~108 GiB free
DATA_ROOT = os.environ.get("WAYMO_ARM_DATA_ROOT", "/media/minh/MINH1/lingbot_waymo_arm")

# --------------------------------------------------------------------------- #
# inherited, asserted at run time by cli/verify.py
# --------------------------------------------------------------------------- #
GRID_DIMS = WP.GRID_DIMS                       # (256, 256, 32)
VOXEL_SIZE_M = WP.VOXEL_SIZE_M                 # 0.2
CLIP_OFFSETS = WP.CLIP_OFFSETS                 # (-4, -3, -2, -1, 0)
STREAM_STRIDE = WP.WAYMO_STREAM_STRIDE         # 5 native frames = 0.50 s
SELECTED_CAMERA = WP.WAYMO_SELECTED_CAMERA     # FRONT
SEEDS: Tuple[int, ...] = BR.SEEDS              # (0, 1, 2)
OCCUPANCY_THRESHOLD = BR.OCCUPANCY_THRESHOLD

#: MEASURED horizontal field of view, degrees. Both numbers are read from calibration, not
#: assumed; they are the headline difference between the two arms.
HFOV_WAYMO_DEG = 50.0
HFOV_K360_IMAGE00_DEG = WP.K360_IMAGE00_HFOV_DEG       # 103.9
#: the test cameras, for the same comparison
HFOV_SEMANTICKITTI_DEG = 82.0                  # 1226 px at fx 707.09
HFOV_OCC3D_CAM_FRONT_DEG = 64.5                # 1600 px at fx 1266.42

# --------------------------------------------------------------------------- #
# anchor policy
# --------------------------------------------------------------------------- #
#: anchors sampled per segment. 1 would be the independent count; 5 matches the KITTI-360
#: arm's training volume (3 drives, 1,358 anchors; 8 drives, 9,672). Both are reported.
ANCHORS_PER_SEGMENT = 5
#: segments held out of training for the loss curve only. The arm is judged on the target
#: benchmarks, never on Waymo.
N_VAL_SEGMENTS = 40
MIN_ANCHOR_SEPARATION_FRAMES = WP.MIN_ANCHOR_SEPARATION_FRAMES     # 40 stream frames
SAMPLE_SEED = 0

#: privileged occupancy window: the anchor plus this many later STREAM frames, the frozen
#: KITTI-360 setting (``delayed_semantic.TargetConfig``). At stride 5 that is 10.0 s, and it
#: is why eligible anchors stop at stream index 18 of a 39-frame segment.
TARGET_FUTURE_STREAM_FRAMES = 20

# --------------------------------------------------------------------------- #
# diagnostic gate (step 1). Numbers to be MEASURED, not thresholds to pass.
# --------------------------------------------------------------------------- #
DIAGNOSTIC_SEGMENTS = 40


@dataclass(frozen=True)
class Paths:
    data_root: str = DATA_ROOT
    artifacts: str = ARTIFACT_DIR
    #: packed hard map (S_H, F_H) per anchor -- the CompletionUNet input
    inputs: str = os.path.join(DATA_ROOT, "inputs")
    #: packed raw-LiDAR occupancy supervision per anchor
    targets: str = os.path.join(DATA_ROOT, "targets")
    checkpoints: str = os.path.join(ARTIFACT_DIR, "checkpoints")
    logs: str = os.path.join(ARTIFACT_DIR, "logs")
    #: read-only, from the pilot
    clips: str = os.path.join(WP.DATA_ROOT, "clips")
    waymo_frames: str = os.path.join(WP.DATA_ROOT, "waymo_frames")
    waymo_lasers: str = os.path.join(WP.DATA_ROOT, "waymo_lasers")

    def clip_path(self, segment: str, t: int) -> str:
        return os.path.join(self.clips, segment, f"{t:05d}.npz")

    def input_path(self, segment: str, t: int) -> str:
        return os.path.join(self.inputs, segment, f"{t:05d}.npz")

    def target_path(self, segment: str, t: int) -> str:
        return os.path.join(self.targets, segment, f"{t:05d}.npz")

    def laser_sidecar(self, segment: str) -> str:
        return os.path.join(self.waymo_lasers, f"{segment}.npz")


def ensure_dirs(p: Paths = None) -> None:
    p = p or Paths()
    for d in (p.data_root, p.inputs, p.targets, p.checkpoints, p.logs, p.artifacts):
        os.makedirs(d, exist_ok=True)


def segments(p: Paths = None) -> List[str]:
    """Fully extracted segments, deterministic order."""
    p = p or Paths()
    return sorted(d for d in os.listdir(p.waymo_frames)
                  if len(os.listdir(os.path.join(p.waymo_frames, d))) >= 190)


def split(segs: Sequence[str]) -> Dict[str, List[str]]:
    """Train / val by segment, so no val anchor shares a scene with a train anchor."""
    segs = list(segs)
    return {"val": segs[-N_VAL_SEGMENTS:], "train": segs[:-N_VAL_SEGMENTS]}


__all__ = [k for k in dir() if not k.startswith("_")]
