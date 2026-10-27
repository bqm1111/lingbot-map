"""Single source of truth for the ONCE training arm: the frozen recipe, trained on ONCE.

The question: does our completion advantage survive training on **ONCE**, one of OccAny's own
five source datasets, with method, grid, causal contract, losses, optimiser and step budget
held fixed?

**Status: EXPLORATORY.** Authorized 2026-10-27 as an exploratory arm with valid geometric
supervision and a documented source-distribution difference -- *not* as a passing arm under
the original preregistration. :data:`PREVELANCE_BAND_STATUS` records the band that failed and
is written into every manifest this arm produces, so no downstream table can quietly promote
it. See ``artifacts/method_vs_data/once_supervision_validation.md``.

Reused byte for byte, never redefined here:

* the hard mapper and network input of ``tools/baseline_reconstruction.current.geometry``;
* the occupancy supervision rule of ``core/supervision/rawtarget.py`` -- ONCE has a single
  roof LiDAR, so the single-sensor function applies directly and the multisensor path Waymo
  needed is not used;
* the grid of ``core.datasets.kitti360_transforms.GRID``, asserted identical below;
* the training loop, losses, recipe yaml and checkpoint rule reached through
  ``tools.waymo_arm.train``, which is itself gated against the frozen KITTI-360 trainer;
* the zero-shot evaluation harness ``baseline_reconstruction.targets.evaluate``.

What differs from the KITTI-360 arm, recorded rather than inherited silently:

* **camera field of view.** KITTI-360 ``image_00`` is 103.9 deg; ONCE ``cam01`` is **90.1 deg**
  (measured, ``once_metadata_audit.md`` §3). This is the arm's advantage over A1: Waymo's
  50.0 deg could not be separated from its failure, and 90.1 deg is wider than either test
  camera (SemanticKITTI 82 deg, Occ3D 64.5 deg).
* **supervision window.** KITTI-360 carves 21 sweeps at 0.523 s spacing over 10.46 s. ONCE
  releases at 2 FPS, and the authorized policy matches **duration** (~2.1 s) rather than sweep
  count, giving **5 real native sweeps over 2.0 s**. Nothing is interpolated or duplicated.
* **supervision density.** ONCE targets carry ~34 % fewer occupied voxels than
  duration-matched KITTI-360, structured by height and range. Measured, preserved, never
  corrected for.
* **sensor.** One 40-beam ~200 m roof LiDAR (66 k points/sweep) against KITTI-360's 64-beam
  80 m HDL-64E (120 k). ONCE's road plane sits 1.73-1.76 m below the sensor, which is
  KITTI-360's velodyne height, so the frozen grid z origin applies unchanged.
* **independent scenes.** Six sequences. :data:`MIN_ANCHOR_SEPARATION_FRAMES` keeps supervision
  windows disjoint, but the independent *scene* count is **6** and no table may report the
  anchor count as independent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from core.datasets import kitti360_transforms as TF
from core.datasets.occupancy import VoxelGrid
from tools.baseline_reconstruction import config as BR

VERSION = "once_arm.1"
REPO_ROOT = BR.REPO_ROOT
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "method_vs_data", "once_arm")
#: bulk caches live on the 7.3 TB volume; manifests, logs and checkpoints live in ARTIFACT_DIR
DATA_ROOT = os.environ.get("ONCE_ARM_DATA_ROOT", "/media/minh/MINH1/lingbot_once_arm")
ONCE_DATA_ROOT = os.environ.get("ONCE_DATA_ROOT",
                                "/media/minh/MINH1/once_data/extracted/data")

# --------------------------------------------------------------------------- #
# exploratory status -- written into every manifest
# --------------------------------------------------------------------------- #
STATUS = "exploratory"
PREVELANCE_BAND_STATUS: Dict[str, object] = {
    "gate": "occupied prevalence in valid",
    "predeclared_band": [0.015, 0.040],
    "once_measured_median": 0.0130,
    "verdict": "FAILED -- not relabelled as passed",
    "sequences_inside_band": "2 of 6",
    "why_it_is_not_disqualifying": (
        "The band was calibrated under the frozen 21-sweep / 10.46 s carve. Applied to the "
        "authorized ~2.1 s window it also rejects KITTI-360, on vertical coverage (21 of 32 "
        "z-levels against a floor of 30). Each corpus fails a different band and the corpus "
        "the bands were calibrated from is one of them."),
    "cause": (
        "Scene content. LiDAR sparsity is falsified as the cause: removing 85 % of "
        "KITTI-360's returns leaves it in band (prevalence elasticity to point count ~0.1)."),
    "reference": "artifacts/method_vs_data/once_supervision_validation.md",
}

# --------------------------------------------------------------------------- #
# inherited, asserted at import
# --------------------------------------------------------------------------- #
GRID: VoxelGrid = TF.GRID                      # 256x256x32 @ 0.2 m, origin (0, -25.6, -2)
GRID_DIMS = GRID.dims
VOXEL_SIZE_M = GRID.voxel_size
CLIP_OFFSETS = BR.CLIP_OFFSETS                 # (-4, -3, -2, -1, 0)
SEEDS: Tuple[int, ...] = BR.SEEDS              # (0, 1, 2)
OCCUPANCY_THRESHOLD = BR.OCCUPANCY_THRESHOLD
assert GRID.dims == (256, 256, 32) and GRID.voxel_size == 0.2, "grid is not the frozen one"
assert tuple(GRID.origin) == (0.0, -25.6, -2.0), "grid origin is not the frozen one"
assert CLIP_OFFSETS == (-4, -3, -2, -1, 0), "clip contract changed"

# --------------------------------------------------------------------------- #
# ONCE specifics
# --------------------------------------------------------------------------- #
SELECTED_CAMERA = "cam01"
#: ONCE's released images are ALREADY RECTIFIED -- measured, 12/12 frames by LiDAR-depth
#: correlation and 11/12 anchors by map-vs-target IoU. ``cv2.undistort`` is NOT applied; doing
#: so is a second undistortion and halves map-target agreement. This retracts
#: ``once_metadata_audit.md`` §5, whose gradient test was biased.
UNDISTORT_IMAGES = False
DISTORTION_EVIDENCE = {
    "decision": "use shipped JPEGs as-is; do NOT call cv2.undistort",
    "logcorr_lidar_vs_mono_depth": {"plain_pinhole": 0.8571, "distortion_applied": 0.6978,
                                    "pinhole_better_in": "12 of 12 frames"},
    "map_vs_target_iou_pct": {"shipped": 3.26, "undistorted": 1.44,
                              "shipped_better_in": "11 of 12 anchors"},
    "retracts": "once_metadata_audit.md §5 (M2 gradient test is centre-biased)",
}
HFOV_ONCE_CAM01_DEG = 90.1                     # measured, once_metadata_audit.md §3
HFOV_ONCE_CAM03_DEG = 52.7                     # available FOV control, not used here
HFOV_K360_IMAGE00_DEG = 103.9
HFOV_WAYMO_DEG = 50.0
HFOV_SEMANTICKITTI_DEG = 82.0
HFOV_OCC3D_CAM_FRONT_DEG = 64.5

#: ONCE's native cadence is 0.500 s, so one native frame IS one stream frame. The five-frame
#: causal input clip therefore spans 2.0 s -- the same as KITTI-360's and Waymo's.
STREAM_STRIDE = 1
INPUT_CLIP_SPAN_S = 2.0
#: authorized privileged supervision window: forward from the anchor, by DURATION
SUPERVISION_SPAN_S = 2.1
#: a window containing an interval longer than this is rejected; 1.7 % of ONCE intervals are
#: gaps of up to 13 s, so this cannot be assumed away
MAX_SWEEP_GAP_S = 0.75

#: supervision windows of two anchors do not overlap at this separation (5 x 0.5 s = 2.5 s)
MIN_ANCHOR_SEPARATION_FRAMES = 5
#: matches the KITTI-360 arm's training volume (3 drives, 1,358 anchors). The INDEPENDENT
#: scene count is 6 regardless.
TARGET_N_ANCHORS = 1400
#: sequence-disjoint development split, for the loss curve only. The arm is judged on the
#: target benchmarks, never on ONCE.
VAL_SEQUENCES: Tuple[str, ...] = ("000121",)
SAMPLE_SEED = 0


@dataclass(frozen=True)
class Paths:
    data_root: str = DATA_ROOT
    once_root: str = ONCE_DATA_ROOT
    artifacts: str = ARTIFACT_DIR
    clips: str = os.path.join(DATA_ROOT, "clips")
    inputs: str = os.path.join(DATA_ROOT, "inputs")
    targets: str = os.path.join(DATA_ROOT, "targets")
    checkpoints: str = os.path.join(ARTIFACT_DIR, "checkpoints")
    logs: str = os.path.join(ARTIFACT_DIR, "logs")
    manifests: str = os.path.join(ARTIFACT_DIR, "manifests")
    runs: str = os.path.join(ARTIFACT_DIR, "runs")

    def clip_path(self, seq: str, t: int) -> str:
        return os.path.join(self.clips, seq, f"{t:05d}.npz")

    def input_path(self, seq: str, t: int) -> str:
        return os.path.join(self.inputs, seq, f"{t:05d}.npz")

    def target_path(self, seq: str, t: int) -> str:
        return os.path.join(self.targets, seq, f"{t:05d}.npz")


def ensure_dirs(p: "Paths | None" = None) -> None:
    p = p or Paths()
    for d in (p.data_root, p.clips, p.inputs, p.targets, p.checkpoints, p.logs,
              p.manifests, p.runs, p.artifacts):
        os.makedirs(d, exist_ok=True)


def sequences(p: "Paths | None" = None) -> List[str]:
    p = p or Paths()
    return sorted(d for d in os.listdir(p.once_root)
                  if os.path.isdir(os.path.join(p.once_root, d)))


def split(seqs: Sequence[str]) -> Dict[str, List[str]]:
    """Train / val by SEQUENCE, so no val anchor shares a scene with a train anchor."""
    seqs = list(seqs)
    val = [s for s in seqs if s in VAL_SEQUENCES]
    return {"train": [s for s in seqs if s not in VAL_SEQUENCES], "val": val}


def provenance() -> Dict[str, object]:
    """Written into every manifest and log this arm produces."""
    return {"version": VERSION, "status": STATUS,
            "prevalence_band": PREVELANCE_BAND_STATUS,
            "camera": SELECTED_CAMERA, "hfov_deg": HFOV_ONCE_CAM01_DEG,
            "undistort_images": UNDISTORT_IMAGES,
            "distortion_evidence": DISTORTION_EVIDENCE,
            "input_clip": {"offsets": list(CLIP_OFFSETS), "stream_stride": STREAM_STRIDE,
                           "span_s": INPUT_CLIP_SPAN_S, "future_frames_used": 0},
            "supervision": {"policy": "duration-matched, native cadence",
                            "span_s": SUPERVISION_SPAN_S,
                            "max_gap_s": MAX_SWEEP_GAP_S,
                            "interpolated_or_duplicated_sweeps": False},
            "grid": {"dims": list(map(int, GRID.dims)), "voxel_size": float(GRID.voxel_size),
                     "origin": [float(x) for x in GRID.origin], "frame": GRID.frame},
            "anchors": {"min_separation_frames": MIN_ANCHOR_SEPARATION_FRAMES,
                        "target_n": TARGET_N_ANCHORS,
                        "independent_scene_count": 6,
                        "note": "the anchor count is NOT the independent count"},
            "val_sequences": list(VAL_SEQUENCES)}


__all__ = [k for k in dir() if not k.startswith("_")]
