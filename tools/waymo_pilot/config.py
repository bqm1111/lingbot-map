"""Single source of truth for the Waymo supervision pilot.

Nothing in this package hard-codes a path, cap, bound or drive list; every one is declared
here so the report, the checks and the CLI read the same objects.

What is reused unchanged (never redefined here):

* the causal input contract, grid and frozen geometry of ``tools/baseline_reconstruction``
  and ``tools/semantic_completion`` (five frames ``t-4..t`` of one camera ending at the
  anchor, SSCBench-KITTI-360 anchor velodyne grid, CompletionUNet seed 0 step 6000);
* the frozen semantic teacher Trident-H and its 18-phrase readout
  (``tools/semantic_completion/evidence/trident.py``);
* the privileged-target machinery of ``tools/delayed_semantic/targets.py`` (camera view,
  z-buffer occlusion rule, voxel binning) and the reveal record of
  ``tools/future_revealability/future_visibility.py``;
* the raw-LiDAR occupancy supervision of ``core/supervision/rawtarget.py``.

The pilot adds: a Waymo reader, a provenance-split privileged target, the §4 population
decomposition stated without human labels, and the statistics that compare the two.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

from tools.delayed_semantic import config as DS
from tools.semantic_completion import config as G15

VERSION = "waymo_pilot.1"
REPO_ROOT = G15.REPO_ROOT
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "waymo_supervision_pilot")
REPORT_PATH = os.path.join(REPO_ROOT, "reports", "waymo_supervision_pilot.md")
DATA_ROOT = os.environ.get("WAYMO_PILOT_DATA_ROOT",
                           "/media/SSD1/MINH_DATASETS/lingbot_waymo_pilot")

# --------------------------------------------------------------------------- #
# the frozen causal contract, inherited (asserted in checks.check_1)
# --------------------------------------------------------------------------- #
GRID_DIMS = G15.GRID_DIMS                   # (256, 256, 32)
N_VOXELS = G15.N_VOXELS                     # 2,097,152
VOXEL_SIZE_M = G15.VOXEL_SIZE_M             # 0.2
GRID_ORIGIN_M = G15.GRID_ORIGIN_M           # (0, -25.6, -2)
N_CLASSES = G15.N_CLASSES                   # 18
TEACHER_DIM = G15.TEACHER_DIM               # 1024
#: the historical causal contract: five frames of ONE camera ending at the anchor
CLIP_OFFSETS: Tuple[int, ...] = DS.HISTORY_OFFSETS      # (-4, -3, -2, -1, 0)
CLIP_LENGTH = len(CLIP_OFFSETS)
GEOMETRY_SEED = DS.GEOMETRY_SEED             # 0

# --------------------------------------------------------------------------- #
# sample size (§2). Caps, not targets: the pilot stops at them and reports coverage.
# --------------------------------------------------------------------------- #
MAX_SEGMENTS = 20
MAX_ANCHORS = 100
MAX_ANCHORS_PER_SEGMENT = 5
SMOKE_SEGMENTS = 2
SMOKE_ANCHORS = 10
#: anchors inside one segment are spread by at least this many stream frames, so two
#: anchors of a segment never share a causal clip or a privileged window
MIN_ANCHOR_SEPARATION_FRAMES = 40
SAMPLE_SEED = 0

#: KITTI-360 reference sample: the official SSCBench *train* drives only. 0006 is
#: development data and is never sampled here (asserted in sample.py).
K360_SAMPLE_DRIVES: Tuple[str, ...] = DS.TRAIN_DRIVES
#: every KITTI-360 training drive with the caches this pilot needs. The frozen recipe trains
#: on the three above; these four are available and unused, and they are what lets the pilot
#: vary SCENE COUNT inside KITTI-360 while the dataset, teacher and recipe stay fixed.
K360_EXTRA_DRIVES: Tuple[str, ...] = ("2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync",
                                      "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync")
K360_ALL_TRAIN_DRIVES: Tuple[str, ...] = tuple(sorted(set(K360_SAMPLE_DRIVES) | set(K360_EXTRA_DRIVES)))
K360_DEV_DRIVE = DS.DEV_DRIVE

# --------------------------------------------------------------------------- #
# engineering caps (§2). Exceeding one stops expansion; it never lowers a threshold.
# --------------------------------------------------------------------------- #
MAX_DERIVED_GIB = 20.0
MAX_PREPROCESS_GPU_HOURS = 2.0

# --------------------------------------------------------------------------- #
# privileged supervision window (§3)
# --------------------------------------------------------------------------- #
#: W-temporal / KITTI-360-reference keep the EXISTING recipe and its original limits.
EXISTING_TARGET_CFG = DS.TargetConfig()      # 20 future stream frames, <= 10 s, z-buffer rule
#: the bounded ADDITIONAL window of the extended variant, before and after the anchor
EXTENDED_WINDOW_S = 2.0
#: maximum |image timestamp - LiDAR timestamp| accepted for one observation, seconds.
#: KITTI-360 measures < 5 ms (tools/delayed_semantic/targets.py); Waymo is recorded per
#: frame from ``pose_timestamp`` and is never assumed.
MAX_SYNC_ERROR_S = 0.05

#: Supervision sources, kept apart so "new environment" is never confused with
#: "new target recipe". Every target voxel records which of these produced it.
SOURCE_FUTURE = "future_selected_camera"       # the existing recipe
SOURCE_PRECLIP = "preclip_history"             # earlier than the causal clip, within the bound
SOURCE_INTERLEAVED = "interleaved_temporal"    # frames the stride-N stream skips, within the bound
SOURCE_OTHER_CAMERA = "other_camera"           # synchronized additional cameras, within the bound
SOURCES: Tuple[str, ...] = (SOURCE_FUTURE, SOURCE_PRECLIP, SOURCE_INTERLEAVED,
                            SOURCE_OTHER_CAMERA)
#: which sources each variant may use
VARIANTS: Dict[str, Tuple[str, ...]] = {
    "k360_reference": (SOURCE_FUTURE,),
    "w_temporal": (SOURCE_FUTURE,),
    "w_extended": SOURCES,
    "k360_7drives": (SOURCE_FUTURE,),
}

# --------------------------------------------------------------------------- #
# Waymo (§2). Every field below was read from the installed SDK or from the local
# reference implementation; see waymo.py for the provenance of each.
# --------------------------------------------------------------------------- #
WAYMO_ROOT = os.environ.get("WAYMO_ROOT", "/media/SSD1/MINH_DATASETS/waymo_perception_v1_4_2")
WAYMO_EXPECTED_VERSION = "perception v1.4.2 (individual_files/training/*.tfrecord)"
WAYMO_SDK_PACKAGE = "waymo-open-dataset-tf-2-12-0"
#: The reader environment, built on SSD1 because ``/`` is nearly full and TF is ~1.8 GiB.
#: Installed with ``--no-deps`` plus tensorflow/numpy/absl/immutabledict: the SDK's own
#: metadata pins ``jaxlib==0.4.13``, which has been withdrawn from PyPI and is unsatisfiable.
#: Nothing this pilot calls (``dataset_pb2``, ``frame_utils``) needs jax.
WAYMO_ENV = os.environ.get("WAYMO_ENV", "/media/SSD1/MINH_DATASETS/envs/waymo")
#: Google Cloud SDK. Standalone (pip) gsutil no longer supports user-account OAuth, so the
#: licence-holder's own account can only authenticate through gcloud.
GCLOUD = os.environ.get("GCLOUD_HOME", "/media/SSD1/MINH_DATASETS/envs/google-cloud-sdk")
WAYMO_SELECTED_CAMERA = "FRONT"                  # CameraName.Name == 1
WAYMO_OTHER_CAMERAS: Tuple[str, ...] = ("FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT")
#: Waymo runs at 10 Hz. The causal clip must keep the existing TEMPORAL protocol, so the
#: stream is decimated to one frame in ``WAYMO_STREAM_STRIDE``; 5 gives 0.50 s against
#: KITTI-360's measured 0.52 s. Recorded, never silently inherited (§1).
WAYMO_STREAM_STRIDE = 5
WAYMO_NOMINAL_HZ = 10.0


@dataclass(frozen=True)
class Paths:
    data_root: str = DATA_ROOT
    artifacts: str = ARTIFACT_DIR
    #: per-frame Waymo cache written by waymo_extract.py (the only TF-dependent step)
    waymo_frames: str = os.path.join(DATA_ROOT, "waymo_frames")
    #: Trident-H token maps of Waymo images, written by waymo_tokens.py
    waymo_tokens: str = os.path.join(DATA_ROOT, "waymo_tokens")
    #: per-anchor pilot measurement, one npz per (variant, scene, anchor)
    pairs: str = os.path.join(DATA_ROOT, "pairs")
    figures: str = os.path.join(ARTIFACT_DIR, "figures")
    examples: str = os.path.join(ARTIFACT_DIR, "examples")
    logs: str = os.path.join(ARTIFACT_DIR, "logs")
    g15: G15.Paths = field(default_factory=G15.Paths)

    def waymo_frame_path(self, segment: str, frame: int) -> str:
        return os.path.join(self.waymo_frames, segment, f"{frame:05d}.npz")

    def waymo_token_path(self, segment: str, frame: int, camera: str) -> str:
        return os.path.join(self.waymo_tokens, segment, f"{frame:05d}_{camera}.npz")

    def pair_path(self, variant: str, scene: str, anchor: int) -> str:
        return os.path.join(self.pairs, variant, scene, f"{anchor:05d}.npz")


# --------------------------------------------------------------------------- #
# leakage control (§6). Evaluation benchmarks never appear in this pilot.
# --------------------------------------------------------------------------- #
#: forbidden while building causal inputs: future frames, LiDAR, GT poses, labels, targets
CAUSAL_FORBIDDEN: Tuple[str, ...] = DS.PREDICT_FORBIDDEN
#: forbidden while building privileged targets: human labels and the target benchmarks.
#: LiDAR, GT poses, calibration and other cameras are allowed here and only here.
TARGET_FORBIDDEN: Tuple[str, ...] = DS.TARGET_FORBIDDEN
#: never opened anywhere in this pilot
BENCHMARK_FORBIDDEN: Tuple[str, ...] = ("data/kitti/dataset", "occ3d_gt", "nuscenes",
                                        "lingbot_occ3d_zeroshot", "scale_gate/cache",
                                        "semantickitti")
#: human-annotation sources. Allowed ONLY inside the marked teacher-correctness diagnostic.
ANNOTATION_PATHS: Tuple[str, ...] = ("eval_masks", "preprocess/labels", "data_3d_semantics",
                                     "samples_gtsem")

#: distance-to-nearest-causal-evidence bins, the frozen ones
DISTANCE_EDGES_M = G15.DISTANCE_EDGES_M      # (1, 2, 4, 8)
DISTANCE_NAMES = G15.DISTANCE_NAMES
#: metric range bins (horizontal distance from the anchor sensor), for §5
RANGE_EDGES_M: Tuple[float, ...] = (10.0, 20.0, 30.0, 40.0)
RANGE_NAMES: Tuple[str, ...] = ("0-10 m", "10-20 m", "20-30 m", "30-40 m", ">40 m")
CLASS_NAMES: Tuple[str, ...] = ("car", "bicycle", "motorcycle", "truck", "other-vehicle",
                                "person", "road", "parking", "sidewalk", "other-ground",
                                "building", "fence", "vegetation", "terrain", "pole",
                                "traffic-sign", "other-structure", "other-object")
THING_LABELS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)


def anchor_frame_budget() -> Dict[str, int]:
    """How many NATIVE Waymo frames one anchor needs around it, and where anchors can sit.

    The causal clip reaches ``(CLIP_LENGTH - 1) * WAYMO_STREAM_STRIDE`` native frames back.
    The privileged side reaches ``horizon_frames * WAYMO_STREAM_STRIDE`` forward under the
    existing recipe, and ``EXTENDED_WINDOW_S * WAYMO_NOMINAL_HZ`` either way under the
    extended one. An extraction shorter than ``span`` supports **no anchors at all**, which
    is worth computing rather than discovering after a download.
    """
    ext = int(round(EXTENDED_WINDOW_S * WAYMO_NOMINAL_HZ))
    back = max((CLIP_LENGTH - 1) * WAYMO_STREAM_STRIDE, ext)
    fwd = max(EXISTING_TARGET_CFG.horizon_frames * WAYMO_STREAM_STRIDE, ext)
    return {"frames_back": back, "frames_forward": fwd, "span": back + fwd + 1,
            "first_anchor_frame": back, "extended_window_frames": ext}


def anchors_supported(n_frames: int) -> int:
    """Anchors an extraction of ``n_frames`` consecutive native frames can support."""
    b = anchor_frame_budget()
    return max(0, n_frames - b["frames_back"] - b["frames_forward"])


def ensure_dirs() -> None:
    p = Paths()
    for d in (ARTIFACT_DIR, p.figures, p.examples, p.logs, os.path.dirname(REPORT_PATH),
              p.waymo_frames, p.waymo_tokens, p.pairs):
        os.makedirs(d, exist_ok=True)


__all__ = [k for k in dir() if not k.startswith("_")]
