"""Gate 8D: the protocol, fixed before any target is built or any weight is trained.

Everything the experiment is allowed to decide in advance lives here as a constant. The
module is hashed into ``artifacts/gate8d/protocol.json``; training refuses to start once a
checkpoint exists and the hash has moved. That is the mechanism that stops the design from
drifting in response to a result.

Nothing in this module may import a target-domain loader or name a target dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict

GATE = "8D"

# ---- sources ------------------------------------------------------------------------
TRAIN_DRIVES = ("2013_05_28_drive_0003_sync",
                "2013_05_28_drive_0007_sync",
                "2013_05_28_drive_0010_sync")
VAL_DRIVE = "2013_05_28_drive_0006_sync"

# ---- privileged future window --------------------------------------------------------
#: unchanged from Gate 8C-1: the target may see 20 stream frames ahead of the anchor
FUTURE_STREAM_FRAMES = 20
#: Gate 8C-1 used only the stride-5 stream sweeps inside that window. Gate 8D uses every
#: native KITTI-360 sweep in the same time interval -- the horizon is identical, the
#: sampling within it is dense.
NATIVE_STRIDE = 1

# ---- LiDAR evidence (unchanged rules, denser sampling) --------------------------------
CARVE_NEAR_M = 1.0
BAND_HALF_M = 0.2
CARVE_DECIMATION = 4
#: How many distinct sweeps must report an endpoint before that endpoint outranks a
#: grazing free carve from another sweep.
#:
#: Gate 8C-1 demoted *any* occupied/free disagreement to UNKNOWN. With 21 sweeps that was
#: sound; with 101 it is degenerate -- a voxel is five times more likely to be grazed by
#: some ray from some viewpoint. The Phase 3 audit measured the consequence on KITTI-360:
#: 49.9 % of Gate 8C-1's occupied voxels became UNKNOWN while **0.0 % became FREE**. The
#: surfaces were never contradicted, only disowned, and the pre-registered occupied-recall
#: gate failed at 50.9 %.
#:
#: The value is 1: an endpoint always outranks a grazing carve. That is not a new rule, it
#: is PRECEDENCE read literally -- "consistent LiDAR endpoint" is rank 1, "consistent LiDAR
#: ray interior" is rank 2. Measured on KITTI-360 across 1/2/3/5, only 1 clears the
#: pre-registered 98 % recall floor (100.00 %, against 95.8 / 91.2 / 82.7), and it raises
#: occupied supervision from 1.72 M to 3.07 M voxels. The collision rate is unchanged at
#: 0.019 % in every case, because it measures pseudo-free voxels, which this does not touch.
#:
#: "Genuine conflict -> unknown" is therefore not vacuous but cross-source: a pseudo-teacher
#: may never overwrite a LiDAR endpoint, and ``n_pseudo_refused_by_lidar_endpoint`` counts
#: every time it tried. Amended after the Phase 3 audit, before any weight was trained.
MIN_OCC_FRAMES = 1

# ---- MoGe-2 free-space evidence ------------------------------------------------------
#: a MoGe ray carves free space only up to ``depth - margin(depth)``; the band
#: ``[depth - margin, depth + margin]`` is left unknown, and nothing beyond is touched.
#: ``margin`` is an empirical quantile of |MoGe - LiDAR| fitted on the TRAIN drives and
#: validated on the val drive -- never a hand-picked constant.
MOGE_MARGIN_QUANTILE = 0.95
MOGE_MARGIN_DEPTH_BINS_M = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 1e9)
MOGE_MIN_MARGIN_M = 0.4
MOGE_MAX_DEPTH_M = 40.0
MOGE_PIXEL_STRIDE = 4
#: a MoGe free vote counts only if this many future views agree the voxel is free
MOGE_MIN_VIEWS = 2
#: LiDAR uses EVERY native sweep in the horizon (NATIVE_STRIDE = 1). Image-based evidence
#: needs a cached depth or sky map per frame and a ray cast per pixel, so it is drawn at
#: the stream stride inside the SAME horizon -- 21 views per anchor. The horizon is
#: identical; only the per-frame cost differs. Amended before any target or weight existed.
IMAGE_EVIDENCE_NATIVE_STRIDE = 5

# ---- Trident-H sky-ray evidence ------------------------------------------------------
#: the KITTI-360 vocabulary has no sky class, so the sky teacher runs with the project's
#: fixed open-vocabulary phrase list extended by this one prompt. Fixed here, never tuned.
SKY_PROMPT = "sky"
SKY_MIN_PROB = 0.80
SKY_MIN_VIEWS = 3
#: rays are cast every ``SKY_PIXEL_STRIDE`` pixels. A KITTI-360-only probe showed this is
#: the binding constraint on ceiling supervision -- at stride 8 too few voxels are crossed
#: by enough distinct views to clear ``SKY_MIN_VIEWS`` (1.3 % of the top layer supervised),
#: at stride 4 it is 31 %. The confidence rule is unchanged; only the sampling is denser.
#: Amended before any target or weight existed, from source-side geometry alone.
SKY_PIXEL_STRIDE = 4
#: the teacher is cached at a finer stride so the used stride can be derived without
#: re-running the frozen model
SKY_CACHE_STRIDE = 2

# ---- evidence precedence -------------------------------------------------------------
PRECEDENCE = ("lidar_endpoint_occupied", "lidar_interior_free",
              "weighted_pseudo_free", "conflict_unknown", "no_evidence_unknown")
#: pseudo-free (MoGe or sky) is accepted only if LiDAR never called the voxel occupied
PSEUDO_FREE_MIN_WEIGHT = 2.0

# ---- surface-distance auxiliary task -------------------------------------------------
TUDF_TRUNCATION_M = 1.0
TUDF_LOSS_WEIGHT = 0.25

# ---- semantics -----------------------------------------------------------------------
SEM_MIN_TEACHER_PROB = 0.50
SEM_MIN_VIEWS = 2

# ---- architecture --------------------------------------------------------------------
ARCH = {"base": "gate8.net.CompletionUNet", "width": 24, "padding_mode": "replicate",
        "extra_outputs": {"surface_distance_m": 1},
        "semantic_head": "existing open-vocabulary logits, unchanged readout"}

# ---- training ------------------------------------------------------------------------
SEEDS = (0, 1, 2)
STEPS = 6000
CROP = (128, 128, 32)
BATCH_SIZE = 4
LR = 1e-3
WEIGHT_DECAY = 0.01

# ---- source-only stress variants (drive 0006) ----------------------------------------
STRESS = {"voxel_size_m": (0.15, 0.20, 0.30, 0.45),
          "history_frames": (5, 10, 20),
          "lidar_beam_keep": (1.0, 0.5),
          "scale_noise_log": (0.0, 0.05),
          "fov_scale": (1.0, 0.9)}

#: preregistered selection score. Weights fixed here, never revisited.
SELECTION_SCORE = {
    "mean_stress_sc_iou": 0.35, "worst_stress_sc_iou": 0.25,
    "volume_ratio_penalty": 0.15, "calibration_ece_penalty": 0.05,
    "semantic_teacher_agreement": 0.10, "completed_semantic_accuracy": 0.05,
    "surface_thickness_penalty": 0.05,
    "note": ("source AUROC is deliberately NOT the selection criterion: the Gate 8C-1 "
             "neighbourhood experiment raised source AUROC 0.68 -> 0.74 while Occ3D IoU "
             "fell 33.28 -> 28.12, so source ranking alone is not a transfer proxy.")}

# ---- target success criteria, fixed in advance ---------------------------------------
SUCCESS = {"semantickitti_sc_iou_min": 25.28,
           "semantickitti_sc_iou_stretch": 25.91,
           "occ3d_matched_forward_sc_iou_min": 30.0,
           "max_pred_over_gt_volume": 1.6,
           "must_improve_ssc_miou_over": ("gate8c1_plain_column", "mapper_dilate"),
           "must_improve_completed_naming_accuracy": True}

#: the target-construction acceptance gate, recorded before any target file is opened
TARGET_ACCEPTANCE = {"max_pseudo_free_collision_rate": 0.02,
                     "min_occupied_recall_vs_gate8c1": 0.98,
                     "note": ("collision = a voxel called free by MoGe/sky evidence that a "
                              "held-out later LiDAR sweep -- one not used to build that "
                              "target -- reports as an endpoint")}

EVAL_PROTOCOLS = ("past5_causal", "stream_all_past_causal", "occany_fwd_non_causal")


def source_digest() -> str:
    """Hash of this module's source: the protocol is what the file says."""
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def as_dict() -> Dict:
    g = globals()
    out = {k: v for k, v in g.items()
           if k.isupper() and not k.startswith("_")
           and isinstance(v, (str, int, float, tuple, dict, bool))}
    out["protocol_source_sha256"] = source_digest()
    return out


def dumps() -> str:
    return json.dumps(as_dict(), indent=2, sort_keys=True, default=str)


__all__ = ["as_dict", "dumps", "source_digest", "GATE", "TRAIN_DRIVES", "VAL_DRIVE"]
