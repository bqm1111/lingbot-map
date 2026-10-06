"""Single source of truth for the OccAny comparison.

The question: is the frozen current baseline (SemanticKITTI 26.25 % / Occ3D 31.56 % seed medians on
our own 163 / 1,182 anchors) comparable to OccAny's published 25.91 % / 23.55 %? Those numbers
differ in sample set, post-processing, temporal direction and (Occ3D) benchmark placement at once.
This experiment removes each difference explicitly:

* **samples** -- every sample OccAny itself scored (805 SemanticKITTI 08 frames, 4,819 Occ3D val
  samples), read from the released model's saved predictions on this machine;
* **ground truth** -- OccAny's stored ``voxel_label`` for *both* methods (checked against an
  independent loader of the official files, per sample);
* **scorer** -- OccAny's own ``SSCMetrics.get_score_completion``;
* **post-processing** -- raw *and* OccAny's geometry pooling (3x3x3 max, pool-then-mask) for both;
* **temporal direction** -- ``forward``: OccAny's own five frames (target first, stream offsets
  0/2/4/6/8); ``causal``: our deployed five frames ending at the target;
* **benchmark placement** (Occ3D only) -- ``occany``: OccAny's static camera extrinsic
  (``T_cam_to_voxel`` in its output); ``ours``: our timestamp-compensated chain. SemanticKITTI
  shares one transform (asserted per sample).

Nothing is trained, selected or tuned: the three frozen checkpoints and threshold 0 come from the
source freeze, which every GPU entry point verifies before it runs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from tools.baseline_reconstruction import config as BR

REPO_ROOT = BR.REPO_ROOT
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "occany_comparison")
EVAL_DIR = os.path.join(ARTIFACT_DIR, "eval")
FIGURE_DIR = os.path.join(ARTIFACT_DIR, "figures")
LOG_DIR = os.path.join(ARTIFACT_DIR, "logs")
DATA_ROOT = os.environ.get("OCCANY_COMPARISON_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_occany_comparison")
#: five-frame LingBot clips for every (sample, protocol), written by cli/build_clips.py
CLIPS_ROOT = os.path.join(DATA_ROOT, "clips")
#: MoGe-B frames the frozen gate-7B cache lacks (frames past our 40-frame Occ3D streams)
MOGE_EXTRA_ROOT = os.path.join(DATA_ROOT, "moge_b")
NUSCENES_ROOT = "/media/SSD1/MINH_DATASETS/nuscenes"
OCCANY_SK_LEARNING_MAP = "/home/minh/workspace/OccAny/occany/datasets/semantic_kitti.yaml"

DATASETS: Tuple[str, ...] = ("semantickitti", "occ3d")
IGNORE = 255


@dataclass(frozen=True)
class OccAnyRun:
    """One released-OccAny extraction on this machine (see artifacts/gate8c1/occany_headtohead_verified.json)."""
    dataset: str
    pred_dir: str
    pred_key: str             # full completion: reconstruction + novel-view generation
    n_classes: int
    empty_class: int
    conf_threshold: float     # OccAny's per-dataset setting, kept as released
    published_sc_iou: float   # OccAny v2 Table 1, five frames
    expected_samples: int


_OCCANY_OUT = "/media/SSD1/MINH_DATASETS/occany_out/ssc_voxel_pred"
OCCANY: Dict[str, OccAnyRun] = {
    "semantickitti": OccAnyRun("semantickitti", f"{_OCCANY_OUT}/OccAny_5frames_kitti512_rot60_vpi10_fwd3_sTrans2",
                               "render_recon_gen_recon2.5_gen2.5", 20, 0, 2.5, 0.2591, 805),
    "occ3d": OccAnyRun("occ3d", f"{_OCCANY_OUT}/OccAny_5frames_nuscenes512_rot60_vpi10_fwd3_sTrans2",
                       "render_recon_gen_recon1.1_gen1.1", 18, 17, 1.1, 0.2355, 4819),
}


@dataclass(frozen=True)
class Protocol:
    name: str
    offsets: Tuple[int, ...]  # stream offsets from the target frame, in temporal order
    anchor_slot: int          # which of the five frames is the target (the map's origin)
    description: str


PROTOCOLS: Dict[str, Protocol] = {
    "forward": Protocol("forward", (0, 2, 4, 6, 8), 0,
                        "OccAny's input: target first, four later frames (SemanticKITTI raw +0..+40, ~4.1 s; "
                        "Occ3D samples +0..+8, ~4 s). Non-causal; matches OccAny exactly."),
    "causal": Protocol("causal", (-4, -3, -2, -1, 0), 4,
                       "our deployed input: five frames ending at the target (SemanticKITTI raw -20..0, "
                       "~2.1 s; Occ3D samples -4..0, ~2 s). No future observation."),
}
#: benchmark placement of the finished map. SemanticKITTI: one shared transform.
CALIBRATIONS: Dict[str, Tuple[str, ...]] = {"semantickitti": ("shared",), "occ3d": ("occany", "ours")}
POSTPROCESS: Tuple[str, ...] = ("raw", "pooled")
SEEDS: Tuple[int, ...] = BR.SEEDS
CHECKPOINT_STEP = 6000
OCCUPANCY_THRESHOLD = BR.OCCUPANCY_THRESHOLD

#: sample subsets every table is reported on
SUBSETS: Tuple[str, ...] = ("occany_all", "causal_eligible", "official_anchors")
SUBSET_DESCRIPTIONS: Dict[str, str] = {
    "occany_all": "every sample OccAny scored (805 / 4,819)",
    "causal_eligible": "samples with four earlier stream frames (the causal protocol needs them)",
    "official_anchors": "our reported anchors (163 / 1,182) that OccAny also scored and both protocols can run",
}

#: predeclared primary comparison (written to predeclaration.md before any score exists)
PRIMARY = {"subset": "occany_all", "protocol": "forward", "post": "pooled",
           "calibration": {"semantickitti": "shared", "occ3d": "occany"}}

#: paired block bootstrap of (ours seed-mean IoU - OccAny IoU)
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 0
#: SemanticKITTI blocks: 25 consecutive target frames (125 raw frames, ~12.5 s); Occ3D blocks: one scene
SK_BLOCK_FRAMES = 25


def method_keys(dataset: str) -> List[str]:
    """Ordered score keys: ``occany|post``, ``ours|protocol|calibration|post|seedN``, ``maponly|protocol|calibration|post``."""
    keys = [f"occany|{post}" for post in POSTPROCESS]
    for p in PROTOCOLS:
        for c in CALIBRATIONS[dataset]:
            for post in POSTPROCESS:
                keys += [f"ours|{p}|{c}|{post}|seed{s}" for s in SEEDS] + [f"maponly|{p}|{c}|{post}"]
    return keys


def clip_path(dataset: str, protocol: str, sample_id: str) -> str:
    return os.path.join(CLIPS_ROOT, dataset, protocol, f"{sample_id}.npz")


def shard_path(dataset: str, shard: int, shards: int) -> str:
    return os.path.join(EVAL_DIR, f"{dataset}_shard{shard}of{shards}.npz")


__all__ = [k for k in dir() if not k.startswith("_")]
