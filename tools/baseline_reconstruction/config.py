"""Single source of truth for the baseline-reconstruction experiment.

Every path, drive list, seed, grid convention, threshold and protocol version that the
experiment depends on is declared here as a frozen dataclass. Modules never hard-code
these; the CLI, the tests and the freeze record all read the same objects.

Two phases share this file:

* **historical** -- the Gate 8C-1 sky-fix baseline as recorded on disk (never rerun here);
* **current**    -- the geometry-only baseline trained on the frozen local-reset geometry.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "baseline_reconstruction")
DATA_ROOT = os.environ.get("BASELINE_RECON_DATA_ROOT",
                           "/media/SSD1/MINH_DATASETS/lingbot_baseline_reconstruction")

PROTOCOL_VERSION = "baseline_reconstruction/2026-09-14"
#: what this experiment calls the model it trains
CURRENT_BASELINE_NAME = "Current Geometry-Only Baseline (L + LB-M + H, CompletionUNet-3ch)"


# --------------------------------------------------------------------------- #
# historical (Gate 8C-1, post ceiling fix)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HistoricalPaths:
    """Where the post-ceiling-fix Gate 8C-1 record lives. Read-only."""
    gate_artifacts: str = os.path.join(REPO_ROOT, "artifacts", "gate8c1")
    #: the adopted post-fix model; the pre-fix record is ``frozen_manifest.json``
    sky_manifest: str = os.path.join(REPO_ROOT, "artifacts", "gate8c1", "frozen_manifest_sky.json")
    prefix_manifest: str = os.path.join(REPO_ROOT, "artifacts", "gate8c1", "frozen_manifest.json")
    padfix_manifest: str = os.path.join(REPO_ROOT, "artifacts", "gate8c1", "frozen_manifest_padfix.json")
    sky2_manifest: str = os.path.join(REPO_ROOT, "artifacts", "gate8c1", "frozen_manifest_sky2.json")
    #: the only on-disk evaluation of the sky model on the full official anchor sets
    #: (gitignored: exists on this machine only)
    sky_eval_dir: str = os.path.join(REPO_ROOT, "output", "skyfix", "eval")
    sky_eval_logs: str = os.path.join(REPO_ROOT, "output", "skyfix")
    #: per-anchor prediction dump of the median seed (SemanticKITTI, past5, seed 2)
    sky_pred_dump: str = os.path.join(REPO_ROOT, "output", "skyfix", "pred_seed2")
    prefix_eval_dir: str = os.path.join(REPO_ROOT, "artifacts", "gate8c1")
    ceiling_fix_note: str = os.path.join(REPO_ROOT, "artifacts", "gate8c1", "ceiling_fix.md")
    ceiling_bug_report: str = os.path.join(REPO_ROOT, "reports", "the_ceiling_bug.md")
    method_report: str = os.path.join(REPO_ROOT, "reports", "METHOD.md")
    #: KITTI-360 supervision roots. ``sky`` trained the canonical model.
    targets_prefix: str = "/media/SSD1/MINH_DATASETS/lingbot_gate8c1/targets"
    targets_sky: str = "/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky/targets"
    targets_sky2: str = "/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky2/targets"
    targets_current: str = "/media/SSD1/MINH_DATASETS/lingbot_k360_7drives/targets"
    evaluator: str = "core/run/eval_target.py"
    #: git commits that bracket the fix (see historical/provenance.py for what each is)
    commit_prefix_code: str = "9a648322e6dc4e34995f0530650c2fa469779676"
    commit_first_tracked_tree: str = "f7b7552"
    commit_sky_rule_measured_fix: str = "647bcf3"
    commit_docs_update: str = "0ba3b5d"


#: the headline protocol of record for the historical numbers
HISTORICAL_PROTOCOL = "past5"
HISTORICAL_SEEDS: Tuple[int, ...] = (0, 1, 2)
HISTORICAL_TARGETS: Tuple[str, ...] = ("semantickitti", "occ3d")


# --------------------------------------------------------------------------- #
# current baseline
# --------------------------------------------------------------------------- #
TRAIN_DRIVES: Tuple[str, ...] = ("2013_05_28_drive_0003_sync", "2013_05_28_drive_0007_sync",
                                 "2013_05_28_drive_0010_sync")
VAL_DRIVE = "2013_05_28_drive_0006_sync"
SEEDS: Tuple[int, ...] = (0, 1, 2)
#: the KITTI-360 anchor grid every source volume lives on (SSCBench-KITTI-360 velodyne frame)
GRID_DIMS: Tuple[int, int, int] = (256, 256, 32)
VOXEL_SIZE_M = 0.2
#: five causal frames t-4..t; offsets are in *stream* index of each dataset's stream
CLIP_OFFSETS: Tuple[int, ...] = (-4, -3, -2, -1, 0)
#: the frozen decision rule of Gates 12/13: occupied := final log-odds > 0 with base 0
OCCUPANCY_THRESHOLD = 0.0
#: range bins by anchor-camera z of the voxel centre (metres)
RANGE_EDGES_M: Tuple[float, ...] = (5.0, 10.0, 20.0, 30.0)
RANGE_NAMES: Tuple[str, ...] = ("0-5 m", "5-10 m", "10-20 m", "20-30 m", ">30 m")


@dataclass(frozen=True)
class CurrentPaths:
    """Frozen caches the current baseline consumes, plus where it writes."""
    #: local-reset five-frame L clips (frozen LingBot + causal MoGe-2 clip scale)
    clips_train_root: str = "/media/SSD1/MINH_DATASETS/lingbot_gate12_suf/local_lingbot"
    clips_val_root: str = "/media/SSD1/MINH_DATASETS/lingbot_local_registration_audit/local_lingbot"
    #: the frozen Gate 12 H input cache: S_H / F_H from the clips through the hard mapper
    inputs_root: str = "/media/SSD1/MINH_DATASETS/lingbot_gate12_suf/inputs"
    #: raw-LiDAR occupancy targets built by core/supervision/rawtarget.py (current rule)
    targets_root: str = "/media/SSD1/MINH_DATASETS/lingbot_k360_7drives/targets"
    #: 0006 evaluation regions from Gate 12 (valid / occupied / past5 / unobs5_later ...)
    regions_root: str = "/media/SSD1/MINH_DATASETS/lingbot_gate12_suf/regions"
    checkpoints: str = os.path.join(ARTIFACT_DIR, "checkpoints")
    runs: str = os.path.join(ARTIFACT_DIR, "runs")
    source_eval: str = os.path.join(ARTIFACT_DIR, "source_eval")
    #: target-side caches written after the source freeze only
    target_clips_root: str = os.path.join(DATA_ROOT, "target_clips")
    target_eval: str = os.path.join(ARTIFACT_DIR, "zero_shot_eval")
    training_config: str = os.path.join(ARTIFACT_DIR, "training_config.yaml")
    #: the adopted recipe (Gate 12 condition H == Gate 13 condition M)
    recipe_source: str = os.path.join(REPO_ROOT, "artifacts", "gate12_suf", "training_config_H.yaml")
    gate13_results: str = os.path.join(REPO_ROOT, "artifacts", "gate13_oracle", "results_table.md")
    gate13_eval_dir: str = os.path.join(REPO_ROOT, "artifacts", "gate13_oracle", "eval")

    def clip_path(self, drive: str, t: int) -> str:
        root = self.clips_val_root if drive == VAL_DRIVE else self.clips_train_root
        return os.path.join(root, drive, f"{t:05d}.npz")

    def input_path(self, drive: str, t: int) -> str:
        return os.path.join(self.inputs_root, drive, f"{t:05d}.npz")

    def target_path(self, drive: str, t: int) -> str:
        return os.path.join(self.targets_root, drive, f"{t:05d}.npz")

    def region_path(self, t: int) -> str:
        return os.path.join(self.regions_root, VAL_DRIVE, f"{t:05d}.npz")

    def checkpoint_path(self, seed: int, step: int) -> str:
        return os.path.join(self.checkpoints, f"geo_seed{seed}_step{step:05d}.pt")

    def target_clip_path(self, dataset: str, segment: str, anchor_index: int) -> str:
        return os.path.join(self.target_clips_root, dataset, segment, f"{anchor_index:05d}.npz")


#: Source-side reproduction tolerance against Gate 13 M (predeclared; Gate 12 H vs Gate 13 M
#: differed by <= 0.0011 per seed on the identical recipe, so this is ~3x that spread).
SOURCE_REPRODUCTION_TOLERANCE: Dict[str, float] = {"mean_full_iou_abs": 0.003, "per_seed_full_iou_abs": 0.005}
#: Gate 13 M full-region IoU per seed (artifacts/gate13_oracle/results_table.md), the expected regime
GATE13_M_FULL_IOU: Dict[int, float] = {0: 0.1354, 1: 0.1336, 2: 0.1346}


# --------------------------------------------------------------------------- #
# zero-shot target protocol (frozen constants inherited from core/, restated for the record)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TargetProtocol:
    dataset: str
    stream_rule: str
    camera: str
    prediction_grid: str
    evaluation_grid: str
    evaluation_mask: str
    occupancy_target: str
    reduction: str
    anchors_expected: int


TARGET_PROTOCOLS: Dict[str, TargetProtocol] = {
    "semantickitti": TargetProtocol(
        dataset="semantickitti",
        stream_rule="sequence 08, image_2, stride-5 stream of 815 unique frames; anchor = last frame of "
                    "each official clip; clip = stream indices i-4..i (native frames 20 apart end to end)",
        camera="image_2 (left colour camera)",
        prediction_grid="256x256x32 @ 0.2 m, origin (0, -25.6, -2), velodyne frame of the anchor",
        evaluation_grid="same as prediction grid",
        evaluation_mask="official voxels/<frame>.invalid == 0 (label != 255)",
        occupancy_target="official label != 0 inside the mask",
        reduction="none",
        anchors_expected=163),
    "occ3d": TargetProtocol(
        dataset="occ3d",
        stream_rule="150 official validation scenes, CAM_FRONT samples in timestamp order, one stream "
                    "per scene; anchor = every sample with a label; clip = stream indices i-4..i",
        camera="CAM_FRONT only (the frozen single-camera protocol)",
        prediction_grid="canonical 400x400x32 @ 0.2 m, origin (-40, -40, -1), ego frame of the anchor sample",
        evaluation_grid="native 200x200x16 @ 0.4 m",
        evaluation_mask="mask_camera == 1, lidar mask off, ego x-index < 100 ignored (single front camera)",
        occupancy_target="semantics != 17 (free) inside the mask",
        reduction="native voxel occupied iff any of its 8 canonical subvoxels is occupied",
        anchors_expected=1182),
}

#: comparability classes, as the brief defines them
COMPARABILITY_CLASSES: Tuple[str, ...] = ("fully comparable", "partially comparable", "not comparable")


def ensure_dirs() -> None:
    p = CurrentPaths()
    for d in (ARTIFACT_DIR, os.path.join(ARTIFACT_DIR, "tables"), os.path.join(ARTIFACT_DIR, "figures"),
              os.path.join(ARTIFACT_DIR, "logs"), p.checkpoints, p.runs, p.source_eval):
        os.makedirs(d, exist_ok=True)


__all__ = [k for k in dir() if not k.startswith("_")]
