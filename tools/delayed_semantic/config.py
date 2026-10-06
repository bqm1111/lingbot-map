"""Causal semantic completion with delayed observation supervision -- pilot. Single source of truth.

Reused, never modified: Gate 15 (``tools/semantic_completion``: grid, evidence lifting, frozen geometry
loader, nearest-evidence rule, confusion metrics) and the Current Geometry-Only Baseline it loads.

Grid (every flat tensor in this package): SSCBench-KITTI-360 velodyne grid of the anchor frame t,
dims (256, 256, 32), 0.2 m, origin (0, -25.6, -2) m, flat = ravel_multi_index((x, y, z)), C order.

Time: a *stream frame* is one frame of the frozen LingBot stream (native KITTI-360 stride 5, ~0.5 s).
``t`` is the stream index of the prediction frame; ``time_s`` is the KITTI-360 image timestamp.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

from tools.semantic_completion import config as G15

REPO_ROOT = G15.REPO_ROOT
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "delayed_semantic_pilot")
DATA_ROOT = os.environ.get("DELAYED_SEM_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_delayed_semantic")
PROTOCOL_VERSION = "delayed_semantic_pilot/2026-09-14"

TRAIN_DRIVES: Tuple[str, ...] = G15.TRAIN_DRIVES          # 0003, 0007, 0010
DEV_DRIVE = G15.VAL_DRIVE                                  # 0006 -- development evaluation, inspected before
GRID_DIMS, N_VOXELS, VOXEL_SIZE_M, GRID_ORIGIN_M = G15.GRID_DIMS, G15.N_VOXELS, G15.VOXEL_SIZE_M, G15.GRID_ORIGIN_M
TEACHER_DIM, N_CLASSES = G15.TEACHER_DIM, G15.N_CLASSES

#: history H_t of every student component: the five stream frames t-4..t (the frozen input budget)
HISTORY_OFFSETS: Tuple[int, ...] = (-4, -3, -2, -1, 0)
CURRENT_SLOT = 4
#: teacher temperature: Trident-H's own logit scale (tools/feature_first_3d_semantics/common.py)
LOGIT_SCALE = 40.0
GEOMETRY_SEED = 0
HEAD_SEED = 0


@dataclass(frozen=True)
class TargetConfig:
    """Delayed target construction (offline, privileged, training/diagnostics only)."""
    horizon_frames: int = 20            # future stream frames t+1..t+20 (the frozen Gate 8 privileged horizon)
    horizon_s: float = 10.0             # and at most 10 s after t (stream gaps never stretch the window)
    min_camera_depth_m: float = 1.0
    zbuffer_cell_px: int = 4            # occlusion test on native 1408x376 pixels
    zbuffer_tol_m: float = 0.5          # a return is camera-visible iff z <= zmin(cell) + max(tol_m, tol_rel * zmin)
    zbuffer_tol_rel: float = 0.05
    persistence_min_span_s: float = 1.0 # static support: LiDAR returns in the voxel at least 1 s apart
    lru_frames: int = 40


@dataclass(frozen=True)
class ModelConfig:
    """Learned evidence association (B1 and B2 share it exactly)."""
    k_candidates: int = 16              # nearest observed evidence voxels (Euclidean) per query
    evidence_proj: int = 16             # vocabulary-free projection of the 1024-d evidence feature
    hidden: int = 64
    path_samples: int = 8               # samples along the query->candidate segment
    range_norm_m: float = 6.0
    trunk_channels: int = 24


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 3000
    anchors_per_step: int = 8
    hidden_queries: int = 2048          # per anchor: hidden observed evidence (observed-only term, B1 and B2)
    delayed_queries: int = 2048         # per anchor: delayed targets (B2 only)
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    hide_block_vox: Tuple[int, int, int] = (8, 8, 4)   # Gate 15 hiding setting
    hide_block_prob: float = 0.5
    log_every: int = 50
    ckpt_every: int = 1000


@dataclass(frozen=True)
class Paths:
    provenance_root: str = os.path.join(DATA_ROOT, "provenance")
    targets_root: str = os.path.join(DATA_ROOT, "delayed_targets")
    predictions_root: str = os.path.join(DATA_ROOT, "predictions_at_t")
    g15: G15.Paths = G15.Paths()
    checkpoints: str = os.path.join(ARTIFACT_DIR, "checkpoints")
    runs: str = os.path.join(ARTIFACT_DIR, "runs")
    eval: str = os.path.join(ARTIFACT_DIR, "eval")
    reliability: str = os.path.join(ARTIFACT_DIR, "reliability")

    def provenance_path(self, drive: str, t: int) -> str:
        return os.path.join(self.provenance_root, drive, f"{t:05d}.npz")

    def target_path(self, drive: str, t: int) -> str:
        return os.path.join(self.targets_root, drive, f"{t:05d}.npz")

    def head_checkpoint(self, method: str, step: int) -> str:
        return os.path.join(self.checkpoints, f"{method}_seed{HEAD_SEED}_step{step:05d}.pt")


#: student prediction path (clips, hard map, evidence, frozen geometry, head): no future, LiDAR, poses, labels
PREDICT_FORBIDDEN: Tuple[str, ...] = G15.TRAIN_FORBIDDEN_TOKENS + ("delayed_targets", "lingbot_k360_7drives", "data_poses")
#: training loop: may read delayed targets (privileged supervision), never human labels, LiDAR or target datasets
TRAIN_FORBIDDEN: Tuple[str, ...] = ("data/kitti/dataset", "occ3d_gt", "nuscenes", "lingbot_occ3d_zeroshot", "scale_gate/cache",
                                   "preprocess/labels", "eval_masks", "samples_gtsem", "data_3d_semantics", "velodyne",
                                   "data_3d_raw", "lingbot_gate12_suf/regions")
#: target construction: LiDAR, GT poses, calibration allowed; human labels and target datasets not
TARGET_FORBIDDEN: Tuple[str, ...] = ("data/kitti/dataset", "occ3d_gt", "nuscenes", "lingbot_occ3d_zeroshot", "scale_gate/cache",
                                     "preprocess/labels", "eval_masks", "samples_gtsem", "data_3d_semantics", "/voxels/",
                                     "_1_1.npy", "lingbot_gate12_suf/regions")
#: development evaluation: target datasets only
EVAL_FORBIDDEN: Tuple[str, ...] = G15.EVAL_FORBIDDEN_TOKENS

DISTANCE_EDGES_M: Tuple[float, ...] = G15.DISTANCE_EDGES_M
DISTANCE_NAMES: Tuple[str, ...] = G15.DISTANCE_NAMES
REVEAL_EDGES_S: Tuple[float, ...] = (1.0, 2.0, 4.0)
REVEAL_NAMES: Tuple[str, ...] = ("0-1 s", "1-2 s", "2-4 s", "4-10 s", "not revealed")


def ensure_dirs() -> None:
    p = Paths()
    for d in (ARTIFACT_DIR, p.checkpoints, p.runs, p.eval, p.reliability, os.path.join(ARTIFACT_DIR, "tables"),
              os.path.join(ARTIFACT_DIR, "figures"), os.path.join(ARTIFACT_DIR, "logs"), p.provenance_root, p.targets_root, p.predictions_root):
        os.makedirs(d, exist_ok=True)


__all__ = [k for k in dir() if not k.startswith("_")]
