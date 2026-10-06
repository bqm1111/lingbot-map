"""Semantic-field experiment (B4) -- single source of truth. Every value is predeclared in
``artifacts/semantic_field/source_predeclaration.md``.

Grid (source): SSCBench-KITTI-360 velodyne grid of the target frame t, 256 x 256 x 32 @ 0.2 m, flat C order.
Feature space: raw Trident-H token features, 1024-d, unit L2; text bank 1024-d unit L2; logits = 40 * cosine.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

from tools.delayed_semantic import config as PILOT

REPO_ROOT = PILOT.REPO_ROOT
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "semantic_field")
DATA_ROOT = os.environ.get("SEMANTIC_FIELD_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_semantic_field")
PROTOCOL_VERSION = "semantic_field_b4/2026-09-15"

TRAIN_DRIVES: Tuple[str, ...] = PILOT.TRAIN_DRIVES
DEV_DRIVE = PILOT.DEV_DRIVE
SEEDS: Tuple[int, ...] = (0, 1, 2)
GEOMETRY_SEED = PILOT.GEOMETRY_SEED


@dataclass(frozen=True)
class FeatureSpace:
    teacher_dim: int = 1024
    text_logit_scale: float = 40.0              # Trident-H's own temperature (tau = 1 / 40)
    centering: str = "normalize(f - mu), mu = mean raw unit evidence feature over training-drive evidence voxels"


@dataclass(frozen=True)
class HeadConfig:
    trunk_channels: int = 24                    # frozen CompletionUNet decoder d1
    geometry_channels: int = 3                  # hard map S, F and frozen occupancy O
    evidence_projection: int = 32
    widths: Tuple[int, int, int] = (32, 48, 64)
    coarse_dilations: Tuple[int, int] = (1, 2)
    groups: int = 8


@dataclass(frozen=True)
class MaskConfig:
    block_voxels: Tuple[int, int, int] = (16, 16, 8)    # 3.2 x 3.2 x 1.6 m cuboids
    hide_probability: float = 0.5


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 6000
    anchors_per_step: int = 4
    crop: Tuple[int, int, int] = (128, 128, 32)
    max_masked_targets: int = 8192
    max_delayed_targets: int = 4096
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    lambda_text: float = 1.0
    lambda_feature: float = 1.0
    lambda_delayed: float = 1.0
    log_every: int = 50
    ckpt_every: int = 1000


@dataclass(frozen=True)
class Paths:
    feature_mean: str = os.path.join(DATA_ROOT, "feature_mean_train_drives.npz")
    checkpoints: str = os.path.join(ARTIFACT_DIR, "checkpoints")
    runs: str = os.path.join(ARTIFACT_DIR, "runs")
    eval: str = os.path.join(ARTIFACT_DIR, "eval")
    figures: str = os.path.join(ARTIFACT_DIR, "figures")
    tables: str = os.path.join(ARTIFACT_DIR, "tables")
    logs: str = os.path.join(ARTIFACT_DIR, "logs")
    predictions: str = os.path.join(DATA_ROOT, "predictions_at_t")

    def checkpoint(self, seed: int, step: int) -> str:
        return os.path.join(self.checkpoints, f"B4_seed{seed}_step{step:05d}.pt")


#: training may not open human labels, LiDAR or target datasets (delayed targets are allowed)
TRAIN_FORBIDDEN: Tuple[str, ...] = PILOT.TRAIN_FORBIDDEN
#: target datasets stay closed until SOURCE PASS (checked by every source entry point)
TARGET_TOKENS: Tuple[str, ...] = ("data/kitti/dataset", "nuscenes", "occ3d", "lingbot_zero_shot_eval", "lingbot_occany_comparison", "occany_out")


def ensure_dirs() -> None:
    p = Paths()
    for d in (ARTIFACT_DIR, DATA_ROOT, p.checkpoints, p.runs, p.eval, p.figures, p.tables, p.logs, p.predictions):
        os.makedirs(d, exist_ok=True)


__all__ = [k for k in dir() if not k.startswith("_")]
