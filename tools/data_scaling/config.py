"""Gate 19 configuration: is the learned completion model data-limited?

Only the amount of training data changes. Architecture, optimizer, losses, crop, input representation, causal
protocol, voxel grid and validation procedure are the frozen Gate 12-H / Gate 13-M recipe, read from the same
``training_config.yaml`` the frozen baseline used.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

from tools.baseline_reconstruction import config as BR

VERSION = "gate19.1"
ARTIFACT_DIR = os.path.join(BR.REPO_ROOT, "artifacts", "gate19_data_scaling")
DATA_ROOT = os.environ.get("GATE19_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_gate19_data_scaling")
SEED_PERMUTATION = 0                       # the one seed that fixes which anchors enter which subset

#: Every drive with BOTH a hard-map input and an occupancy target. All eight share one ``rules_sha256``,
#: so the added drives were built with the same hard mapper as the frozen three (checked in the audit).
FROZEN_TRAIN_DRIVES: Tuple[str, ...] = BR.TRAIN_DRIVES          # 0003 / 0007 / 0010, the current pool
ADDED_DRIVES: Tuple[str, ...] = ("2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync",
                                 "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync")
VAL_DRIVE = BR.VAL_DRIVE                                        # 0006, frozen, never trained on

#: Target sizes. D_small is EXACTLY the frozen pool, so the smallest point of the curve is the published baseline
#: and can be validated against its recorded draw digest.
SCALES: Dict[str, int] = {"D_small": 1358, "D_medium": 3000, "D_large": 6000, "D_all": 10 ** 9}
SEEDS: Tuple[int, ...] = BR.SEEDS                               # 0, 1, 2

#: Two comparisons, because they answer different questions and can disagree:
#:   fixed_steps  -- every scale gets the frozen 6,000 updates; more data = more unique anchors per update
#:   fixed_epochs -- every scale gets the same number of passes over its data as D_small had (6000*4/1358)
FIXED_STEPS = 6000
BATCH_SIZE = 4                                                  # from the frozen recipe; used only to convert epochs


def epochs_of(n_anchors: int, steps: int = FIXED_STEPS) -> float:
    return steps * BATCH_SIZE / max(n_anchors, 1)


def steps_for_fixed_epochs(n_anchors: int, reference_anchors: int = 1358) -> int:
    """The step count that gives ``n_anchors`` the same number of epochs D_small got at 6,000 steps."""
    return int(round(FIXED_STEPS * n_anchors / reference_anchors))


def pool_path(name: str) -> str:
    return os.path.join(ARTIFACT_DIR, "pools", f"{name}.json")


def run_tag(scale: str, regime: str, seed: int) -> str:
    return f"{scale}_{regime}_seed{seed}"


def checkpoint_dir() -> str:
    return os.path.join(DATA_ROOT, "checkpoints")


__all__ = ["VERSION", "ARTIFACT_DIR", "DATA_ROOT", "SEED_PERMUTATION", "FROZEN_TRAIN_DRIVES", "ADDED_DRIVES",
           "VAL_DRIVE", "SCALES", "SEEDS", "FIXED_STEPS", "BATCH_SIZE", "epochs_of", "steps_for_fixed_epochs",
           "pool_path", "run_tag", "checkpoint_dir"]
