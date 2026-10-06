"""Final zero-shot semantic completion evaluation -- single source of truth for paths and frozen inputs.

Nothing here is tuned on a target dataset. Protocol parameters (region, ranges, metrics, statistics) live in
``practical/protocol.py`` and are hash-frozen with ``practical_evaluation_predeclaration.md`` before any
method comparison is scored.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "final_zero_shot")
DATA_ROOT = os.environ.get("ZERO_SHOT_EVAL_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_zero_shot_eval")
DATASETS: Tuple[str, ...] = ("semantickitti", "occ3d")

TOKEN_ROOT = os.path.join(DATA_ROOT, "trident_tokens")        # per-frame Trident-H token maps of target frames
PREDICTION_ROOT = os.path.join(DATA_ROOT, "predictions")      # canonical predictions (one file per sample)
FRAME_LIST = os.path.join(DATA_ROOT, "frames_{dataset}.json")
TEXT_EMBEDDINGS = os.path.join(DATA_ROOT, "text_embeddings_{dataset}.npz")
LOG_DIR = os.path.join(ARTIFACT_DIR, "logs")
TABLE_DIR = os.path.join(ARTIFACT_DIR, "tables")
FIGURE_DIR = os.path.join(ARTIFACT_DIR, "figures")

#: frozen semantic heads: Gate 16 B3 seeds (trained on KITTI-360 only, before any target access)
GATE16_CHECKPOINTS = os.path.join(REPO_ROOT, "artifacts", "gate16_candidate_selection", "checkpoints")
B3_SEEDS: Tuple[int, ...] = (0, 1, 2)
GEOMETRY_SEED = 0          # the geometry seed B3 was trained on; its occupancy and trunk feed every method


def b3_checkpoint(seed: int) -> str:
    return os.path.join(GATE16_CHECKPOINTS, f"B3_seed{seed}_step03000.pt")


def token_path(dataset: str, key: str) -> str:
    return os.path.join(TOKEN_ROOT, dataset, f"{key}.npz")


def prediction_path(dataset: str, sample_id: str) -> str:
    return os.path.join(PREDICTION_ROOT, dataset, f"{sample_id}.npz")


def ensure_dirs() -> None:
    for d in (ARTIFACT_DIR, LOG_DIR, TABLE_DIR, FIGURE_DIR, TOKEN_ROOT, PREDICTION_ROOT):
        os.makedirs(d, exist_ok=True)


__all__ = [k for k in dir() if not k.startswith("_")]
