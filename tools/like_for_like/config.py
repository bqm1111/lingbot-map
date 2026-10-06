"""Paths and settings for the like-for-like semantic evaluation.

The frozen final zero-shot evaluation scored our CAUSAL five-frame window on the samples that also have such a window
(801 / 4,219). OccAny's published semantics use its own five frames starting at the target (non-causal) on every sample
it scored (805 / 4,819). This package re-runs OUR unchanged heads under that setting so the sample set, frames, labels
and placement match the geometry comparison; nothing about the method changes.
"""
from __future__ import annotations

import os

from tools.occany_comparison import config as OC
from tools.zero_shot_eval import config as ZC

PROTOCOL = OC.PROTOCOLS["forward"]
ARTIFACT_DIR = os.path.join(ZC.REPO_ROOT, "artifacts", "like_for_like")
DATA_ROOT = os.path.join(ZC.DATA_ROOT, "forward")
PREDICTION_ROOT = os.path.join(DATA_ROOT, "predictions")
FRAME_LIST = os.path.join(DATA_ROOT, "frames_{dataset}.json")
SCORE_DIR = os.path.join(ARTIFACT_DIR, "scores")
LOG_DIR = os.path.join(ARTIFACT_DIR, "logs")


def prediction_path(dataset: str, sample_id: str) -> str:
    return os.path.join(PREDICTION_ROOT, dataset, f"{sample_id}.npz")


def ensure_dirs() -> None:
    for d in (ARTIFACT_DIR, DATA_ROOT, PREDICTION_ROOT, SCORE_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


__all__ = ["PROTOCOL", "ARTIFACT_DIR", "DATA_ROOT", "PREDICTION_ROOT", "FRAME_LIST", "SCORE_DIR", "LOG_DIR", "prediction_path", "ensure_dirs"]
