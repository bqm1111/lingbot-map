"""Gate 16 -- candidate-level delayed supervision. Single source of truth; everything else comes from the pilot config.

Candidate convention (unchanged from the pilot): for a query voxel, K = 16 nearest observed evidence voxel centres
(Euclidean, ascending), ``model.nearest_candidates``. Candidate 0 is the nearest.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

from tools.delayed_semantic import config as PC

REPO_ROOT = PC.REPO_ROOT
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "gate16_candidate_selection")
DATA_ROOT = os.path.join(PC.DATA_ROOT, "gate16")
PROTOCOL_VERSION = "gate16_candidate_selection/2026-09-14"

K = PC.ModelConfig().k_candidates
assert K == 16, "Gate 16 fixes the candidate count at 16"
SEEDS: Tuple[int, ...] = (0, 1, 2)
TAU_TARGET = 1.0 / PC.LOGIT_SCALE            # candidate-target temperature: the teacher's own scale, reused, never swept
STEPS = PC.TrainConfig().steps

#: new conditions -> scorer option
VARIANTS: Dict[str, Dict] = {"B3": {"semantic_projection": True}, "B3geom": {"semantic_projection": False}}
PILOT_HEADS: Dict[str, str] = {m: os.path.join(PC.ARTIFACT_DIR, "checkpoints", f"{m}_seed0_step{STEPS:05d}.pt") for m in ("B1", "B2")}

CHECKPOINTS = os.path.join(ARTIFACT_DIR, "checkpoints")
RUNS = os.path.join(ARTIFACT_DIR, "runs")
EVAL = os.path.join(ARTIFACT_DIR, "eval")
TABLES = os.path.join(ARTIFACT_DIR, "tables")
FIGURES = os.path.join(ARTIFACT_DIR, "figures")
LOGS = os.path.join(ARTIFACT_DIR, "logs")
PREDICTIONS = os.path.join(DATA_ROOT, "predictions_at_t")


def head_path(method: str, seed: int) -> str:
    return os.path.join(CHECKPOINTS, f"{method}_seed{seed}_step{STEPS:05d}.pt")


def head_methods() -> Dict[str, str]:
    """Every learned head scored by Gate 16: pilot B1/B2 (seed 0) + B3/B3geom seeds."""
    out = dict(PILOT_HEADS)
    out.update({f"{v}_s{s}": head_path(v, s) for v in VARIANTS for s in SEEDS})
    return out


def ensure_dirs() -> None:
    for d in (ARTIFACT_DIR, CHECKPOINTS, RUNS, EVAL, TABLES, FIGURES, LOGS, PREDICTIONS):
        os.makedirs(d, exist_ok=True)


__all__ = [k for k in dir() if not k.startswith("_")]
