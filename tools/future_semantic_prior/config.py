"""Gate 21 configuration: the future-semantic-prior learning experiment. Single source of truth.

One question: trained ONLY on hidden voxels that a later frame happens to re-observe (Gate 20's
``visible`` level), does a small semantic-completion head generalise to hidden voxels that are NEVER
re-observed?

Everything upstream is frozen and loaded read-only: the LingBot causal clip, the MoGe-2 clip scale, the
hard mapper, CompletionUNet geometry seed 0 step 6000, Trident-H and its 18 KITTI-360 phrases, the
occupancy threshold, the text readout, the voxel grid, the evaluation masks, Gate 20's future-visibility
definition and Gate 20's ``mean_probability`` target. This gate adds exactly one trainable module.

Grid (every flat tensor here): SSCBench-KITTI-360 anchor velodyne grid, dims (256, 256, 32), 0.2 m,
origin (0, -25.6, -2) m, flat = ravel_multi_index((x, y, z)), C order.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

from tools.delayed_semantic import config as C
from tools.future_revealability import config as G20

VERSION = "gate21.1"
ARTIFACT_DIR = os.path.join(C.REPO_ROOT, "artifacts", "gate21_future_semantic_prior")
DATA_ROOT = os.environ.get("GATE21_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_gate21_semantic_prior")

TRAIN_DRIVES: Tuple[str, ...] = C.TRAIN_DRIVES      # 0003, 0007, 0010
DEV_DRIVE = C.DEV_DRIVE                              # 0006 -- development + decision, never trained on
SEEDS: Tuple[int, ...] = (0, 1, 2)
GEOMETRY_SEED = C.GEOMETRY_SEED                      # 0, fixed for every control and seed: B0 must not move
N_CLASSES = C.N_CLASSES                              # 18
GRID_DIMS, N_VOXELS = C.GRID_DIMS, C.N_VOXELS

#: The supervision level. Gate 20's ``visible``: a future LiDAR return in the voxel that passes the frozen
#: z-buffer test and carries a bilinear Trident-H token. Gate 21 does not redefine it and does not relax it.
REVEAL_LEVEL = G20.DECISION_LEVEL                     # "visible"
REVEAL_READOUT = G20.DECISION_READOUT                 # "mean_probability"
#: index of the 10 s horizon in ``G20.HORIZONS_S`` -- the whole frozen privileged window
EVENTUAL_H = len(G20.HORIZONS_S) - 1

CLASS_NAMES = G20.CLASS_NAMES
THING_LABELS = G20.THING_LABELS
DISTANCE_EDGES_M = C.DISTANCE_EDGES_M
DISTANCE_NAMES = C.DISTANCE_NAMES

# ---- input channels (section 9: written down before training) --------------------------------------
#: Two DISJOINT groups, so C2/C3/C4 is a clean factorial rather than three overlapping models.
#:
#:   geometry   what the frozen geometry stack knows: the CompletionUNet decoder feature, the hard map,
#:              the completed occupancy, and the voxel's own place in the grid.
#:   semantics  what the causal Trident-H evidence knows: where it is, how far it is, and what it says.
#:
#: ``evidence_dist`` is grouped with SEMANTICS, not geometry: it is a function of where semantic evidence
#: lies, so a "geometry-only" model that used it would not be geometry-only.
GEOMETRY_CHANNELS: Tuple[Tuple[str, int], ...] = (
    ("trunk", 24),          # frozen CompletionUNet decoder feature d1
    ("hard_S", 1),          # hard mapper surface
    ("hard_F", 1),          # hard mapper carved-free
    ("occupancy", 1),       # frozen predicted occupancy O (the completed geometry)
    ("height", 1),          # voxel-centre z in metres / 2.0
    ("range", 1),           # sqrt(x^2 + y^2) of the voxel centre / 25.6
)
SEMANTIC_CHANNELS: Tuple[Tuple[str, int], ...] = (
    ("evidence_mask", 1),   # 1 where a causal Trident-H evidence voxel is
    ("evidence_dist", 1),   # log1p(metres to nearest evidence) / log1p(25.6)
    ("evidence_proj", 32),  # learned 1024 -> 32 projection of the unit evidence feature, 0 elsewhere
)
#: C4 keeps only the one geometry channel the brief allows it ("masks/coordinates required for indexing").
C4_GEOMETRY_KEEP: Tuple[str, ...] = ("occupancy",)

CONTROLS: Dict[str, str] = {
    "C1_prior": "input-independent class logits -- how much of any gain is the training class frequency",
    "C2_geometry": "geometry channels only, no causal semantic evidence at all",
    "C3_full": "geometry + causal semantic evidence -- the PRIMARY Gate 21 model",
    "C4_semantics": "causal semantic evidence + completed occupancy only, no frozen latent features",
}
PRIMARY_CONTROL = "C3_full"
LEARNED_CONTROLS: Tuple[str, ...] = ("C1_prior", "C2_geometry", "C3_full", "C4_semantics")


@dataclass(frozen=True)
class HeadConfig:
    """Reuses Gate 15's propagation-head body unchanged (width 32, dilations 1/2/4/8, receptive-field
    radius 2*(1+2+4+8) = 30 voxels = 6 m). Only the output changes: 18 class logits instead of a 1024-d
    feature, because Gate 21 distils a probability distribution, not an embedding (section 8)."""
    width: int = 32
    dilations: Tuple[int, ...] = (1, 2, 4, 8)
    evidence_dim: int = 32
    teacher_dim: int = C.TEACHER_DIM


@dataclass(frozen=True)
class TrainConfig:
    """The established Gate 15 / B5 / pilot conventions. No sweep, no architecture search (section 18)."""
    steps: int = 3000                # the pilot's schedule
    anchors_per_step: int = 2        # full-grid samples; no crop, so train and eval see the same tensor
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 50
    ckpt_every: int = 500            # every checkpoint is a point on the training curve (section 14)
    curve_stride: int = 12           # curves use every 12th dev anchor -> 50 anchors


@dataclass(frozen=True)
class Thresholds:
    """Pre-registered in ``decision_rule.md`` and hashed BEFORE the first training run. Never moved.

    The primary metric is mIoU on E2 = the never-revealed part of P0 on 0006, where B0 scores 10.18."""
    b0_e2_miou: float = 10.180049696284014   # read from Gate 20's analysis.json before registering
    a_delta_e2_min: float = 1.0              # Outcome A: mean over 3 seeds, absolute pp
    a_all_seeds_positive: bool = True
    a_ci_lower_above: float = 0.0
    c_delta_e2_max: float = 0.25             # Outcome C fires at or below this, or if a seed is negative
    #: integrity, not an outcome: if either fails the run is void and no outcome is declared
    require_geometry_identical: bool = True
    require_leakage_clean: bool = True
    #: reported, not decisive: a composite SSC read must not collapse the rest of the scene
    report_ssc_change_floor_pp: float = -0.5


BOOTSTRAP_BLOCK = G20.BOOTSTRAP_BLOCK          # 20-anchor temporal blocks
BOOTSTRAP_RESAMPLES = G20.BOOTSTRAP_RESAMPLES  # 2000
BOOTSTRAP_SEED = 0

#: Training may read the Gate 21 target cache (privileged future supervision) and the frozen causal caches.
#: It may NEVER read human labels, LiDAR, GT poses, any target dataset -- or ANY FILE OF THE DEV DRIVE.
#:
#: The last of those is why the training curves are not measured inside the training loop: evaluating 0006
#: between optimiser steps would have to open 0006's ground truth, and the audit could then no longer say
#: that the training process never saw the development drive. ``curves.py`` sweeps the saved checkpoints
#: afterwards instead, which measures the same thing and keeps the claim absolute.
TRAIN_FORBIDDEN: Tuple[str, ...] = C.TRAIN_FORBIDDEN + ("sscbench_kitti360/preprocess", "lingbot_k360_7drives",
                                                        "drive_0006")
#: Building the head INPUT must touch nothing privileged at all -- not even the Gate 21 target cache.
INPUT_FORBIDDEN: Tuple[str, ...] = C.PREDICT_FORBIDDEN + (DATA_ROOT, "lingbot_gate20_future_reveal")
#: Development evaluation on 0006: target datasets only.
EVAL_FORBIDDEN: Tuple[str, ...] = C.EVAL_FORBIDDEN


def target_path(drive: str, t: int) -> str:
    """Per-anchor U^R cache: the revealed subset of P0 and its soft future target."""
    return os.path.join(DATA_ROOT, "targets", drive, f"{t:05d}.npz")


def checkpoint_path(control: str, seed: int, step: int) -> str:
    return os.path.join(DATA_ROOT, "checkpoints", f"{control}_seed{seed}_step{step:05d}.pt")


def seed_dir(seed: int) -> str:
    return os.path.join(ARTIFACT_DIR, f"seed_{seed}")


def ensure_dirs() -> None:
    for d in (ARTIFACT_DIR, os.path.join(ARTIFACT_DIR, "figures"), os.path.join(ARTIFACT_DIR, "logs"),
              os.path.join(ARTIFACT_DIR, "tables"), os.path.join(ARTIFACT_DIR, "training_curves"),
              os.path.join(DATA_ROOT, "targets"), os.path.join(DATA_ROOT, "checkpoints"),
              os.path.join(DATA_ROOT, "eval")) + tuple(seed_dir(s) for s in SEEDS):
        os.makedirs(d, exist_ok=True)


__all__ = [k for k in dir() if not k.startswith("_")]
