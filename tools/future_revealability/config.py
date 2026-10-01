"""Gate 20 configuration: the future-revealability audit. No training, no new model.

One question: of the geometry our causal system *correctly completes but has never observed*, how much
becomes genuinely observable later in the same drive, and is the later observation a trustworthy semantic
target?

Grid, distance bins, class list and the privileged future window are the frozen ones
(``tools/delayed_semantic.config`` / ``tools/semantic_completion.config``); nothing here redefines them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

from tools.delayed_semantic import config as C

VERSION = "gate20.1"
ARTIFACT_DIR = os.path.join(C.REPO_ROOT, "artifacts", "gate20_future_revealability")
DATA_ROOT = os.environ.get("GATE20_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_gate20_future_reveal")
SEED = 0

TRAIN_DRIVES: Tuple[str, ...] = C.TRAIN_DRIVES          # 0003, 0007, 0010 -- the audit's PRIMARY population
DEV_DRIVE = C.DEV_DRIVE                                  # 0006 -- reported separately, never used to set a threshold
STREAM_DT_S = 0.520                                      # one stream frame (measured in Gate 18/B5)

#: The privileged future window is the frozen one: t+1..t+20 and at most 10 s (``C.TargetConfig``).
#: "eventual" is therefore exactly the 10 s horizon, not an unbounded future.
HORIZONS_S: Tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 6.0, 10.0)
EVENTUAL_S = 10.0
ZERO_HORIZON_S = 0.0                                     # must come out exactly 0 by definition (validation #1)

#: Three nested reveal levels. ``visible`` is the DECISION level: it is the evidence rule the semantic
#: lifting pipeline already uses (camera-visible surface return with a valid Trident-H token), applied to a
#: future frame and registered back into the grid of t. ``geometric`` drops the camera test, ``deployable``
#: adds the frozen delayed-target occupancy/persistence selection on top.
LEVELS: Tuple[str, ...] = ("geometric", "visible", "deployable")
DECISION_LEVEL = "visible"
#: ``in_frustum`` sits between ``geometric`` and ``visible`` and exists only to say which of the two
#: filters -- the camera's field of view or occlusion -- costs the coverage. It never decides anything.
DIAG_LEVELS: Tuple[str, ...] = ("geometric", "in_frustum", "visible", "deployable")

#: Readouts of the future target, each using only the views inside the horizon.
READOUTS: Tuple[str, ...] = ("first", "latest", "mean_probability")
DECISION_READOUT = "mean_probability"

DISTANCE_EDGES_M = C.DISTANCE_EDGES_M                    # (1, 2, 4, 8)
DISTANCE_NAMES = C.DISTANCE_NAMES                        # 0-1, 1-2, 2-4, 4-8, >8 m
FAR_BINS = (3, 4)                                        # 4-8 m and >8 m -> the ">4 m" row of the brief
VERY_FAR_BINS = (4,)                                     # >8 m

CLASS_NAMES: Tuple[str, ...] = ("car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "road", "parking",
                                "sidewalk", "other-ground", "building", "fence", "vegetation", "terrain", "pole",
                                "traffic-sign", "other-structure", "other-object")
THING_LABELS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
N_CLASSES = 18

#: Region analysis: 6-connected same-class GT components (Gate 17's definition, reused unchanged).
REGION_FRACTIONS: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)
MIN_REGION_HIDDEN_VOX = 8                                # a component with fewer hidden voxels is not a region

#: Block bootstrap over anchors, matching every earlier gate (20-anchor temporal blocks, 2000 resamples, seed 0).
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_RESAMPLES = 2000

POPULATIONS: Dict[str, str] = {
    "P0": "GT-occupied AND frozen-predicted-occupied AND never causally observed (== regions completed_never_observed)",
    "P1": "P0 AND the frozen B0 nearest-evidence class is wrong",
    "P0_correct": "P0 AND the frozen B0 nearest-evidence class is right (sanity check)",
}


@dataclass(frozen=True)
class Thresholds:
    """Pre-registered in ``decision_rule.md`` before the full run. Never moved afterwards."""
    # Outcome A needs ALL of these.
    a_p0_eventual: float = 50.0
    a_p1_eventual: float = 40.0
    a_far_eventual: float = 25.0          # pooled 4-8 m and >8 m
    a_teacher_top1: float = 55.0
    a_teacher_miou: float = 12.0
    a_min_classes_20pct: int = 12         # of 18
    # Outcome C fires on ANY of these.
    c_p0_eventual: float = 20.0
    c_p1_eventual: float = 15.0
    c_far_eventual: float = 10.0
    c_teacher_top1: float = 40.0
    c_teacher_miou: float = 8.0


#: Gate 18's frozen current-evidence teacher, for the comparison section 10 asks for. Read-only reference.
GATE18_TEACHER = {"top1": 62.57, "miou": 16.91, "thing_miou": 32.58, "stuff_miou": 17.83}


def mask_path(drive: str, t: int) -> str:
    return os.path.join(DATA_ROOT, "eval_masks", drive, f"{t:05d}.npz")


def reveal_path(drive: str, t: int) -> str:
    return os.path.join(DATA_ROOT, "reveal", drive, f"{t:05d}.npz")


def population_path(drive: str, t: int) -> str:
    return os.path.join(DATA_ROOT, "population", drive, f"{t:05d}.npz")


def ensure_dirs() -> None:
    for d in (ARTIFACT_DIR, os.path.join(ARTIFACT_DIR, "figures"), os.path.join(ARTIFACT_DIR, "logs"),
              os.path.join(ARTIFACT_DIR, "tables"), os.path.join(DATA_ROOT, "eval_masks"),
              os.path.join(DATA_ROOT, "reveal"), os.path.join(DATA_ROOT, "population")):
        os.makedirs(d, exist_ok=True)


__all__ = [k for k in dir() if not k.startswith("_")]
