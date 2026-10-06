"""Gate 17 configuration: the instance-grounding oracle diagnostic.

One question: if instance association between visible semantic evidence and completed unobserved voxels were
perfect, how much semantic mIoU would that recover? This is a go/no-go gate, not a model.

Every frozen component stays frozen. The only new information is the oracle grouping described below.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

from tools.delayed_semantic import config as C

VERSION = "gate17.1"
ARTIFACT_DIR = os.path.join(C.REPO_ROOT, "artifacts", "gate17_instance_oracle")
DATA_ROOT = os.environ.get("GATE17_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_gate17_instance_oracle")
DEV_DRIVE = C.DEV_DRIVE
SEED = 0

#: sem_gt labels are 1..18; vocab index = label - 1.
CLASS_NAMES: Tuple[str, ...] = ("car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "road", "parking",
                                "sidewalk", "other-ground", "building", "fence", "vegetation", "terrain", "pole",
                                "traffic-sign", "other-structure", "other-object")
#: Classes where a connected same-class component is a plausible OBJECT. Everything else is a region, and a
#: "component" there means a connected piece of road or building, not an instance. Poles and traffic signs are
#: discrete too, but the benchmark treats them as stuff and they are reported inside stuff for consistency.
THING_LABELS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
STUFF_LABELS: Tuple[int, ...] = tuple(c for c in range(1, 19) if c not in THING_LABELS)

#: The three populations of the brief, expressed as regions that already exist in tools/delayed_semantic/regions.py.
#: P2 is not a new mask: on ``completed_never_observed`` predicted occupancy and GT occupancy are already both true,
#: so "correctly completed" is exactly that region.
POPULATIONS: Dict[str, str] = {
    "P1_ssc_deployable": "ssc_never_observed",        # prediction = class where predicted-occupied else empty, over labelled voxels
    "P2_correct_completion": "completed_never_observed",  # GT-occupied AND predicted-occupied, no evidence
    "P3_oracle_geometry": "never_observed",           # GT-occupied, no evidence, regardless of prediction (ORACLE)
}

#: Methods. Semantics come from the frozen teacher everywhere except C2, which is the absolute ceiling.
METHODS: Tuple[str, ...] = ("C0_b0", "A_oracle_instance", "C1_shuffled_instance", "C3_visible_majority", "C2_gt_class")
ORACLE_USE: Dict[str, str] = {
    "C0_b0": "none -- the deployed baseline",
    "A_oracle_instance": "grouping only; the label is the teacher's, aggregated over the instance's own visible evidence",
    "C1_shuffled_instance": "grouping only, then aggregates permuted between instances (control)",
    "C3_visible_majority": "grouping only; majority teacher class over the instance's visible evidence",
    "C2_gt_class": "FULL semantic oracle -- the ceiling, never a method",
}

#: Instance visible fraction bins (visible GT instance voxels / total GT instance voxels in the volume).
VISIBILITY_EDGES: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
VISIBILITY_NAMES: Tuple[str, ...] = ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")
#: Distance-to-evidence bins reuse the frozen pilot definition so the axis matches earlier studies.
DISTANCE_EDGES_M = C.DISTANCE_EDGES_M
DISTANCE_NAMES = C.DISTANCE_NAMES

#: 6-connectivity: faces only. 26-connectivity would merge objects across diagonal touches far more aggressively.
CONNECTIVITY = 6


def shard_path(shard: int, shards: int) -> str:
    return os.path.join(DATA_ROOT, "eval", f"gate17_shard{shard}of{shards}.npz")


__all__ = ["VERSION", "ARTIFACT_DIR", "DATA_ROOT", "DEV_DRIVE", "SEED", "CLASS_NAMES", "THING_LABELS", "STUFF_LABELS",
           "POPULATIONS", "METHODS", "ORACLE_USE", "VISIBILITY_EDGES", "VISIBILITY_NAMES", "DISTANCE_EDGES_M",
           "DISTANCE_NAMES", "CONNECTIVITY", "shard_path"]
