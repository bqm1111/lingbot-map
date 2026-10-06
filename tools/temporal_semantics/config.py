"""Gate 18 configuration: temporal semantic evidence scaling. No training.

Rows with K <= 5 are fully deployable. Rows with K > 5 use GT-pose registration to relate locally-reset clips and are
an UPPER BOUND, labelled as such everywhere (see plan.md).
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

from tools.delayed_semantic import config as C

VERSION = "gate18.1"
ARTIFACT_DIR = os.path.join(C.REPO_ROOT, "artifacts", "gate18_temporal_semantics")
DATA_ROOT = os.environ.get("GATE18_DATA_ROOT", "/media/SSD1/MINH_DATASETS/lingbot_gate18_temporal")
DEV_DRIVE = C.DEV_DRIVE
SEED = 0
STREAM_DT_S = 0.520                      # measured in Gate B5; one stream frame

#: History lengths in FRAMES. <=5 come from the anchor's own clip; >5 need earlier anchors (GT-pose registered).
HISTORY_K: Tuple[int, ...] = (1, 3, 5, 10, 20)
ALL_PAST = "all"                          # every earlier anchor, fused in a persistent world map
DEPLOYABLE_MAX_K = 5                      # above this, rows are a registration upper bound

AGGREGATIONS: Tuple[str, ...] = ("latest", "mean_feature", "mean_probability", "count_weighted")
#: The frozen evidence rule is the count-weighted mean of unit features; `mean_feature` reproduces it at K=5.
FROZEN_AGGREGATION = "mean_feature"

#: Observation-count bins as CLOSED integer ranges, so the mapping is explicit rather than an off-by-one waiting to
#: happen. (torch.bucketize with right=True shifts every bin by one against these names.)
OBS_RANGES: Tuple[Tuple[int, int], ...] = ((1, 1), (2, 2), (3, 5), (6, 10), (11, 10 ** 9))
OBS_NAMES: Tuple[str, ...] = ("1", "2", "3-5", "6-10", ">10")
DISTANCE_EDGES_M = C.DISTANCE_EDGES_M
DISTANCE_NAMES = C.DISTANCE_NAMES

CLASS_NAMES: Tuple[str, ...] = ("car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "road", "parking",
                                "sidewalk", "other-ground", "building", "fence", "vegetation", "terrain", "pole",
                                "traffic-sign", "other-structure", "other-object")
THING_LABELS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)

POPULATIONS: Dict[str, str] = {
    "T1_observed_evidence": "evidence voxels with a GT label -- teacher reliability, no registration",
    "T2_observed_map": "observed_current + observed_past_only -- does fusion improve the map",
    "T3_completed_unobserved": "completed_never_observed via the existing B0 rule on the accumulated evidence",
}


def shard_path(tag: str, shard: int, shards: int) -> str:
    return os.path.join(DATA_ROOT, "eval", f"{tag}_shard{shard}of{shards}.npz")


__all__ = ["VERSION", "ARTIFACT_DIR", "DATA_ROOT", "DEV_DRIVE", "SEED", "STREAM_DT_S", "HISTORY_K", "ALL_PAST",
           "DEPLOYABLE_MAX_K", "AGGREGATIONS", "FROZEN_AGGREGATION", "OBS_BINS", "OBS_NAMES", "DISTANCE_EDGES_M",
           "DISTANCE_NAMES", "CLASS_NAMES", "THING_LABELS", "POPULATIONS", "shard_path"]
