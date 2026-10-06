"""Track O evaluator: the official benchmark computation on every valid voxel, from the same canonical predictions.

SC (binary): TP / FP / FN over ``label != 255`` with occupied = label not empty (OccAny ``get_score_completion``).
SSC: confusion of the official label against the method label where predicted occupied, else the empty class, over the same
voxels; mIoU per OccAny ``SSCMetrics.get_stats`` is computed at analysis time from the pooled confusion
(``canonical/metrics.occany_ssc_miou``), plus the superclass variant with OccAny's mapping.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from tools.zero_shot_eval.canonical.metrics import OccupancyCounts, confusion
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction
from tools.zero_shot_eval.practical.protocol import DatasetSpec

IGNORE = 255


def score_sample_official(pred: CanonicalPrediction, label: np.ndarray, spec: DatasetSpec, methods: List[str]) -> Dict[str, np.ndarray]:
    valid = label != IGNORE
    gt_occ = valid & (label != spec.empty_class)
    out = {"official_occ": OccupancyCounts.of(pred.occupied[valid], gt_occ[valid]).as_array()}
    for method in methods:
        ssc = np.where(pred.occupied, pred.dense_labels(method, spec.empty_class), spec.empty_class)
        out[f"official_conf__{method}"] = confusion(label[valid], ssc[valid], spec.n_classes)
    return out


def score_occupancy_official(occupied: np.ndarray, label: np.ndarray, spec: DatasetSpec) -> np.ndarray:
    valid = label != IGNORE
    return OccupancyCounts.of(occupied[valid], (label != spec.empty_class)[valid]).as_array()


__all__ = ["score_sample_official", "score_occupancy_official"]
