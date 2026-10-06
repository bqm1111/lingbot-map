"""Track P evaluator: canonical prediction + official label -> per-sample confusions and occupancy counts.

Per sample and per region (``P`` = Track P operational region, ``full`` = every valid voxel):
    occupancy counts [TP, FP, FN] over the region, within geometrically observed voxels, within unobserved voxels;
    per method, 19x19 / 18x18 confusions for the populations of ``POPULATIONS``.
Pooled over samples only (not needed per sample): Track P occupancy and central-population confusions by range bin, central
confusions by distance to nearest evidence. OccAny's released geometry (Level A) is scored with the same region and partitions.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from tools.zero_shot_eval.canonical.metrics import OccupancyCounts, confusion
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction
from tools.zero_shot_eval.practical.masks import TrackPMasks, populations, track_p_masks
from tools.zero_shot_eval.practical.protocol import EVIDENCE_DISTANCE_EDGES_M, EVIDENCE_DISTANCE_NAMES, REGION, DatasetSpec

REGIONS = ("P", "full")
POPULATIONS = ("ssc", "ssc_no_evidence", "gt_occupied", "observed_evidence", "no_evidence", "completed_no_evidence", "geometrically_unobserved")
CENTRAL = "completed_no_evidence"


def region_mask(m: TrackPMasks, region: str) -> np.ndarray:
    return m.region if region == "P" else m.valid


def occupancy_block(occupied: np.ndarray, observed: np.ndarray, m: TrackPMasks, region: str) -> Dict[str, np.ndarray]:
    r = region_mask(m, region)
    return {f"occ__{region}__all": OccupancyCounts.of(occupied[r], m.gt_occupied[r]).as_array(),
            f"occ__{region}__observed": OccupancyCounts.of(occupied[r & observed], m.gt_occupied[r & observed]).as_array(),
            f"occ__{region}__unobserved": OccupancyCounts.of(occupied[r & ~observed], m.gt_occupied[r & ~observed]).as_array()}


def score_sample(pred: CanonicalPrediction, label: np.ndarray, spec: DatasetSpec, methods: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """``{"per_sample": {key: array}, "pooled": {key: array}}``; label: flat uint8 official label on the evaluation grid."""
    m = track_p_masks(pred, label, spec)
    C = spec.n_classes; nD, nR = len(EVIDENCE_DISTANCE_NAMES), len(REGION.range_names)
    per_sample: Dict[str, np.ndarray] = {}
    pooled: Dict[str, np.ndarray] = {}
    for region in REGIONS:
        per_sample.update(occupancy_block(pred.occupied, pred.observed, m, region))
    pooled["occ_range"] = np.stack([OccupancyCounts.of(pred.occupied[m.region & (m.range_bin == b)], m.gt_occupied[m.region & (m.range_bin == b)]).as_array()
                                    for b in range(nR)])
    dist = np.full(pred.n_voxels, np.inf, np.float32); dist[pred.query_index] = pred.evidence_distance
    dbin = np.digitize(dist, np.asarray(EVIDENCE_DISTANCE_EDGES_M))
    for method in methods:
        lab = pred.dense_labels(method, spec.empty_class)
        ssc = np.where(pred.occupied, lab, spec.empty_class)
        for region in REGIONS:
            r = region_mask(m, region); pops = populations(pred, m, r)
            per_sample[f"conf__{region}__ssc__{method}"] = confusion(label[r], ssc[r], C)
            rn = r & ~pred.evidence
            per_sample[f"conf__{region}__ssc_no_evidence__{method}"] = confusion(label[rn], ssc[rn], C)
            for name, mask in pops.items():
                per_sample[f"conf__{region}__{name}__{method}"] = confusion(label[mask], lab[mask], C)
        central = populations(pred, m, m.region)[CENTRAL]
        pooled[f"dist__{method}"] = np.stack([confusion(label[central & (dbin == b)], lab[central & (dbin == b)], C) for b in range(nD)])
        pooled[f"range__{method}"] = np.stack([confusion(label[central & (m.range_bin == b)], lab[central & (m.range_bin == b)], C) for b in range(nR)])
    pooled["dist_voxels"] = np.asarray([int((populations(pred, m, m.region)[CENTRAL] & (dbin == b)).sum()) for b in range(nD)])
    return {"per_sample": per_sample, "pooled": pooled}


def score_occany_geometry(occupied: np.ndarray, pred: CanonicalPrediction, label: np.ndarray, spec: DatasetSpec, tag: str) -> Dict[str, np.ndarray]:
    """OccAny occupancy through the same regions; ``observed`` partitions use OUR causal observation mask of the same sample."""
    m = track_p_masks(pred, label, spec)
    out = {}
    for region in REGIONS:
        out.update({k.replace("occ__", f"occany_{tag}__"): v for k, v in occupancy_block(occupied, pred.observed, m, region).items()})
    out[f"occany_{tag}__range"] = np.stack([OccupancyCounts.of(occupied[m.region & (m.range_bin == b)], m.gt_occupied[m.region & (m.range_bin == b)]).as_array()
                                           for b in range(len(REGION.range_names))])
    return out


__all__ = ["REGIONS", "POPULATIONS", "CENTRAL", "score_sample", "score_occany_geometry"]
