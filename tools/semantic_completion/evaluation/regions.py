"""Evaluation regions on 0006. All masks are flat [N_VOXELS] bool on the velodyne grid.

E  = evidence voxels (direct causal semantic observation), from the evidence cache
O  = frozen predicted occupancy of the geometry seed
G  = GT occupied with a class label (sem_gt in 1..18);  L = any labelled voxel (sem_gt != 255)

semantic-only regions (prediction = class of the method, GT = class):
    oracle_observed      G &  E        teacher / copy quality where evidence exists
    oracle_unobserved    G & ~E        oracle-occupancy diagnostic without evidence
    completed_observed   G &  O &  E
    completed_unobserved G &  O & ~E   PRIMARY: correctly completed occupied voxels with no direct evidence
SSC regions (prediction = class if O else 0):
    ssc_full             L
    ssc_no_evidence      L & ~E
"""
from __future__ import annotations

from typing import Dict

import torch

SEMANTIC_REGIONS = ("oracle_observed", "oracle_unobserved", "completed_observed", "completed_unobserved")
SSC_REGIONS = ("ssc_full", "ssc_no_evidence")
PRIMARY = "completed_unobserved"


def build_regions(sem_gt: torch.Tensor, evidence: torch.Tensor, occupied_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
    G = (sem_gt >= 1) & (sem_gt <= 18); L = sem_gt != 255; E, O = evidence, occupied_pred
    return {"oracle_observed": G & E, "oracle_unobserved": G & ~E, "completed_observed": G & O & E, "completed_unobserved": G & O & ~E,
            "ssc_full": L, "ssc_no_evidence": L & ~E}


__all__ = ["SEMANTIC_REGIONS", "SSC_REGIONS", "PRIMARY", "build_regions"]
