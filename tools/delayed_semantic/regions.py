"""Development-evaluation regions on 0006. Flat [N_VOXELS] bool masks; GT is used only here, after prediction.

Symbols
    G    GT occupied with a class (sem_gt in 1..18)            L    labelled voxel (sem_gt != 255)
    E    student evidence voxel (observed semantics anywhere in H_t = t-4..t)
    Ec   evidence observed in frame t itself                     Ep   E & ~Ec (observed only in t-4..t-1)
    Gobs student geometry observed the voxel: hard map S (surface) or F (carved free) from H_t
    O    frozen predicted occupancy (geometry seed 0)
    R    0006 delayed-target voxel that is not evidence (a later camera view of a static LiDAR surface)
    Rh   R whose voxel had a camera-visible LiDAR return in some H_t frame (privileged; "missed", not unseen)

Semantic-only regions (score = class of the method on GT-occupied voxels; denominator = region voxels)
    observed_current              G & Ec                    recognition (identical for all methods by the evidence path)
    observed_past_only            G & Ep                    memory      (identical for all methods by the evidence path)
    never_observed                G & ~E                    completion, oracle occupancy
    completed_never_observed      G & O & ~E                PRIMARY (== Gate 15 completed_unobserved)
    completed_geo_observed        G & O & ~E & Gobs         geometry saw it, no semantic support
    completed_geo_unobserved      G & O & ~E & ~Gobs        geometry never saw it either
    completed_later_revealed      G & O & ~E & R            later measured by a camera (bins by time to reveal)
    completed_revealed_unseen     G & O & ~E & R & ~Rh      ... and no H_t camera ever saw a return there
    completed_not_revealed        G & O & ~E & ~R
SSC regions (prediction = class where O, else empty 0; 19x19)
    ssc_full                      L
    ssc_never_observed            L & ~E
"""
from __future__ import annotations

from typing import Dict

import torch

SEMANTIC_REGIONS = ("observed_current", "observed_past_only", "never_observed", "completed_never_observed", "completed_geo_observed",
                    "completed_geo_unobserved", "completed_later_revealed", "completed_revealed_unseen", "completed_not_revealed")
SSC_REGIONS = ("ssc_full", "ssc_never_observed")
PRIMARY = "completed_never_observed"


def build_regions(sem_gt: torch.Tensor, E: torch.Tensor, Ec: torch.Tensor, S: torch.Tensor, F_: torch.Tensor, O: torch.Tensor,
                  R: torch.Tensor, Rh: torch.Tensor) -> Dict[str, torch.Tensor]:
    G = (sem_gt >= 1) & (sem_gt <= 18); L = sem_gt != 255; Gobs = S | F_; U = G & O & ~E
    return {"observed_current": G & Ec, "observed_past_only": G & E & ~Ec, "never_observed": G & ~E, "completed_never_observed": U,
            "completed_geo_observed": U & Gobs, "completed_geo_unobserved": U & ~Gobs, "completed_later_revealed": U & R,
            "completed_revealed_unseen": U & R & ~Rh, "completed_not_revealed": U & ~R, "ssc_full": L, "ssc_never_observed": L & ~E}


__all__ = ["SEMANTIC_REGIONS", "SSC_REGIONS", "PRIMARY", "build_regions"]
