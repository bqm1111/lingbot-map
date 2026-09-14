"""Target-consistency study: is there a persistent voxel-level semantic identity under the
per-view Trident-H observations, recoverable from *independent* future-view consensus?

No model is trained here. The frozen future mapper is replayed per anchor over three
temporally disjoint sub-windows of the permitted 20-frame future horizon; candidate targets
are built from A+B alone and validated against C alone.
"""
from __future__ import annotations
import os
from tools.future_semantic_completion import study as ST

ART = os.path.join(ST.ART, "target_consistency")
#: bulk per-anchor evidence cache (hundreds of MB); the repo filesystem is nearly full
DATA_ROOT = os.environ.get("FSC_DATA_ROOT",
                           "/media/SSD1/MINH_DATASETS/lingbot_future_semantic_completion")
EVIDENCE_DIR = os.path.join(DATA_ROOT, "target_consistency", "evidence")

#: disjoint future sub-windows, in stream frames after the anchor (inclusive bounds);
#: early / middle / late over the frozen 20-frame horizon of core.supervision.targets
SUBWINDOWS = {"A": (1, 7), "B": (8, 14), "C": (15, 20)}
GROUPS = ("A", "B", "C")
#: the existing target's minimum fused teacher weight (core/run/build_samples.SEM_W_MIN)
SEM_W_MIN = 1.0

TAUS = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8)
GAMMAS = (0.0, 0.7, 0.8, 0.9, 0.95)


def evidence_path(drive: str, idx: int) -> str:
    return os.path.join(EVIDENCE_DIR, drive, f"{idx:05d}.npz")


__all__ = ["ART", "DATA_ROOT", "EVIDENCE_DIR", "SUBWINDOWS", "GROUPS", "SEM_W_MIN", "TAUS",
           "GAMMAS", "evidence_path"]
