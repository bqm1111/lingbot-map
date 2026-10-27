"""The prediction sets compared, and how to read each one onto the evaluation grid.

Every OccAny directory here was produced on this machine from the released checkpoints; the
reproduction gates are in ``reports/occany_comparison_honest_report.md``. ``window`` records whether
the inputs are causal (frames at or before the target) or forward (frames after it).
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

OURS_CAUSAL_ROOT = "/media/SSD1/MINH_DATASETS/lingbot_zero_shot_eval/predictions"
_CAUSAL = "/media/SSD1/occany_causal/ssc_voxel_pred"
_FORWARD = "/media/SSD1/MINH_DATASETS/occany_out/ssc_voxel_pred"


@dataclass(frozen=True)
class Run:
    key: str
    kind: str          # "ours" | "occany"
    window: str        # "causal" | "forward"
    path: str
    semantic: bool
    frames: int
    note: str


RUNS: Dict[str, Tuple[Run, ...]] = {
    "semantickitti": (
        Run("ours", "ours", "causal", os.path.join(OURS_CAUSAL_ROOT, "semantickitti"), True, 5,
            "frozen causal pipeline, 5 past frames, B0 / B3 seeds 0-2"),
        Run("occany_1f_geom", "occany", "causal",
            f"{_CAUSAL}/OccAny_1frame_kitti512_resized_1216_box5_text5_DINOB_boxth5_rot60_vpi50_fwd1_sTrans2", False, 1,
            "their EXP_ID 1 verbatim (geometry); reproduces the published 24.03 monocular IoU"),
        Run("occany_1f_sem", "occany", "causal",
            f"{_CAUSAL}/OccAny_1frame_kitti512_resized_1216_box5_text5_DINOB_boxth15_rot60_vpi50_fwd1_sTrans2_nsr_mm", True, 1,
            "1-frame semantic; constructed here -- OccAny publishes no such number"),
        Run("occany_5f_geom", "occany", "forward",
            f"{_FORWARD}/OccAny_5frames_kitti512_rot60_vpi10_fwd3_sTrans2", False, 5,
            "their published 5-frame geometry; inputs are t .. t+40 raw frames"),
        Run("occany_5f_sem", "occany", "forward",
            f"{_CAUSAL}/OccAny_5frames_kitti512_resized_1216_box5_text5_DINOB_boxth15_rot60_vpi10_fwd1_sTrans2_nsr_mm", True, 5,
            "their EXP_ID 4 verbatim (semantic); reproduces the published 7.28 / 13.53"),
    ),
}

#: image the method actually consumes, used only to define the visibility frustum
IMAGE_SIZE = {"semantickitti": (1226, 370)}


def occany_path(run: Run, sample_id: str) -> str:
    return os.path.join(run.path, sample_id, "voxel_predictions.pkl")


def load_occany(run: Run, sample_id: str, dims) -> Tuple[np.ndarray, np.ndarray]:
    """(label, volume) flat uint8 -- OccAny's stored ground truth and its full completion."""
    with open(occany_path(run, sample_id), "rb") as fh:
        d = pickle.load(fh)
    keys = [k for k in d if k.startswith("render_recon_gen_recon")]
    assert len(keys) == 1, (run.key, sample_id, keys)
    label = np.asarray(d["voxel_label"]).reshape(dims)
    assert label.min() >= 0 and label.max() <= 255
    return label.astype(np.uint8).reshape(-1), np.asarray(d[keys[0]]).reshape(-1).astype(np.uint8)


def samples(dataset: str) -> Tuple[str, ...]:
    """Sample ids every run in ``RUNS[dataset]`` covers, in dataset order."""
    sets = []
    for r in RUNS[dataset]:
        if r.kind == "ours":
            sets.append({f[:-4] for f in os.listdir(r.path) if f.endswith(".npz")})
        else:
            sets.append(set(os.listdir(r.path)))
    return tuple(sorted(set.intersection(*sets)))


__all__ = ["Run", "RUNS", "IMAGE_SIZE", "load_occany", "occany_path", "samples", "OURS_CAUSAL_ROOT"]
