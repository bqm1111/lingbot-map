"""Loading the cached C3 voxel clips.

The cache stores model inputs and LiDAR-derived targets in the *same file*, so the
separation is enforced here, in one place: :func:`inputs` returns only the five C3
evidence arrays, :func:`targets` returns the LiDAR-derived volumes, and nothing calls
both into the same tensor. ``tests/voxel_gate`` asserts the input path is bit-identical
when the targets are mutated.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import numpy as np
import torch

from gates.scale_gate.config import REPO_ROOT
from prompted_lingbot.occupancy import SEMANTICKITTI_GRID as G

from .voxels import build_input, correction_region, dense_from_sparse, unpack

N_VOXELS = int(np.prod(G.dims))

# Keys the model may read; everything else in the npz is target/evaluation material.
INPUT_KEYS = ("c3_flat", "c3_count", "c3_n_frames", "c3_sum_conf", "c3_sum_depth")
TARGET_KEYS = ("gt_flat", "vc_flat", "valid_bits")


def clip_index(cfg, split: str) -> List[dict]:
    path = os.path.join(REPO_ROOT, cfg.experiment.output_dir, f"c3_{split}.json")
    return json.load(open(path))["clips"]


def select_clips(cfg, split: str, sequences: Sequence[str]) -> List[dict]:
    s = set(sequences)
    return [c for c in clip_index(cfg, split) if c["sequence"] in s]


def load_npz(cfg, clip_id: str):
    return np.load(os.path.join(REPO_ROOT, cfg.experiment.output_dir, "cache_c3",
                                f"{clip_id}.npz"), allow_pickle=False)


def inputs(d, device) -> Dict[str, torch.Tensor]:
    """Frozen-C3 evidence only. No key from :data:`TARGET_KEYS` is touched."""
    sp = {"flat": d["c3_flat"].astype(np.int64), "count": d["c3_count"],
          "n_frames": d["c3_n_frames"], "sum_conf": d["c3_sum_conf"],
          "sum_depth": d["c3_sum_depth"]}
    feat5 = dense_from_sparse(sp, device)
    return {"feat5": feat5, "occupied": feat5[0] > 0}


def targets(d, device):
    """LiDAR-derived volumes: evaluation mask, GT occupancy, visible-ceiling occupancy."""
    keep = torch.from_numpy(unpack(d["valid_bits"])).to(device)
    gt = torch.zeros(N_VOXELS, dtype=torch.bool, device=device)
    vc = torch.zeros(N_VOXELS, dtype=torch.bool, device=device)
    if d["gt_flat"].size:
        gt[torch.from_numpy(d["gt_flat"].astype(np.int64)).to(device)] = True
    if d["vc_flat"].size:
        vc[torch.from_numpy(d["vc_flat"].astype(np.int64)).to(device)] = True
    return keep, gt.view(G.dims), vc.view(G.dims) & keep


def sample(cfg, clip_id: str, radius: int, norm, device):
    """One training/evaluation sample.

    ``y`` is plain binary LiDAR occupancy. The anti-completion guarantee is the
    *supervision region*, not the label: the model only ever sees loss inside the local
    band around observed C3 surfaces, so it cannot be trained to populate occluded or
    unobserved space. ``vc`` (the five-frame visible-reconstruction occupancy) is carried
    alongside as a **diagnostic only** -- it is never a model input, never gates
    inference, and no longer restricts the label.
    """
    d = load_npz(cfg, clip_id)
    inp = inputs(d, device)
    region = correction_region(inp["occupied"], radius)
    keep, gt, vc = targets(d, device)
    region = region & keep
    x = build_input(inp["feat5"], region, norm) if norm is not None else None
    return {"x": x, "occupied": inp["occupied"], "region": region, "keep": keep,
            "gt": gt, "vc": vc, "y": gt.float(), "clip_id": clip_id}


def compute_norm(cfg, clip_ids: Sequence[str]) -> Dict[str, float]:
    """Input normalisation from **source-training clips only**."""
    acc = {k: [] for k in ("log1p_count", "n_frames", "mean_confidence",
                           "mean_point_depth")}
    for cid in clip_ids:
        d = load_npz(cfg, cid)
        c = d["c3_count"].astype(np.float64)
        if not c.size:
            continue
        acc["log1p_count"].append(np.log1p(c))
        acc["n_frames"].append(d["c3_n_frames"].astype(np.float64))
        acc["mean_confidence"].append(d["c3_sum_conf"].astype(np.float64) / c)
        acc["mean_point_depth"].append(d["c3_sum_depth"].astype(np.float64) / c)
    out = {}
    for k, v in acc.items():
        a = np.concatenate(v)
        out[f"{k}_mean"], out[f"{k}_std"] = float(a.mean()), float(a.std() + 1e-6)
    out["n_clips"] = len(clip_ids)
    return out


def scores(pred: torch.Tensor, gt: torch.Tensor, keep: torch.Tensor) -> Dict[str, float]:
    """Binary occupancy tp/fp/fn/IoU, identical in definition to the frozen evaluator."""
    p, t = pred & keep, gt & keep
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    den = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "iou": tp / den if den else 0.0,
            "n_pred_occupied": int(p.sum())}
