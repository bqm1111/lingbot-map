"""Clean sample construction: the inference path can never see a target.

Contrast with the Gate-3 implementation (``voxel_gate/data.py:81``), which computed

    region = correction_region(occupied, radius) & keep          # keep == SSC valid mask

and then passed that region both as input channel 5 and as the output gate. Here:

    R_infer = binary_dilation(C3_occupied, radius=3)             # C3 evidence only
    S_train = R_infer AND valid                                  # loss only
    S_eval  = valid                                              # metrics only

``R_infer`` may contain evaluation-invalid voxels; that is expected and tested.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import numpy as np
import torch

from gates.scale_gate.config import REPO_ROOT
from prompted_lingbot.occupancy import SEMANTICKITTI_GRID as G

from gates.voxel_gate.voxels import build_input, dilate, dense_from_sparse, unpack

N_VOXELS = int(np.prod(G.dims))

INPUT_KEYS = ("c3_flat", "c3_count", "c3_n_frames", "c3_sum_conf", "c3_sum_depth")
TARGET_KEYS = ("gt_flat", "vc_flat", "valid_bits")


def cache_dir(cfg, geometry: str) -> str:
    return os.path.join(REPO_ROOT, cfg.experiment.output_dir, f"cache_{geometry}")


def clip_index(cfg, geometry: str, split: str) -> List[dict]:
    p = os.path.join(REPO_ROOT, cfg.experiment.output_dir, f"{geometry}_{split}.json")
    return json.load(open(p))["clips"]


def select_clips(cfg, geometry: str, split: str, sequences: Sequence[str]) -> List[dict]:
    s = set(sequences)
    return [c for c in clip_index(cfg, geometry, split) if c["sequence"] in s]


def load_npz(cfg, geometry: str, clip_id: str):
    return np.load(os.path.join(cache_dir(cfg, geometry), f"{clip_id}.npz"),
                   allow_pickle=False)


def input_view(d) -> Dict[str, np.ndarray]:
    """Copy out **only** :data:`INPUT_KEYS`. Everything downstream sees this, not ``d``.

    This is the structural fix for the Gate-3 leak: the inference path is handed a dict
    that physically does not contain ``valid_bits``, ``gt_flat`` or ``vc_flat``, so no
    later edit can reintroduce a target dependency without failing a KeyError.
    """
    return {k: np.asarray(d[k]) for k in INPUT_KEYS}


def inference_inputs(d, radius: int, device) -> Dict[str, torch.Tensor]:
    """Everything the model may see. Reads only :data:`INPUT_KEYS`; no target, ever."""
    v = input_view(d)
    sp = {"flat": v["c3_flat"].astype(np.int64), "count": v["c3_count"],
          "n_frames": v["c3_n_frames"], "sum_conf": v["c3_sum_conf"],
          "sum_depth": v["c3_sum_depth"]}
    feat5 = dense_from_sparse(sp, device)
    occupied = feat5[0] > 0
    return {"feat5": feat5, "occupied": occupied,
            "R_infer": dilate(occupied, radius)}          # pure dilation, no valid mask


def evaluation_targets(d, device):
    """Label-side volumes. Used for loss restriction, metrics and diagnostics only."""
    keep = torch.from_numpy(unpack(d["valid_bits"])).to(device)
    gt = torch.zeros(N_VOXELS, dtype=torch.bool, device=device)
    vc = torch.zeros(N_VOXELS, dtype=torch.bool, device=device)
    if d["gt_flat"].size:
        gt[torch.from_numpy(d["gt_flat"].astype(np.int64)).to(device)] = True
    if d["vc_flat"].size:
        vc[torch.from_numpy(d["vc_flat"].astype(np.int64)).to(device)] = True
    return keep, gt.view(G.dims), vc.view(G.dims) & keep


def sample(cfg, geometry: str, clip_id: str, radius: int, norm, device,
           occ_only: bool = False):
    d = load_npz(cfg, geometry, clip_id)
    inp = inference_inputs(d, radius, device)
    keep, gt, vc = evaluation_targets(d, device)
    x = None
    if norm is not None:
        x = build_input(inp["feat5"], inp["R_infer"], norm)
        if occ_only:                       # ablation A_occ_only: keep shape, zero 1-4
            x = x.clone(); x[1:5] = 0.0
    return {"x": x, "occupied": inp["occupied"], "R_infer": inp["R_infer"],
            "keep": keep, "gt": gt, "vc": vc, "y": gt.float(),
            "S_train": inp["R_infer"] & keep, "clip_id": clip_id}


def compute_norm(cfg, geometry: str, clip_ids: Sequence[str]) -> Dict[str, float]:
    """Standardisation from source-training clips only; no target array is opened."""
    acc = {k: [] for k in ("log1p_count", "n_frames", "mean_confidence",
                           "mean_point_depth")}
    for cid in clip_ids:
        d = load_npz(cfg, geometry, cid)
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
    """Binary occupancy metrics. ``keep`` restricts the *metric*, never the prediction."""
    p, t = pred & keep, gt & keep
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    den = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "iou": tp / den if den else 0.0,
            "n_pred_occupied": int(p.sum())}
