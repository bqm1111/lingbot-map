"""Our prediction for the surround setting: six cameras, one timestep, scored on the 360-degree Occ3D label.

Everything downstream of the window is the verified body ``like_for_like.predict.predict_window`` -- frozen geometry,
evidence lifting, B0 / B3 readout, grids, reductions. Only three things are the surround setting's own:

* the window is the six ring views of one timestep (``surround_clips``), anchor slot 0 = ``CAM_FRONT``;
* the benchmark placement is OccAny's own ``T_cam_to_voxel`` from the saved output of the SAME sample, which is
  ``cam0_to_ego`` for ``CAM_FRONT`` in both of OccAny's settings, so no rig calibration is needed here either;
* the ground truth is the 360-degree label (``surround.surround_label``), matching OccAny's ``use_surround_label=True``.

    PYTHONPATH=$PWD python -m tools.like_for_like.surround_predict --device cuda:0 --shard 0 --shards 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from core.datasets import grids as CG
from tools.like_for_like import config as LC, surround as SR, surround_clips as SCL
from tools.like_for_like.predict import predict_window
from tools.occany_comparison import scoring as SC
from tools.occany_comparison.geometry import WindowClip
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction
from tools.zero_shot_eval.prediction.grid import GridSpec
from tools.zero_shot_eval.prediction.pipeline import FrozenModels, load_models


def prediction_path(sample_id: str) -> str:
    return os.path.join(SR.PREDICTION_ROOT, f"{sample_id}.npz")


def predict_surround(sample, models: FrozenModels, device) -> CanonicalPrediction:
    native = GridSpec.of(CG.EVAL_GRID["occ3d"])
    T = SC.load_occany("occ3d", sample.occany_pkl, native.dims).T_cam_to_grid
    clip = WindowClip.load(SCL.clip_path(sample.sample_id)); assert clip.anchor_slot == SCL.ANCHOR_SLOT
    keys = [v["key"] for v in SR.ring_views(SR.sample_token(sample))]
    return predict_window(sample, "occ3d", models, device, clip, keys, SR.surround_label(sample).reshape(-1), T,
                          {"protocol": "surround", "calibration": "occany", "cameras": list(SR.RING), "label": "surround (360 deg)"})


def run_shard(device: str, shard: int = 0, shards: int = 1, limit: Optional[int] = None) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    os.makedirs(SR.PREDICTION_ROOT, exist_ok=True)
    models = load_models("occ3d", dev)
    samples = SR.samples()[shard::shards][:limit]
    t0, written = time.time(), 0
    for i, s in enumerate(samples):
        out = prediction_path(s.sample_id)
        if os.path.exists(out):
            continue
        predict_surround(s, models, dev).save(out); written += 1
        if (i + 1) % 50 == 0:
            print(f"  [surround s{shard}] {i + 1}/{len(samples)} {(time.time() - t0) / max(written, 1):.2f}s each", flush=True)
    rec = {"setting": "surround", "shard": shard, "shards": shards, "samples": len(samples), "written": written,
           "seconds": time.time() - t0}
    os.makedirs(LC.LOG_DIR, exist_ok=True)
    json.dump(rec, open(os.path.join(LC.LOG_DIR, f"surround_predict_s{shard}of{shards}.json"), "w"), indent=1, default=str)
    return rec


def check_placement(n: int = 5) -> Dict:
    """OccAny's stored ``T_cam_to_voxel`` must be the CAM_FRONT-to-ego transform the surround setting also uses."""
    native = GridSpec.of(CG.EVAL_GRID["occ3d"]); out = []
    for s in SR.samples()[:n]:
        stored = SC.load_occany("occ3d", s.occany_pkl, native.dims).T_cam_to_grid
        rig = SR.ring_views(SR.sample_token(s))[0]["T_cam_to_ego"]
        out.append({"sample_id": s.sample_id, "max_abs_diff_rotation": float(np.abs(stored[:3, :3] - rig[:3, :3]).max()),
                    "translation_diff_m": (stored[:3, 3] - rig[:3, 3]).round(4).tolist()})
    return {"samples": out}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1); ap.add_argument("--limit", type=int)
    ap.add_argument("--check-placement", action="store_true")
    a = ap.parse_args()
    print(json.dumps(check_placement() if a.check_placement else run_shard(a.device, a.shard, a.shards, a.limit), indent=1, default=str))


if __name__ == "__main__":
    main()
