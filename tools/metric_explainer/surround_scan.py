"""Per-sample occupancy counts for the single-camera and six-camera settings on the SAME 360-degree label.

The surround figure picks its rows at fixed quantiles of the per-sample improvement, so the improvement has to exist
per sample first. This also answers a question the pooled numbers cannot: on how many samples does adding the five
other cameras actually help?

    PYTHONPATH=$PWD python -m tools.metric_explainer.surround_scan --shard 0 --shards 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import numpy as np

from tools.like_for_like import config as LC, surround as SR, surround_predict as SP
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS

OUT = os.path.join(LC.ARTIFACT_DIR, "surround_scan.npz")
IGNORE = 255


def counts(occ: np.ndarray, gt_occ: np.ndarray, valid: np.ndarray):
    p, g = occ & valid, gt_occ & valid
    return int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())


def run(shard: int = 0, shards: int = 1, limit: int | None = None) -> Dict:
    spec = DATASET_SPECS["occ3d"]
    samples = SR.samples()[shard::shards][:limit]
    ids, one, six = [], [], []
    t0 = time.time()
    for i, s in enumerate(samples):
        p1, p6 = LC.prediction_path("occ3d", s.sample_id), SP.prediction_path(s.sample_id)
        if not (os.path.exists(p1) and os.path.exists(p6)):
            continue
        lab = SR.surround_label(s)
        valid = lab != IGNORE
        gt_occ = valid & (lab != spec.empty_class)
        a, b = CanonicalPrediction.load(p1), CanonicalPrediction.load(p6)
        ids.append(s.sample_id)
        one.append(counts(a.occupied.reshape(lab.shape), gt_occ, valid))
        six.append(counts(b.occupied.reshape(lab.shape), gt_occ, valid))
        if (i + 1) % 250 == 0:
            print(f"  [scan s{shard}] {i + 1}/{len(samples)} {(time.time() - t0) / len(ids):.3f}s each", flush=True)
    rec = {"sample_id": np.asarray(ids), "one_camera": np.asarray(one), "six_cameras": np.asarray(six)}
    path = OUT if shards == 1 else OUT.replace(".npz", f"_s{shard}of{shards}.npz")
    np.savez_compressed(path, **rec)
    return {"shard": shard, "n": len(ids), "path": path, "seconds": time.time() - t0}


def merge(shards: int) -> Dict:
    parts = [np.load(OUT.replace(".npz", f"_s{s}of{shards}.npz"), allow_pickle=True) for s in range(shards)]
    rec = {k: np.concatenate([p[k] for p in parts]) for k in ("sample_id", "one_camera", "six_cameras")}
    order = np.argsort(rec["sample_id"], kind="stable")
    rec = {k: v[order] for k, v in rec.items()}
    np.savez_compressed(OUT, **rec)
    iou = lambda c: c[:, 0] / np.maximum(c[:, 0] + c[:, 1] + c[:, 2], 1)
    d = iou(rec["six_cameras"]) - iou(rec["one_camera"])
    out = {"n_samples": int(len(d)), "mean_delta_iou_pp": float(100 * d.mean()),
           "median_delta_iou_pp": float(100 * np.median(d)), "frac_improved": float((d > 0).mean()),
           "path": OUT}
    json.dump(out, open(os.path.join(LC.ARTIFACT_DIR, "surround_scan.json"), "w"), indent=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int); ap.add_argument("--merge", type=int)
    a = ap.parse_args()
    print(json.dumps(merge(a.merge) if a.merge else run(a.shard, a.shards, a.limit), indent=1, default=str))


if __name__ == "__main__":
    main()
