#!/usr/bin/env python
"""Phase 3 diagnostic — occupancy IoU inside vs outside the anchor camera frustum.

The brief asks for this split *and* warns that out-of-frustum is not the same thing as
unobserved. Both splits are therefore reported, and crossed:

  in_frustum       voxel centre in front of the anchor camera and inside the image rectangle
  out_frustum      everything else in the grid
  observed         ``pred.observed`` -- some subvoxel was resolved by the causal hard map
  completed        NOT ``pred.observed`` -- the network had to invent it

``observed`` is a property of the accumulated map (it includes voxels the anchor camera
cannot see but an earlier frame did), so the two axes are genuinely independent; the 2x2
table shows how far apart they are.

Scope, stated because it differs from the Phase 3 primary result: this reads the canonical
zero-shot predictions, i.e. GEOMETRY_SEED 0 only, on the canonical anchor population
(SemanticKITTI 801, Occ3D 4219) with the OccAny valid mask. The Phase 3 primary numbers come
from the frozen ``baseline_reconstruction`` evaluator on its own population (163 / 1182) with
its own masks and three seeds. The two are not interchangeable; what makes this diagnostic
valid is that BOTH arms are scored here under one protocol.

    PYTHONPATH=$PWD python tools/waymo_arm/cli/frustum_split.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.observability_split.visibility import frustum                                # noqa: E402
from tools.occany_comparison import scoring as SC                                       # noqa: E402
from tools.waymo_arm import config as A                                                 # noqa: E402
from tools.waymo_arm.cli.predict_semantic import PREDICTION_ROOT as WAYMO_PRED          # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS                            # noqa: E402
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction               # noqa: E402
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS                       # noqa: E402

K360_PRED = ZC.PREDICTION_ROOT
#: native image rectangle of the anchor camera. SemanticKITTI seq 08 image_2 is 1226 x 370;
#: nuScenes CAM_FRONT is 1600 x 900.
IMAGE_SIZE = {"semantickitti": (1226, 370), "occ3d": (1600, 900)}
IGNORE = 255


def split(dataset: str, limit: int | None = None) -> dict:
    spec = DATASET_SPECS[dataset]
    W, H = IMAGE_SIZE[dataset]
    assert spec.image_width == W, (spec.image_width, W)
    acc = defaultdict(lambda: 0)
    n = 0
    for s in ZS.evaluated_samples(dataset)[:limit]:
        pk = os.path.join(K360_PRED, dataset, f"{s.sample_id}.npz")
        pw = os.path.join(WAYMO_PRED, dataset, f"{s.sample_id}.npz")
        if not (os.path.exists(pk) and os.path.exists(pw)):
            continue
        a_k, a_w = CanonicalPrediction.load(pk), CanonicalPrediction.load(pw)
        # validity control: the pose and intrinsics come from the sample, not the checkpoint,
        # so the frustum mask must be shared. If it is not, the split is meaningless.
        assert np.allclose(a_k.T_cam_to_grid, a_w.T_cam_to_grid), s.sample_id
        assert np.allclose(a_k.K, a_w.K), s.sample_id
        assert a_k.K[1, 2] < H, (a_k.K[1, 2], H)
        label = SC.load_occany(dataset, s.occany_pkl, a_k.dims).label.reshape(-1)
        valid = label != IGNORE
        gt_occ = valid & (label != spec.empty_class)
        fov = frustum(a_k.T_cam_to_grid, a_k.K, (W, H), a_k.dims, a_k.voxel_size, a_k.origin)
        for arm, p in (("k360", a_k), ("waymo", a_w)):
            occ = p.occupied
            for fname, fm in (("in_frustum", fov), ("out_frustum", ~fov)):
                for oname, om in (("all", np.ones_like(fov)), ("observed", p.observed),
                                  ("completed", ~p.observed)):
                    m = valid & fm & om
                    acc[f"{arm}__{fname}__{oname}"] = acc[f"{arm}__{fname}__{oname}"] + np.array(
                        [int((occ & gt_occ & m).sum()), int((occ & ~gt_occ & m).sum()),
                         int((~occ & gt_occ & m).sum())], np.int64)
        # GT composition is arm-independent; take it from the k360 observability for reference
        for fname, fm in (("in_frustum", fov), ("out_frustum", ~fov)):
            acc[f"gt__{fname}"] = acc[f"gt__{fname}"] + np.array(
                [int((valid & fm).sum()), int((gt_occ & fm).sum())], np.int64)
        n += 1

    def scores(c):
        tp, fp, fn = (float(x) for x in c)
        return {"iou": 100 * tp / max(tp + fp + fn, 1.0),
                "precision": 100 * tp / max(tp + fp, 1.0),
                "recall": 100 * tp / max(tp + fn, 1.0),
                "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    valid_tot = sum(int(acc[f"gt__{f}"][0]) for f in ("in_frustum", "out_frustum"))
    occ_tot = sum(int(acc[f"gt__{f}"][1]) for f in ("in_frustum", "out_frustum"))
    out = {"dataset": dataset, "n_samples": n, "image_size": [W, H],
           "k360_predictions": K360_PRED, "waymo_predictions": WAYMO_PRED,
           "geometry_seed": ZC.GEOMETRY_SEED, "composition": {}, "iou": {}}
    for f in ("in_frustum", "out_frustum"):
        v, o = (int(x) for x in acc[f"gt__{f}"])
        out["composition"][f] = {"valid_voxels": v, "share_of_valid": v / max(valid_tot, 1),
                                 "gt_occupied": o, "share_of_gt_occupied": o / max(occ_tot, 1),
                                 "prevalence_in_region": o / max(v, 1)}
    for f in ("in_frustum", "out_frustum"):
        for o in ("all", "observed", "completed"):
            rec = {arm: scores(acc[f"{arm}__{f}__{o}"]) for arm in ("k360", "waymo")}
            rec["delta_iou"] = rec["waymo"]["iou"] - rec["k360"]["iou"]
            out["iou"][f"{f}/{o}"] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="semantickitti,occ3d")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    res = {"version": A.VERSION, "results": {}}
    for ds in a.datasets.split(","):
        r = split(ds, a.limit)
        res["results"][ds] = r
        print(f"\n=== {ds}  ({r['n_samples']} samples, geometry seed {r['geometry_seed']})")
        for k, v in r["composition"].items():
            print(f"  {k:12s} {v['share_of_valid']*100:5.1f}% of valid, "
                  f"{v['share_of_gt_occupied']*100:5.1f}% of GT occupied, "
                  f"prevalence {v['prevalence_in_region']*100:5.2f}%")
        print(f"  {'population':26s} {'k360':>7s} {'waymo':>7s} {'delta':>7s} "
              f"{'P k360':>7s} {'P waymo':>7s} {'R k360':>7s} {'R waymo':>7s}")
        for k, v in r["iou"].items():
            print(f"  {k:26s} {v['k360']['iou']:7.2f} {v['waymo']['iou']:7.2f} "
                  f"{v['delta_iou']:+7.2f} {v['k360']['precision']:7.2f} "
                  f"{v['waymo']['precision']:7.2f} {v['k360']['recall']:7.2f} "
                  f"{v['waymo']['recall']:7.2f}")
    d = os.path.join(A.ARTIFACT_DIR, "results")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "frustum_split.json")
    json.dump(res, open(path, "w"), indent=1, default=float)
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
