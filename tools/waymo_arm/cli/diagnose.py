#!/usr/bin/env python
"""Step 1, before anything is trained: does a Waymo hard map look like a KITTI-360 one?

Every Waymo camera is 50.0 deg horizontal (MEASURED); KITTI-360's ``image_00`` is 103.9 deg.
The completion head is trained on the hard map ``(S, F)`` that a five-frame causal clip of
that camera produces, so a narrower camera moves supervised volume out of the frustum and
into never-observed space. The observed/unobserved analysis of
``reports/occany_observed_vs_unobserved.md`` showed the system's entire advantage over
OccAny lives in exactly that never-observed region, which makes this shift the central
question of swapping the training set -- not a detail.

So this measures, for every arm on the same grid, two things and nothing else:

* **frustum share** -- the fraction of the 256x256x32 grid the anchor camera can see at all,
  closed form from that arm's own K, image size and camera-to-grid transform;
* **map coverage** -- the fraction of the grid the five-frame hard map actually resolves
  (``S | F``), and how much of its surface ``S`` sits OUTSIDE the anchor frustum, i.e. comes
  from the earlier frames of the clip rather than the current view.

Train arms read their input caches; the two test arms read ``observed`` out of the frozen
canonical predictions, which is defined as "any subvoxel in hard-map S or F" -- the same
quantity. No target label is opened and nothing is trained.

    PYTHONPATH=$PWD python tools/waymo_arm/cli/diagnose.py --limit 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from core.datasets import kitti360 as K3                                               # noqa: E402
from core.datasets import kitti360_transforms as TF                                    # noqa: E402
from tools.baseline_reconstruction import config as BR                                 # noqa: E402
from tools.observability_split import visibility as V                                   # noqa: E402
from tools.waymo_arm import config as A                                                # noqa: E402
from tools.waymo_pilot import waymo_geometry as WG                                      # noqa: E402
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction               # noqa: E402

SSCBENCH_ROOT = "/media/SSD1/MINH_DATASETS/sscbench_kitti360"
#: the image each arm's clip is actually built from
IMAGE_SIZE = {"waymo": (1920, 1280), "k360": (1408, 376),
              "semantickitti": (1226, 370), "occ3d": (1600, 900)}
ZS_PRED = "/media/SSD1/MINH_DATASETS/lingbot_zero_shot_eval/predictions"


def _unpack(packed: np.ndarray, n: int) -> np.ndarray:
    return np.unpackbits(packed)[:n].astype(bool)


def frustum_share(T_cam_to_grid, K, image, dims, voxel_size, origin) -> np.ndarray:
    return V.frustum(T_cam_to_grid, K, image, dims, voxel_size, origin)


def waymo_arm(limit: int) -> dict:
    p = A.Paths()
    grid = WG.waymo_grid()
    dims, vs, org = grid.dims, grid.voxel_size, grid.origin
    n = int(np.prod(dims))
    segs = sorted(d for d in os.listdir(p.inputs) if os.path.isdir(os.path.join(p.inputs, d)))
    fov_cache, rows = {}, []
    for seg in segs:
        for f in sorted(os.listdir(os.path.join(p.inputs, seg))):
            if not f.endswith(".npz") or len(rows) >= limit:
                continue
            cam = WG.WaymoSegment(seg).calibration()
            if seg not in fov_cache:
                fov_cache[seg] = frustum_share(WG.camera_to_grid(cam), cam.K,
                                               IMAGE_SIZE["waymo"], dims, vs, org)
            fov = fov_cache[seg]
            with np.load(os.path.join(p.inputs, seg, f)) as z:
                S, F = _unpack(z["S_H"], n), _unpack(z["F_H"], n)
            rows.append(_row(S, F, fov, n))
    return _summary("waymo", rows, A.HFOV_WAYMO_DEG, len(segs))


def k360_arm(limit: int) -> dict:
    cp = BR.CurrentPaths()
    grid = TF.GRID
    dims, vs, org = grid.dims, grid.voxel_size, grid.origin
    n = int(np.prod(dims))
    cal = K3.parse_calibration(os.path.join(SSCBENCH_ROOT, "calibration"))
    T = np.asarray(cal.rect_cam_to_velo, np.float64)
    K = np.asarray(cal.K, np.float64) if hasattr(cal, "K") else np.asarray(cal.P_rect, np.float64)[:3, :3]
    fov = frustum_share(T, K, IMAGE_SIZE["k360"], dims, vs, org)
    rows, per = [], max(1, limit // max(len(BR.TRAIN_DRIVES), 1))
    for drive in BR.TRAIN_DRIVES:
        d = os.path.join(cp.inputs_root, drive)
        if not os.path.isdir(d):
            continue
        fs = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
        for f in fs[:: max(1, len(fs) // per)][:per]:
            with np.load(os.path.join(d, f)) as z:
                S, F = _unpack(z["S_H"], n), _unpack(z["F_H"], n)
            rows.append(_row(S, F, fov, n))
    return _summary("k360", rows, A.HFOV_K360_IMAGE00_DEG, len(BR.TRAIN_DRIVES))


def test_arm(dataset: str, limit: int) -> dict:
    d = os.path.join(ZS_PRED, dataset)
    fs = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
    step = max(1, len(fs) // limit)
    rows = []
    for f in fs[::step][:limit]:
        pred = CanonicalPrediction.load(os.path.join(d, f))
        n = pred.n_voxels
        fov = frustum_share(pred.T_cam_to_grid, pred.K, IMAGE_SIZE[dataset], pred.dims,
                            pred.voxel_size, pred.origin)
        rows.append({"map_coverage": float(pred.observed.mean()),
                     "frustum_share": float(fov.mean()),
                     "coverage_outside_frustum": float((pred.observed & ~fov).sum()
                                                       / max(int(pred.observed.sum()), 1)),
                     "S_share": None, "S_outside_frustum": None})
    hfov = A.HFOV_SEMANTICKITTI_DEG if dataset == "semantickitti" else A.HFOV_OCC3D_CAM_FRONT_DEG
    return _summary(dataset, rows, hfov, None)


def _row(S, F, fov, n) -> dict:
    obs = S | F
    return {"map_coverage": float(obs.mean()), "frustum_share": float(fov.mean()),
            "coverage_outside_frustum": float((obs & ~fov).sum() / max(int(obs.sum()), 1)),
            "S_share": float(S.mean()),
            "S_outside_frustum": float((S & ~fov).sum() / max(int(S.sum()), 1))}


def _summary(name, rows, hfov, n_scenes) -> dict:
    keys = ["map_coverage", "frustum_share", "coverage_outside_frustum", "S_share",
            "S_outside_frustum"]
    out = {"arm": name, "n_anchors": len(rows), "n_scenes": n_scenes, "hfov_deg": hfov}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        out[k] = {"mean": float(np.mean(v)), "p10": float(np.percentile(v, 10)),
                  "p90": float(np.percentile(v, 90))} if v else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--arms", default="waymo,k360,semantickitti,occ3d")
    a = ap.parse_args()
    A.ensure_dirs()
    out = {"version": A.VERSION, "limit": a.limit, "arms": {}}
    for arm in a.arms.split(","):
        fn = {"waymo": waymo_arm, "k360": k360_arm}.get(arm)
        try:
            out["arms"][arm] = fn(a.limit) if fn else test_arm(arm, a.limit)
        except Exception as exc:                                               # noqa: BLE001
            out["arms"][arm] = {"arm": arm, "error": f"{type(exc).__name__}: {exc}"}
        print(f"  [{arm}] done", flush=True)
    path = os.path.join(A.Paths().logs, "diagnose_composition.json")
    json.dump(out, open(path, "w"), indent=2, default=float)

    cols = [("map_coverage", "map cov"), ("frustum_share", "frustum"),
            ("coverage_outside_frustum", "cov out"), ("S_share", "S"),
            ("S_outside_frustum", "S out")]
    hdr = f"{'arm':14s} {'HFOV':>6s} {'n':>5s} " + " ".join(f"{t:>9s}" for _, t in cols)
    print("\n" + hdr)
    print("-" * len(hdr))
    for arm, r in out["arms"].items():
        if "error" in r:
            print(f"{arm:14s} {r['error'][:70]}")
            continue
        cells = [f"{r[k]['mean']:9.4f}" if r.get(k) else f"{'-':>9s}" for k, _ in cols]
        print(f"{r['arm']:14s} {r['hfov_deg']:6.1f} {r['n_anchors']:5d} " + " ".join(cells))
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
