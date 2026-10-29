#!/usr/bin/env python
"""Cross-tabulate the region masks, to settle whether they may be used interchangeably.

Three different notions are in use in this directory and they are routinely conflated:

* **frustum membership** -- does the anchor camera's view volume contain the voxel centre;
* **strict surface visibility** -- does a GT raycast from one of the five permitted frames
  reach the voxel, stopping at the first GT-occupied voxel it meets;
* **hard-map evidence state** -- did OUR accumulated reconstruction resolve the voxel.

Each is a partition of the valid mask on its own, but they are not the same partition and
they are not additive. This script prints the joint counts so a claim about one cannot be
restated as a claim about another by accident.

    PYTHONPATH=$PWD python tools/method_vs_data/region_crosstab.py --limit 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch                                                                           # noqa: E402
from tools.method_vs_data.observed_strict import (frame_transforms, sk_pose_chain,      # noqa: E402
                                                  strict_regions)
from tools.observability_split import runs as R, visibility as V                        # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS                            # noqa: E402
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction               # noqa: E402
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS                       # noqa: E402

IGNORE = 255
STRICT = ("anchor_observed", "earlier_only", "never_observed")


def run(limit=None, device="cuda:0") -> dict:
    ds = "semantickitti"
    spec = DATASET_SPECS[ds]
    sids = list(R.samples(ds))[:limit]
    by_sid = {s.sample_id: s for s in ZS.evaluated_samples(ds)}
    sids = [s for s in sids if s in by_sid]
    runs = {r.key: r for r in R.RUNS[ds]}
    ref = CanonicalPrediction.load(os.path.join(ZC.PREDICTION_ROOT, ds, f"{sids[0]}.npz"))
    dims, vs, origin = ref.dims, ref.voxel_size, ref.origin
    W, H = 1226, 370
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    poses, cam_to_velo = sk_pose_chain(ZC.REPO_ROOT, "08")

    cells = defaultdict(lambda: np.zeros(2, np.int64))       # (valid, gt_occupied)
    part = defaultdict(lambda: 0)
    for i, sid in enumerate(sids):
        s = by_sid[sid]
        p = CanonicalPrediction.load(os.path.join(ZC.PREDICTION_ROOT, ds, f"{sid}.npz"))
        label, _ = R.load_occany(runs["occany_1f_geom"], sid, dims)
        valid = label != IGNORE
        gt = valid & (label != spec.empty_class)
        natives = [int(k.split("_")[1]) for k in (f["key"] for f in ZS.history_frames(s))]
        Ts = frame_transforms(poses, cam_to_velo, natives, natives[-1])
        m, _ = strict_regions(gt, valid, Ts, p.K, (W, H), dims, vs, origin, dev, 2, 0.1)
        fov = V.frustum(p.T_cam_to_grid, p.K, (W, H), dims, vs, origin)
        obs = p.observed.astype(bool)
        for k in STRICT:
            for fk, fm in (("in_frustum", fov), ("out_frustum", ~fov)):
                for ok, om in (("map_observed", obs), ("map_completed", ~obs)):
                    mm = m[k] & fm & om
                    cells[f"{k}|{fk}|{ok}"] += np.array([int(mm.sum()),
                                                         int((mm & gt).sum())], np.int64)
        # partition properties, asserted rather than assumed
        part["valid"] += int(valid.sum())
        part["strict_sum"] += int(sum(int(m[k].sum()) for k in STRICT))
        part["frustum_sum"] += int((valid & fov).sum()) + int((valid & ~fov).sum())
        part["map_sum"] += int((valid & obs).sum()) + int((valid & ~obs).sum())
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sids)}", flush=True)

    tot_v = sum(int(v[0]) for v in cells.values())
    tot_g = sum(int(v[1]) for v in cells.values())
    out = {"dataset": ds, "n_samples": len(sids),
           "partition_checks": {
               "valid_voxels": part["valid"],
               "strict_regions_sum_equals_valid": part["strict_sum"] == part["valid"],
               "frustum_split_sum_equals_valid": part["frustum_sum"] == part["valid"],
               "map_evidence_split_sum_equals_valid": part["map_sum"] == part["valid"]},
           "joint": {k: {"valid_voxels": int(v[0]), "gt_occupied": int(v[1]),
                         "share_of_valid": int(v[0]) / max(tot_v, 1),
                         "share_of_gt_occupied": int(v[1]) / max(tot_g, 1)}
                     for k, v in sorted(cells.items())}}
    # the two headline marginal confusions, spelled out
    g = lambda k: out["joint"][k]["gt_occupied"]                               # noqa: E731
    nev = sum(g(f"never_observed|{f}|{o}") for f in ("in_frustum", "out_frustum")
              for o in ("map_observed", "map_completed"))
    outf = sum(g(f"{k}|out_frustum|{o}") for k in STRICT
               for o in ("map_observed", "map_completed"))
    both = sum(g(f"never_observed|out_frustum|{o}") for o in ("map_observed", "map_completed"))
    out["overlap_of_gt_occupied"] = {
        "never_observed": nev, "out_frustum": outf, "both": both,
        "fraction_of_never_observed_that_is_out_of_frustum": both / max(nev, 1),
        "fraction_of_out_of_frustum_that_is_never_observed": both / max(outf, 1),
        "note": ("the two masks are NOT interchangeable: an out-of-frustum result is a "
                 "statement about a different voxel set than a never-observed result")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="artifacts/method_vs_data/waymo_canonical_rescore/"
                                    "region_crosstab.json")
    a = ap.parse_args()
    r = run(a.limit, a.device)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(r, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(r["partition_checks"], indent=1))
    print(json.dumps(r["overlap_of_gt_occupied"], indent=1))
    print(f"{'cell':52s} {'%valid':>8s} {'%gt_occ':>8s}")
    for k, v in r["joint"].items():
        print(f"{k:52s} {100 * v['share_of_valid']:8.2f} {100 * v['share_of_gt_occupied']:8.2f}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
