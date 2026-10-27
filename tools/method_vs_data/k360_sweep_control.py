#!/usr/bin/env python
"""KITTI-360 supervision control — brief step 10, measurement only, no training.

Two jobs.

**1. Validate the measurement path.** The ONCE gates judge occupancy statistics against bands
predeclared from KITTI-360's 0.0276 prevalence and Waymo's 0.0222. Before any ONCE number is
believed, the same statistics must be reproduced for KITTI-360 through the same code. The
frozen window is ``[anchor] + future_natives`` -- 21 sweeps at 5-native-frame spacing, which
at KITTI-360's 0.1046 s velodyne period is 0.523 s per sweep and a **10.46 s** span.

**2. Decimate the window** to isolate sweep count from dataset. ``--decimate k`` keeps every
k-th sweep, so ``k=5`` gives 5 sweeps over the same span and ``k=1`` is the frozen recipe.
Comparing the two separates "ONCE differs" from "fewer carves differ".

No training is launched here and no checkpoint is touched.

    PYTHONPATH=$PWD python tools/method_vs_data/k360_sweep_control.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.datasets import kitti360_splits as KS, kitti360_transforms as TF                # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402
from core.supervision import rawtarget as RT                                              # noqa: E402

BANDS = {"prevalence": (0.015, 0.040), "conflict_max": 0.12,
         "valid": (0.45, 0.75), "z_levels_min": 30}


def run(drive: str, n_anchors: int, decimate: int, device,
        max_span_s: float | None = None) -> dict:
    geo = TF.DriveGeometry(drive, KS.SSCBENCH_ROOT, KS.KITTI360_ROOT)
    ts = geo.timestamps("velodyne")
    anchors = KS.anchors(drive, REPO_ROOT)
    if not anchors:
        return {"drive": drive, "error": "no anchors"}
    step = max(len(anchors) // n_anchors, 1)
    sel = anchors[::step][:n_anchors]
    recs = []
    for a in sel:
        sweeps = [a.native_frame] + list(a.future_natives)
        sweeps = sweeps[::decimate]
        if max_span_s is not None:
            t0 = ts[sweeps[0]]
            sweeps = [i for i in sweeps if ts[i] - t0 <= max_span_s]
        if len(sweeps) < 2:
            continue
        t = RT.build(geo, a.native_frame, sweeps, device, grid=TF.GRID)
        s = dict(t.stats)
        s.update({"anchor_native": int(a.native_frame), "n_sweeps_requested": len(sweeps),
                  "sweep_dt_s": float(np.median(np.diff([ts[i] for i in sweeps]))),
                  "span_s": float(ts[sweeps[-1]] - ts[sweeps[0]]),
                  "z_levels_occupied": int((t.state.reshape(TF.DIMS) == RT.OCCUPIED).any((0, 1)).sum())})
        recs.append(s)
    f = lambda k: np.array([r[k] for r in recs], np.float64)
    med = {"prevalence": float(np.median(f("prevalence_in_valid"))),
           "valid": float(np.median(f("valid_fraction"))),
           "conflict": float(np.median(f("conflict_fraction_of_touched"))),
           "z_levels": float(np.median(f("z_levels_occupied"))),
           "n_frames_used": float(np.median(f("n_frames_used"))),
           "span_s": float(np.median(f("span_s"))),
           "sweep_dt_s": float(np.median(f("sweep_dt_s")))}
    ok = {"prevalence": BANDS["prevalence"][0] <= med["prevalence"] <= BANDS["prevalence"][1],
          "valid": BANDS["valid"][0] <= med["valid"] <= BANDS["valid"][1],
          "conflict": med["conflict"] <= BANDS["conflict_max"],
          "z_levels": med["z_levels"] >= BANDS["z_levels_min"]}
    return {"drive": drive, "decimate": decimate, "max_span_s": max_span_s,
            "n_targets": len(recs),
            "median": med, "per_band_pass": ok, "in_bands": bool(all(ok.values())),
            "ranges": {k: [float(np.min(f(kk))), float(np.max(f(kk)))]
                       for k, kk in (("prevalence", "prevalence_in_valid"),
                                     ("valid", "valid_fraction"),
                                     ("conflict", "conflict_fraction_of_touched"))}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drives", default=None, help="default: the frozen train drives")
    ap.add_argument("--n-anchors", type=int, default=30)
    ap.add_argument("--decimations", default="1,5")
    ap.add_argument("--max-span-s", default="",
                    help="truncate each window to this duration; '2.1' gives the "
                         "duration-matched reference for the authorized ONCE policy "
                         "(5 sweeps at KITTI-360's 0.523 s spacing)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="artifacts/method_vs_data/results/k360_sweep_control.json")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    drives = a.drives.split(",") if a.drives else list(KS.TRAIN_DRIVES)
    res = {"bands": BANDS, "note": ("decimate=1 is the frozen recipe (21 sweeps, 0.523 s "
                                    "spacing, 10.46 s span); decimate=5 keeps 5 sweeps over "
                                    "the same span"), "runs": {}}
    print(f"bands: {BANDS}")
    spans = [None] + [float(x) for x in a.max_span_s.split(",") if x.strip()]
    for sp in spans:
      for dec in [int(x) for x in a.decimations.split(",")]:
        for d in drives:
            r = run(d, a.n_anchors, dec, dev, max_span_s=sp)
            res["runs"][f"{d}|dec{dec}|span{sp}"] = r
            if "median" in r:
                m, pb = r["median"], r["per_band_pass"]
                print(f"  {d[-14:]:14s} dec={dec} maxspan={sp}: "
                      f"sweeps {m['n_frames_used']:4.1f} "
                      f"dt {m['sweep_dt_s']:.3f}s span {m['span_s']:6.2f}s | "
                      f"prev {m['prevalence']:.4f}{'' if pb['prevalence'] else ' OUT'} "
                      f"valid {m['valid']:.4f}{'' if pb['valid'] else ' OUT'} "
                      f"conflict {m['conflict']:.4f}{'' if pb['conflict'] else ' OUT'} "
                      f"z {m['z_levels']:.0f} | {'IN BANDS' if r['in_bands'] else 'OUT'} "
                      f"({r['n_targets']} targets)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
