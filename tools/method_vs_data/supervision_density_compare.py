#!/usr/bin/env python
"""Brief §8 comparative supervision statistics: ONCE against KITTI-360, duration-matched.

The authorized policy fixes the supervision **duration** at ~2.1 s rather than the sweep
count. That has a consequence the brief's own table does not state: at KITTI-360's frozen
0.523 s sweep spacing a 2.1 s window holds **5 sweeps**, the same number ONCE gets at its
native 2 FPS. So the policy makes the two corpora sweep-matched as well as duration-matched,
and the correct reference for ONCE is KITTI-360 **at 2.1 s**, not KITTI-360 at the frozen
10.46 s.

This script reports, for both corpora under the authorized window, the quantities §8 asks for:
occupied / free / unknown voxel counts, occupancy by height, occupancy by range from the
anchor, contributing sweeps and actual temporal coverage. Measurement only; the carve rule,
grid and bands are untouched.

    PYTHONPATH=$PWD python tools/method_vs_data/supervision_density_compare.py --device cuda:3
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
from tools.method_vs_data.once_reader import OnceSequence, sequences                      # noqa: E402

SPAN_S = 2.1
Z_BINS = 8          # 32 z-levels -> 8 bins of 4 (0.8 m each)
R_BINS = (0, 10, 20, 30, 40, 51.2)


def profile(t, dims, grid) -> dict:
    """Occupancy by height and by horizontal range from the grid's anchor column."""
    st = t.state
    st = st.cpu().numpy() if hasattr(st, "cpu") else np.asarray(st)
    occ = (st.reshape(dims) == RT.OCCUPIED)
    nx, ny, nz = dims
    by_z = occ.reshape(nx, ny, Z_BINS, nz // Z_BINS).sum((0, 1, 3))
    # range from the anchor: grid is x-forward from 0, y-left centred on -25.6
    xs = (np.arange(nx) + 0.5) * grid["size"] + grid["origin"][0]
    ys = (np.arange(ny) + 0.5) * grid["size"] + grid["origin"][1]
    r = np.sqrt(xs[:, None] ** 2 + ys[None, :] ** 2)
    col = occ.sum(2)
    by_r = [int(col[(r >= a) & (r < b)].sum()) for a, b in zip(R_BINS[:-1], R_BINS[1:])]
    return {"occupied_by_z_bin": [int(v) for v in by_z], "occupied_by_range_m": by_r,
            "z_levels_occupied": int(occ.any((0, 1)).sum())}


def agg(recs: list, extra: dict | None = None) -> dict:
    f = lambda k: np.array([r[k] for r in recs], np.float64)
    out = {"n_targets": len(recs)}
    for k in ("n_occupied", "n_free", "n_unknown", "n_points_in_grid",
              "prevalence_in_valid", "valid_fraction",
              "conflict_fraction_of_touched", "n_frames_used", "span_s", "z_levels_occupied"):
        if k in recs[0]:
            v = f(k)
            out[k] = {"median": float(np.median(v)), "mean": float(v.mean()),
                      "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
    pz = np.array([r["occupied_by_z_bin"] for r in recs], np.float64)
    pr = np.array([r["occupied_by_range_m"] for r in recs], np.float64)
    out["occupied_by_z_bin_median"] = [float(x) for x in np.median(pz, 0)]
    out["occupied_by_range_median"] = [float(x) for x in np.median(pr, 0)]
    out["frac_anchors_prevalence_in_band"] = float(
        ((f("prevalence_in_valid") >= 0.015) & (f("prevalence_in_valid") <= 0.040)).mean())
    if extra:
        out.update(extra)
    return out


def run_k360(n_anchors: int, device) -> dict:
    res = {}
    grid = {"origin": TF.GRID.origin if hasattr(TF.GRID, "origin") else (0.0, -25.6, -2.0),
            "size": 0.2}
    try:
        grid = {"origin": tuple(TF.GRID["origin"]), "size": float(TF.GRID["size"])}
    except Exception:
        grid = {"origin": (0.0, -25.6, -2.0), "size": 0.2}
    for d in KS.TRAIN_DRIVES:
        geo = TF.DriveGeometry(d, KS.SSCBENCH_ROOT, KS.KITTI360_ROOT)
        ts = geo.timestamps("velodyne")
        anchors = KS.anchors(d, REPO_ROOT)
        sel = anchors[:: max(len(anchors) // n_anchors, 1)][:n_anchors]
        recs = []
        for a in sel:
            sw = [a.native_frame] + list(a.future_natives)
            t0 = ts[sw[0]]
            sw = [i for i in sw if ts[i] - t0 <= SPAN_S]
            if len(sw) < 2:
                continue
            t = RT.build(geo, a.native_frame, sw, device, grid=TF.GRID)
            r = dict(t.stats)
            r["span_s"] = float(ts[sw[-1]] - ts[sw[0]])
            pf = profile(t, TF.DIMS, grid)
            r["z_levels_occupied"] = pf.pop("z_levels_occupied")
            r.update(pf)
            recs.append(r)
        res[d] = agg(recs)
        m = res[d]
        print(f"  K360 {d[-14:]:14s} sweeps {m['n_frames_used']['median']:.0f} "
              f"span {m['span_s']['median']:.2f}s prev {m['prevalence_in_valid']['median']:.4f} "
              f"occ {m['n_occupied']['median']:,.0f} free {m['n_free']['median']:,.0f} "
              f"unk {m['n_unknown']['median']:,.0f} z {m['z_levels_occupied']['median']:.0f} "
              f"| anchors in band {100*m['frac_anchors_prevalence_in_band']:.0f}%")
    return res


def run_once(data_root: str, n_anchors: int, device) -> dict:
    grid = {"origin": (0.0, -25.6, -2.0), "size": 0.2}
    res = {}
    for s in sequences(data_root):
        seq = OnceSequence(data_root, s)
        have = sorted(i for i in seq.T_sensor_to_world if seq.has_cloud(i))
        sel = have[:: max(len(have) // n_anchors, 1)][:n_anchors]
        recs = []
        for a in sel:
            sw = seq.window(a, span_s=SPAN_S)
            if len(sw) < 2:
                continue
            t = RT.build(seq, a, sw, device, grid=TF.GRID)
            r = dict(t.stats)
            r["span_s"] = float(seq.t_s[sw[-1]] - seq.t_s[sw[0]])
            pf = profile(t, TF.DIMS, grid)
            r["z_levels_occupied"] = pf.pop("z_levels_occupied")
            r.update(pf)
            recs.append(r)
        if not recs:
            continue
        res[s] = agg(recs)
        m = res[s]
        print(f"  ONCE {s:14s} sweeps {m['n_frames_used']['median']:.0f} "
              f"span {m['span_s']['median']:.2f}s prev {m['prevalence_in_valid']['median']:.4f} "
              f"occ {m['n_occupied']['median']:,.0f} free {m['n_free']['median']:,.0f} "
              f"unk {m['n_unknown']['median']:,.0f} z {m['z_levels_occupied']['median']:.0f} "
              f"| anchors in band {100*m['frac_anchors_prevalence_in_band']:.0f}%")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/media/minh/MINH1/once_data/extracted/data")
    ap.add_argument("--n-anchors", type=int, default=40)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--out",
                    default="artifacts/method_vs_data/results/supervision_density_compare.json")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    print(f"authorized window: {SPAN_S} s, native cadence, no interpolation\n")
    out = {"span_s": SPAN_S, "bands_unchanged": {"prevalence": [0.015, 0.040],
                                                 "conflict_max": 0.12,
                                                 "valid": [0.45, 0.75], "z_levels_min": 30},
           "note": ("At KITTI-360's frozen 0.523 s sweep spacing a 2.1 s window holds 5 "
                    "sweeps, the same count ONCE gets at 2 FPS, so the authorized policy is "
                    "sweep-matched as well as duration-matched."),
           "kitti360": run_k360(a.n_anchors, dev),
           "once": run_once(a.data_root, a.n_anchors, dev)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
