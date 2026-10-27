#!/usr/bin/env python
"""KITTI-360 beam-decimation control — measurement only, no training.

**The question.** ONCE fails the predeclared occupied-prevalence floor (0.0133 median against
0.015, `once_feasibility.md` §16) while passing every geometry gate. Before that is read as
"ONCE is unsuitable", the cause has to be established. Two candidates:

  1. something wrong in our ONCE adapter (origin, axes, windows) -- already excluded, G1-G4
     and G6 pass 6/6 and the round-trip is 3.7e-13;
  2. ONCE's LiDAR is simply sparser. Measured: ONCE ships **40 beams / ~66 k points** per
     sweep against KITTI-360's HDL-64E **64 beams / ~120 k points**.

This script tests (2) directly. It decimates KITTI-360's own sweeps from 64 rings to 40 --
ONCE's beam count -- and re-measures the same statistics through the same frozen builder. If
40-beam KITTI-360 lands on ONCE's prevalence, beam count explains the shortfall and the gate
failure is a sensor property rather than an adapter fault.

Rings are recovered by elevation quantile (the HDL-64E's beams are evenly spaced in elevation),
then 40 of every 64 are kept with an even stride. No pose, calibration, window or grid is
touched, and `rawtarget.build` runs unmodified: only `read_velodyne` is wrapped.

    PYTHONPATH=$PWD python tools/method_vs_data/k360_beam_control.py --device cuda:0
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
#: measured from 5 clouds in each of 3 sequences, 2026-10-27
MEASURED = {"kitti360_points_per_cloud": 120_055, "kitti360_rings": 64,
            "once_points_per_cloud": 66_563, "once_rings": 40,
            "once_prevalence_median": 0.0133, "once_valid_median": 0.694}


def keep_rings(n_src: int, n_keep: int) -> np.ndarray:
    """Which of ``n_src`` elevation rings survive, evenly spaced."""
    return np.unique(np.round(np.linspace(0, n_src - 1, n_keep)).astype(int))


class BeamDecimated(TF.DriveGeometry):
    """KITTI-360 geometry with each sweep thinned.

    Two independent thinning rules, because they probe different things:

    * ``n_beams`` drops whole elevation rings -- the physical difference between a 64-beam and
      a 40-beam sensor. Measured to be nearly free at 0.2 m voxels (the surviving rings hit the
      same voxels), which is why it does not explain ONCE.
    * ``keep_points`` drops a uniform random fraction of returns -- a proxy for lower angular
      resolution, which is what ONCE's 200 m range budget actually costs it inside our 51.2 m
      grid (~25 k in-grid points per sweep against KITTI-360's ~56 k).
    """

    def __init__(self, drive, sscbench_root, kitti360_root, n_beams: int, n_src: int = 64,
                 keep_points: float = 1.0, seed: int = 0):
        super().__init__(drive, sscbench_root, kitti360_root)
        self.n_beams, self.n_src = int(n_beams), int(n_src)
        self.keep_points = float(keep_points)
        self._rng = np.random.default_rng(seed)
        self._keep = set(keep_rings(self.n_src, self.n_beams).tolist())
        self.kept_fraction: list[float] = []

    def read_velodyne(self, native_frame: int) -> np.ndarray:
        pts = super().read_velodyne(native_frame)
        if self.keep_points < 1.0:
            sel = self._rng.random(len(pts)) < self.keep_points
            self.kept_fraction.append(float(sel.mean()))
            pts = pts[sel]
        if self.n_beams >= self.n_src:
            return pts
        r = np.linalg.norm(pts, axis=1)
        ok = r > 1e-6
        el = np.full(len(pts), -np.inf)
        el[ok] = np.arcsin(np.clip(pts[ok, 2] / r[ok], -1, 1))
        # rank points by elevation, cut into n_src equal-count rings
        order = np.argsort(el, kind="stable")
        ring = np.empty(len(pts), np.int64)
        ring[order] = (np.arange(len(pts)) * self.n_src) // max(len(pts), 1)
        mask = np.isin(ring, list(self._keep)) & ok
        self.kept_fraction.append(float(mask.mean()))
        return pts[mask]


def run(drive: str, n_anchors: int, n_beams: int, device, keep_points: float = 1.0) -> dict:
    geo = BeamDecimated(drive, KS.SSCBENCH_ROOT, KS.KITTI360_ROOT, n_beams,
                        keep_points=keep_points)
    anchors = KS.anchors(drive, REPO_ROOT)
    if not anchors:
        return {"drive": drive, "error": "no anchors"}
    sel = anchors[:: max(len(anchors) // n_anchors, 1)][:n_anchors]
    recs = []
    for a in sel:
        sweeps = [a.native_frame] + list(a.future_natives)
        t = RT.build(geo, a.native_frame, sweeps, device, grid=TF.GRID)
        s = dict(t.stats)
        s["z_levels_occupied"] = int(
            (t.state.reshape(TF.DIMS) == RT.OCCUPIED).any((0, 1)).sum())
        recs.append(s)
    f = lambda k: np.array([r[k] for r in recs], np.float64)
    V = int(np.prod(TF.DIMS))
    prev, valid = f("prevalence_in_valid"), f("valid_fraction")
    med = {"prevalence": float(np.median(prev)), "valid": float(np.median(valid)),
           "conflict": float(np.median(f("conflict_fraction_of_touched"))),
           "z_levels": float(np.median(f("z_levels_occupied"))),
           "occupied_voxels": float(np.median(prev * valid * V))}
    ok = {"prevalence": BANDS["prevalence"][0] <= med["prevalence"] <= BANDS["prevalence"][1],
          "valid": BANDS["valid"][0] <= med["valid"] <= BANDS["valid"][1],
          "conflict": med["conflict"] <= BANDS["conflict_max"],
          "z_levels": med["z_levels"] >= BANDS["z_levels_min"]}
    return {"drive": drive, "n_beams": n_beams, "keep_points": keep_points,
            "n_targets": len(recs), "median": med,
            "points_kept_fraction": float(np.mean(geo.kept_fraction)) if geo.kept_fraction else 1.0,
            "per_band_pass": ok, "in_bands": bool(all(ok.values()))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drives", default=None, help="default: the frozen train drives")
    ap.add_argument("--n-anchors", type=int, default=30)
    ap.add_argument("--beams", default="64,40", help="64 = unmodified; 40 = ONCE's count")
    ap.add_argument("--keep-points", default="1.0",
                    help="comma list of uniform keep fractions; 0.45 matches ONCE's in-grid "
                         "point count per sweep")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="artifacts/method_vs_data/results/k360_beam_control.json")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    drives = a.drives.split(",") if a.drives else list(KS.TRAIN_DRIVES)
    res = {"bands": BANDS, "measured_sensor_facts": MEASURED, "runs": {},
           "note": ("Only read_velodyne is wrapped; poses, calibration, the 21-sweep window "
                    "and the grid are the frozen ones. 64 beams reproduces k360_sweep_control.")}
    print(f"ONCE reference: prevalence {MEASURED['once_prevalence_median']:.4f}, "
          f"valid {MEASURED['once_valid_median']:.3f}, 40 beams, "
          f"{MEASURED['once_points_per_cloud']:,} pts/cloud")
    for kp in [float(x) for x in a.keep_points.split(",")]:
      for nb in [int(x) for x in a.beams.split(",")]:
        for d in drives:
            r = run(d, a.n_anchors, nb, dev, keep_points=kp)
            res["runs"][f"{d}|beams{nb}|keep{kp}"] = r
            if "median" in r:
                m, pb = r["median"], r["per_band_pass"]
                print(f"  {d[-14:]:14s} beams={nb:3d} keep={kp:.2f} "
                      f"(pts {r['points_kept_fraction']:.3f}): "
                      f"prev {m['prevalence']:.4f}{'' if pb['prevalence'] else ' OUT'} "
                      f"valid {m['valid']:.4f} occ_vox {m['occupied_voxels']:8,.0f} "
                      f"conflict {m['conflict']:.4f} z {m['z_levels']:.0f} | "
                      f"{'IN BANDS' if r['in_bands'] else 'OUT OF BANDS'}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
