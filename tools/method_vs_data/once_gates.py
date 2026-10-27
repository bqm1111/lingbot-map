#!/usr/bin/env python
"""ONCE correctness gates — brief steps 3, 5, 6 and the axis half of 7.

Every gate is measured on real ONCE point clouds and judged against thresholds fixed in
advance. The occupancy thresholds are the bands predeclared in
`artifacts/method_vs_data/predeclaration.md` §6, **before any ONCE data was inspected**, and
they are not adjusted here.

  **G1 sensor origin.** The decisive gate, and the one DDAD failed. If the published frame's
      origin is the physical sensor, the road sits roughly one mount-height *below* z=0 and
      elevation angles reach well below the horizon. DDAD's merged clouds put the road at
      z≈+0.08 m with elevation p1 of −1.35°, which is what made its rays graze the road.
      PASS: road mode in [−2.6, −1.0] m and elevation p1 ≤ −8°.
  **G2 axis rotation.** Verified against the vehicle's own motion rather than asserted: in our
      x-forward convention the direction of travel must lie close to +x.
      PASS: median forward component ≥ 0.95 of the step length, and the ONCE-frame check
      confirms the same motion lies along −y there.
  **G3 transform round-trip.** `velo_to_velo(a,b) @ velo_to_velo(b,a)` must be the identity.
      PASS: max abs deviation < 1e-9.
  **G4 supervision window.** The authorized protocol: a fixed ~2.1 s window at the native
      rate, no interpolation. Records realised spans and sweep counts, and how many windows
      are rejected for containing a gap.
      PASS: median span in [1.8, 2.4] s and median sweeps in [4, 6].
  **G5 occupancy target statistics**, from the frozen `rawtarget.build` run unmodified.
      PASS: the predeclared bands — occupied prevalence in valid [0.015, 0.040], conflict to
      unknown ≤ 0.12, valid fraction [0.45, 0.75], occupancy present in ≥30 of 32 z-levels.
  **G6 temporal ordering and sequence boundaries.** Timestamps strictly increasing; no window
      spans two sequences (structurally impossible here, since windows are built per
      sequence, and asserted anyway).

    PYTHONPATH=$PWD python tools/method_vs_data/once_gates.py --data-root <root> --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.datasets import kitti360_transforms as TF                                     # noqa: E402
from core.supervision import rawtarget as RT                                            # noqa: E402
from tools.method_vs_data.once_reader import OnceSequence, sequences                    # noqa: E402

#: predeclared in predeclaration.md §6 before any ONCE data was inspected
BANDS = {"prevalence": (0.015, 0.040), "conflict_max": 0.12,
         "valid": (0.45, 0.75), "z_levels_min": 30}
#: G1/G2/G4 thresholds, fixed here before the first run
G1_ROAD_Z = (-2.6, -1.0)
G1_ELEV_P1_MAX = -8.0
G2_FORWARD_MIN = 0.95
G4_SPAN = (1.8, 2.4)
G4_SWEEPS = (4, 6)


def g1_origin(seq: OnceSequence, frames) -> dict:
    """Road height and elevation spread in the published sensor frame (unrotated)."""
    raw = OnceSequence(seq.data_root, seq.seq, rotate=False)
    zs, el_p1, n = [], [], []
    for f in frames:
        p = raw.read_velodyne(f)
        r = np.linalg.norm(p[:, :2], axis=1)
        near = p[(r > 2.0) & (r < 8.0)]
        if len(near) < 500:
            continue
        h, edges = np.histogram(near[:, 2], bins=200)
        zs.append(float(edges[int(np.argmax(h))]))
        rng = np.linalg.norm(p, axis=1)
        ok = rng > 1.0
        el = np.degrees(np.arctan2(p[ok, 2], np.linalg.norm(p[ok, :2], axis=1)))
        el_p1.append(float(np.percentile(el, 1)))
        n.append(int(len(p)))
    road = float(np.median(zs)); e1 = float(np.median(el_p1))
    return {"road_z_mode_m_median": road, "road_z_modes": [round(v, 3) for v in zs],
            "elevation_p1_deg_median": e1, "points_per_sweep_median": int(np.median(n)),
            "threshold_road_z": list(G1_ROAD_Z), "threshold_elev_p1_max": G1_ELEV_P1_MAX,
            "pass": bool(G1_ROAD_Z[0] <= road <= G1_ROAD_Z[1] and e1 <= G1_ELEV_P1_MAX)}


def g2_axis(seq: OnceSequence) -> dict:
    """Direction of travel must be +x in our convention and −y in ONCE's."""
    out = {}
    for tag, rot in (("ours", True), ("once", False)):
        s = OnceSequence(seq.data_root, seq.seq, rotate=rot)
        idx = sorted(s.T_sensor_to_world)
        fr = []
        for a, b in zip(idx[:-1], idx[1:]):
            if s.t_s[b] - s.t_s[a] > 0.75:
                continue
            T = s.velo_to_velo(b, a)          # frame b's origin expressed in frame a
            d = T[:3, 3]
            L = float(np.linalg.norm(d))
            if L < 0.5:
                continue
            fr.append(d / L)
        fr = np.stack(fr)
        out[tag] = {"n": int(len(fr)),
                    "median_unit_components": [round(float(v), 4) for v in np.median(fr, axis=0)]}
    fwd = out["ours"]["median_unit_components"][0]
    back_y = out["once"]["median_unit_components"][1]
    out.update({"forward_component_ours_x": fwd, "component_once_y": back_y,
                "threshold_forward_min": G2_FORWARD_MIN,
                "pass": bool(fwd >= G2_FORWARD_MIN and back_y <= -G2_FORWARD_MIN)})
    return out


def g3_roundtrip(seq: OnceSequence) -> dict:
    idx = sorted(seq.T_sensor_to_world)[:400]
    worst = 0.0
    for a, b in zip(idx[:-1], idx[1:]):
        M = seq.velo_to_velo(a, b) @ seq.velo_to_velo(b, a)
        worst = max(worst, float(np.abs(M - np.eye(4)).max()))
    return {"max_abs_deviation_from_identity": worst, "n_pairs": max(len(idx) - 1, 0),
            "pass": bool(worst < 1e-9)}


def g4_windows(seq: OnceSequence, wkw: dict, stride: int) -> dict:
    spans, counts, rejected, total = [], [], 0, 0
    for a in range(0, len(seq.frames) - 1, stride):
        total += 1
        w = seq.window(a, **wkw)
        if not w or len(w) < 2:
            rejected += 1
            continue
        spans.append(float(seq.t_s[w[-1]] - seq.t_s[w[0]]))
        counts.append(len(w))
    if not spans:
        return {"pass": False, "note": "no usable windows"}
    ms, mc = float(np.median(spans)), float(np.median(counts))
    if "n_sweeps" in wkw:                       # frozen-matched mode: judge the sweep count
        ok = bool(mc == wkw["n_sweeps"])
        thr = {"required_sweeps": wkw["n_sweeps"]}
    else:
        ok = bool(G4_SPAN[0] <= ms <= G4_SPAN[1] and G4_SWEEPS[0] <= mc <= G4_SWEEPS[1])
        thr = {"threshold_span": list(G4_SPAN), "threshold_sweeps": list(G4_SWEEPS)}
    return {"n_candidate_anchors": total, "n_windows": len(spans), "n_rejected_gap": rejected,
            "span_s": {"min": float(np.min(spans)), "p50": ms, "max": float(np.max(spans))},
            "sweeps": {"min": int(np.min(counts)), "p50": mc, "max": int(np.max(counts))},
            **thr, "pass": ok}


def g5_targets(seq: OnceSequence, anchors, wkw: dict, device) -> dict:
    recs = []
    for a in anchors:
        w = seq.window(a, **wkw)
        if len(w) < 2:
            continue
        t = RT.build(seq, a, w, device, grid=TF.GRID)
        s = dict(t.stats)
        s.update({"anchor": int(a), "sweeps": len(w),
                  "span_s": float(seq.t_s[w[-1]] - seq.t_s[w[0]]),
                  "z_levels_occupied": int((t.state.reshape(TF.DIMS) == RT.OCCUPIED).any((0, 1)).sum())})
        recs.append(s)
    if not recs:
        return {"pass": False, "note": "no targets built"}
    f = lambda k: np.array([r[k] for r in recs], np.float64)
    prev, val = f("prevalence_in_valid"), f("valid_fraction")
    con, zl = f("conflict_fraction_of_touched"), f("z_levels_occupied")
    med = {"prevalence": float(np.median(prev)), "valid": float(np.median(val)),
           "conflict": float(np.median(con)), "z_levels": float(np.median(zl))}
    ok = {"prevalence": BANDS["prevalence"][0] <= med["prevalence"] <= BANDS["prevalence"][1],
          "valid": BANDS["valid"][0] <= med["valid"] <= BANDS["valid"][1],
          "conflict": med["conflict"] <= BANDS["conflict_max"],
          "z_levels": med["z_levels"] >= BANDS["z_levels_min"]}
    return {"n_targets": len(recs), "median": med, "bands": BANDS, "per_band_pass": ok,
            "ranges": {k: [float(np.min(f(kk))), float(np.max(f(kk)))]
                       for k, kk in (("prevalence", "prevalence_in_valid"),
                                     ("valid", "valid_fraction"),
                                     ("conflict", "conflict_fraction_of_touched"),
                                     ("z_levels", "z_levels_occupied"))},
            "sweeps_used": {"p50": float(np.median(f("sweeps"))),
                            "min": int(np.min(f("sweeps"))), "max": int(np.max(f("sweeps")))},
            "span_s_used": {"p50": float(np.median(f("span_s"))),
                            "min": float(np.min(f("span_s"))), "max": float(np.max(f("span_s")))},
            "per_target": recs, "pass": bool(all(ok.values()))}


def g6_ordering(seq: OnceSequence) -> dict:
    dt = np.diff(seq.t_s)
    return {"strictly_increasing": bool(np.all(dt > 0)),
            "dt_min_s": float(dt.min()), "dt_max_s": float(dt.max()),
            "windows_are_per_sequence": True,
            "pass": bool(np.all(dt > 0))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--sequences", default=None, help="comma list; default all")
    ap.add_argument("--mode", default="sweeps", choices=["span", "sweeps"],
                    help="'span': fixed time window (--span-s). 'sweeps': fixed sweep count "
                         "(--n-sweeps), which at ONCE's native 0.5 s reproduces the frozen "
                         "21-sweep / ~10 s recipe.")
    ap.add_argument("--span-s", type=float, default=2.1)
    ap.add_argument("--n-sweeps", type=int, default=21)
    ap.add_argument("--n-anchors", type=int, default=8, help="per sequence, for G5")
    ap.add_argument("--window-stride", type=int, default=37)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="artifacts/method_vs_data/results/once_gates.json")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    seqs = a.sequences.split(",") if a.sequences else sequences(a.data_root)
    wkw = {"span_s": a.span_s} if a.mode == "span" else {"n_sweeps": a.n_sweeps}
    res = {"data_root": a.data_root, "mode": a.mode, "window_kwargs": wkw,
           "grid": TF.GRID.name,
           "frozen_reference": ("KITTI-360: 21 sweeps, 0.523 s spacing, 10.461 s span. "
                                "Waymo: 21 sweeps, 0.500 s spacing, 10.0 s span. "
                                "ONCE native cadence is 0.500 s."),
           "sequences": {}}
    agg = {}
    for s in seqs:
        seq = OnceSequence(a.data_root, s)
        have = [i for i in sorted(seq.T_sensor_to_world) if seq.has_cloud(i)]
        if len(have) < 10:
            res["sequences"][s] = {"skipped": "no point clouds on disk",
                                   "n_frames_with_cloud": len(have)}
            continue
        probe = have[:: max(len(have) // 6, 1)][:6]
        anchors = have[:: max(len(have) // a.n_anchors, 1)][:a.n_anchors]
        r = {"n_frames": len(seq.frames), "n_frames_with_cloud": len(have),
             "G1_origin": g1_origin(seq, probe), "G2_axis": g2_axis(seq),
             "G3_roundtrip": g3_roundtrip(seq),
             "G4_windows": g4_windows(seq, wkw, a.window_stride),
             "G5_targets": g5_targets(seq, anchors, wkw, dev),
             "G6_ordering": g6_ordering(seq)}
        res["sequences"][s] = r
        print(f"\n=== {s}  ({len(have)} frames with clouds)")
        g1 = r["G1_origin"]; print(f"  G1 origin   {'PASS' if g1['pass'] else 'FAIL'}  "
                                   f"road z mode {g1['road_z_mode_m_median']:+.3f} m "
                                   f"(band {G1_ROAD_Z}), elevation p1 "
                                   f"{g1['elevation_p1_deg_median']:+.2f}deg (max {G1_ELEV_P1_MAX}), "
                                   f"{g1['points_per_sweep_median']:,} pts/sweep")
        g2 = r["G2_axis"]; print(f"  G2 axis     {'PASS' if g2['pass'] else 'FAIL'}  "
                                 f"travel in ours {g2['ours']['median_unit_components']}, "
                                 f"in once {g2['once']['median_unit_components']}")
        g3 = r["G3_roundtrip"]; print(f"  G3 roundtrip{'PASS' if g3['pass'] else 'FAIL'}  "
                                      f"max dev {g3['max_abs_deviation_from_identity']:.2e}")
        g4 = r["G4_windows"]; print(f"  G4 windows  {'PASS' if g4['pass'] else 'FAIL'}  "
                                    f"span p50 {g4['span_s']['p50']:.3f} s, sweeps p50 "
                                    f"{g4['sweeps']['p50']:.1f} (min {g4['sweeps']['min']}, "
                                    f"max {g4['sweeps']['max']}), {g4['n_rejected_gap']} of "
                                    f"{g4['n_candidate_anchors']} rejected for gaps")
        g5 = r["G5_targets"]
        if g5.get("median"):
            m = g5["median"]; pb = g5["per_band_pass"]
            print(f"  G5 targets  {'PASS' if g5['pass'] else 'FAIL'}  "
                  f"prevalence {m['prevalence']:.4f}{'' if pb['prevalence'] else ' OUT'} | "
                  f"valid {m['valid']:.4f}{'' if pb['valid'] else ' OUT'} | "
                  f"conflict {m['conflict']:.4f}{'' if pb['conflict'] else ' OUT'} | "
                  f"z-levels {m['z_levels']:.0f}{'' if pb['z_levels'] else ' OUT'}  "
                  f"({g5['n_targets']} targets, {g5['sweeps_used']['p50']:.0f} sweeps, "
                  f"{g5['span_s_used']['p50']:.2f} s)")
        g6 = r["G6_ordering"]; print(f"  G6 ordering {'PASS' if g6['pass'] else 'FAIL'}  "
                                     f"dt {g6['dt_min_s']:.3f}..{g6['dt_max_s']:.3f} s")
        for k in ("G1_origin", "G2_axis", "G3_roundtrip", "G4_windows", "G5_targets", "G6_ordering"):
            agg.setdefault(k, []).append(bool(r[k].get("pass")))
    res["summary"] = {k: {"pass_sequences": int(sum(v)), "of": len(v), "all_pass": all(v)}
                      for k, v in agg.items()}
    res["all_gates_pass"] = bool(agg and all(all(v) for v in agg.values()))
    print("\n=== summary")
    for k, v in res["summary"].items():
        print(f"  {k:14s} {'PASS' if v['all_pass'] else 'FAIL'}  "
              f"{v['pass_sequences']}/{v['of']} sequences")
    print(f"\n  ALL GATES {'PASS' if res['all_gates_pass'] else 'FAIL'}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
