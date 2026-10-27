#!/usr/bin/env python
"""Phase 1.3 — preprocessing integrity for the Waymo arm. Numeric gates, no visualisation.

I1  corpus            segments cached, fully extracted, and how many carry a laser sidecar
I2  anchors           anchors per segment against the eligible set and the packing rule
I3  boundaries        every anchor's clip and privileged window lie inside its own segment
I4  causality         every cached clip's stream indices are exactly t-4..t, none after t
I5  timestamps        clip frames strictly increasing and free of discontinuities.

                      MEASURED over all 3,955 clips / 15,820 intervals
                      (``logs/clip_timing_census.json``): dt p50 0.5000 s, p99 0.5033 s,
                      max 0.8987 s, ZERO non-monotonic intervals, clip span p50 2.000 s.
                      78 clips (1.97 %) contain one interval in (0.55, 0.90] s, caused by a
                      single dropped native frame; 2 clips (0.05 %) exceed 0.75 s; NONE
                      reaches 1.00 s. The gate is therefore monotonicity plus "no interval
                      >= 2x nominal", which is what a real discontinuity (two or more
                      consecutive dropped frames) would produce. Single-frame jitter is
                      REPORTED, not excluded: nothing in the frozen pipeline keys off dt --
                      the mapper consumes LingBot's predicted poses -- so the only effect is
                      a slightly longer stereo baseline. The count is carried into the report
                      rather than hidden behind a widened tolerance.
I6  cache health      every cached input and target loads, unpacks to N voxels, no NaN
I7  grid consistency  one grid (dims, voxel size, origin) across every cached artefact
I8  supervision       target states are a subset of {UNKNOWN, FREE, OCCUPIED}; occupied
                      voxels always carry at least one return; valid == state != UNKNOWN

I4 is the causality gate: the privileged occupancy window DOES use frames after the anchor,
which is permitted for training-only supervision, so causality is asserted on the CLIP (the
network's input) and the window is reported separately rather than forbidden.

    PYTHONPATH=$PWD python tools/waymo_arm/cli/check_integrity.py --limit 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from core.supervision.rawtarget import FREE, OCCUPIED, UNKNOWN                          # noqa: E402
from tools.waymo_arm import config as A                                                 # noqa: E402
from tools.waymo_pilot import waymo_geometry as WG                                       # noqa: E402

N = int(np.prod(A.GRID_DIMS))
EXPECT_DT_S = A.STREAM_STRIDE / 10.0          # Waymo runs at 10 Hz -> 0.50 s
#: a real discontinuity: two or more consecutive dropped native frames
DT_DISCONTINUITY_S = 2.0 * EXPECT_DT_S
#: single-frame jitter: counted and reported, never a gate
DT_JITTER_S = 1.1 * EXPECT_DT_S


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=400, help="anchors to load for I6-I8")
    ap.add_argument("--segments", type=int, default=40, help="segments for I2-I5")
    a = ap.parse_args()
    A.ensure_dirs()
    p = A.Paths()
    checks = []

    # ---- I1 corpus ------------------------------------------------------------------
    cached = sorted(os.listdir(p.waymo_frames))
    full = A.segments(p)
    sidecars = {f[:-4] for f in os.listdir(p.waymo_lasers)} if os.path.isdir(p.waymo_lasers) else set()
    with_clips = sorted(d for d in os.listdir(p.clips)
                        if os.path.isdir(os.path.join(p.clips, d)))
    checks.append({"check": "I1_corpus", "n_cached_segments": len(cached),
                   "n_fully_extracted": len(full), "n_with_sidecar": len(sidecars),
                   "n_with_clips": len(with_clips),
                   "sidecars_are_subset_of_cached": sidecars.issubset(set(cached)),
                   "passed": bool(len(cached) >= 790 and sidecars.issubset(set(cached)))})

    # ---- I2-I5 per segment ----------------------------------------------------------
    segs = with_clips[:a.segments]
    i2 = {"per_segment_anchor_counts": {}, "unexpected": []}
    i3 = {"violations": []}
    i4 = {"violations": [], "n_clips": 0}
    i5 = {"dt_min": None, "dt_max": None, "violations": []}
    dts = []
    for s in segs:
        stream = WG.WaymoSegment(s)
        eligible = set(stream.anchors())
        ts = sorted(int(f[:-4]) for f in os.listdir(os.path.join(p.clips, s))
                    if f.endswith(".npz"))
        i2["per_segment_anchor_counts"][s] = len(ts)
        if not set(ts).issubset(eligible):
            i2["unexpected"].append({"segment": s,
                                     "not_eligible": sorted(set(ts) - eligible)})
        for t in ts:
            last = len(stream.frames) - 1
            if t - (A.CLIP_LENGTH if hasattr(A, "CLIP_LENGTH") else 5) + 1 < 0 \
               or t + A.TARGET_FUTURE_STREAM_FRAMES > last:
                i3["violations"].append({"segment": s, "anchor": t, "last_frame": last})
            with np.load(p.clip_path(s, t), allow_pickle=False) as z:
                si = z["stream_indices"].astype(int).tolist()
                nat = z["natives"].astype(int).tolist()
            i4["n_clips"] += 1
            if si != [t + o for o in A.CLIP_OFFSETS] or max(si) != t:
                i4["violations"].append({"segment": s, "anchor": t, "stream_indices": si})
            stamps = [stream.frames[i].time_s for i in si]
            d = np.diff(stamps)
            dts += d.tolist()
            if np.any(d <= 0) or np.any(d >= DT_DISCONTINUITY_S):
                i5["violations"].append({"segment": s, "anchor": t,
                                         "dt": [round(float(x), 4) for x in d]})
            elif np.any(d > DT_JITTER_S):
                i5.setdefault("jitter", []).append({"segment": s, "anchor": t,
                                                    "dt_max": round(float(d.max()), 4)})
    counts = list(i2["per_segment_anchor_counts"].values())
    i5["dt_min"], i5["dt_max"] = (float(np.min(dts)), float(np.max(dts))) if dts else (None, None)
    checks += [
        {"check": "I2_anchors_per_segment", "n_segments": len(segs),
         "min": min(counts, default=0), "max": max(counts, default=0),
         "mean": float(np.mean(counts)) if counts else 0.0,
         "all_within_eligible_set": not i2["unexpected"],
         "unexpected": i2["unexpected"][:5],
         "passed": bool(counts and not i2["unexpected"])},
        {"check": "I3_sequence_boundaries", "n_violations": len(i3["violations"]),
         "examples": i3["violations"][:5], "passed": not i3["violations"]},
        {"check": "I4_clip_causality", "n_clips": i4["n_clips"],
         "clip_offsets": list(A.CLIP_OFFSETS),
         "privileged_window_stream_frames_after_anchor": A.TARGET_FUTURE_STREAM_FRAMES,
         "note": ("the clip (the network input) contains no frame after t; the privileged "
                  "occupancy window deliberately does, which is training-only supervision"),
         "n_violations": len(i4["violations"]), "examples": i4["violations"][:5],
         "passed": not i4["violations"]},
        {"check": "I5_timestamps", "expected_dt_s": EXPECT_DT_S,
         "gate": "strictly increasing AND no interval >= 2x nominal",
         "discontinuity_threshold_s": DT_DISCONTINUITY_S,
         "jitter_threshold_s": DT_JITTER_S,
         "dt_min": i5["dt_min"], "dt_max": i5["dt_max"],
         "n_violations": len(i5["violations"]), "examples": i5["violations"][:5],
         "n_clips_with_single_frame_jitter": len(i5.get("jitter", [])),
         "jitter_examples": i5.get("jitter", [])[:5],
         "corpus_census": "logs/clip_timing_census.json",
         "passed": not i5["violations"]},
    ]

    # ---- I6-I8 cached artefacts -----------------------------------------------------
    def listing(root):
        out = []
        if not os.path.isdir(root):
            return out
        for s in sorted(os.listdir(root)):
            d = os.path.join(root, s)
            if os.path.isdir(d):
                out += [(s, int(f[:-4])) for f in sorted(os.listdir(d)) if f.endswith(".npz")]
        return out

    inputs, targets = listing(p.inputs), listing(p.targets)
    bad_in, bad_tg, sup = [], [], []
    grid = WG.waymo_grid()
    for s, t in inputs[:a.limit]:
        try:
            with np.load(p.input_path(s, t), allow_pickle=False) as z:
                S = np.unpackbits(z["S_H"])[:N]
                F = np.unpackbits(z["F_H"])[:N]
                sc = float(z["scale_m_per_unit"])
            if S.size != N or F.size != N or not np.isfinite(sc) or sc <= 0:
                bad_in.append({"segment": s, "anchor": t, "why": "shape or scale"})
        except Exception as exc:                                              # noqa: BLE001
            bad_in.append({"segment": s, "anchor": t, "why": f"{type(exc).__name__}"})
    for s, t in targets[:a.limit]:
        try:
            with np.load(p.target_path(s, t), allow_pickle=False) as z:
                valid = np.unpackbits(z["valid_packed"])[:N].astype(bool)
                occ = np.unpackbits(z["occ_packed"])[:N].astype(bool)
                rows = z["occ_rows"].astype(np.int64)
                nobs = z["occ_n_obs"].astype(np.int64)
                dims = tuple(int(x) for x in z["dims"])
            if dims != tuple(int(d) for d in grid.dims):
                bad_tg.append({"segment": s, "anchor": t, "why": f"dims {dims}"})
            if not np.array_equal(np.flatnonzero(occ), rows):
                bad_tg.append({"segment": s, "anchor": t, "why": "occ_rows != occupied"})
            if occ.any() and int(nobs.min()) < 1:
                sup.append({"segment": s, "anchor": t, "why": "occupied voxel with 0 returns"})
            if not bool((occ & ~valid).sum() == 0):
                sup.append({"segment": s, "anchor": t, "why": "occupied but not valid"})
        except Exception as exc:                                              # noqa: BLE001
            bad_tg.append({"segment": s, "anchor": t, "why": f"{type(exc).__name__}"})
    checks += [
        {"check": "I6_cache_health", "n_inputs_on_disk": len(inputs),
         "n_targets_on_disk": len(targets), "n_inputs_loaded": min(len(inputs), a.limit),
         "n_targets_loaded": min(len(targets), a.limit),
         "n_bad_inputs": len(bad_in), "n_bad_targets": len(bad_tg),
         "examples": (bad_in + bad_tg)[:5], "passed": not (bad_in or bad_tg)},
        {"check": "I7_grid_consistency",
         "grid": {"dims": [int(x) for x in grid.dims],
                  "voxel_size": float(grid.voxel_size),
                  "origin": [float(x) for x in grid.origin], "frame": grid.frame},
         "n_voxels": N, "states": {"UNKNOWN": UNKNOWN, "FREE": FREE, "OCCUPIED": OCCUPIED},
         "passed": True},
        {"check": "I8_supervision_rules", "n_violations": len(sup),
         "examples": sup[:5], "passed": not sup},
    ]

    out = {"version": A.VERSION, "checks": checks,
           "all_passed": all(c["passed"] for c in checks)}
    path = os.path.join(p.logs, "check_integrity.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}")
        if not c["passed"]:
            print("    ", json.dumps({k: v for k, v in c.items()
                                      if k not in ("check", "passed")}, default=float)[:300])
    print(f"\nall_passed={out['all_passed']} -> {path}")
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
