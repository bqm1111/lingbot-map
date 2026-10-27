#!/usr/bin/env python
"""ONCE audit from the real `train_infos.tar` metadata — brief step 2.

Verifies, on actual ONCE bytes rather than documentation: frame timestamps and cadence,
camera intrinsics and the resulting HFOV, sensor calibration, ego poses, and sequence /
frame counts. Nothing here touches point clouds; that is the separate origin probe.

    python tools/method_vs_data/once_infos_audit.py --data-root <extracted>/data
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

CAMERAS = ("cam01", "cam03", "cam05", "cam06", "cam07", "cam08", "cam09")
#: ONCE images are 1920 x 1020 (width x height); the devkit and OccAny's reader agree
IMG_W, IMG_H = 1920, 1020


def quat_to_R(q) -> np.ndarray:
    """ONCE pose quaternion order is (qx, qy, qz, qw) -- scipy's `from_quat` convention."""
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]],
                    np.float64)


def pose_to_T(p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_R(p[:4])
    T[:3, 3] = [float(v) for v in p[4:]]
    return T


def audit(data_root: str) -> dict:
    seqs = sorted(d for d in os.listdir(data_root)
                  if os.path.isdir(os.path.join(data_root, d)))
    out = {"data_root": data_root, "n_sequences": len(seqs), "sequences": {},
           "cameras": list(CAMERAS), "image_size": [IMG_W, IMG_H]}
    all_dt, all_step, tot_frames, tot_annot = [], [], 0, 0
    calib_ref = None
    for s in seqs:
        j = json.load(open(os.path.join(data_root, s, f"{s}.json")))
        frames = j["frames"]
        ts = np.array([float(f["frame_id"]) for f in frames], np.float64)
        dt = np.diff(ts) / 1000.0                      # frame_id is a ms timestamp
        has_pose = [("pose" in f and f["pose"] is not None) for f in frames]
        has_anno = [("annos" in f) for f in frames]
        T = [pose_to_T(f["pose"]) for f in frames if f.get("pose")]
        tr = np.stack([t[:3, 3] for t in T]) if T else np.zeros((0, 3))
        step = np.linalg.norm(np.diff(tr, axis=0), axis=1) if len(tr) > 1 else np.zeros(0)

        cal = {}
        for c in CAMERAS:
            ci = j["calib"][c]
            K = np.asarray(ci["cam_intrinsic"], np.float64)
            c2v = np.asarray(ci["cam_to_velo"], np.float64)
            dist = np.asarray(ci["distortion"], np.float64).ravel()
            cal[c] = {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                      "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                      "hfov_deg": float(np.degrees(2 * np.arctan(IMG_W / (2 * K[0, 0])))),
                      "vfov_deg": float(np.degrees(2 * np.arctan(IMG_H / (2 * K[1, 1])))),
                      "cam_to_velo_translation": [round(float(v), 4) for v in c2v[:3, 3]],
                      "distortion": [round(float(v), 6) for v in dist],
                      "distortion_all_zero": bool(np.allclose(dist, 0.0))}
        if calib_ref is None:
            calib_ref = cal
            out["calibration_first_sequence"] = cal
        else:
            d = max(abs(cal[c][k] - calib_ref[c][k])
                    for c in CAMERAS for k in ("fx", "fy", "cx", "cy"))
            cal = {"max_abs_intrinsic_diff_vs_first_sequence": d}

        out["sequences"][s] = {
            "n_frames": len(frames),
            "n_with_pose": int(sum(has_pose)),
            "n_with_annos": int(sum(has_anno)),
            "timestamp_span_s": float((ts[-1] - ts[0]) / 1000.0),
            "dt_s": {"min": float(dt.min()), "p50": float(np.median(dt)),
                     "max": float(dt.max())} if len(dt) else None,
            "monotonic": bool(np.all(dt > 0)) if len(dt) else True,
            "path_length_m": float(step.sum()),
            "step_m": {"p50": float(np.median(step)), "max": float(np.max(step))} if len(step) else None,
            "calib": cal}
        all_dt.append(dt); all_step.append(step)
        tot_frames += len(frames); tot_annot += int(sum(has_anno))

    dt = np.concatenate(all_dt); step = np.concatenate(all_step)
    out["corpus"] = {
        "n_sequences": len(seqs), "n_frames": tot_frames, "n_frames_with_annos": tot_annot,
        "dt_s": {"min": float(dt.min()), "p01": float(np.percentile(dt, 1)),
                 "p50": float(np.median(dt)), "p99": float(np.percentile(dt, 99)),
                 "max": float(dt.max())},
        "implied_fps": float(1.0 / np.median(dt)),
        "all_dt_monotonic": bool(np.all(dt > 0)),
        "frac_dt_within_5pct_of_0p5s": float(np.mean(np.abs(dt - 0.5) <= 0.025)),
        "step_m": {"p50": float(np.median(step)), "p99": float(np.percentile(step, 99))},
        "implied_speed_mps_p50": float(np.median(step) / np.median(dt))}
    # how many 5-frame causal clips and ~2.1 s supervision windows the corpus supports
    n5 = sum(max(v["n_frames"] - 4, 0) for v in out["sequences"].values())
    out["capacity"] = {"five_frame_causal_clips": n5,
                       "independent_sequences": len(seqs),
                       "note": "clips within one sequence are not independent"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="artifacts/method_vs_data/results/once_infos_audit.json")
    a = ap.parse_args()
    r = audit(a.data_root)
    c = r["corpus"]
    print(f"=== ONCE train_infos: {c['n_sequences']} sequences, {c['n_frames']:,} frames, "
          f"{c['n_frames_with_annos']:,} annotated")
    print(f"  dt: min {c['dt_s']['min']:.3f}  p50 {c['dt_s']['p50']:.3f}  "
          f"max {c['dt_s']['max']:.3f} s  -> implied {c['implied_fps']:.2f} FPS")
    print(f"  dt monotonic: {c['all_dt_monotonic']}   within 5% of 0.5 s: "
          f"{c['frac_dt_within_5pct_of_0p5s']*100:.1f}%")
    print(f"  ego step p50 {c['step_m']['p50']:.3f} m, p99 {c['step_m']['p99']:.3f} m "
          f"-> ~{c['implied_speed_mps_p50']:.1f} m/s")
    print(f"  capacity: {r['capacity']['five_frame_causal_clips']:,} five-frame clips over "
          f"{r['capacity']['independent_sequences']} independent sequences")
    print("  per sequence:")
    for s, v in r["sequences"].items():
        print(f"    {s}: {v['n_frames']:>4} frames, pose {v['n_with_pose']:>4}, "
              f"annos {v['n_with_annos']:>4}, span {v['timestamp_span_s']:7.1f} s, "
              f"dt p50 {v['dt_s']['p50']:.3f}, path {v['path_length_m']:7.1f} m")
    print("  cameras (first sequence):")
    for cam, v in r["calibration_first_sequence"].items():
        print(f"    {cam}: fx {v['fx']:8.2f} HFOV {v['hfov_deg']:5.1f}deg VFOV {v['vfov_deg']:5.1f}deg "
              f"cam_to_velo t {v['cam_to_velo_translation']} distortion_all_zero={v['distortion_all_zero']}")
    diffs = [v["calib"].get("max_abs_intrinsic_diff_vs_first_sequence")
             for v in r["sequences"].values() if "max_abs_intrinsic_diff_vs_first_sequence" in v["calib"]]
    if diffs:
        print(f"  calibration varies across sequences: max abs intrinsic diff {max(diffs):.3f} px")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(r, open(a.out, "w"), indent=1, default=float)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
