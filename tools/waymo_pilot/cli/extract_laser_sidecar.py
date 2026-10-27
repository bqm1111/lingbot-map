#!/usr/bin/env python
"""Recover the per-point laser identity that the frame cache lost, as a small sidecar.

``waymo.read_segment`` concatenates ``convert_range_image_to_point_cloud``'s five per-laser
arrays into one ``points_vehicle``, so the cache cannot say which sensor measured a point.
Ray carving needs that: ``core/supervision/rawtarget.build`` marks ray interiors free from
ONE sensor origin, and Waymo's TOP laser sits ~2.1 m above the vehicle origin while the four
short-range lasers sit elsewhere. At the measured median planar range of 8.4 m, using the
wrong origin misdirects a ray by ~15 deg, which carves free space through real surfaces.

The SDK orders that list deterministically -- ``sorted(frame.context.laser_calibrations,
key=lambda c: c.name)``, verified in the installed 1.6.7 source -- so the per-laser point
*counts* plus the laser calibrations are sufficient to label every cached point after the
fact. This writes exactly that, one small npz per segment, and leaves the 599 GiB frame
cache untouched.

The raw ``.tfrecord`` is a transient here as in ``fetch_corpus.py``: downloaded, parsed,
deleted. Every segment is verified against the cache before it is accepted -- the per-frame
counts must sum to the cached point count, or the segment is rejected rather than stored.

    /media/SSD1/MINH_DATASETS/envs/waymo/bin/python \\
        tools/waymo_pilot/cli/extract_laser_sidecar.py --segments 1 --workers 1
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.waymo_pilot import config as C                                     # noqa: E402

BUCKET = "gs://waymo_open_dataset_v_1_4_2/individual_files/training"
GSUTIL = os.path.join(C.GCLOUD, "bin", "gsutil")
#: one npz per segment, beside the frame cache
SIDECAR_DIR = os.path.join(C.DATA_ROOT, "waymo_lasers")
LASER_NAME = {1: "TOP", 2: "FRONT", 3: "SIDE_LEFT", 4: "SIDE_RIGHT", 5: "REAR"}


def sidecar_path(segment: str) -> str:
    return os.path.join(SIDECAR_DIR, f"{segment}.npz")


def cached_point_counts(segment: str) -> dict:
    """{frame_index: n_points} from the existing frame cache, without loading the arrays."""
    out = {}
    for p in sorted(glob.glob(os.path.join(C.Paths().waymo_frames, segment, "*.npz"))):
        with np.load(p) as z:
            out[int(z["frame_index"])] = int(z["points_vehicle"].shape[0])
    return out


def read_laser_counts(path: str):
    """Yield ``(frame_index, names, counts, laser_calibrations)`` for one tfrecord."""
    import tensorflow.compat.v1 as tf                                          # noqa: PLC0415
    from waymo_open_dataset import dataset_pb2                                 # noqa: PLC0415
    from waymo_open_dataset.utils import frame_utils                           # noqa: PLC0415

    tf.enable_eager_execution()
    for i, data in enumerate(tf.data.TFRecordDataset(path, compression_type="")):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytes(data.numpy()))
        cals = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        names = [LASER_NAME.get(int(c.name), f"LASER_{int(c.name)}") for c in cals]
        parsed = frame_utils.parse_range_image_and_camera_projection(frame)
        range_images, camera_projections, _seg, top_pose = parsed
        if range_images and top_pose is not None and len(top_pose.data):
            points, _cp = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, top_pose)
            counts = [int(p.shape[0]) for p in points]
        else:
            counts = [0] * len(cals)
        calib = {LASER_NAME.get(int(c.name), f"LASER_{int(c.name)}"): {
            "name": int(c.name),
            "extrinsic": list(c.extrinsic.transform),          # laser -> VEHICLE, row major
            "beam_inclination_min": float(c.beam_inclination_min),
            "beam_inclination_max": float(c.beam_inclination_max),
            "beam_inclinations": list(c.beam_inclinations)} for c in cals}
        yield i, names, counts, calib


def _one(task) -> dict:
    segment, staging = task
    t0 = time.time()
    if os.path.isfile(sidecar_path(segment)):
        return {"segment": segment, "status": "skipped", "seconds": 0.0}
    cached = cached_point_counts(segment)
    if not cached:
        return {"segment": segment, "status": "no_frame_cache"}
    raw = os.path.join(staging, segment)
    try:
        if not os.path.isfile(raw):
            r = subprocess.run([GSUTIL, "-q", "cp", f"{BUCKET}/{segment}", raw],
                               capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                return {"segment": segment, "status": "download_failed",
                        "error": (r.stderr or "")[-300:]}
        dl = time.time() - t0
        frames, names_ref, calib_ref, rows, mismatch = [], None, None, [], []
        for i, names, counts, calib in read_laser_counts(raw):
            if names_ref is None:
                names_ref, calib_ref = names, calib
            elif names != names_ref:
                return {"segment": segment, "status": "laser_set_changed", "frame": i}
            if i in cached and sum(counts) != cached[i]:
                mismatch.append({"frame": i, "sidecar": sum(counts), "cache": cached[i]})
            frames.append(i)
            rows.append(counts)
        if mismatch:
            return {"segment": segment, "status": "count_mismatch",
                    "n_mismatch": len(mismatch), "examples": mismatch[:3]}
        os.makedirs(SIDECAR_DIR, exist_ok=True)
        tmp = sidecar_path(segment) + ".tmp.npz"
        np.savez_compressed(tmp, segment=segment, frame_index=np.asarray(frames, np.int32),
                            laser_names=np.asarray(names_ref), counts=np.asarray(rows, np.int64),
                            laser_calibration_json=json.dumps(calib_ref),
                            n_frames_verified=np.int64(len(set(frames) & set(cached))))
        os.replace(tmp, sidecar_path(segment))
        return {"segment": segment, "status": "ok", "n_frames": len(frames),
                "n_verified": len(set(frames) & set(cached)), "lasers": names_ref,
                "download_s": dl, "seconds": time.time() - t0,
                "bytes": os.path.getsize(sidecar_path(segment))}
    except Exception as exc:                                                   # noqa: BLE001
        return {"segment": segment, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if os.path.isfile(raw):
            os.remove(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", type=int, default=0, help="0 = every cached segment")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--staging", default=C.WAYMO_ROOT)
    ap.add_argument("--tag", default="laser_sidecar")
    a = ap.parse_args()

    os.makedirs(a.staging, exist_ok=True)
    os.makedirs(SIDECAR_DIR, exist_ok=True)
    segs = sorted(os.listdir(C.Paths().waymo_frames))
    want = segs[:a.segments] if a.segments else segs
    have = sum(1 for s in want if os.path.isfile(sidecar_path(s)))
    print(f"[laser] {len(segs)} cached segments; taking {len(want)}; {have} sidecars present",
          flush=True)

    t0, out = time.time(), []
    with mp.Pool(a.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_one, [(s, a.staging) for s in want]), 1):
            out.append(rec)
            if rec["status"] not in ("ok", "skipped"):
                print(f"  [{i}/{len(want)}] {rec['status']}: {rec['segment'][:34]} "
                      f"{rec.get('error', rec.get('examples', ''))}", flush=True)
            elif i % 10 == 0 or i == 1:
                el = (time.time() - t0) / 60
                ok = sum(1 for r in out if r["status"] == "ok")
                print(f"  [{i}/{len(want)}] ok={ok} {el:.1f} min "
                      f"({el / max(i, 1) * len(want):.0f} min projected)", flush=True)

    ok = [r for r in out if r["status"] == "ok"]
    rec = {"version": C.VERSION, "requested": len(want), "workers": a.workers,
           "n_ok": len(ok), "n_skipped": sum(1 for r in out if r["status"] == "skipped"),
           "n_failed": sum(1 for r in out if r["status"] not in ("ok", "skipped")),
           "n_frames_verified": sum(r.get("n_verified", 0) for r in ok),
           "minutes": (time.time() - t0) / 60,
           "mib": sum(r.get("bytes", 0) for r in ok) / 2 ** 20, "segments": out}
    path = os.path.join(C.Paths().logs, f"{a.tag}.json")
    os.makedirs(C.Paths().logs, exist_ok=True)
    json.dump(rec, open(path, "w"), indent=2, default=float)
    print(f"\n[laser] {rec['n_ok']} ok (+{rec['n_skipped']} present), {rec['n_failed']} failed, "
          f"{rec['n_frames_verified']} frames verified against the cache, "
          f"{rec['mib']:.1f} MiB, {rec['minutes']:.1f} min -> {path}")
    return 0 if rec["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
