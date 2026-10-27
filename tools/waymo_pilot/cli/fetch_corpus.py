#!/usr/bin/env python
"""Fetch a Waymo training corpus: download, extract, delete the raw file, repeat.

The only writable large volume on this machine is SSD1, so the raw ``.tfrecord`` is a
transient: each segment is downloaded, converted into the per-frame cache, and the 1 GiB
source deleted before the next one starts. Steady-state cost is therefore the derived cache
alone (~0.62 GiB per segment measured), not raw + derived.

Resumable: a segment whose frames are already on disk is skipped without downloading.
Bounded: the run stops when free space would fall below ``--min-free-gib``, reports the
coverage it actually reached, and never silently fills the volume.

    /media/SSD1/MINH_DATASETS/envs/waymo/bin/python tools/waymo_pilot/cli/fetch_corpus.py \\
        --segments 100 --workers 8
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.waymo_pilot import config as C                                     # noqa: E402

BUCKET = "gs://waymo_open_dataset_v_1_4_2/individual_files/training"
GSUTIL = os.path.join(C.GCLOUD, "bin", "gsutil")


def free_gib(path: str) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2 ** 30

def already_done(seg: str, expect_min: int) -> int:
    d = os.path.join(C.Paths().waymo_frames, seg)
    return len(glob.glob(os.path.join(d, "*.npz"))) if os.path.isdir(d) else 0


def _one(task) -> dict:
    """Download one segment, extract it, delete the raw file."""
    seg, staging, stride, min_free = task
    from tools.waymo_pilot.cli.waymo_extract import extract_segment
    t0 = time.time()
    have = already_done(seg, 1)
    if have >= 150:                       # a full Waymo segment is ~199 frames
        return {"segment": seg, "status": "skipped", "n_frames": have, "seconds": 0.0}
    if free_gib(staging) < min_free:
        return {"segment": seg, "status": "stopped_low_disk",
                "free_gib": free_gib(staging)}
    raw = os.path.join(staging, seg)
    try:
        if not os.path.isfile(raw):
            r = subprocess.run([GSUTIL, "-q", "cp", f"{BUCKET}/{seg}", raw],
                               capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                return {"segment": seg, "status": "download_failed",
                        "error": (r.stderr or "")[-300:]}
        dl = time.time() - t0
        rec = extract_segment(raw, seg, stride=stride)
        return {"segment": seg, "status": "ok", "n_frames": rec["n_frames"],
                "gib": rec["bytes"] / 2 ** 30, "download_s": dl,
                "seconds": time.time() - t0}
    except Exception as exc:                                        # noqa: BLE001
        return {"segment": seg, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if os.path.isfile(raw):
            os.remove(raw)                # the raw file is a transient, never kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min-free-gib", type=float, default=40.0)
    ap.add_argument("--staging", default=C.WAYMO_ROOT)
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()

    C.ensure_dirs()
    os.makedirs(a.staging, exist_ok=True)
    listing = subprocess.run([GSUTIL, "ls", BUCKET + "/"], capture_output=True, text=True,
                             timeout=600)
    if listing.returncode != 0:
        print(f"cannot list the bucket: {listing.stderr[-400:]}", file=sys.stderr)
        return 2
    segs = sorted(os.path.basename(x) for x in listing.stdout.split() if x.endswith(".tfrecord"))
    want = segs[:a.segments]
    print(f"[corpus] {len(segs)} segments in the split; taking the first {len(want)} "
          f"(deterministic, sorted)")
    done = sum(1 for s in want if already_done(s, 1) >= 150)
    print(f"[corpus] already extracted: {done}; free on SSD1: {free_gib(a.staging):.0f} GiB; "
          f"floor {a.min_free_gib:.0f} GiB")
    if a.list_only:
        return 0

    t0 = time.time()
    tasks = [(s, a.staging, a.stride, a.min_free_gib) for s in want]
    out = []
    with mp.Pool(a.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_one, tasks), 1):
            out.append(rec)
            if rec["status"] not in ("ok", "skipped"):
                print(f"  [{i}/{len(tasks)}] {rec['status']}: {rec['segment'][:28]} "
                      f"{rec.get('error', rec.get('free_gib', ''))}", flush=True)
            elif i % 5 == 0 or rec["status"] == "ok":
                el = time.time() - t0
                ok = sum(1 for r in out if r["status"] == "ok")
                print(f"  [{i}/{len(tasks)}] ok={ok} "
                      f"{el / 60:.1f} min elapsed, free {free_gib(a.staging):.0f} GiB",
                      flush=True)

    shutil.rmtree(a.staging, ignore_errors=True)
    ok = [r for r in out if r["status"] == "ok"]
    rec = {"version": C.VERSION, "requested": a.segments, "workers": a.workers,
           "n_ok": len(ok), "n_skipped": sum(1 for r in out if r["status"] == "skipped"),
           "n_failed": sum(1 for r in out if r["status"] not in ("ok", "skipped")),
           "n_frames": sum(r.get("n_frames", 0) for r in out),
           "gib": sum(r.get("gib", 0.0) for r in ok),
           "minutes": (time.time() - t0) / 60,
           "free_gib_after": free_gib(C.DATA_ROOT),
           "segments": out}
    path = os.path.join(C.Paths().logs, "fetch_corpus.json")
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2, default=float)
    print(f"\n[corpus] {rec['n_ok']} extracted (+{rec['n_skipped']} already present), "
          f"{rec['n_failed']} failed, {rec['n_frames']} frames, {rec['gib']:.1f} GiB, "
          f"{rec['minutes']:.1f} min -> {path}")
    return 0 if rec["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
