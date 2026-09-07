#!/usr/bin/env python
"""Gate 6 prediction phase — target-free, audited, and hashed before evaluation.

Runs the whole prediction inside :class:`gate6.audit.Gate6Audit`: the moment any code path
opens a ``.label``, ``.invalid``, ``labels.npz``, ``_1_1.npy``, a velodyne sweep, a
``voxels/`` volume or an oracle scale table, the run aborts. On success the per-clip
predictions are written to immutable files and a SHA-256 manifest is emitted. The
evaluator refuses to run against an unpinned manifest.

    python tools/gate6/predict.py --dataset kitti360
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                        # noqa: E402
from gates.gate6 import audit as g6audit, frames as F, grids, pipelines, vocab   # noqa: E402

PRED_ROOT = "/media/SSD1/MINH_DATASETS/lingbot_gate6/predictions"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(vocab.DATASETS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default=PRED_ROOT)
    ap.add_argument("--no-audit", action="store_true",
                    help="diagnostic only; the pinned run never uses it")
    a = ap.parse_args()

    ds, dev = a.dataset, torch.device(a.device)
    v = vocab.load(ds)
    out_dir = os.path.join(a.out_root, ds)
    os.makedirs(out_dir, exist_ok=True)
    G = grids.EVAL_GRID[ds]

    rows, skipped, t0 = [], {"no_scale": 0, "no_semantics": 0, "no_cache": 0}, time.time()
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(dev)

    ctx = (g6audit.Gate6Audit() if not a.no_audit else None)
    auditor = ctx.__enter__() if ctx else None
    try:
        for i, clip in enumerate(pipelines.iter_clips(ds, REPO_ROOT)):
            if a.limit and i >= a.limit:
                break
            if clip.scale is None:
                skipped["no_scale"] += 1
                continue
            sem = pipelines.load_semantics(ds, clip.frame_keys, dev)
            if sem is None:
                skipped["no_semantics"] += 1
                continue
            p = pipelines.predict_clip(ds, clip, sem, dev)
            del sem
            dst = os.path.join(out_dir, f"{p.clip_id}.npz")
            tmp = dst + ".tmp.npz"
            np.savez(tmp, raw_flat=p.raw_flat, raw_channel=p.raw_channel,
                     dil_flat=p.dil_flat, dil_channel=p.dil_channel,
                     dil_support=p.dil_support,
                     labels=np.asarray(v.labels, np.int32),
                     scale=np.float64(p.scale), n_points=np.int64(p.n_points),
                     grid=np.asarray(G.name), dims=np.asarray(G.dims, np.int32))
            os.replace(tmp, dst)
            rows.append({"clip_id": p.clip_id, "group": p.group, "scale": p.scale,
                         "n_points": p.n_points, "n_raw": int(len(p.raw_flat)),
                         "n_dil": int(len(p.dil_flat)),
                         "n_dilation_only": int((p.dil_support == 0).sum())})
            if len(rows) % 200 == 0:
                el = time.time() - t0
                print(f"[{ds}] {len(rows)} clips {el/len(rows):.2f}s/clip", flush=True)
    finally:
        if ctx:
            ctx.__exit__(None, None, None)

    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".npz"))
    per_file = {f: sha256_file(os.path.join(out_dir, f)) for f in files}
    roll = hashlib.sha256()
    for f in files:
        roll.update(f"{f}\0{per_file[f]}\0".encode())

    manifest = {
        "dataset": ds, "grid": G.name, "dims": list(G.dims),
        "n_clips": len(rows), "skipped": skipped,
        "vocabulary_labels": list(v.labels), "vocabulary_phrases": list(v.phrases),
        "frozen": {"gauge": "G51-B", "conf_threshold": pipelines.CONF_THRESHOLD,
                   "depth_range_m": [pipelines.MIN_DEPTH_M, pipelines.MAX_DEPTH_M],
                   "dilate_radius_voxels": pipelines.DILATE_RADIUS_VOXELS},
        "audit": (auditor.summary() if auditor else {"disabled": True}),
        "prediction_root": out_dir,
        "n_files": len(files), "rollup_sha256": roll.hexdigest(),
        "per_file_sha256": per_file,
        "seconds": time.time() - t0,
        "peak_gpu_gib": (torch.cuda.max_memory_allocated(dev) / 2**30
                         if dev.type == "cuda" else None),
        "targets_opened": "none (auditor active)" if auditor else "AUDIT DISABLED",
    }
    write_json(os.path.join(REPO_ROOT, "artifacts", "gate6",
                            f"prediction_manifest_{ds}.json"), manifest)
    import csv as _csv
    cp = os.path.join(REPO_ROOT, "artifacts", "gate6", f"predict_per_clip_{ds}.csv")
    with open(cp, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0])) if rows else None
        if w:
            w.writeheader()
            w.writerows(rows)
    print(json.dumps({k: manifest[k] for k in
                      ("dataset", "n_clips", "skipped", "n_files", "rollup_sha256",
                       "seconds", "peak_gpu_gib")}, indent=1))
    print("audit:", json.dumps(manifest["audit"])[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
