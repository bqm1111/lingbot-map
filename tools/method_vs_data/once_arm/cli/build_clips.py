#!/usr/bin/env python
"""Build the frozen causal clips (LingBot + MoGe-2 scale) for ONCE anchors.

The deployable half of the ONCE arm: five cam01 images ending at the anchor, LingBot replayed
on them alone with a cleared KV cache, and the causal MoGe-2 clip scale. **No LiDAR, no second
camera, no future frame and no ONCE pose enters this** -- a FileAudit enforces it, and the
clip's poses are LingBot's own predictions.

ONCE's released images are **already rectified**, so the shipped JPEGs go straight through and
``cv2.undistort`` is NOT applied -- measured 12/12 by LiDAR-depth correlation and 11/12 by
map-vs-target IoU (``config.DISTORTION_EVIDENCE``). ``--undistort`` reruns the losing arm of
that ablation.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD python \\
        tools/method_vs_data/once_arm/cli/build_clips.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from core.datasets.audit import FileAudit                                            # noqa: E402
from tools.method_vs_data.once_arm import anchors as AN, config as A, geometry as G   # noqa: E402
from tools.method_vs_data.once_reader import OnceSequence                             # noqa: E402

#: the causal half may not open a sweep or a second camera
FORBIDDEN = ("lidar_roof", "/targets/", "cam03", "cam05", "cam06", "cam07", "cam08", "cam09")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--tag", default="build_clips")
    ap.add_argument("--undistort", action="store_true",
                    help="ablation only: apply cv2.undistort first. MEASURED WORSE -- the "
                         "shipped images are already rectified (config.DISTORTION_EVIDENCE)")
    ap.add_argument("--clips-dir", default=None, help="ablation only: override the clip root")
    a = ap.parse_args()

    A.ensure_dirs()
    p = A.Paths()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    print(f"[clips] device {dev} ({torch.cuda.get_device_name(dev)}); "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}", flush=True)

    man = AN.load_or_build(p)
    rows_all = man["anchors"][a.shard::a.shards]
    if a.limit:
        rows_all = rows_all[:a.limit]
    print(f"[clips] manifest digest {man['digest'][:12]}, {len(rows_all)} anchors this shard "
          f"of {man['n_selected_total']} total", flush=True)

    from gates.gate7b import replay                                              # noqa: PLC0415
    from gates.scale_gate.config import load_config                              # noqa: PLC0415
    from moge_gauge.calibrated import CalibratedMoGe                              # noqa: PLC0415

    t_load = time.time()
    model, info = replay.build_model(dev)
    g5 = load_config("configs/gate5/moge_metric_gauge.yaml")
    moge = CalibratedMoGe(dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo,
                          revision=g5.moge.hf_revision)
    prov = moge.provenance()
    assert prov["hf_revision"] == "39c4d5e957afe587e04eec59dc2bcc3be5ecd968", \
        "MoGe-2 revision differs from the frozen gauge"
    print(f"[clips] frozen models loaded in {time.time() - t_load:.0f}s", flush=True)

    und = os.path.join(p.data_root, "undistorted")
    clip_path = (lambda sq, tt: os.path.join(a.clips_dir, sq, f"{tt:05d}.npz")) \
        if a.clips_dir else p.clip_path
    cache = {}

    def moge_fn(path: str, fov: float):
        if path not in cache:
            rgb = replay.load_images([path])[0].numpy()
            o = moge.infer_calibrated(rgb, fov_x_deg=float(fov))
            cache[path] = (np.asarray(o.depth_z, np.float32), np.asarray(o.mask, bool))
            if len(cache) > 64:
                cache.pop(next(iter(cache)))
        return cache[path]

    seqs, fovs = {}, {}
    rows, t0, n = [], time.time(), 0
    with FileAudit(patterns=FORBIDDEN, strict=True):
        for i, r in enumerate(rows_all):
            s, t = r["sequence"], int(r["anchor"])
            out = clip_path(s, t)
            if os.path.exists(out):
                rows.append({"sequence": s, "t": t, "skipped": True})
                continue
            if s not in seqs:
                seqs[s] = OnceSequence(A.ONCE_DATA_ROOT, s)
                fovs[s] = G.clip_fov_x_deg(seqs[s])
            ts = time.time()
            try:
                rec = G.build_clip(seqs[s], t, model, moge_fn, dev, fovs[s], und,
                                   undistort=a.undistort)
            except Exception as exc:                                          # noqa: BLE001
                rows.append({"sequence": s, "t": t, "error": f"{type(exc).__name__}: {exc}"})
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            tmp = out + ".tmp.npz"
            np.savez_compressed(tmp, **rec)
            os.replace(tmp, out)
            n += 1
            rows.append({"sequence": s, "t": t, "scale_m_per_unit": float(rec["scale"]),
                         "n_moge_ok": int(np.isfinite(rec["log_s_frames"]).sum()),
                         "fov_x_deg": float(rec["fov_x_deg"]),
                         "seconds": time.time() - ts, "bytes": os.path.getsize(out)})
            if n % 25 == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(rows_all)} built {n}, {el / n:.2f}s each, "
                      f"eta {(len(rows_all) - i - 1) * el / n / 60:.0f} min", flush=True)

    built = [r for r in rows if "scale_m_per_unit" in r]
    errs = [r for r in rows if "error" in r]
    sc = np.array([r["scale_m_per_unit"] for r in built]) if built else np.array([])
    rec = {"provenance": A.provenance(), "manifest_digest": man["digest"],
           "shard": a.shard, "shards": a.shards,
           "n_built": len(built), "n_skipped": sum(1 for r in rows if r.get("skipped")),
           "n_failed": len(errs), "minutes": (time.time() - t0) / 60,
           "scale_m_per_unit": ({"min": float(sc.min()), "p50": float(np.median(sc)),
                                 "max": float(sc.max()),
                                 "n_nonfinite": int((~np.isfinite(sc)).sum())}
                                if len(sc) else None),
           "fov_x_deg_per_sequence": {k: float(v) for k, v in fovs.items()},
           "errors": errs[:10]}
    os.makedirs(p.logs, exist_ok=True)
    f = os.path.join(p.logs, f"{a.tag}_shard{a.shard}.json")
    json.dump(rec, open(f, "w"), indent=2, default=float)
    print(f"\n[clips] {rec['n_built']} built, {rec['n_skipped']} skipped, "
          f"{rec['n_failed']} failed, {rec['minutes']:.1f} min -> {f}")
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
