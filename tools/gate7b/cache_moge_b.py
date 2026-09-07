#!/usr/bin/env python
"""Per-frame G51-B MoGe depth for the two benchmarks whose dense cache is the A variant.

Phase-0 finding: the Gate-5.1 dense MoGe caches for SemanticKITTI and Occ3D-nuScenes
reproduce ``scales_G51-A_*.csv`` to 2.5e-5 and miss the pinned ``scales_G51-B_*.csv`` by
4 % and 18 %. They are MoGe's own inferred FOV, not the calibrated one, so they are the
wrong gauge for this gate. The Gate-5.2 KITTI-360 cache under ``moge/B`` *is* the
calibrated variant (verified to 9e-6) and is reused untouched.

This regenerates the calibrated-FOV variant **once per unique stream frame**, on the same
processed lattice the LingBot stream uses. MoGe is frozen and deterministic; the only
extra input is one scalar horizontal FOV in degrees, exactly as G51-B allows.

    python tools/gate7b/cache_moge_b.py --dataset semantickitti
"""
from __future__ import annotations

import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, load_config, write_json                  # noqa: E402
from moge_gauge.calibrated import CalibratedMoGe                                  # noqa: E402
from moge_gauge.calibration import calibrated_fov_x_deg                           # noqa: E402
from gates.gate6 import frames as G6F                                                   # noqa: E402
from gates.gate7b import depth as D7, replay, streams                                   # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate7b")


def frame_fovs(dataset: str) -> dict:
    """``frame_key -> calibrated horizontal FOV (deg)`` on the processed lattice.

    The calibrated intrinsics come from the same per-clip LingBot cache every previous
    gate used: SemanticKITTI and KITTI-360 carry one rectified ``K_processed`` per
    sequence, Occ3D carries a per-sample ``K_native`` that must be scaled to the processed
    lattice. No target, LiDAR or pose information is read.
    """
    out = {}
    for rec in G6F.read_manifest(dataset, REPO_ROOT):
        p = G6F.lingbot_cache_path(dataset, rec.clip_id, REPO_ROOT)
        if not os.path.exists(p):
            continue
        with np.load(p) as z:
            ph, pw = (int(x) for x in z["proc_hw"])
            if dataset == "occ3d":
                nh, nw = (int(x) for x in z["native_hw"])
                Kn = z["K_native"].astype(np.float64)          # [T, 3, 3]
                fx = Kn[:, 0, 0] * (pw / nw)
                fovs = [calibrated_fov_x_deg(float(f), pw) for f in fx]
            else:
                Kp = z["K_processed"].astype(np.float64)
                fovs = [calibrated_fov_x_deg(float(Kp[0, 0]), pw)] * len(rec.keys)
        for k, f in zip(rec.keys, fovs):
            prev = out.get(k)
            if prev is not None and abs(prev - f) > 1e-6:
                raise ValueError(f"{dataset}/{k}: two calibrated FOVs {prev} vs {f}")
            out[k] = f
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    choices=["semantickitti", "occ3d", "kitti360"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    ds = a.dataset
    if ds in D7.MOGE_CLIP_CACHE:
        print(f"{ds}: the Gate-5.2 calibrated-FOV cache is reused; nothing to regenerate")
        return 0
    out_dir = os.path.join(D7.MOGE_B_ROOT, ds)
    os.makedirs(out_dir, exist_ok=True)

    g5 = load_config("configs/gate5/moge_metric_gauge.yaml")
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    moge = CalibratedMoGe(dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo,
                          revision=g5.moge.hf_revision)
    prov = moge.provenance()
    assert prov["hf_revision"] == "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"

    fovs = frame_fovs(ds)
    segs = streams.build(ds, REPO_ROOT)
    frames = [f for s in segs for f in s.frames]
    if a.limit:
        frames = frames[:a.limit]
    todo = [f for f in frames
            if a.overwrite or not os.path.exists(D7.moge_frame_path(ds, f.key))]
    print(f"{ds}: {len(frames)} stream frames, {len(todo)} to compute")

    t0, done = time.time(), 0
    torch.cuda.reset_peak_memory_stats(dev)
    for f in todo:
        rgb = replay.load_images([f.path])[0].numpy()          # (3, H, W) in [0, 1]
        o = moge.infer_calibrated(rgb, fov_x_deg=float(fovs[f.key]))
        dz = np.asarray(o.depth_z, np.float32)
        mk = np.asarray(o.mask, bool)
        np.savez_compressed(D7.moge_frame_path(ds, f.key),
                            moge_depth=dz.astype(np.float16),
                            moge_mask=np.packbits(mk.reshape(-1)),
                            mask_shape=np.asarray(mk.shape, np.int32),
                            fov_x_deg=np.float32(fovs[f.key]),
                            fov_out_deg=np.float32(o.fov_x_deg))
        done += 1
        if done % 250 == 0:
            print(f"  {done}/{len(todo)} {(time.time()-t0)/done:.2f}s/frame", flush=True)

    meta = {"dataset": ds, "n_stream_frames": len(frames), "n_computed": done,
            "seconds": time.time() - t0,
            "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
            "moge_provenance": prov, "variant": "G51-B (full lattice, calibrated FOV)",
            "root": out_dir}
    write_json(os.path.join(ART, f"moge_b_cache_{ds}.json"), meta)
    print(f"{ds}: {done} frames in {meta['seconds']/60:.1f} min -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
