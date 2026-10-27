#!/usr/bin/env python
"""Clip -> packed hard map ``(S_H, F_H)``: the CompletionUNet input, on Waymo.

The clips written by ``waymo_pilot/cli/build_waymo_clips.py`` hold MoGe depth, LingBot poses
and the causal clip scale, not the voxel volumes the network eats. This runs the *frozen*
incremental mapper (``baseline_reconstruction.current.geometry.build_local_geometry``) on
each clip and writes the same npz contract the KITTI-360 arm's input cache uses, so
``current/dataset.py::HInputDataset`` reads either arm unchanged.

Causal: the clip and the camera calibration only. The grid's z offset comes from the config
constant, never from a per-segment LiDAR measurement, so no sweep is opened here.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD python tools/waymo_arm/cli/build_inputs.py \\
        --device cuda:0 --segments 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.baseline_reconstruction.current.geometry import (LocalClip,                 # noqa: E402
                                                            build_local_geometry)
from tools.waymo_arm import config as A                                                # noqa: E402
from tools.waymo_pilot import waymo_geometry as WG                                      # noqa: E402

N_VOXELS = int(np.prod(A.GRID_DIMS))


def build_one(segment: str, t: int, grid, T_cam_to_grid: np.ndarray, device,
              p: A.Paths) -> dict:
    clip = LocalClip.load(p.clip_path(segment, t))
    S, F, stats = build_local_geometry(clip, grid, T_cam_to_grid, device)
    out = p.input_path(segment, t)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, S_H=np.packbits(S.cpu().numpy().reshape(-1)),
                        F_H=np.packbits(F.cpu().numpy().reshape(-1)),
                        segment=segment, t=np.int64(t),
                        scale_m_per_unit=np.float64(clip.scale_m_per_unit),
                        stats_json=json.dumps(stats))
    os.replace(tmp, out)
    return {"segment": segment, "t": t, **stats, "bytes": os.path.getsize(out)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--segments", type=int, default=0, help="0 = every segment with clips")
    ap.add_argument("--skip-segments", type=int, default=0)
    ap.add_argument("--tag", default="build_inputs")
    a = ap.parse_args()

    A.ensure_dirs()
    p = A.Paths()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    segs = sorted(d for d in os.listdir(p.clips) if os.path.isdir(os.path.join(p.clips, d)))
    segs = segs[a.skip_segments:a.skip_segments + a.segments] if a.segments else segs[a.skip_segments:]
    grid = WG.waymo_grid()                       # config road-z constant: no LiDAR read
    print(f"[inputs] {len(segs)} segments; grid {grid.dims} @ {grid.voxel_size} m "
          f"origin {np.round(grid.origin, 3).tolist()} frame {grid.frame}", flush=True)

    rows, t0, n_done = [], time.time(), 0
    for si, seg in enumerate(segs):
        ts = sorted(int(f[:-4]) for f in os.listdir(os.path.join(p.clips, seg))
                    if f.endswith(".npz"))
        T = WG.camera_to_grid(WG.WaymoSegment(seg).calibration())
        for t in ts:
            if os.path.exists(p.input_path(seg, t)):
                continue
            try:
                rows.append(build_one(seg, t, grid, T, dev, p))
                n_done += 1
            except Exception as exc:                                       # noqa: BLE001
                rows.append({"segment": seg, "t": t,
                             "error": f"{type(exc).__name__}: {exc}"})
        if (si + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [{si + 1}/{len(segs)}] {n_done} inputs, "
                  f"{el / max(n_done, 1):.2f}s each", flush=True)

    ok = [r for r in rows if "error" not in r]
    rec = {"version": A.VERSION, "n_built": len(ok),
           "n_failed": len(rows) - len(ok), "minutes": (time.time() - t0) / 60,
           "grid": {"dims": list(map(int, grid.dims)), "voxel_size": float(grid.voxel_size),
                    "origin": [float(x) for x in grid.origin], "frame": grid.frame},
           "occupancy_of_grid": {
               "S_mean": float(np.mean([r["n_S"] for r in ok]) / N_VOXELS) if ok else None,
               "F_mean": float(np.mean([r["n_F"] for r in ok]) / N_VOXELS) if ok else None},
           "bytes_per_input": float(np.mean([r["bytes"] for r in ok])) if ok else None,
           "errors": [r for r in rows if "error" in r][:10], "rows": ok[:50]}
    os.makedirs(p.logs, exist_ok=True)
    path = os.path.join(p.logs, f"{a.tag}.json")
    json.dump(rec, open(path, "w"), indent=2, default=float)
    print(f"\n[inputs] {rec['n_built']} built, {rec['n_failed']} failed, "
          f"{rec['minutes']:.1f} min; S {rec['occupancy_of_grid']['S_mean']:.4f} "
          f"F {rec['occupancy_of_grid']['F_mean']:.4f} of the grid -> {path}")
    return 0 if rec["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
