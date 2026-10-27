#!/usr/bin/env python
"""Clip -> packed hard map ``(S_H, F_H)``: the CompletionUNet input, on ONCE.

Runs the **frozen** incremental mapper
(``baseline_reconstruction.current.geometry.build_local_geometry``) on each clip and writes
the same npz contract the KITTI-360 and Waymo arms' caches use, so
``current/dataset.py::HInputDataset`` reads any arm unchanged.

Causal: the clip and the camera calibration only. The grid is the frozen SSCBench-KITTI-360
grid; ONCE's road plane sits 1.73-1.76 m below its roof LiDAR, which is KITTI-360's velodyne
height, so the frozen z origin applies without a per-sequence LiDAR measurement and no sweep
is opened here.

    PYTHONPATH=$PWD python tools/method_vs_data/once_arm/cli/build_inputs.py --device cuda:0
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
from tools.baseline_reconstruction.current.geometry import (LocalClip,                # noqa: E402
                                                            build_local_geometry)
from tools.method_vs_data.once_arm import config as A, geometry as G                  # noqa: E402
from tools.method_vs_data.once_reader import OnceSequence                             # noqa: E402

N_VOXELS = int(np.prod(A.GRID_DIMS))


def build_one(seq: str, t: int, grid, T_cam_to_grid: np.ndarray, device, p: A.Paths) -> dict:
    clip = LocalClip.load(p.clip_path(seq, t))
    S, F, stats = build_local_geometry(clip, grid, T_cam_to_grid, device)
    out = p.input_path(seq, t)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, S_H=np.packbits(S.cpu().numpy().reshape(-1)),
                        F_H=np.packbits(F.cpu().numpy().reshape(-1)),
                        segment=seq, t=np.int64(t),
                        scale_m_per_unit=np.float64(clip.scale_m_per_unit),
                        stats_json=json.dumps(stats))
    os.replace(tmp, out)
    return {"sequence": seq, "t": t, **stats, "bytes": os.path.getsize(out)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="build_inputs")
    a = ap.parse_args()

    A.ensure_dirs()
    p = A.Paths()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    grid = A.GRID
    seqs = sorted(d for d in os.listdir(p.clips) if os.path.isdir(os.path.join(p.clips, d)))
    print(f"[inputs] {len(seqs)} sequences with clips; grid {grid.dims} @ "
          f"{grid.voxel_size} m origin {list(grid.origin)} frame {grid.frame}", flush=True)

    rows, t0, n = [], time.time(), 0
    todo = []
    Ts = {}
    for s in seqs:
        Ts[s] = G.camera_to_grid(OnceSequence(A.ONCE_DATA_ROOT, s))
        for f in sorted(os.listdir(os.path.join(p.clips, s))):
            if f.endswith(".npz"):
                todo.append((s, int(f[:-4])))
    if a.limit:
        todo = todo[:a.limit]
    for i, (s, t) in enumerate(todo):
        if os.path.exists(p.input_path(s, t)):
            continue
        try:
            rows.append(build_one(s, t, grid, Ts[s], dev, p))
            n += 1
        except Exception as exc:                                              # noqa: BLE001
            rows.append({"sequence": s, "t": t, "error": f"{type(exc).__name__}: {exc}"})
        if n and n % 100 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(todo)} built {n}, {el / n:.2f}s each, "
                  f"eta {(len(todo) - i - 1) * el / n / 60:.0f} min", flush=True)

    ok = [r for r in rows if "error" not in r]
    rec = {"provenance": A.provenance(), "n_built": len(ok),
           "n_failed": len(rows) - len(ok), "minutes": (time.time() - t0) / 60,
           "grid": {"dims": list(map(int, grid.dims)), "voxel_size": float(grid.voxel_size),
                    "origin": [float(x) for x in grid.origin], "frame": grid.frame},
           "camera_to_grid_per_sequence": {k: np.round(v, 6).tolist() for k, v in Ts.items()},
           "occupancy_of_grid": {
               "S_mean": float(np.mean([r["n_S"] for r in ok]) / N_VOXELS) if ok else None,
               "F_mean": float(np.mean([r["n_F"] for r in ok]) / N_VOXELS) if ok else None},
           "errors": [r for r in rows if "error" in r][:10]}
    os.makedirs(p.logs, exist_ok=True)
    fp = os.path.join(p.logs, f"{a.tag}.json")
    json.dump(rec, open(fp, "w"), indent=2, default=float)
    print(f"\n[inputs] {rec['n_built']} built, {rec['n_failed']} failed, "
          f"{rec['minutes']:.1f} min; S {rec['occupancy_of_grid']['S_mean']} "
          f"F {rec['occupancy_of_grid']['F_mean']} -> {fp}")
    return 0 if rec["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
