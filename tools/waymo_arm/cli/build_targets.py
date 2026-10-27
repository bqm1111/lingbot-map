#!/usr/bin/env python
"""Raw-LiDAR occupancy supervision for Waymo anchors, under the frozen Gate 8C-1 rule.

One sweep per (frame, laser) over the anchor plus ``TARGET_FUTURE_STREAM_FRAMES`` later
stream frames, carved from each laser's own origin by
``core/supervision/rawtarget_multisensor.py`` -- which ``cli/check_equivalence.py`` gates
against the single-sensor function the KITTI-360 targets were built with.

The npz contract is the KITTI-360 one (``core/run/build_targets.save``), so
``baseline_reconstruction/current/dataset.py::HInputDataset`` reads either arm unchanged.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD python tools/waymo_arm/cli/build_targets.py \\
        --device cuda:0 --shard 0 --shards 1
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
from core.supervision import rawtarget_multisensor as MS                                # noqa: E402
from tools.waymo_arm import config as A                                                 # noqa: E402
from tools.waymo_arm import geometry as WAG                                             # noqa: E402
from tools.waymo_pilot import waymo_geometry as WG                                       # noqa: E402


def save(path: str, t, segment: str, anchor: int, frames: list) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    occ = t.occupied
    rows = np.flatnonzero(occ).astype(np.int32)
    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp, valid_packed=np.packbits(t.valid), occ_packed=np.packbits(occ),
        occ_rows=rows, occ_n_obs=t.n_obs[rows].astype(np.uint16),
        occ_n_frames=t.n_frames_occ[rows].astype(np.uint8),
        dims=np.asarray(A.GRID_DIMS, np.int32),
        stream_index=np.int32(anchor), native_frame=np.int32(anchor * A.STREAM_STRIDE),
        future_stream_indices=np.asarray(frames[1:], np.int32),
        future_natives=np.asarray([f * A.STREAM_STRIDE for f in frames[1:]], np.int32),
        frames_used=np.asarray(t.frames_used, np.int32), segment=segment,
        stats_json=json.dumps(t.stats))
    os.replace(tmp, path)
    return os.path.getsize(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--segments", type=int, default=0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    A.ensure_dirs()
    p = A.Paths()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(dev)
    grid = WG.waymo_grid()
    segs = [s for s in sorted(os.listdir(p.clips))
            if os.path.isdir(os.path.join(p.clips, s)) and os.path.isfile(p.laser_sidecar(s))]
    if a.segments:
        segs = segs[:a.segments]
    segs = segs[a.shard::a.shards]
    print(f"[targets] shard {a.shard}/{a.shards}: {len(segs)} segments with clips + sidecar; "
          f"grid origin {np.round(grid.origin, 3).tolist()}", flush=True)

    rows, t0, nb = [], time.time(), 0
    for si, seg in enumerate(segs):
        try:
            wag = WAG.WaymoAnchorGeometry(seg)
        except FileNotFoundError as exc:
            rows.append({"segment": seg, "error": str(exc)})
            continue
        ts = sorted(int(f[:-4]) for f in os.listdir(os.path.join(p.clips, seg))
                    if f.endswith(".npz"))
        for t in ts:
            out = p.target_path(seg, t)
            if os.path.exists(out):
                continue
            try:
                frames = wag.target_stream_frames(t)
                sw = wag.sweeps(t)
                tgt = MS.build(sw, grid.dims, grid.origin, grid.voxel_size, dev)
                nb += save(out, tgt, seg, t, frames)
                rows.append(dict(tgt.stats, segment=seg, anchor=t, n_sweeps=len(sw),
                                 n_frames=len(frames),
                                 laser_shares=wag.laser_shares(t)))
            except Exception as exc:                                        # noqa: BLE001
                rows.append({"segment": seg, "anchor": t,
                             "error": f"{type(exc).__name__}: {exc}"})
        done = [r for r in rows if "error" not in r]
        if (si + 1) % 10 == 0 and done:
            el = time.time() - t0
            print(f"  [{si + 1}/{len(segs)}] {len(done)} targets, "
                  f"{el / len(done):.2f}s each, {nb / 2**30:.1f} GiB", flush=True)

    ok = [r for r in rows if "error" not in r]
    keys = ("valid_fraction", "prevalence_in_valid", "conflict_fraction_of_touched",
            "n_frames_used", "n_occupied", "n_free", "n_unknown", "n_sky_free")
    rec = {"version": A.VERSION, "shard": a.shard, "shards": a.shards,
           "n_built": len(ok), "n_failed": len(rows) - len(ok),
           "minutes": (time.time() - t0) / 60, "gib": nb / 2 ** 30,
           "grid": {"dims": list(map(int, grid.dims)), "voxel_size": float(grid.voxel_size),
                    "origin": [float(x) for x in grid.origin]},
           "aggregate": {k: float(np.mean([r[k] for r in ok])) for k in keys} if ok else {},
           "laser_shares": ({k: float(np.mean([r["laser_shares"][k] for r in ok]))
                             for k in ok[0]["laser_shares"]} if ok else {}),
           "errors": [r for r in rows if "error" in r][:10]}
    tag = a.tag or f"build_targets_s{a.shard}of{a.shards}"
    path = os.path.join(p.logs, f"{tag}.json")
    json.dump(rec, open(path, "w"), indent=2, default=float)
    print(f"\n[targets] {rec['n_built']} built, {rec['n_failed']} failed, "
          f"{rec['minutes']:.1f} min, {rec['gib']:.1f} GiB")
    if ok:
        agg = rec["aggregate"]
        print(f"[targets] valid {agg['valid_fraction']:.3f} of the grid, "
              f"occupied prevalence {agg['prevalence_in_valid']:.4f}, "
              f"conflict {agg['conflict_fraction_of_touched']:.4f}, "
              f"frames {agg['n_frames_used']:.1f}")
        print(f"[targets] laser point shares {json.dumps({k: round(v, 4) for k, v in rec['laser_shares'].items()})}")
    print(f"-> {path}")
    return 0 if rec["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
