#!/usr/bin/env python
"""Gate 8C-1: rebuild every KITTI-360 occupancy target from raw Velodyne.

Reads only KITTI-360 raw sweeps, KITTI-360 ground-truth poses and the frozen LingBot
stream cache (for the anchor list). The SSCBench ``_1_1.npy`` completion labels are never
opened -- ``gate8c1.sources.assert_no_target_access`` fails the run if any forbidden path
is reached.

    python tools/gate8c1/build_targets.py --drive 2013_05_28_drive_0003_sync --device cuda:1
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402
from gates.gate8c0 import transforms as TF                                             # noqa: E402
from gates.gate8c1 import rawtarget as RT, sources as SRC                              # noqa: E402


def save(path: str, t: RT.RawTarget, anc: SRC.Anchor) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    occ = t.occupied
    rows = np.flatnonzero(occ).astype(np.int32)
    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        valid_packed=np.packbits(t.valid), occ_packed=np.packbits(occ),
        occ_rows=rows, occ_n_obs=t.n_obs[rows].astype(np.uint16),
        occ_n_frames=t.n_frames_occ[rows].astype(np.uint8),
        dims=np.asarray(TF.DIMS, np.int32),
        stream_index=np.int32(anc.stream_index),
        native_frame=np.int32(anc.native_frame),
        future_stream_indices=np.asarray(anc.future_stream_indices, np.int32),
        future_natives=np.asarray(anc.future_natives, np.int32),
        frames_used=np.asarray(t.frames_used, np.int32))
    os.replace(tmp, path)
    return os.path.getsize(path)


def load(path: str):
    """``valid`` / ``occupied`` boolean volumes plus the per-voxel support counts."""
    with np.load(path, allow_pickle=False) as z:
        n = int(np.prod(z["dims"]))
        # counts are stored only for occupied voxels; callers want them dense
        rows = z["occ_rows"].astype(np.int64)
        n_obs = np.zeros(n, np.uint16); n_obs[rows] = z["occ_n_obs"]
        n_fr = np.zeros(n, np.uint8); n_fr[rows] = z["occ_n_frames"]
        return {"valid": np.unpackbits(z["valid_packed"])[:n].astype(bool),
                "occupied": np.unpackbits(z["occ_packed"])[:n].astype(bool),
                "occ_rows": rows, "occ_n_obs": z["occ_n_obs"],
                "occ_n_frames": z["occ_n_frames"],
                "n_obs": n_obs, "n_frames_occ": n_fr,
                "dims": tuple(int(x) for x in z["dims"]),
                "native_frame": int(z["native_frame"]),
                "stream_index": int(z["stream_index"]),
                "future_natives": z["future_natives"],
                "future_stream_indices": z["future_stream_indices"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True, help="full KITTI-360 drive name")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--stride", type=int, default=1, help="anchor decimation")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    drive = a.drive
    assert drive in list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE], f"unknown drive {drive!r}"
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    all_anc = SRC.anchors(drive, REPO_ROOT)
    ancs = all_anc[::a.stride][a.shard::a.shards]
    rows, t0, nb, n_skip = [], time.time(), 0, 0
    for anc in ancs:
        p = SRC.target_path(drive, anc.stream_index)
        if os.path.exists(p):
            n_skip += 1
            continue
        SRC.assert_no_target_access([p, geo.velodyne_path(anc.native_frame)])
        tgt = RT.build(geo, anc.native_frame,
                       [anc.native_frame] + list(anc.future_natives), dev)
        nb += save(p, tgt, anc)
        rows.append(dict(tgt.stats, stream_index=anc.stream_index,
                         native_frame=anc.native_frame, n_future=anc.n_future))
        if len(rows) % 100 == 0:
            print(f"  [{drive[-9:-5]} s{a.shard}] {len(rows)}/{len(ancs)} "
                  f"{(time.time()-t0)/len(rows):.2f}s each", flush=True)
    os.makedirs(ART, exist_ok=True)
    agg = ({k: float(np.mean([r[k] for r in rows])) for k in
            ("valid_fraction", "prevalence_in_valid", "conflict_fraction_of_touched",
             "n_frames_used", "n_occupied", "n_free")} if rows else {})
    write_json(os.path.join(ART, f"targets_{drive[-9:-5]}_s{a.shard}.json"),
               {"drive": drive, "partition": SRC.partition_of(drive),
                "n_anchors_total": len(all_anc), "n_selected": len(ancs),
                "n_built": len(rows), "n_already_present": n_skip, "stride": a.stride,
                "bytes": nb, "carve_decimation": RT.CARVE_DECIMATION,
                "band_half_m": RT.BAND_HALF_M, "carve_near_m": RT.CARVE_NEAR_M,
                "future_frames": SRC.FUTURE_FRAMES, "seconds": time.time() - t0,
                "aggregate": agg, "rows": rows[:200]})
    print(f"[{drive[-9:-5]} s{a.shard}] {len(rows)} built (+{n_skip} present) in "
          f"{(time.time()-t0)/60:.1f} min, {nb/1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
