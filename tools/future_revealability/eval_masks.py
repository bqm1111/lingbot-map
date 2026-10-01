"""Ground-truth semantics on the anchor grid, for the three training drives.

The frozen development masks exist for 0006 only (`tools/kitti360_data_scaling_7drives/build_eval_masks.py`,
590 anchors). Gate 20's primary population lives on the *training* drives, so the same three arrays this
gate consumes -- ``valid``, ``occupied`` and ``sem_gt`` -- are built here for 0003/0007/0010 with that
builder's logic unchanged, including its frame conversion:

    label index = floor((voxel centre - t_cam0_to_velo - origin) / 0.2)

``verify()`` rebuilds a sample of 0006 and requires the three arrays to match the frozen masks bit-exactly;
nothing in this gate runs until that passes.

GT is used for ANALYSIS ONLY. Nothing here enters target construction or any prediction path.

    PYTHONPATH=$PWD python -m tools.future_revealability.eval_masks --verify
    PYTHONPATH=$PWD python -m tools.future_revealability.eval_masks --drive 2013_05_28_drive_0007_sync
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from core.datasets import kitti360_splits as SRC, kitti360_transforms as TF
from core.run.build_targets import load as load_target
from tools.delayed_semantic import config as C
from tools.future_revealability import config as G
from tools.semantic_completion.evidence import build as G15B

#: The frozen builder reads SSCBench human labels from here, and its targets from the seven-drive cache
#: (``drives7.apply`` sets ``G8C1_DATA_ROOT`` to it). Both are reproduced exactly.
LABELS = "/media/SSD1/MINH_DATASETS/sscbench_kitti360/preprocess/labels"
TARGETS = "/media/SSD1/MINH_DATASETS/lingbot_k360_7drives/targets"
FROZEN_MASKS = "/media/SSD1/MINH_DATASETS/lingbot_k360_7drives/eval_masks"
DIMS = tuple(int(d) for d in C.GRID_DIMS)
ORIGIN = np.asarray(C.GRID_ORIGIN_M, np.float64)
VS = float(C.VOXEL_SIZE_M)


def _target_path(drive: str, t: int) -> str:
    return os.path.join(TARGETS, drive, f"{t:05d}.npz")


def build_anchor(drive: str, t: int, geo: TF.DriveGeometry, t_c2v: np.ndarray) -> Dict[str, np.ndarray]:
    """``valid``, ``occupied``, ``sem_gt`` for one anchor. 0 = empty, 1..18 = class, 255 = ignore."""
    tgt = load_target(_target_path(drive, t))
    valid, occ = tgt["valid"], tgt["occupied"]
    assert not (occ & ~valid).any(), "occupied must be a subset of valid"
    n = int(np.prod(DIMS))
    sem = np.full(n, 255, np.uint8)
    sem[valid & ~occ] = 0
    k = np.flatnonzero(geo.pose_frames[1:] == int(tgt["native_frame"]))
    lp = os.path.join(LABELS, drive, f"{int(k[0]):06d}_1_1.npy") if len(k) else ""
    has_label = os.path.exists(lp)
    if has_label:
        L = np.load(lp).astype(np.int32)
        oi = np.flatnonzero(valid & occ)
        ctr = (np.stack(np.unravel_index(oi, DIMS), 1) + 0.5) * VS + ORIGIN
        li = np.floor((ctr - t_c2v - ORIGIN) / VS).astype(int)
        ins = (li >= 0).all(1) & (li < np.asarray(DIMS)).all(1)
        lab = np.zeros(len(oi), np.int32)
        lab[ins] = L[tuple(li[ins].T)]
        good = (lab >= 1) & (lab <= 18)
        sem[oi[good]] = lab[good].astype(np.uint8)
    return {"valid": np.packbits(valid), "occupied": np.packbits(occ), "sem_gt": sem,
            "has_label": np.bool_(has_label), "native_frame": np.int32(tgt["native_frame"]),
            "stream_index": np.int32(t)}


def load_mask(drive: str, t: int) -> Dict[str, np.ndarray]:
    """``sem_gt`` [N] uint8 and ``valid`` [N] bool, from the frozen 0006 masks or this gate's rebuild."""
    p = os.path.join(FROZEN_MASKS, drive, f"{t:05d}.npz") if drive == G.DEV_DRIVE else G.mask_path(drive, t)
    with np.load(p) as z:
        n = int(np.prod(DIMS))
        return {"sem_gt": z["sem_gt"].astype(np.int64),
                "valid": np.unpackbits(z["valid"])[:n].astype(bool),
                "has_label": bool(z["has_label"])}


def verify(n_anchors: int = 24) -> Dict:
    """Rebuild a stride sample of 0006 and require bit-exact agreement with the frozen masks."""
    drive = G.DEV_DRIVE
    geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    t_c2v = np.asarray(geo.calib.cam0_to_velo, np.float64)[:3, 3]
    ids = G15B.anchor_ids(drive, C.Paths().g15)
    sel = ids[:: max(1, len(ids) // n_anchors)][:n_anchors]
    rows = []
    for t in sel:
        got = build_anchor(drive, int(t), geo, t_c2v)
        with np.load(os.path.join(FROZEN_MASKS, drive, f"{int(t):05d}.npz")) as z:
            rows.append({"t": int(t),
                         "sem_gt": bool(np.array_equal(got["sem_gt"], z["sem_gt"])),
                         "valid": bool(np.array_equal(got["valid"], z["valid"])),
                         "occupied": bool(np.array_equal(got["occupied"], z["occupied"])),
                         "n_labelled_occ": int(((got["sem_gt"] >= 1) & (got["sem_gt"] <= 18)).sum())})
    ok = all(r["sem_gt"] and r["valid"] and r["occupied"] for r in rows)
    return {"drive": drive, "n_checked": len(rows), "bit_exact": ok, "rows": rows}


def build_drive(drive: str, shard: int = 0, shards: int = 1, limit: Optional[int] = None) -> Dict:
    assert drive in G.TRAIN_DRIVES, f"0006 masks are frozen; {drive} is not a training drive"
    geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    t_c2v = np.asarray(geo.calib.cam0_to_velo, np.float64)[:3, 3]
    ids = G15B.anchor_ids(drive, C.Paths().g15)[shard::shards][:limit]
    rows: List[Dict] = []
    t0 = time.time()
    for k, t in enumerate(ids):
        out = G.mask_path(drive, int(t))
        if os.path.exists(out):
            continue
        a = build_anchor(drive, int(t), geo, t_c2v)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp.npz"
        np.savez_compressed(tmp, **a)
        os.replace(tmp, out)
        rows.append({"t": int(t), "has_label": bool(a["has_label"]),
                     "n_labelled_occ": int(((a["sem_gt"] >= 1) & (a["sem_gt"] <= 18)).sum())})
        if (k + 1) % 100 == 0:
            print(f"  [masks {drive[17:21]} s{shard}] {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.2f}s each", flush=True)
    return {"drive": drive, "shard": shard, "n_anchors": len(ids), "n_built": len(rows),
            "n_with_label": sum(r["has_label"] for r in rows), "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive"); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    G.ensure_dirs()
    if a.verify:
        r = verify()
        json.dump(r, open(os.path.join(G.ARTIFACT_DIR, "logs", "eval_mask_verification.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=1))
        assert r["bit_exact"], "rebuilt 0006 masks differ from the frozen ones"
        return
    drives = [a.drive] if a.drive else list(G.TRAIN_DRIVES)
    for d in drives:
        print(json.dumps(build_drive(d, a.shard, a.shards, a.limit), indent=1))


if __name__ == "__main__":
    main()
