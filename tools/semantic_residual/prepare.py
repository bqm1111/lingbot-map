"""B5 query cache: everything the residual head needs at an anchor, computed once.

Training needs, per query voxel: the frozen trunk feature, the hard-map/occupancy channels, and which evidence voxel
B0 would copy from. All of that is a deterministic function of the causal state at t, so it is computed once here
rather than re-running the frozen CompletionUNet on every training step.

The query set is defined **causally and before any target is read**:

    query = predicted occupied (frozen occupancy O)  AND  not an evidence voxel

Future data then selects which of those voxels carry a loss, and supplies the target. It never enters the query
definition, the context, or the B0 assignment. ``target_row`` is an index into the anchor's B5 target file, so the
teacher feature is stored exactly once. B0's own feature IS copied in, because training reads it every step and the
evidence file it comes from is ~40 MB per anchor.

    PYTHONPATH=$PWD python -m tools.semantic_residual.prepare --drive 2013_05_28_drive_0003_sync --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import numpy as np
import torch
from scipy import ndimage

from tools.delayed_semantic import config as C
from tools.semantic_completion.evidence import build as G15B
from tools.semantic_completion.frozen_geometry import geometry_input, load_frozen_geometry, load_hard_map, occupancy_and_trunk
from tools.semantic_residual import config as B5

DIMS = tuple(int(d) for d in C.GRID_DIMS)
VOXEL = float(C.VOXEL_SIZE_M)
ORIGIN = np.asarray(C.GRID_ORIGIN_M, np.float64)


def query_cache_path(drive: str, t: int) -> str:
    return os.path.join(B5.DATA_ROOT, "queries", drive, f"{t:05d}.npz")


def nearest_evidence(evidence_flat: np.ndarray) -> tuple:
    """B0's rule on the source grid: 3D Euclidean distance transform, scipy index order.

    Mirrors ``tools/zero_shot_eval/prediction/semantics.py::nearest_evidence`` exactly (same scipy call, same ravel
    order); only the grid constants are the source grid's."""
    mask = np.zeros(DIMS, bool)
    mask.reshape(-1)[evidence_flat] = True
    dist, idx = ndimage.distance_transform_edt(~mask, return_indices=True)
    flat = np.ravel_multi_index((idx[0], idx[1], idx[2]), DIMS).reshape(-1)
    return flat.astype(np.int64), (dist.reshape(-1) * VOXEL).astype(np.float32)


def centres(flat: np.ndarray) -> np.ndarray:
    X, Y, Z = DIMS
    k = flat.astype(np.int64)
    i = k // (Y * Z); j = (k // Z) % Y; l = k % Z
    return (np.stack([i, j, l], 1) + 0.5) * VOXEL + ORIGIN


@torch.no_grad()
def prepare_anchor(net, drive: str, t: int, device) -> Dict:
    g15 = C.Paths().g15
    with np.load(g15.evidence_path(drive, t)) as z:
        ev_flat = z["flat_index"].astype(np.int64)
        ev_count = z["count"].astype(np.float32)
        ev_feature = z["feature"]            # [E, 1024] fp16, unit in the raw teacher space
    with np.load(B5.target_path(drive, t)) as z:
        tgt_flat = z["flat_index"].astype(np.int64)
        tgt_hist_seen = z["hist_seen"]; tgt_tier = z["tier"]; tgt_dt = z["first_dt_s"].astype(np.float32)

    S, F_ = load_hard_map(drive, t, device)
    occ, trunk = occupancy_and_trunk(net, geometry_input(S, F_))
    O = occ[0].cpu().numpy()                                   # [N] bool, frozen predicted occupancy
    trunk = trunk[0].reshape(24, -1)                           # [24, N]

    is_ev = np.zeros(len(O), bool); is_ev[ev_flat] = True
    # the causal query set -- no future information has been read at this point
    is_query = O & ~is_ev

    # loss locations: query voxels that a later frame revealed
    keep = is_query[tgt_flat]
    q_flat = tgt_flat[keep]
    target_row = np.nonzero(keep)[0].astype(np.int32)

    nn_flat, dist = nearest_evidence(ev_flat)
    row_of = np.full(len(O), -1, np.int64); row_of[ev_flat] = np.arange(len(ev_flat))
    nn_row = row_of[nn_flat[q_flat]]
    assert (nn_row >= 0).all() or len(ev_flat) == 0, "nearest evidence must resolve to an evidence row"

    delta = centres(ev_flat[nn_row]) - centres(q_flat) if len(ev_flat) else np.zeros((len(q_flat), 3), np.float64)
    Sn, Fn = S.cpu().numpy(), F_.cpu().numpy()
    arrays = {
        "flat_index": q_flat.astype(np.int32),
        "target_row": target_row,
        "trunk": trunk[:, q_flat].T.cpu().numpy().astype(np.float16),
        "geometry": np.stack([Sn[q_flat], Fn[q_flat], O[q_flat]], 1).astype(np.uint8),
        "nn_row": nn_row.astype(np.int32),
        "delta_m": delta.astype(np.float16),
        "distance_m": dist[q_flat].astype(np.float16),
        "nn_count": ev_count[nn_row].astype(np.float32) if len(ev_flat) else np.zeros(len(q_flat), np.float32),
        # B0's feature, stored here so training never has to read the 40 MB evidence file per step
        "b0_feature": (ev_feature[nn_row] if len(ev_flat) else np.zeros((len(q_flat), 1024), np.float16)),
        "hist_seen": tgt_hist_seen[keep],
        "tier": tgt_tier[keep],
        "first_dt_s": tgt_dt[keep].astype(np.float16),
        "n_evidence": np.int64(len(ev_flat)),
        "t": np.int64(t),
    }
    stats = {"t": t, "n_occupied": int(O.sum()), "n_evidence": int(len(ev_flat)), "n_query_total": int(is_query.sum()),
             "n_query_supervised": int(len(q_flat)),
             "n_query_supervised_unobserved": int((~tgt_hist_seen[keep]).sum()),
             "median_distance_m": float(np.median(dist[q_flat])) if len(q_flat) else float("nan")}
    return {"arrays": arrays, "stats": stats}


def prepare_drive(drive: str, device: str, limit: int | None = None, overwrite: bool = False) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    net = load_frozen_geometry(C.GEOMETRY_SEED, dev)
    g15 = C.Paths().g15
    os.makedirs(os.path.join(B5.DATA_ROOT, "queries", drive), exist_ok=True)
    ids = [t for t in G15B.anchor_ids(drive, g15) if os.path.exists(B5.target_path(drive, t))][:limit]
    written = skipped = 0; agg = []; t0 = time.time()
    for i, t in enumerate(ids):
        path = query_cache_path(drive, t)
        if os.path.exists(path) and not overwrite:
            skipped += 1; continue
        rec = prepare_anchor(net, drive, t, dev)
        tmp = path + ".tmp.npz"; np.savez_compressed(tmp, **rec["arrays"]); os.replace(tmp, path)
        written += 1; agg.append(rec["stats"])
        if written % 50 == 0:
            print(f"  [{drive[-9:-5]}] {i + 1}/{len(ids)} {(time.time() - t0) / written:.2f}s each", flush=True)
    out = {"drive": drive, "n_anchors": len(ids), "written": written, "skipped": skipped, "seconds": time.time() - t0}
    if agg:
        for k in ("n_occupied", "n_evidence", "n_query_total", "n_query_supervised", "n_query_supervised_unobserved"):
            out[f"mean_{k}"] = float(np.mean([a[k] for a in agg]))
    os.makedirs(B5.ARTIFACT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(B5.ARTIFACT_DIR, f"queries_{drive[-9:-5]}.json"), "w"), indent=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    print(json.dumps(prepare_drive(a.drive, a.device, a.limit, a.overwrite), indent=1))


if __name__ == "__main__":
    main()
