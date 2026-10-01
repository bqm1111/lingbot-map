"""The Gate 20 per-anchor record: who is in P0/P1, and when (if ever) each voxel is revealed again.

Order of operations per anchor, and it matters:

 1. **causal state** -- hard map, frozen CompletionUNet occupancy, Trident-H evidence, B0 nearest-evidence
    label and distance. Computed inside a ``FileAudit`` with ``C.PREDICT_FORBIDDEN``, so LiDAR, GT poses,
    delayed targets and human labels cannot be opened. Digested.
 2. **ground truth**, for analysis only: ``sem_gt`` on the anchor grid. P0/P1 are formed here.
 3. **future reveal record**, restricted to the P0 rows.
 4. **no-feedback check** -- the step-1 digests are recomputed and must be unchanged.

P0 is asserted to be identical to the frozen region ``completed_never_observed``, which is the primary
region of the pilot, Gate 16, Gate 17 and Gate 18. This gate invents no new mask.

    PYTHONPATH=$PWD python -m tools.future_revealability.build_targets --drive 2013_05_28_drive_0007_sync --device cuda:0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from core.datasets import kitti360_splits as SRC, kitti360_transforms as TF
from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as C
from tools.delayed_semantic.regions import build_regions
from tools.delayed_semantic.stream import DriveStream
from tools.delayed_semantic.targets import ViewCache
from tools.future_revealability import config as G, eval_masks as EM
from tools.future_revealability.future_visibility import nested_levels_ok, reveal_record
from tools.instance_oracle import oracle as OR
from tools.semantic_completion.evaluation import baseline_nearest as B0
from tools.semantic_completion.evidence import build as G15B, trident as TR
from tools.semantic_completion.evidence.lift import load_evidence
from tools.semantic_completion.frozen_geometry import (geometry_input, load_frozen_geometry, load_hard_map,
                                                       occupancy_and_trunk)

NONE = 255
INF = float("inf")
NH = len(G.HORIZONS_S)


def _digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


@torch.no_grad()
def causal_state(drive: str, t: int, net, text, device, paths: C.Paths, want_trunk: bool = False) -> Dict:
    """Everything available at t, and nothing else. The audit makes that a runtime guarantee.

    ``want_trunk`` additionally returns the frozen decoder feature ``d1`` from the SAME forward (Gate 21
    needs it as a head input; computing it in a second forward would be the same number twice)."""
    with FileAudit(patterns=C.PREDICT_FORBIDDEN, strict=True):
        S, F_ = load_hard_map(drive, t, device, paths.g15)
        occ, d1 = occupancy_and_trunk(net, geometry_input(S, F_))
        O = occ[0]
        ev = load_evidence(paths.g15.evidence_path(drive, t), device)
    E = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=device)
    E[ev.flat_index] = True
    ev_label = torch.zeros(C.N_VOXELS, dtype=torch.long, device=device)
    ev_label[ev.flat_index] = TR.classify(ev.feature, text)
    nearest, dist_m = B0.nearest_evidence(E)
    out = {"S": S, "F": F_, "O": O, "E": E, "ev": ev, "dist_m": dist_m,
           "b0": B0.nearest_label(ev_label, nearest)}
    if want_trunk:
        out["trunk"] = d1[0]
    return out


def _state_digest(st: Dict) -> str:
    return _digest(st["O"].cpu().numpy(), st["E"].cpu().numpy(), st["b0"].cpu().numpy().astype(np.int32), st["dist_m"])


@torch.no_grad()
def build_anchor(drive: str, t: int, stream: DriveStream, geo: TF.DriveGeometry, net, text, views: ViewCache,
                 device, cfg: C.TargetConfig, paths: C.Paths) -> Dict:
    # ---- 1. causal ------------------------------------------------------------------------------
    st = causal_state(drive, t, net, text, device, paths)
    d_before = _state_digest(st)

    # ---- 2. ground truth, analysis only ---------------------------------------------------------
    m = EM.load_mask(drive, t)
    sem_gt = torch.from_numpy(m["sem_gt"]).to(device)
    valid_np = m["valid"]
    Gocc = (sem_gt >= 1) & (sem_gt <= 18)
    P0 = Gocc & st["O"] & ~st["E"]
    reg = build_regions(sem_gt, st["E"], st["E"], st["S"], st["F"], st["O"],
                        torch.zeros_like(st["E"]), torch.zeros_like(st["E"]))
    assert torch.equal(P0, reg["completed_never_observed"]), "P0 must be the frozen primary region"
    assert not bool((P0 & st["E"]).any()), "P0 contains a causally observed voxel"
    b0_correct = st["b0"] == sem_gt

    # GT connected components (Gate 17's definition, reused): per-class 6-connected
    inst_np, inst_label = OR.instance_map(m["sem_gt"].astype(np.int64), valid_np)
    n_inst = int(len(inst_label) - 1)
    inst = torch.from_numpy(inst_np.astype(np.int64)).to(device)
    comp_total = torch.bincount(inst[Gocc], minlength=n_inst + 1)
    comp_hidden = torch.bincount(inst[P0], minlength=n_inst + 1)

    # ---- 3. future, restricted to P0 ------------------------------------------------------------
    r = reveal_record(stream, geo, t, views, P0, cfg, device)
    assert nested_levels_ok(r), "a camera-visible future return without a future return"

    # the deployable level: the frozen delayed-target selector, read not recomputed
    deploy = torch.full((C.N_VOXELS,), INF, dtype=torch.float32, device=device)
    with np.load(paths.target_path(drive, t)) as z:
        tidx = torch.from_numpy(z["flat_index"].astype(np.int64)).to(device)
        deploy[tidx] = torch.from_numpy(z["first_dt_s"].astype(np.float32)).to(device)
    rows = P0.nonzero(as_tuple=True)[0]
    assert torch.equal(rows.to(torch.int32).cpu(), torch.from_numpy(r.flat_index)), "row alignment"

    # ---- 4. no feedback -------------------------------------------------------------------------
    assert _state_digest(st) == d_before, "the causal state changed after the future was opened"

    dist_bin = np.digitize(st["dist_m"], np.asarray(G.DISTANCE_EDGES_M)).astype(np.uint8)
    R = rows.cpu().numpy()
    arrays = {
        "flat_index": R.astype(np.int32),
        "sem_gt": m["sem_gt"][R].astype(np.uint8),
        "dist_bin": dist_bin[R],
        "dist_m": st["dist_m"][R].astype(np.float16),
        "b0_label": st["b0"][rows].cpu().numpy().astype(np.uint8),
        "b0_correct": b0_correct[rows].cpu().numpy(),
        "geo_first_dt": r.geo_first_dt.astype(np.float32),
        "img_first_dt": r.img_first_dt.astype(np.float32),
        "vis_first_dt": r.vis_first_dt.astype(np.float32),
        "dep_first_dt": deploy[rows].cpu().numpy().astype(np.float32),
        "vis_n_views": r.vis_n_views.clip(0, 255).astype(np.uint8),
        "mean_top1": r.mean_top1, "last_top1": r.last_top1, "first_top1": r.first_top1,
        "comp_id": inst_np[R].astype(np.int32),
        "comp_class": inst_label.astype(np.uint8),
        "comp_total": comp_total.cpu().numpy().astype(np.int32),
        "comp_hidden": comp_hidden.cpu().numpy().astype(np.int32),
    }
    stats = {"drive": drive, "t": t, "has_label": bool(m["has_label"]), "n_P0": int(P0.sum()),
             "n_P1": int((P0 & ~b0_correct).sum()), "n_gt_occ": int(Gocc.sum()),
             "n_evidence": int(st["E"].sum()), "n_pred_occ": int(st["O"].sum()),
             "n_future_frames": len(r.frames), "horizon_dt_s": r.frames[-1]["dt_s"] if r.frames else 0.0,
             "state_digest": d_before, "n_components": n_inst,
             "frames": r.frames}
    return {"arrays": arrays, "stats": stats}


def build_drive(drive: str, device: str, shard: int = 0, shards: int = 1, limit: Optional[int] = None,
                cfg: C.TargetConfig = C.TargetConfig(), paths: C.Paths = C.Paths()) -> Dict:
    dev = torch.device(device)
    torch.cuda.set_device(dev)
    G.ensure_dirs()
    stream = DriveStream(drive)
    geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    net = load_frozen_geometry(C.GEOMETRY_SEED, dev, paths.g15)
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings)
    text = text.to(dev)
    views = ViewCache(geo, text, cfg, dev)
    ids = G15B.anchor_ids(drive, paths.g15)[shard::shards][:limit]
    rows: List[Dict] = []
    t0 = time.time()
    for k, t in enumerate(ids):
        out = G.population_path(drive, int(t))
        if os.path.exists(out):
            continue
        if not EM.load_mask(drive, int(t))["has_label"]:
            rows.append({"drive": drive, "t": int(t), "has_label": False, "n_P0": 0, "n_P1": 0})
            continue
        r = build_anchor(drive, int(t), stream, geo, net, text, views, dev, cfg, paths)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp.npz"
        st = dict(r["stats"]); frames = st.pop("frames")
        np.savez_compressed(tmp, **r["arrays"], stats_json=np.asarray(json.dumps(r["stats"])))
        os.replace(tmp, out)
        rows.append(st)
        if (k + 1) % 50 == 0:
            print(f"  [g20 {drive[17:21]} s{shard}] {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.2f}s each", flush=True)
    return {"drive": drive, "shard": shard, "shards": shards, "n_anchors": len(ids), "n_built": len(rows),
            "seconds": round(time.time() - t0, 1), "version": G.VERSION}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    print(json.dumps(build_drive(a.drive, a.device, a.shard, a.shards, a.limit), indent=1))


if __name__ == "__main__":
    main()
