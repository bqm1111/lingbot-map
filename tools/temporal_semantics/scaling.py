"""Gate 18 sections 3-7: the K ladder on T1 / T2 / T3.

T1  evidence voxels at this K with a GT label        — teacher reliability; the population GROWS with K
T2  evidence voxels at K=1 with a GT label           — population FIXED across K, so only the fused feature changes
T3  completed_never_observed, via the existing B0 rule on the accumulated evidence; the region is defined from the
    FROZEN 5-frame evidence so the population is identical for every K and only the propagated label changes.

    PYTHONPATH=$PWD python -m tools.temporal_semantics.scaling --device cuda:0 --shard 0 --shards 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
from scipy import ndimage

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as C
from tools.delayed_semantic.memory import load_observed
from tools.delayed_semantic.regions import build_regions
from tools.semantic_completion.evaluation import semantic_metrics as SM
from tools.semantic_completion.evidence import build as G15B, trident as TR
from tools.semantic_completion.frozen_geometry import geometry_input, load_frozen_geometry, occupancy_and_trunk
from tools.temporal_semantics import config as G18
from tools.temporal_semantics.accumulate import HistoryBuilder, label_of

DIMS = tuple(int(d) for d in C.GRID_DIMS)
LADDER = list(G18.HISTORY_K)
RULES_AT_K5 = ("latest", "mean_feature", "mean_probability")


def nearest_label(label_dense: torch.Tensor, ev_mask: np.ndarray):
    """B0's rule, unchanged: the label of the nearest evidence voxel, plus the distance to it in metres."""
    m = ev_mask.reshape(DIMS)
    if not m.any():
        return (torch.zeros(C.N_VOXELS, dtype=torch.long, device=label_dense.device),
                np.full(C.N_VOXELS, np.inf, np.float32))
    dist, idx = ndimage.distance_transform_edt(~m, return_indices=True)
    flat = np.ravel_multi_index((idx[0], idx[1], idx[2]), DIMS).reshape(-1)
    return (label_dense[torch.from_numpy(flat.astype(np.int64)).to(label_dense.device)],
            (dist.reshape(-1) * float(C.VOXEL_SIZE_M)).astype(np.float32))


def evaluate(device: str, shard: int = 0, shards: int = 1, limit: Optional[int] = None,
             paths: C.Paths = C.Paths()) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings); text = text.to(dev)
    net = load_frozen_geometry(C.GEOMETRY_SEED, dev, paths.g15)
    hb = HistoryBuilder(G18.DEV_DRIVE, dev, paths)
    ids = G15B.anchor_ids(G18.DEV_DRIVE, paths.g15)[shard::shards][:limit]
    cols = [f"K{k}" for k in LADDER] + [f"rule_{r}" for r in RULES_AT_K5]

    conf = {c: {p: [] for p in ("T1", "T2", "T3")} for c in cols}
    cdist = {c: [] for c in cols}
    nD = len(G18.DISTANCE_NAMES)
    stats = {c: [] for c in cols}          # [n_evidence, n_scored_T1, mean_frames_per_voxel]
    t0 = time.time()
    with FileAudit(patterns=C.EVAL_FORBIDDEN, strict=True) as audit:
        for n, t in enumerate(ids):
            with np.load(paths.g15.input_path(G18.DEV_DRIVE, t)) as z:
                S = torch.from_numpy(np.unpackbits(z["S_H"])[:C.N_VOXELS].astype(bool)).to(dev)
                F_ = torch.from_numpy(np.unpackbits(z["F_H"])[:C.N_VOXELS].astype(bool)).to(dev)
            occ, _ = occupancy_and_trunk(net, geometry_input(S, F_)); O = occ[0]
            with np.load(paths.g15.eval_mask_path(t)) as z:
                sem_gt = torch.from_numpy(z["sem_gt"].astype(np.int64)).to(dev)
            frozen = load_observed(G18.DEV_DRIVE, t, dev, paths)
            E_frozen = frozen.dense_mask()
            Ec = torch.zeros_like(E_frozen); Ec[frozen.flat_index[frozen.current()]] = True
            R = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=dev)
            reg = build_regions(sem_gt, E_frozen, Ec, S, F_, O, R, R)
            T3_pop = reg["completed_never_observed"]
            labelled = (sem_gt >= 1) & (sem_gt <= 18)

            fixed_pop = None
            for col, (k, rule) in zip(cols, [(k, G18.FROZEN_AGGREGATION) for k in LADDER] + [(5, r) for r in RULES_AT_K5]):
                ev = hb.build(t, k, rule, S, text)
                lab = torch.zeros(C.N_VOXELS, dtype=torch.long, device=dev)
                lab[ev["flat_index"]] = label_of(ev, text)
                mask = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=dev); mask[ev["flat_index"]] = True
                if col == "K1":
                    fixed_pop = mask & labelled
                conf[col]["T1"].append(SM.confusion(sem_gt, lab, mask & labelled))
                conf[col]["T2"].append(SM.confusion(sem_gt, lab, fixed_pop))
                prop, dist_m = nearest_label(lab, mask.cpu().numpy())
                conf[col]["T3"].append(SM.confusion(sem_gt, prop, T3_pop))
                dbin = torch.from_numpy(np.digitize(dist_m, np.asarray(G18.DISTANCE_EDGES_M))).to(dev)
                cdist[col].append(np.stack([SM.confusion(sem_gt, prop, T3_pop & (dbin == b)) for b in range(nD)]))
                stats[col].append([int(mask.sum()), int((mask & labelled).sum()), float(ev["n_frames"].float().mean())])
            if (n + 1) % 25 == 0:
                print(f"  [scaling s{shard}] {n+1}/{len(ids)} {(time.time()-t0)/(n+1):.2f}s each", flush=True)

    out = {"ids": np.asarray(ids)}
    for c in cols:
        for p in ("T1", "T2", "T3"):
            out[f"conf__{c}__{p}"] = np.stack(conf[c][p])
        out[f"stats__{c}"] = np.asarray(stats[c])
        out[f"confdist__{c}"] = np.stack(cdist[c])
    p = G18.shard_path("scaling", shard, shards); os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, **out)
    meta = {"version": G18.VERSION, "shard": shard, "shards": shards, "n_anchors": len(ids), "columns": cols,
            "seconds": time.time() - t0, "firewall": {"opened": len(audit.opened), "violations": audit.violations}}
    json.dump(meta, open(p[:-4] + ".json", "w"), indent=1)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    print(json.dumps(evaluate(a.device, a.shard, a.shards, a.limit), indent=1))


if __name__ == "__main__":
    main()
