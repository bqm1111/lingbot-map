"""Gate 17 evaluation on 0006: five label maps, three populations, with breakdowns.

Reuses the frozen prediction path and regions exactly as the pilot and Gate B5 do. The only addition is the oracle
instance grouping, which is built from ground truth and is therefore opened only AFTER the causal prediction.

    PYTHONPATH=$PWD python -m tools.instance_oracle.evaluate --device cuda:0 --shard 0 --shards 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as C
from tools.delayed_semantic.memory import load_observed
from tools.delayed_semantic.regions import build_regions
from tools.delayed_semantic.stream import DriveStream
from tools.instance_oracle import config as G, oracle as OR
from tools.semantic_completion.evaluation import baseline_nearest as B0, semantic_metrics as SM
from tools.semantic_completion.evaluation.evaluate import REGIONS_ROOT
from tools.semantic_completion.evidence import build as G15B, trident as TR
from tools.semantic_completion.frozen_geometry import geometry_input, load_frozen_geometry, load_hard_map, occupancy_and_trunk


@torch.no_grad()
def causal_state(stream: DriveStream, t: int, net, text, device, paths: C.Paths):
    """Everything available at t. Inside the firewall: no future frame, pose, target or label may be opened."""
    with FileAudit(patterns=C.PREDICT_FORBIDDEN, strict=True):
        S, F_ = load_hard_map(stream.drive, t, device, paths.g15)
        occ, _ = occupancy_and_trunk(net, geometry_input(S, F_)); O = occ[0]
        ev = load_observed(stream.drive, t, device, paths)
    E = ev.dense_mask()
    Ec = torch.zeros_like(E); Ec[ev.flat_index[ev.current()]] = True
    ev_label = torch.zeros(C.N_VOXELS, dtype=torch.long, device=device)
    ev_label[ev.flat_index] = TR.classify(ev.feature, text)
    nearest, dist_m = B0.nearest_evidence(E)
    return {"S": S, "F": F_, "O": O, "E": E, "Ec": Ec, "ev": ev, "ev_label": ev_label,
            "b0": B0.nearest_label(ev_label, nearest), "dist_m": dist_m}


def evaluate(device: str, shard: int = 0, shards: int = 1, limit: Optional[int] = None,
             paths: C.Paths = C.Paths()) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings); text = text.to(dev)
    stream = DriveStream(G.DEV_DRIVE)
    net = load_frozen_geometry(C.GEOMETRY_SEED, dev, paths.g15)
    ids = G15B.anchor_ids(G.DEV_DRIVE, paths.g15)[shard::shards][:limit]
    nD, nV = len(G.DISTANCE_NAMES), len(G.VISIBILITY_NAMES)
    pops = G.POPULATIONS

    conf = {m: {p: [] for p in pops} for m in G.METHODS}
    cdist = {m: [] for m in G.METHODS}; cvis = {m: [] for m in G.METHODS}
    cthing = {m: {"thing": [], "stuff": []} for m in G.METHODS}
    acc = {"ids": [], "coverage": [], "n_pop": [], "n_instances": [], "n_inst_with_evidence": []}
    t0 = time.time()
    with FileAudit(patterns=C.EVAL_FORBIDDEN, strict=True) as audit:
        for k, t in enumerate(ids):
            with np.load(paths.g15.eval_mask_path(t)) as z:
                sem_gt_np = z["sem_gt"].astype(np.int64)
                valid_np = np.unpackbits(z["valid"])[:C.N_VOXELS].astype(bool)
            sem_gt = torch.from_numpy(sem_gt_np).to(dev)
            st = causal_state(stream, t, net, text, dev, paths)
            # ---- ground truth grouping is built only below this line ----
            gt_occ_mask = (sem_gt_np >= 1) & (sem_gt_np <= 18) & valid_np
            inst_np, inst_label = OR.instance_map(sem_gt_np, valid_np)
            n_inst = int(len(inst_label) - 1)
            E_np = st["E"].cpu().numpy()
            stats = OR.instance_stats(inst_np, E_np & gt_occ_mask, n_inst)
            inst = torch.from_numpy(inst_np.astype(np.int64)).to(dev)
            sem = OR.instance_semantics(inst, st["ev"].flat_index, st["ev"].feature, st["ev_label"], n_inst, text, dev)
            gen = torch.Generator(device=dev).manual_seed(G.SEED * 100003 + t)
            labs = OR.method_labels(st["b0"], inst, sem, sem_gt, gen)
            covered = labs.pop("_covered")

            with np.load(os.path.join(REGIONS_ROOT, G.DEV_DRIVE, f"{t:05d}.npz")) as z:
                zvalid = torch.from_numpy(np.unpackbits(z["valid"])[:C.N_VOXELS].astype(bool)).to(dev)
            R = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=dev)          # unused here; regions need the argument
            reg = build_regions(sem_gt, st["E"], st["Ec"], st["S"], st["F"], st["O"], R, R)
            O = st["O"]
            dbin = torch.from_numpy(np.digitize(st["dist_m"], np.asarray(G.DISTANCE_EDGES_M))).to(dev)
            vfrac = torch.from_numpy(stats["visible_fraction"]).to(dev)[inst]
            vbin = torch.bucketize(vfrac.float(), torch.tensor(G.VISIBILITY_EDGES, device=dev), right=False)
            thing = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=dev)
            for c in G.THING_LABELS:
                thing |= (sem_gt == c)

            P2 = reg[pops["P2_correct_completion"]]
            acc["ids"].append(t)
            acc["coverage"].append([int((covered & P2).sum()), int(P2.sum())])
            acc["n_pop"].append([int(reg[r].sum()) for r in pops.values()])
            acc["n_instances"].append(n_inst)
            acc["n_inst_with_evidence"].append(int(sem["has_evidence"].sum()))
            for m in G.METHODS:
                lab = labs[m]
                for pname, rname in pops.items():
                    # P1 is the deployable SSC read: class where predicted-occupied, else empty
                    pred = torch.where(O, lab, torch.zeros_like(lab)) if pname == "P1_ssc_deployable" else lab
                    conf[m][pname].append(SM.confusion(sem_gt, pred, reg[rname]))
                cdist[m].append(np.stack([SM.confusion(sem_gt, lab, P2 & (dbin == b)) for b in range(nD)]))
                cvis[m].append(np.stack([SM.confusion(sem_gt, lab, P2 & (vbin == b)) for b in range(nV)]))
                cthing[m]["thing"].append(SM.confusion(sem_gt, lab, P2 & thing))
                cthing[m]["stuff"].append(SM.confusion(sem_gt, lab, P2 & ~thing))
            if (k + 1) % 50 == 0:
                print(f"  [gate17 s{shard}] {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.2f}s each", flush=True)

    out = {"ids": np.asarray(acc["ids"]), "coverage": np.asarray(acc["coverage"], np.int64),
           "n_pop": np.asarray(acc["n_pop"], np.int64), "n_instances": np.asarray(acc["n_instances"]),
           "n_inst_with_evidence": np.asarray(acc["n_inst_with_evidence"])}
    for m in G.METHODS:
        for p in pops:
            out[f"conf__{m}__{p}"] = np.stack(conf[m][p])
        out[f"confdist__{m}"] = np.stack(cdist[m]); out[f"confvis__{m}"] = np.stack(cvis[m])
        out[f"confthing__{m}"] = np.stack(cthing[m]["thing"]); out[f"confstuff__{m}"] = np.stack(cthing[m]["stuff"])
    p = G.shard_path(shard, shards); os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, **out)
    meta = {"version": G.VERSION, "shard": shard, "shards": shards, "n_anchors": len(ids),
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
