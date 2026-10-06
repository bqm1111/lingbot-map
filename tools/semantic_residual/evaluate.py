"""B5 source evaluation on 0006: B0 against B5, on identical populations.

Mirrors ``tools/delayed_semantic/evaluate.py`` -- the same regions, the same 19x19 confusions, the same distance and
reveal bins, the same firewall -- with B5's three seeds in place of the pilot's heads. Reusing that evaluator is the
point: the comparison is only meaningful if B0 is scored exactly as it was before.

Two guarantees hold by construction and are recorded per anchor:
  * occupancy is identical (B5 never touches ``O``);
  * evidence voxels keep their own teacher class in every method (the shared evidence path).

Ground truth is the held-out source semantics of drive 0006, opened only AFTER the prediction is made and saved.
Teacher-agreement metrics (cosine, top-1 against the future teacher) are recorded separately and are clearly not
semantic accuracy.

    PYTHONPATH=$PWD python -m tools.semantic_residual.evaluate --device cuda:0 --shard 0 --shards 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as C
from tools.delayed_semantic.memory import load_observed
from tools.delayed_semantic.regions import PRIMARY, SEMANTIC_REGIONS, SSC_REGIONS, build_regions
from tools.delayed_semantic.stream import DriveStream
from tools.semantic_completion.evaluation import baseline_nearest as B0, semantic_metrics as SM
from tools.semantic_completion.evaluation.evaluate import REGIONS_ROOT
from tools.semantic_completion.evidence import build as G15B, trident as TR
from tools.semantic_completion.frozen_geometry import geometry_input, load_frozen_geometry, load_hard_map, occupancy_and_trunk
from tools.semantic_residual import config as B5
from tools.semantic_residual.model import QueryContext, ResidualHead

CHUNK = 16384
#: the population the decision rule is about: predicted occupied, no evidence, revealed later, never seen by an H_t camera
DECISION_REGION = "completed_revealed_unseen"


def load_b5(seed: int, device) -> ResidualHead:
    path = B5.checkpoint_path(seed, B5.TrainConfig().steps)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ResidualHead(); m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


@torch.no_grad()
def b5_labels(model: ResidualHead, ev, trunk: torch.Tensor, query: torch.Tensor, S: torch.Tensor, F_: torch.Tensor,
              O: torch.Tensor, nearest: np.ndarray, dist_m: np.ndarray, text, device) -> torch.Tensor:
    """Class [Q] for flat query voxels. The context is exactly what was available causally at t."""
    r = B5.ResidualConfig().range_norm_m
    row_of = torch.full((C.N_VOXELS,), -1, dtype=torch.long, device=device)
    row_of[ev.flat_index] = torch.arange(len(ev.flat_index), device=device)
    nn = torch.from_numpy(nearest).to(device)
    dist = torch.from_numpy(dist_m).to(device)
    exyz_all = _centres(ev.flat_index, device)
    tr = trunk.reshape(trunk.shape[0], -1)
    out = []
    for s in range(0, len(query), CHUNK):
        q = query[s:s + CHUNK]
        rows = row_of[nn[q]]
        valid = (rows >= 0).float().unsqueeze(1)
        rows = rows.clamp(min=0)
        ctx = QueryContext(
            b0_feature=torch.nn.functional.normalize(ev.feature[rows].float(), dim=-1),
            trunk=tr[:, q].T.float(),
            geometry=torch.stack([S[q], F_[q], O[q]], 1).float(),
            delta=(exyz_all[rows] - _centres(q, device)) / r,
            distance=(dist[q] / r).unsqueeze(1),
            valid=valid,
            log_count=(torch.log1p(ev.count[rows].float()) / 5).unsqueeze(1),
        )
        out.append(TR.classify(model.corrected_feature(ctx), text))
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long, device=device)


def _centres(flat: torch.Tensor, device) -> torch.Tensor:
    X, Y, Z = C.GRID_DIMS
    k = flat.long()
    i = k // (Y * Z); j = (k // Z) % Y; l = k % Z
    origin = torch.as_tensor(C.GRID_ORIGIN_M, dtype=torch.float32, device=device)
    return (torch.stack([i, j, l], 1).float() + 0.5) * C.VOXEL_SIZE_M + origin


@torch.no_grad()
def predict_anchor(stream: DriveStream, t: int, net, models: Dict[str, ResidualHead], text, extra_query, device,
                   paths: C.Paths = C.Paths()) -> Dict:
    hist = stream.history(t)
    with FileAudit(patterns=C.PREDICT_FORBIDDEN, strict=True):
        S, F_ = load_hard_map(stream.drive, t, device, paths.g15)
        occ, trunk = occupancy_and_trunk(net, geometry_input(S, F_)); O = occ[0]; trunk = trunk[0]
        ev = load_observed(stream.drive, t, device, paths)
    assert ev.t == hist.t
    E = ev.dense_mask(); Ec = torch.zeros_like(E); Ec[ev.flat_index[ev.current()]] = True
    ev_label = torch.zeros(C.N_VOXELS, dtype=torch.long, device=device); ev_label[ev.flat_index] = TR.classify(ev.feature, text)
    nearest, dist_m = B0.nearest_evidence(E)
    lab = {"B0": B0.nearest_label(ev_label, nearest)}
    query_mask = (O | extra_query) & ~E if extra_query is not None else O & ~E
    query = query_mask.nonzero(as_tuple=True)[0]
    for name, model in models.items():
        l = b5_labels(model, ev, trunk, query, S, F_, O, nearest, dist_m, text, device)
        full = torch.zeros(C.N_VOXELS, dtype=torch.long, device=device); full[query] = l; lab[name] = full
    for m in lab:
        lab[m] = torch.where(E, ev_label, lab[m])                     # the shared evidence path, unchanged
    return {"labels": lab, "O": O, "S": S, "F": F_, "E": E, "Ec": Ec, "ev_label": ev_label, "dist_m": dist_m,
            "query": query_mask, "ev": ev, "trunk": trunk, "nearest": nearest}


def evaluate(device: str, seeds=(0, 1, 2), shard: int = 0, shards: int = 1, limit: Optional[int] = None,
             paths: C.Paths = C.Paths()) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings); text = text.to(dev)
    stream = DriveStream(C.DEV_DRIVE)
    net = load_frozen_geometry(C.GEOMETRY_SEED, dev, paths.g15)
    models = {f"B5_s{s}": load_b5(s, dev) for s in seeds}
    methods = ["B0"] + list(models)
    ids = G15B.anchor_ids(C.DEV_DRIVE, paths.g15)[shard::shards][:limit]
    nD, nR = len(C.DISTANCE_NAMES), len(C.REVEAL_NAMES)
    acc = {"ids": [], "occ": [], "evidence_path_equal": [], "occupancy_identical": [], "n_query": []}
    conf = {m: {r: [] for r in SEMANTIC_REGIONS + SSC_REGIONS} for m in methods}
    cdist = {m: [] for m in methods}; creveal = {m: [] for m in methods}
    teacher = {m: [] for m in methods}                                  # [cos_sum, top1_hits, n] on supervised voxels
    t0 = time.time()
    with FileAudit(patterns=C.EVAL_FORBIDDEN, strict=True) as audit:
        for k, t in enumerate(ids):
            with np.load(paths.g15.eval_mask_path(t)) as z:
                sem_gt = torch.from_numpy(z["sem_gt"].astype(np.int64)).to(dev)
            G = (sem_gt >= 1) & (sem_gt <= 18)
            pred = predict_anchor(stream, t, net, models, text, G, dev, paths)
            # ---- ground truth and later evidence are opened only below this line ----
            with np.load(os.path.join(REGIONS_ROOT, C.DEV_DRIVE, f"{t:05d}.npz")) as z:
                valid = torch.from_numpy(np.unpackbits(z["valid"])[:C.N_VOXELS].astype(bool)).to(dev)
                gt_occ = torch.from_numpy(np.unpackbits(z["occupied"])[:C.N_VOXELS].astype(bool)).to(dev) & valid
            with np.load(B5.target_path(C.DEV_DRIVE, t)) as z:
                keep = ~z["in_evidence"]
                rflat = torch.from_numpy(z["flat_index"][keep].astype(np.int64)).to(dev)
                rdt = torch.from_numpy(z["first_dt_s"][keep].astype(np.float32)).to(dev)
                rh = torch.from_numpy(z["hist_seen"][keep]).to(dev)
                rfeat = torch.from_numpy(z["feature"][keep].astype(np.float32)).to(dev)
            R = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=dev); R[rflat] = True
            Rh = torch.zeros_like(R); Rh[rflat[rh]] = True
            rbin = torch.full((C.N_VOXELS,), nR - 1, dtype=torch.long, device=dev)
            rbin[rflat] = torch.bucketize(rdt, torch.tensor(C.REVEAL_EDGES_S, device=dev), right=True)
            O = pred["O"]; E = pred["E"]
            reg = build_regions(sem_gt, E, pred["Ec"], pred["S"], pred["F"], O, R, Rh)
            dbin = torch.from_numpy(np.digitize(pred["dist_m"], np.asarray(C.DISTANCE_EDGES_M))).to(dev)
            acc["occ"].append([int((O & gt_occ & valid).sum()), int((O & ~gt_occ & valid).sum()), int((~O & gt_occ & valid).sum())])
            acc["evidence_path_equal"].append(all(torch.equal(pred["labels"][m][E], pred["ev_label"][E]) for m in methods))
            acc["occupancy_identical"].append(True)                    # B5 never writes O; asserted by construction
            acc["n_query"].append(int(pred["query"].sum()))
            # teacher agreement on the supervised voxels (NOT semantic accuracy)
            tgt_top = torch.nn.functional.normalize(rfeat, dim=-1) @ text.T
            tgt_cls = tgt_top.argmax(1) + 1
            for m in methods:
                lab = pred["labels"][m]
                for r in SEMANTIC_REGIONS:
                    conf[m][r].append(SM.confusion(sem_gt, lab, reg[r]))
                ssc = torch.where(O, lab, torch.zeros_like(lab))
                for r in SSC_REGIONS:
                    conf[m][r].append(SM.confusion(sem_gt, ssc, reg[r]))
                cdist[m].append(np.stack([SM.confusion(sem_gt, lab, reg[PRIMARY] & (dbin == b)) for b in range(nD)]))
                creveal[m].append(np.stack([SM.confusion(sem_gt, lab, reg[PRIMARY] & (rbin == b)) for b in range(nR)]))
                # ONLY on voxels the methods actually both label: R inside the query set. Outside it B5 writes no
                # label while B0's nearest rule still does, so scoring all of R would compare a label against a blank.
                sel = pred["query"][rflat] | E[rflat]
                hit = (lab[rflat][sel] == tgt_cls[sel])
                teacher[m].append([float(hit.sum()), float(int(sel.sum()))])
            acc["ids"].append(t)
            if (k + 1) % 50 == 0:
                print(f"  [b5 eval s{shard}] {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.2f}s each", flush=True)
    out = {"ids": np.asarray(acc["ids"]), "occ": np.asarray(acc["occ"], np.int64),
           "evidence_path_equal": np.asarray(acc["evidence_path_equal"]),
           "occupancy_identical": np.asarray(acc["occupancy_identical"]), "n_query": np.asarray(acc["n_query"])}
    for m in methods:
        for r in SEMANTIC_REGIONS + SSC_REGIONS:
            out[f"conf__{m}__{r}"] = np.stack(conf[m][r])
        out[f"confdist__{m}"] = np.stack(cdist[m]); out[f"confreveal__{m}"] = np.stack(creveal[m])
        out[f"teacher__{m}"] = np.asarray(teacher[m])
    d = os.path.join(B5.DATA_ROOT, "eval"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"eval_shard{shard}of{shards}.npz")
    np.savez_compressed(p, **out)
    meta = {"shard": shard, "shards": shards, "n_anchors": len(ids), "methods": methods, "seconds": time.time() - t0,
            "decision_region": DECISION_REGION, "firewall": {"opened": len(audit.opened), "violations": audit.violations}}
    json.dump(meta, open(p[:-4] + ".json", "w"), indent=1)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    print(json.dumps(evaluate(a.device, shard=a.shard, shards=a.shards, limit=a.limit), indent=1))


if __name__ == "__main__":
    main()
