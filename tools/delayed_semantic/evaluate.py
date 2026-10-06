"""Development evaluation on 0006: B0, B0g, B1, B2 with one shared evidence path, geometry and vocabulary.

Per anchor, in this order:
1. ``predict_anchor`` (inside a PREDICT_FORBIDDEN file audit: no future frame, LiDAR, pose, target or label):
   frozen geometry forward -> O; H_t evidence + provenance; every method labels every voxel of the query set.
   The query set is O plus the GT-occupied voxels, passed in by the evaluator so oracle-occupancy regions can be
   scored; each query is computed independently of the others, so adding queries cannot change a prediction
   (``verify.py`` checks it). Evidence voxels take their own teacher class in every method.
2. the prediction-at-t record (labels on O, B2 confidence) is written before any GT or target is opened.
3. scoring: GT (eval masks), Gate 12 occupancy regions, 0006 delayed targets (regions R / Rh, time to reveal).

Saved per anchor: 19x19 confusions per (method, region), per distance bin and per reveal bin on the primary
region, occupancy TP/FP/FN, label-free agreement with the delayed target class, evidence-path equality flag.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as C
from tools.delayed_semantic.baselines import geodesic_label
from tools.delayed_semantic.memory import load_observed
from tools.delayed_semantic.model import EvidenceAssociation, nearest_candidates, voxel_centres
from tools.delayed_semantic.regions import PRIMARY, SEMANTIC_REGIONS, SSC_REGIONS, build_regions
from tools.delayed_semantic.stream import DriveStream
from tools.semantic_completion.evaluation import baseline_nearest as B0, semantic_metrics as SM
from tools.semantic_completion.evaluation.evaluate import REGIONS_ROOT
from tools.semantic_completion.evidence import build as G15B, trident as TR
from tools.semantic_completion.frozen_geometry import geometry_input, load_frozen_geometry, load_hard_map, occupancy_and_trunk

METHODS = ("B0", "B0g", "B1", "B2")


def load_head(path: str, device) -> EvidenceAssociation:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = EvidenceAssociation(C.ModelConfig(**ck["model_config"])); m.load_state_dict(ck["state_dict"]); return m.to(device).eval()


@torch.no_grad()
def head_predict(head: EvidenceAssociation, ev, trunk: torch.Tensor, query: torch.Tensor, masks: torch.Tensor, text, chunk: int = 16384):
    """Class [Q] and top weight [Q] for flat query voxels ``query`` (none of them evidence)."""
    exyz = voxel_centres(ev.flat_index)
    bits = ev.frame_bits.long()
    evd = {"feature": ev.feature, "trunk": trunk.reshape(trunk.shape[0], -1)[:, ev.flat_index].T, "xyz": exyz, "count": ev.count,
           "current": ev.current(), "n_frames": sum((bits >> q) & 1 for q in range(5))}
    tr = trunk.reshape(trunk.shape[0], -1)
    labels, wmax = [], []
    for s in range(0, len(query), chunk):
        q = query[s:s + chunk]; qxyz = voxel_centres(q)
        idx, _ = nearest_candidates(qxyz, exyz, head.cfg.k_candidates)
        f, w = head(tr[:, q].T, qxyz, idx, evd, masks)
        labels.append(TR.classify(f, text)); wmax.append(w.max(1).values)
    return (torch.cat(labels), torch.cat(wmax)) if labels else (torch.zeros(0, dtype=torch.long, device=trunk.device), torch.zeros(0, device=trunk.device))


@torch.no_grad()
def predict_anchor(stream: DriveStream, t: int, net, heads: Dict[str, EvidenceAssociation], text, extra_query: Optional[torch.Tensor], device,
                   paths: C.Paths = C.Paths()) -> Dict:
    hist = stream.history(t)                                                           # the only frame accessor used
    with FileAudit(patterns=C.PREDICT_FORBIDDEN, strict=True):
        S, F_ = load_hard_map(stream.drive, t, device, paths.g15)
        occ, trunk = occupancy_and_trunk(net, geometry_input(S, F_)); O = occ[0]; trunk = trunk[0]
        ev = load_observed(stream.drive, t, device, paths)
    assert ev.t == hist.t
    E = ev.dense_mask(); Ec = torch.zeros_like(E); Ec[ev.flat_index[ev.current()]] = True
    ev_label = torch.zeros(C.N_VOXELS, dtype=torch.long, device=device); ev_label[ev.flat_index] = TR.classify(ev.feature, text)
    nearest, dist_m = B0.nearest_evidence(E)
    lab = {"B0": B0.nearest_label(ev_label, nearest)}
    lab["B0g"], reached = geodesic_label(ev_label, E, O, lab["B0"])
    query_mask = (O | extra_query) & ~E if extra_query is not None else O & ~E
    query = query_mask.nonzero(as_tuple=True)[0]
    masks = torch.stack([F_, S, O]).reshape(3, *C.GRID_DIMS)
    conf = {}
    for m, head in heads.items():
        l, w = head_predict(head, ev, trunk, query, masks, text)
        full = torch.zeros(C.N_VOXELS, dtype=torch.long, device=device); full[query] = l; lab[m] = full
        wf = torch.zeros(C.N_VOXELS, device=device); wf[query] = w; conf[m] = wf
    for m in lab:
        lab[m] = torch.where(E, ev_label, lab[m])                                     # the shared evidence path
    return {"labels": lab, "top_weight": conf, "O": O, "S": S, "F": F_, "E": E, "Ec": Ec, "ev_label": ev_label, "dist_m": dist_m,
            "geodesic_reached": reached, "query": query_mask}


def save_prediction_record(drive: str, t: int, time_s: float, pred: Dict, head_paths: Dict[str, str], paths: C.Paths = C.Paths()) -> str:
    """The prediction made at t, stored before any later evidence is opened; later verification reads it, never rewrites it."""
    p = os.path.join(paths.predictions_root, drive, f"{t:05d}.npz")
    O = pred["O"]; idx = O.nonzero(as_tuple=True)[0]
    arrays = {f"label_{m}": v[idx].to(torch.uint8).cpu().numpy() for m, v in pred["labels"].items()}
    arrays.update({f"top_weight_{m}": v[idx].half().cpu().numpy() for m, v in pred["top_weight"].items()})
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p):
        with np.load(p) as z:
            same = all(np.array_equal(z[k], v) for k, v in arrays.items()) and np.array_equal(z["O"], np.packbits(O.cpu().numpy()))
        assert same, f"a stored prediction at t={t} would change: {p}"
        return p
    np.savez_compressed(p, O=np.packbits(O.cpu().numpy()), t=np.int64(t), time_s=np.float64(time_s), heads=json.dumps(head_paths), **arrays)
    return p


def evaluate(head_paths: Dict[str, str], device: str, shard: int = 0, shards: int = 1, limit: Optional[int] = None, out_path: Optional[str] = None,
             paths: C.Paths = C.Paths()) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    text, _, _ = TR.load_text_embeddings(paths.g15.text_embeddings); text = text.to(dev)
    stream = DriveStream(C.DEV_DRIVE)
    net = load_frozen_geometry(C.GEOMETRY_SEED, dev, paths.g15)
    heads = {m: load_head(p, dev) for m, p in head_paths.items()}
    methods = ["B0", "B0g"] + list(heads)
    ids = G15B.anchor_ids(C.DEV_DRIVE, paths.g15)[shard::shards][:limit]
    nD, nR = len(C.DISTANCE_NAMES), len(C.REVEAL_NAMES)
    acc = {"ids": [], "occ": [], "evidence_path_equal": [], "n_query": [], "geodesic_reached": []}
    conf = {m: {r: [] for r in SEMANTIC_REGIONS + SSC_REGIONS} for m in methods}
    cdist = {m: [] for m in methods}; creveal = {m: [] for m in methods}; ctarget = {m: [] for m in methods}
    t0 = time.time()
    with FileAudit(patterns=C.EVAL_FORBIDDEN, strict=True) as audit:
        for k, t in enumerate(ids):
            with np.load(paths.g15.eval_mask_path(t)) as z:
                sem_gt = torch.from_numpy(z["sem_gt"].astype(np.int64)).to(dev)
            G = (sem_gt >= 1) & (sem_gt <= 18)
            pred = predict_anchor(stream, t, net, heads, text, G, dev, paths)
            save_prediction_record(C.DEV_DRIVE, t, stream.frames[t].time_s, pred, head_paths, paths)
            # ---- later evidence and labels are opened only below this line ----
            with np.load(os.path.join(REGIONS_ROOT, C.DEV_DRIVE, f"{t:05d}.npz")) as z:
                valid = torch.from_numpy(np.unpackbits(z["valid"])[:C.N_VOXELS].astype(bool)).to(dev)
                gt_occ = torch.from_numpy(np.unpackbits(z["occupied"])[:C.N_VOXELS].astype(bool)).to(dev) & valid
            with np.load(paths.target_path(C.DEV_DRIVE, t)) as z:
                keep = ~z["in_evidence"]
                rflat = torch.from_numpy(z["flat_index"][keep].astype(np.int64)).to(dev)
                rtop = torch.from_numpy(z["prob"][keep].astype(np.float32).argmax(1) + 1).to(dev)
                rdt = torch.from_numpy(z["first_dt_s"][keep].astype(np.float32)).to(dev)
                rh = torch.from_numpy(z["hist_seen"][keep]).to(dev)
            R = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=dev); R[rflat] = True
            Rh = torch.zeros_like(R); Rh[rflat[rh]] = True
            rbin = torch.full((C.N_VOXELS,), nR - 1, dtype=torch.long, device=dev)
            rbin[rflat] = torch.bucketize(rdt, torch.tensor(C.REVEAL_EDGES_S, device=dev), right=True)
            O = pred["O"]
            reg = build_regions(sem_gt, pred["E"], pred["Ec"], pred["S"], pred["F"], O, R, Rh)
            dbin = torch.from_numpy(np.digitize(pred["dist_m"], np.asarray(C.DISTANCE_EDGES_M))).to(dev)
            acc["occ"].append([int((O & gt_occ & valid).sum()), int((O & ~gt_occ & valid).sum()), int((~O & gt_occ & valid).sum())])
            E = pred["E"]
            acc["evidence_path_equal"].append(all(torch.equal(pred["labels"][m][E], pred["ev_label"][E]) for m in methods))
            acc["n_query"].append(int(pred["query"].sum()))
            acc["geodesic_reached"].append([int((pred["geodesic_reached"] & O & ~E).sum()), int((O & ~E).sum())])   # [reached, all completed non-evidence]
            tgt_dense = torch.zeros(C.N_VOXELS, dtype=torch.long, device=dev); tgt_dense[rflat] = rtop
            for m in methods:
                lab = pred["labels"][m]
                for r in SEMANTIC_REGIONS:
                    conf[m][r].append(SM.confusion(sem_gt, lab, reg[r]))
                ssc = torch.where(O, lab, torch.zeros_like(lab))
                for r in SSC_REGIONS:
                    conf[m][r].append(SM.confusion(sem_gt, ssc, reg[r]))
                cdist[m].append(np.stack([SM.confusion(sem_gt, lab, reg[PRIMARY] & (dbin == b)) for b in range(nD)]))
                creveal[m].append(np.stack([SM.confusion(sem_gt, lab, reg[PRIMARY] & (rbin == b)) for b in range(nR)]))
                ctarget[m].append(SM.confusion(tgt_dense, lab, R & (pred["query"] | E)))       # label-free: rows = delayed target class
            acc["ids"].append(t)
            if (k + 1) % 50 == 0:
                print(f"  [eval s{shard}] {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.2f}s each", flush=True)
    out = {"ids": np.asarray(acc["ids"]), "occ": np.asarray(acc["occ"], np.int64), "evidence_path_equal": np.asarray(acc["evidence_path_equal"]),
           "n_query": np.asarray(acc["n_query"]), "geodesic_reached": np.asarray(acc["geodesic_reached"])}
    for m in methods:
        for r in SEMANTIC_REGIONS + SSC_REGIONS:
            out[f"conf__{m}__{r}"] = np.stack(conf[m][r])
        out[f"confdist__{m}"] = np.stack(cdist[m]); out[f"confreveal__{m}"] = np.stack(creveal[m]); out[f"conftarget__{m}"] = np.stack(ctarget[m])
    out_path = out_path or os.path.join(paths.eval, f"eval_shard{shard}of{shards}.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True); np.savez_compressed(out_path, **out)
    meta = {"head_checkpoints": head_paths, "shard": shard, "shards": shards, "n_anchors": len(ids), "seconds": time.time() - t0,
            "firewall": {"opened": len(audit.opened), "violations": audit.violations}}
    json.dump(meta, open(out_path[:-4] + ".json", "w"), indent=1)
    return meta


__all__ = ["METHODS", "load_head", "head_predict", "predict_anchor", "save_prediction_record", "evaluate"]
