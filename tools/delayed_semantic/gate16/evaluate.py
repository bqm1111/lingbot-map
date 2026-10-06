"""Gate 16 development evaluation on 0006: B0, pilot B1/B2 (seed 0), B3 and B3geom (seeds 0-2) on ONE candidate set per anchor.

Per anchor:
1. semantic GT defines the query list only ((O | G) & ~E, the pilot's list; predictions are per-query independent);
2. student side (``context.student_context``, PREDICT_FORBIDDEN audit) -> candidate sets in 16,384-query chunks (the pilot's
   chunking, so B1/B2 labels reproduce the pilot exactly) -> every head scores the same candidates;
3. the prediction-at-t record (labels and top-weight candidate on O) is written;
4. scoring (evaluation only): pilot regions + candidate-hit / candidate-miss split of the primary region; readout and top-weight
   candidate confusions; C0 and oracle confusions (must equal candidate_ceiling.json); pooled distance / reveal / class tallies;
   label-free candidate-target agreement on the anchor's delayed-target voxels (+ GT correctness there); occupancy counts.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as PC
from tools.delayed_semantic.gate16 import config as G, context as CX
from tools.delayed_semantic.gate16.candidate_metrics import FIELDS, SelectionTally
from tools.delayed_semantic.gate16.candidate_regions import candidate_hit, oracle_label
from tools.delayed_semantic.gate16.candidate_targets import build_target
from tools.delayed_semantic.model import EvidenceAssociation, voxel_centres
from tools.delayed_semantic.regions import PRIMARY, SEMANTIC_REGIONS, SSC_REGIONS, build_regions
from tools.delayed_semantic.stream import DriveStream
from tools.semantic_completion.evaluation import semantic_metrics as SM
from tools.semantic_completion.evidence import build as G15B, trident as TR
from tools.semantic_completion.frozen_geometry import load_frozen_geometry

CAND_REGIONS = ("primary_candidate_hit", "primary_candidate_miss")
TOP_REGIONS = (PRIMARY,) + CAND_REGIONS


def load_head(path: str, device) -> EvidenceAssociation:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = EvidenceAssociation(PC.ModelConfig(**ck["model_config"]), semantic_projection=ck.get("semantic_projection", True))
    m.load_state_dict(ck["state_dict"]); return m.to(device).eval()


@torch.no_grad()
def score_heads(heads: Dict[str, EvidenceAssociation], ctx: Dict, chunks, text) -> Dict[str, Dict[str, torch.Tensor]]:
    """Per head: readout label [Q], top-weight candidate index [Q], weights [Q, K] (kept only when ``keep_weights``)."""
    tr = ctx["trunk"].reshape(ctx["trunk"].shape[0], -1)
    out = {m: {"label": [], "top": [], "weights": []} for m in heads}
    for q, cs in chunks:
        qxyz = voxel_centres(q)
        for m, h in heads.items():
            f, w = h(tr[:, q].T, qxyz, cs.rows, ctx["evd"], ctx["masks"])
            out[m]["label"].append(TR.classify(f, text)); out[m]["top"].append(w.argmax(-1)); out[m]["weights"].append(w)
    return {m: {k: torch.cat(v) for k, v in d.items()} for m, d in out.items()}


def evaluate(device: str, shard: int = 0, shards: int = 1, limit: Optional[int] = None, head_paths: Optional[Dict[str, str]] = None,
             out_path: Optional[str] = None, predictions_root: str = G.PREDICTIONS) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    text, _, _ = TR.load_text_embeddings(); text = text.to(dev)
    stream = DriveStream(PC.DEV_DRIVE); net = load_frozen_geometry(PC.GEOMETRY_SEED, dev)
    head_paths = head_paths or G.head_methods()
    heads = {m: load_head(p, dev) for m, p in head_paths.items()}
    methods = ["B0"] + list(heads)
    ids = G15B.anchor_ids(PC.DEV_DRIVE)[shard::shards][:limit]
    nD, nR = len(PC.DISTANCE_NAMES), len(PC.REVEAL_NAMES)
    regions = SEMANTIC_REGIONS + CAND_REGIONS + SSC_REGIONS
    per_anchor = {f"conf__{m}__{r}": [] for m in methods for r in regions}
    per_anchor.update({f"conftop__{m}__{r}": [] for m in methods for r in TOP_REGIONS})
    per_anchor.update({f"conf__{o}__{r}": [] for o in ("C0", "oracle") for r in TOP_REGIONS})
    acc = {"ids": [], "occ": [], "evidence_path_equal": [], "n_query": []}
    pooled = {f"dist_label__{m}": np.zeros((nD, 19, 19), np.int64) for m in methods}
    pooled.update({f"dist_top__{m}": np.zeros((nD, 19, 19), np.int64) for m in methods})
    pooled.update({f"reveal_label__{m}": np.zeros((nR, 19, 19), np.int64) for m in methods})
    pooled.update({f"reveal_top__{m}": np.zeros((nR, 19, 19), np.int64) for m in methods})
    pooled.update({"dist_hit": np.zeros((nD, 2), np.int64), "reveal_hit": np.zeros((nR, 2), np.int64), "class_hit": np.zeros((19, 2), np.int64)})
    tally = SelectionTally(); t0 = time.time()
    with FileAudit(patterns=PC.EVAL_FORBIDDEN, strict=True) as audit:
        for k, t in enumerate(ids):
            tr = CX.truth(t, dev); gt = tr["sem_gt"]; Gm = (gt >= 1) & (gt <= 18)
            ctx = CX.student_context(stream, t, net, text, dev)
            E, O = ctx["E"], ctx["O"]
            query = ((O | Gm) & ~E).nonzero(as_tuple=True)[0]
            chunks = CX.query_candidates(ctx, query, text)
            cs = CX.cat_sets([c for _, c in chunks])
            scored = score_heads(heads, ctx, chunks, text)
            labels = {"B0": ctx["b0"].clone()}; top = {"B0": torch.zeros(len(query), dtype=torch.long, device=dev)}
            for m in heads:
                full = torch.zeros(PC.N_VOXELS, dtype=torch.long, device=dev); full[query] = scored[m]["label"]; labels[m] = full; top[m] = scored[m]["top"]
            for m in labels:
                labels[m] = torch.where(E, ctx["ev_label"], labels[m])
            # prediction-at-t record (before scoring)
            p = os.path.join(predictions_root, PC.DEV_DRIVE, f"{t:05d}.npz"); os.makedirs(os.path.dirname(p), exist_ok=True)
            oq = O[query]; oi = O.nonzero(as_tuple=True)[0]
            rec = {f"label_{m}": labels[m][oi].to(torch.uint8).cpu().numpy() for m in labels}
            rec.update({f"top_{m}": top[m][oq].to(torch.uint8).cpu().numpy() for m in heads})
            if os.path.exists(p):
                with np.load(p) as z:
                    assert all(np.array_equal(z[kk], v) for kk, v in rec.items()), f"stored prediction at t={t} would change"
            else:
                tmp = p + ".tmp.npz"; np.savez_compressed(tmp, O=np.packbits(O.cpu().numpy()), t=np.int64(t), heads=json.dumps(head_paths), **rec); os.replace(tmp, p)
            # ---- scoring ----
            reg = build_regions(gt, E, ctx["Ec"], ctx["S"], ctx["F"], O, tr["R"], tr["Rh"])
            gq = gt[query].clamp(0, 18); lab_q = (gq >= 1) & (gq <= 18)
            hit_q = candidate_hit(cs, gq) & lab_q
            HIT = torch.zeros(PC.N_VOXELS, dtype=torch.bool, device=dev); HIT[query[hit_q]] = True
            reg["primary_candidate_hit"] = reg[PRIMARY] & HIT; reg["primary_candidate_miss"] = reg[PRIMARY] & ~HIT
            dense = lambda v: torch.zeros(PC.N_VOXELS, dtype=torch.long, device=dev).index_put_((query,), v)
            topcls = {m: dense(cs.classes.gather(1, top[m][:, None]).squeeze(1)) for m in labels}
            c0 = dense(cs.classes[:, 0]); orc = dense(torch.where(lab_q, oracle_label(cs, gq.clamp(min=1)), cs.classes[:, 0]))
            for m in labels:
                for r in SEMANTIC_REGIONS + CAND_REGIONS:
                    per_anchor[f"conf__{m}__{r}"].append(SM.confusion(gt, labels[m], reg[r]))
                ssc = torch.where(O, labels[m], torch.zeros_like(labels[m]))
                for r in SSC_REGIONS:
                    per_anchor[f"conf__{m}__{r}"].append(SM.confusion(gt, ssc, reg[r]))
                for r in TOP_REGIONS:
                    per_anchor[f"conftop__{m}__{r}"].append(SM.confusion(gt, topcls[m], reg[r]))
            for o, lab in (("C0", c0), ("oracle", orc)):
                for r in TOP_REGIONS:
                    per_anchor[f"conf__{o}__{r}"].append(SM.confusion(gt, lab, reg[r]))
            P = reg[PRIMARY]
            dbin = torch.from_numpy(np.digitize(ctx["dist_m"], np.asarray(PC.DISTANCE_EDGES_M))).to(dev)
            for b in range(nD):
                mb = P & (dbin == b)
                pooled["dist_hit"][b] += [int(mb.sum()), int((mb & HIT).sum())]
                for m in labels:
                    pooled[f"dist_label__{m}"][b] += SM.confusion(gt, labels[m], mb); pooled[f"dist_top__{m}"][b] += SM.confusion(gt, topcls[m], mb)
            for b in range(nR):
                mb = P & (tr["rbin"] == b)
                pooled["reveal_hit"][b] += [int(mb.sum()), int((mb & HIT).sum())]
                for m in labels:
                    pooled[f"reveal_label__{m}"][b] += SM.confusion(gt, labels[m], mb); pooled[f"reveal_top__{m}"][b] += SM.confusion(gt, topcls[m], mb)
            gp = gt[P]
            pooled["class_hit"] += np.stack([np.bincount(gp.cpu().numpy(), minlength=19), np.bincount(gt[P & HIT].cpu().numpy(), minlength=19)], 1)
            # label-free candidate-target agreement on this anchor's delayed-target voxels
            rq = tr["rflat"]
            if len(rq):
                rchunks = CX.query_candidates(ctx, rq, text)
                rcs = CX.cat_sets([c for _, c in rchunks]); rsc = score_heads(heads, ctx, rchunks, text)
                ct = build_target(rcs.semantic_probs, tr["rprob"])
                tally.add("B0", rcs, ct, None, gt[rq])
                for m in heads:
                    tally.add(m, rcs, ct, rsc[m]["weights"], gt[rq])
            acc["occ"].append([int((O & tr["gt_occ"] & tr["valid"]).sum()), int((O & ~tr["gt_occ"] & tr["valid"]).sum()), int((~O & tr["gt_occ"] & tr["valid"]).sum())])
            acc["evidence_path_equal"].append(all(torch.equal(labels[m][E], ctx["ev_label"][E]) for m in labels))
            acc["n_query"].append(len(query)); acc["ids"].append(t)
            if (k + 1) % 25 == 0:
                print(f"  [g16 eval s{shard}] {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.2f}s each", flush=True)
    out = {k: np.stack(v) for k, v in per_anchor.items()}
    out.update({k: np.asarray(v) for k, v in acc.items()}); out.update(pooled)
    out["selection"] = tally.array(methods); out["methods"] = np.asarray(methods); out["selection_fields"] = np.asarray(FIELDS)
    out_path = out_path or os.path.join(G.EVAL, f"eval_shard{shard}of{shards}.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True); np.savez_compressed(out_path, **out)
    meta = {"heads": head_paths, "shard": shard, "shards": shards, "n_anchors": len(ids), "seconds": time.time() - t0,
            "firewall": {"opened": len(audit.opened), "violations": audit.violations}}
    json.dump(meta, open(out_path[:-4] + ".json", "w"), indent=1)
    return meta


__all__ = ["CAND_REGIONS", "TOP_REGIONS", "load_head", "score_heads", "evaluate"]
