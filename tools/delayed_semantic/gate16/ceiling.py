"""Candidate ceilings on 0006, reconstructed from the frozen candidate rule BEFORE any B3 prediction is scored.

Per region (primary = GT occupied & predicted occupied & no evidence, and its pilot splits):
    C0       candidate 0 (nearest) readout class == GT
    C16      any of the 16 candidates' readout class == GT  (candidate-hit)
    oracle   GT where hit, else the class of the candidate with the largest p_i[GT]  (accuracy, 19x19 confusion)
Also per class, per distance bin (pilot EDT distance to nearest evidence) and per reveal bin; the selection threshold
T = N + 0.5 (O - N) is written into candidate_ceiling.json and frozen (stage 'ceiling').
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as PC
from tools.delayed_semantic.analysis import fast_miou
from tools.delayed_semantic.gate16 import config as G, context as CX
from tools.delayed_semantic.gate16.candidate_regions import candidate_hit, oracle_label
from tools.delayed_semantic.regions import build_regions
from tools.delayed_semantic.stream import DriveStream
from tools.semantic_completion.evaluation import semantic_metrics as SM
from tools.semantic_completion.evidence import build as G15B, trident as TR
from tools.semantic_completion.frozen_geometry import load_frozen_geometry

REGIONS = ("completed_never_observed", "completed_geo_observed", "completed_geo_unobserved", "completed_later_revealed",
           "completed_revealed_unseen", "completed_not_revealed")
NAMES = ["car", "bicycle", "motorcycle", "truck", "other vehicle", "person", "road", "parking", "sidewalk", "other ground", "building",
         "fence", "vegetation", "terrain", "pole", "traffic sign", "other structure", "other object"]


def primary_candidates(ctx, tr, text):
    """Candidate set of every primary voxel, computed on the pilot query list ((O | G) & ~E) with the pilot chunking."""
    G_ = (tr["sem_gt"] >= 1) & (tr["sem_gt"] <= 18)
    query = ((ctx["O"] | G_) & ~ctx["E"]).nonzero(as_tuple=True)[0]
    cs = CX.cat_sets([c for _, c in CX.query_candidates(ctx, query, text)])
    return query, cs


def run(device: str, limit: int | None = None) -> Dict:
    G.ensure_dirs(); dev = torch.device(device); torch.cuda.set_device(dev)
    text, _, _ = TR.load_text_embeddings(); text = text.to(dev)
    net = load_frozen_geometry(PC.GEOMETRY_SEED, dev); stream = DriveStream(PC.DEV_DRIVE)
    ids = G15B.anchor_ids(PC.DEV_DRIVE)[:limit]
    acc = {r: np.zeros(4, np.int64) for r in REGIONS}                       # voxels, C0, hit, oracle
    conf = {k: np.zeros((19, 19), np.int64) for k in ("C0", "oracle", "C0_hit", "oracle_hit", "C0_miss", "oracle_miss")}
    per_class = np.zeros((19, 4), np.int64)
    nD, nR = len(PC.DISTANCE_NAMES), len(PC.REVEAL_NAMES)
    dist = np.zeros((nD, 4), np.int64); reveal = np.zeros((nR, 4), np.int64)
    t0 = time.time()
    with FileAudit(patterns=PC.EVAL_FORBIDDEN, strict=True) as audit:
        for k, t in enumerate(ids):
            ctx = CX.student_context(stream, t, net, text, dev); tr = CX.truth(t, dev)
            query, cs = primary_candidates(ctx, tr, text)
            reg = build_regions(tr["sem_gt"], ctx["E"], ctx["Ec"], ctx["S"], ctx["F"], ctx["O"], tr["R"], tr["Rh"])
            prim = reg["completed_never_observed"][query]
            gq = tr["sem_gt"][query].clamp(0, 18)
            gp = gq[prim]; csp = cs.subset(prim)
            c0 = csp.classes[:, 0]; hit = candidate_hit(csp, gp); orc = oracle_label(csp, gp)
            row = torch.stack([torch.ones_like(hit), c0 == gp, hit, orc == gp], 1).long()
            for r in REGIONS:
                m = reg[r][query][prim]
                acc[r] += row[m].sum(0).cpu().numpy()
            for name, lab, m in (("C0", c0, torch.ones_like(hit)), ("oracle", orc, torch.ones_like(hit)), ("C0_hit", c0, hit), ("oracle_hit", orc, hit),
                                 ("C0_miss", c0, ~hit), ("oracle_miss", orc, ~hit)):
                conf[name] += np.bincount((gp[m] * 19 + lab[m]).cpu().numpy(), minlength=361).reshape(19, 19)
            per_class += torch.zeros(19, 4, dtype=torch.long, device=dev).index_add_(0, gp, row).cpu().numpy()
            dbin = torch.from_numpy(np.digitize(ctx["dist_m"], np.asarray(PC.DISTANCE_EDGES_M))).to(dev)[csp.query_flat]
            dist += torch.zeros(nD, 4, dtype=torch.long, device=dev).index_add_(0, dbin, row).cpu().numpy()
            reveal += torch.zeros(nR, 4, dtype=torch.long, device=dev).index_add_(0, tr["rbin"][csp.query_flat], row).cpu().numpy()
            if (k + 1) % 100 == 0:
                print(f"  [ceiling] {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.2f}s each", flush=True)
    frac = lambda v: {"voxels": int(v[0]), "C0_acc": v[1] / max(v[0], 1), "C16_hit": v[2] / max(v[0], 1), "oracle_acc": v[3] / max(v[0], 1)}
    P = acc["completed_never_observed"]
    N, O = P[1] / P[0], P[3] / P[0]
    rec = {"note": "reconstructed candidate ceilings on 0006 (evaluation only); frozen before any B3 prediction is scored",
           "n_anchors": len(ids), "regions": {r: frac(v) for r, v in acc.items()},
           "miou": {k: fast_miou(v) for k, v in conf.items()}, "confusions": {k: v.tolist() for k, v in conf.items()},
           "candidate_miss_fraction": 1 - P[2] / P[0],
           "per_class": {NAMES[c - 1]: frac(per_class[c]) for c in range(1, 19)},
           "per_distance": {PC.DISTANCE_NAMES[b]: frac(dist[b]) for b in range(nD)}, "per_reveal": {PC.REVEAL_NAMES[b]: frac(reveal[b]) for b in range(nR)},
           "selection_threshold": {"N_nearest_candidate_acc": N, "O_oracle_acc": O, "H": O - N, "T": N + 0.5 * (O - N), "required_recovered_fraction": 0.5},
           "history_wide_context": json.load(open(os.path.join(PC.ARTIFACT_DIR, "diagnostic_candidate_ceiling.json")))["regions"]["primary"]["anywhere_ceiling"],
           "seconds": time.time() - t0, "firewall": {"opened": len(audit.opened), "violations": audit.violations}}
    if limit is None:
        json.dump(rec, open(os.path.join(G.ARTIFACT_DIR, "candidate_ceiling.json"), "w"), indent=1)
    return rec


__all__ = ["REGIONS", "NAMES", "primary_candidates", "run"]
