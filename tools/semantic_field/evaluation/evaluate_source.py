"""Source evaluation on KITTI-360 0006: B0, B3 (seeds 0-2), B4 (seeds 0-2) with one candidate set and one occupancy per anchor.

Per anchor:
    student context (causal) -> candidates of the query list (O | G) & ~E -> B3 labels, B4 labels; evidence voxels keep their own class
    regions: pilot regions + oracle_observed (G & E), completed_observed (G & O & E), candidate-supported / unsupported split of the primary
    per-anchor confusions (bootstrap), pooled distance-bin confusions, occupancy TP / FP / FN
    masked-evidence diagnostic: deterministic 16x16x8 cuboids (p = 0.5, generator seeded by the anchor id) hide evidence;
        hidden voxels are predicted by nearest remaining evidence, B3 over remaining candidates and B4 with the masked input;
        scored against the hidden voxel's own teacher class (label-free) and, where labelled, GT
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as PILOT
from tools.delayed_semantic.gate16 import candidates as CS, config as G16, context as CX
from tools.delayed_semantic.gate16.candidate_regions import candidate_hit
from tools.delayed_semantic.gate16.evaluate import load_head, score_heads
from tools.delayed_semantic.model import voxel_centres
from tools.delayed_semantic.regions import PRIMARY, SEMANTIC_REGIONS, SSC_REGIONS, build_regions
from tools.delayed_semantic.stream import DriveStream
from tools.semantic_completion.dataset import hidden_block_mask
from tools.semantic_completion.evaluation import baseline_nearest as B0N, semantic_metrics as SM
from tools.semantic_completion.evidence import build as G15B, trident as TR
from tools.semantic_completion.frozen_geometry import load_frozen_geometry
from tools.semantic_field import freeze
from tools.semantic_field.config.schema import GEOMETRY_SEED, SEEDS, TARGET_TOKENS, MaskConfig, Paths
from tools.semantic_field.data.feature_mean import load_feature_mean
from tools.semantic_field.data.semantic_evidence import SparseSemanticEvidence
from tools.semantic_field.evaluation.predict import b4_features, load_b4

GRID = PILOT.GRID_DIMS
EXTRA_REGIONS = ("oracle_observed", "completed_observed", "primary_candidate_supported", "primary_candidate_unsupported")
REGIONS = SEMANTIC_REGIONS + EXTRA_REGIONS + SSC_REGIONS


def method_names():
    return ["B0"] + [f"B3_s{s}" for s in SEEDS] + [f"B4_s{s}" for s in SEEDS]


def evaluate(device: str, shard: int = 0, shards: int = 1, limit: Optional[int] = None, out_path: Optional[str] = None, paths: Paths = Paths()) -> Dict:
    freeze.check("source_predeclaration"); freeze.check("centering"); freeze.check("source_code")
    dev = torch.device(device); torch.cuda.set_device(dev)
    text, _, _ = TR.load_text_embeddings(); text = text.to(dev); mu = load_feature_mean(dev)
    stream = DriveStream(PILOT.DEV_DRIVE); net = load_frozen_geometry(GEOMETRY_SEED, dev)
    b3 = {f"B3_s{s}": load_head(G16.head_path("B3", s), dev) for s in SEEDS}
    b4 = {f"B4_s{s}": load_b4(paths.checkpoint(s, 6000), dev) for s in SEEDS}
    methods = method_names(); mask_cfg = MaskConfig()
    ids = G15B.anchor_ids(PILOT.DEV_DRIVE)[shard::shards][:limit]
    nD = len(PILOT.DISTANCE_NAMES)
    per = {f"conf__{m}__{r}": [] for m in methods for r in REGIONS}
    per.update({"occ": [], "evidence_path_equal": [], "ids": []})
    masked = {f"masked_teacher__{m}": np.zeros(2, np.int64) for m in ["B0", *b3, *b4]}
    masked.update({f"masked_gt__{m}": np.zeros((19, 19), np.int64) for m in ["B0", *b3, *b4]})
    dist = {f"dist__{m}": np.zeros((nD, 19, 19), np.int64) for m in methods}
    t0 = time.time()
    with FileAudit(patterns=PILOT.EVAL_FORBIDDEN + TARGET_TOKENS, strict=True) as audit:
        for k, t in enumerate(ids):
            tr = CX.truth(t, dev); gt = tr["sem_gt"]; G = (gt >= 1) & (gt <= 18)
            ctx = CX.student_context(stream, t, net, text, dev); E, O = ctx["E"], ctx["O"]
            query = ((O | G) & ~E).nonzero(as_tuple=True)[0]
            chunks = CX.query_candidates(ctx, query, text); cs = CX.cat_sets([c for _, c in chunks])
            labels = {"B0": ctx["b0"].clone()}
            for m, v in score_heads(b3, ctx, chunks, text).items():
                full = torch.zeros(PILOT.N_VOXELS, dtype=torch.long, device=dev); full[query] = v["label"]; labels[m] = full
            for m, model in b4.items():
                full = torch.zeros(PILOT.N_VOXELS, dtype=torch.long, device=dev); full[query] = TR.classify(b4_features(model, ctx, query, mu, GRID), text); labels[m] = full
            for m in labels:
                labels[m] = torch.where(E, ctx["ev_label"], labels[m])
            reg = build_regions(gt, E, ctx["Ec"], ctx["S"], ctx["F"], O, tr["R"], tr["Rh"])
            gq = gt[query].clamp(0, 18); hit = candidate_hit(cs, gq) & (gq >= 1)
            HIT = torch.zeros_like(E); HIT[query[hit]] = True
            reg.update({"oracle_observed": G & E, "completed_observed": G & O & E,
                        "primary_candidate_supported": reg[PRIMARY] & HIT, "primary_candidate_unsupported": reg[PRIMARY] & ~HIT})
            dbin = torch.from_numpy(np.digitize(ctx["dist_m"], np.asarray(PILOT.DISTANCE_EDGES_M))).to(dev)
            for m, lab in labels.items():
                for r in SEMANTIC_REGIONS + EXTRA_REGIONS:
                    per[f"conf__{m}__{r}"].append(SM.confusion(gt, lab, reg[r]))
                ssc = torch.where(O, lab, torch.zeros_like(lab))
                for r in SSC_REGIONS:
                    per[f"conf__{m}__{r}"].append(SM.confusion(gt, ssc, reg[r]))
                for b in range(nD):
                    dist[f"dist__{m}"][b] += SM.confusion(gt, lab, reg[PRIMARY] & (dbin == b))
            per["occ"].append([int((O & tr["gt_occ"] & tr["valid"]).sum()), int((O & ~tr["gt_occ"] & tr["valid"]).sum()), int((~O & tr["gt_occ"] & tr["valid"]).sum())])
            per["evidence_path_equal"].append(all(torch.equal(labels[m][E], ctx["ev_label"][E]) for m in labels)); per["ids"].append(t)
            # ---- masked-evidence diagnostic ----
            ev = ctx["ev"]; gen = torch.Generator().manual_seed(int(t))
            hide = hidden_block_mask(ev.flat_index, GRID, mask_cfg.block_voxels, mask_cfg.hide_probability, gen)
            if hide.any() and (~hide).any():
                hq = ev.flat_index[hide]; teacher = ctx["ev_label"][hq]; gh = gt[hq]; lab_h = (gh >= 1) & (gh <= 18)
                vis_mask = torch.zeros_like(E); vis_mask[ev.flat_index[~hide]] = True
                near, _ = B0N.nearest_evidence(vis_mask)
                preds = {"B0": ctx["ev_label"][torch.from_numpy(near).to(dev)[hq]]}
                vis_rows = (~hide).nonzero(as_tuple=True)[0]
                mchunks = [(hq[s:s + CX.CHUNK], CS.build(hq[s:s + CX.CHUNK], ctx["evd"], text, allowed_rows=vis_rows)) for s in range(0, len(hq), CX.CHUNK)]
                for m, v in score_heads(b3, ctx, mchunks, text).items():
                    preds[m] = v["label"]
                visible = SparseSemanticEvidence(ev.flat_index[~hide], ev.feature[~hide], ev.count[~hide])
                for m, model in b4.items():
                    preds[m] = TR.classify(b4_features(model, ctx, hq, mu, GRID, evidence=visible), text)
                for m, p in preds.items():
                    masked[f"masked_teacher__{m}"] += [len(hq), int((p == teacher).sum())]
                    masked[f"masked_gt__{m}"] += SM.confusion(gh, p, lab_h)
            if (k + 1) % 25 == 0:
                print(f"  [B4 source s{shard}] {k + 1}/{len(ids)} {(time.time() - t0) / (k + 1):.2f}s each", flush=True)
    out = {k: np.stack(v) for k, v in per.items() if k.startswith("conf__")}
    out.update({"occ": np.asarray(per["occ"], np.int64), "evidence_path_equal": np.asarray(per["evidence_path_equal"]), "ids": np.asarray(per["ids"])})
    out.update(masked); out.update(dist); out["methods"] = np.asarray(methods)
    out_path = out_path or os.path.join(paths.eval, f"source_eval_shard{shard}of{shards}.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True); np.savez_compressed(out_path, **out)
    meta = {"shard": shard, "shards": shards, "n_anchors": len(ids), "seconds": time.time() - t0,
            "b4_checkpoints": {m: paths.checkpoint(s, 6000) for s, m in zip(SEEDS, b4)}, "firewall": {"opened": len(audit.opened), "violations": audit.violations}}
    json.dump(meta, open(out_path[:-4] + ".json", "w"), indent=1)
    return meta


__all__ = ["REGIONS", "method_names", "evaluate"]
