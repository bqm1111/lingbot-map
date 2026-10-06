"""POST-HOC diagnostic (written after the decision; not part of any criterion): the evidence-transfer ceiling.

For each 0006 primary-region voxel (GT occupied & predicted occupied & no evidence) and its geometric split:
    nearest1   the nearest evidence voxel's teacher class equals GT            (= B0's accuracy)
    any16      any of the 16 candidates the head can weight carries the GT class (ceiling of B1/B2's selection)
    anywhere   the GT class appears on any evidence voxel of H_t              (ceiling of any evidence-transfer rule)
An evidence-transfer method cannot exceed ``any16`` (B1/B2) or ``anywhere`` (any copy rule) in accuracy.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as C
from tools.delayed_semantic.memory import load_observed
from tools.delayed_semantic.model import nearest_candidates, voxel_centres
from tools.semantic_completion.evidence import build as G15B, trident as TR
from tools.semantic_completion.frozen_geometry import geometry_input, load_frozen_geometry, load_hard_map, occupancy_and_trunk


@torch.no_grad()
def candidate_ceiling(device: str, paths: C.Paths = C.Paths()) -> dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    text, _, _ = TR.load_text_embeddings(); text = text.to(dev)
    net = load_frozen_geometry(C.GEOMETRY_SEED, dev)
    K = C.ModelConfig().k_candidates
    names = ("primary", "geo_observed", "geo_unobserved")
    tot = {n: np.zeros(4, np.int64) for n in names}                       # voxels, nearest1, any16, anywhere
    per_class = np.zeros((19, 4), np.int64)
    with FileAudit(patterns=C.EVAL_FORBIDDEN, strict=True):
        for t in G15B.anchor_ids(C.DEV_DRIVE):
            S, F_ = load_hard_map(C.DEV_DRIVE, t, dev); occ, _ = occupancy_and_trunk(net, geometry_input(S, F_)); O = occ[0]
            ev = load_observed(C.DEV_DRIVE, t, dev); E = ev.dense_mask(); lab = TR.classify(ev.feature, text)
            with np.load(paths.g15.eval_mask_path(t)) as z:
                gt = torch.from_numpy(z["sem_gt"].astype(np.int64)).to(dev)
            prim = (gt >= 1) & (gt <= 18) & O & ~E
            q = prim.nonzero(as_tuple=True)[0]
            if not len(q) or len(ev.flat_index) < K:
                continue
            idx, _ = nearest_candidates(voxel_centres(q), voxel_centres(ev.flat_index), K)
            g = gt[q]; cl = lab[idx]
            present = torch.zeros(19, dtype=torch.bool, device=dev); present[lab] = True
            hit = torch.stack([torch.ones_like(g, dtype=torch.bool), cl[:, 0] == g, (cl == g[:, None]).any(1), present[g]], 1).long()
            geo = (S | F_)[q]
            for n, m in (("primary", torch.ones_like(geo)), ("geo_observed", geo), ("geo_unobserved", ~geo)):
                tot[n] += hit[m].sum(0).cpu().numpy()
            per_class += torch.zeros(19, 4, dtype=torch.long, device=dev).index_add_(0, g, hit).cpu().numpy()
    names18 = ["car", "bicycle", "motorcycle", "truck", "other vehicle", "person", "road", "parking", "sidewalk", "other ground", "building",
               "fence", "vegetation", "terrain", "pole", "traffic sign", "other structure", "other object"]
    rec = {"note": "post-hoc diagnostic, not a decision criterion",
           "regions": {n: {"voxels": int(v[0]), "nearest1_acc": v[1] / v[0], "any16_ceiling": v[2] / v[0], "anywhere_ceiling": v[3] / v[0]} for n, v in tot.items()},
           "per_class": {names18[c - 1]: {"voxels": int(per_class[c, 0]), "nearest1_acc": per_class[c, 1] / max(per_class[c, 0], 1),
                                          "any16_ceiling": per_class[c, 2] / max(per_class[c, 0], 1), "anywhere_ceiling": per_class[c, 3] / max(per_class[c, 0], 1)}
                         for c in range(1, 19)}}
    json.dump(rec, open(os.path.join(C.ARTIFACT_DIR, "diagnostic_candidate_ceiling.json"), "w"), indent=1)
    return rec


__all__ = ["candidate_ceiling"]
