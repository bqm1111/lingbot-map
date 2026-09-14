"""Source evaluation on KITTI-360 0006: B0 (nearest evidence), B1 (propagation head, every seed), teacher ceiling.

Per anchor (shared by all seeds): hard map, evidence E, teacher label of every evidence voxel, EDT to E
(B0 labels + propagation distance), GT (evaluation-only ``sem_gt``). Per seed: frozen occupancy O + trunk
(one no-grad forward), head features on the full grid with ALL evidence given, open-vocabulary readout.
Also a label-free held-out-evidence test: evidence hidden in deterministic (8, 8, 4) blocks (seeded by the
anchor id) and the cosine to the teacher at the hidden evidence voxels, B1 vs B0 (nearest VISIBLE evidence).

Saved per anchor: confusion matrices for every (method, region), for the primary region per distance bin,
and occupancy TP/FP/FN on the Gate 12 regions' ``full`` mask (must equal the baseline exactly).
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from core.datasets.audit import FileAudit
from tools.semantic_completion import config as C
from tools.semantic_completion.dataset import hidden_block_mask
from tools.semantic_completion.evaluation import baseline_nearest as B0, semantic_metrics as SM
from tools.semantic_completion.evaluation.regions import SEMANTIC_REGIONS, SSC_REGIONS, PRIMARY, build_regions
from tools.semantic_completion.evidence.build import anchor_ids
from tools.semantic_completion.evidence.lift import load_evidence
from tools.semantic_completion.evidence.trident import classify, load_text_embeddings
from tools.semantic_completion.frozen_geometry import geometry_input, load_frozen_geometry, load_hard_map, occupancy_and_trunk
from tools.semantic_completion.model.propagation_head import PropagationHead

REGIONS_ROOT = "/media/SSD1/MINH_DATASETS/lingbot_gate12_suf/regions"      # Gate 12 occupancy regions (valid, occupied)
METHODS = ("B0", "B1")


def load_head(path: str, device) -> PropagationHead:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = PropagationHead(C.HeadConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["head_config"].items()}))
    head.load_state_dict(ck["state_dict"]); return head.to(device).eval()


@torch.no_grad()
def head_labels(head: PropagationHead, trunk: torch.Tensor, index: torch.Tensor, feature: torch.Tensor, text: torch.Tensor,
                gather: Optional[torch.Tensor] = None):
    """Full-grid head readout: ``(label [N_VOXELS] long, feature at ``gather`` [k, 1024] or None)``."""
    dense = head.scatter_evidence(index, feature, C.GRID_DIMS)[None]
    mask = torch.zeros(C.N_VOXELS, device=trunk.device); mask[index] = 1.0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = head(trunk, dense, mask.reshape(1, 1, *C.GRID_DIMS))
    flat = out[0].reshape(C.TEACHER_DIM, -1)
    label = torch.empty(C.N_VOXELS, dtype=torch.long, device=trunk.device)
    for s in range(0, C.N_VOXELS, 1 << 19):                                  # chunked cosine readout
        label[s:s + (1 << 19)] = classify(flat[:, s:s + (1 << 19)].T.float(), text)
    feats = F.normalize(flat[:, gather].T.float(), dim=-1) if gather is not None else None
    return label, feats


def evaluate_source(head_paths: Dict[int, str], device: str, stride: int = 1, out_path: Optional[str] = None,
                    paths: C.Paths = C.Paths(), shard: int = 0, shards: int = 1) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    text, phrases, labels = load_text_embeddings(paths.text_embeddings); text = text.to(dev)
    nets = {s: load_frozen_geometry(s, dev, paths) for s in head_paths}
    heads = {s: load_head(p, dev) for s, p in head_paths.items()}
    ids = anchor_ids(C.VAL_DRIVE, paths)[::stride][shard::shards]
    nD = len(C.DISTANCE_NAMES); tcfg = C.TrainConfig()
    rec = {"ids": [], "occ": {s: [] for s in heads}, "teacher_obs": [], "heldout": {s: [] for s in heads}, "n_evidence": []}
    keys = [f"{m}_seed{s}" for s in heads for m in METHODS]
    conf = {k_: {r: [] for r in SEMANTIC_REGIONS + SSC_REGIONS} for k_ in keys}; conf_dist = {k_: [] for k_ in keys}
    t0 = time.time()
    with FileAudit(patterns=C.EVAL_FORBIDDEN_TOKENS, strict=True) as audit:
        for k, t in enumerate(ids):
            S, F_ = load_hard_map(C.VAL_DRIVE, t, dev, paths); x = geometry_input(S, F_)
            ev = load_evidence(paths.evidence_path(C.VAL_DRIVE, t), dev)
            E = ev.dense_mask()
            with np.load(paths.eval_mask_path(t)) as z:
                sem_gt = torch.from_numpy(z["sem_gt"].astype(np.int64)).to(dev)
            with np.load(os.path.join(REGIONS_ROOT, C.VAL_DRIVE, f"{t:05d}.npz")) as z:
                valid = torch.from_numpy(np.unpackbits(z["valid"])[:C.N_VOXELS].astype(bool)).to(dev)
                gt_occ = torch.from_numpy(np.unpackbits(z["occupied"])[:C.N_VOXELS].astype(bool)).to(dev) & valid
            ev_label = torch.zeros(C.N_VOXELS, dtype=torch.long, device=dev); ev_label[ev.flat_index] = classify(ev.feature, text)
            nearest, dist_m = B0.nearest_evidence(E)
            b0 = B0.nearest_label(ev_label, nearest)
            dbin = torch.from_numpy(np.digitize(dist_m, np.asarray(C.DISTANCE_EDGES_M))).to(dev)
            # held-out evidence blocks (label-free propagation test), deterministic per anchor
            gen = torch.Generator().manual_seed(int(t))
            hid = hidden_block_mask(ev.flat_index, C.GRID_DIMS, tcfg.hide_block_vox, tcfg.hide_block_prob, gen)
            vis_mask = torch.zeros(C.N_VOXELS, dtype=torch.bool, device=dev); vis_mask[ev.flat_index[~hid]] = True
            near_vis, _ = B0.nearest_evidence(vis_mask)
            feat_dense_idx = torch.full((C.N_VOXELS,), -1, dtype=torch.long, device=dev); feat_dense_idx[ev.flat_index] = torch.arange(len(ev.flat_index), device=dev)
            hidden_idx = ev.flat_index[hid]
            b0_vis_feat = ev.feature[feat_dense_idx[torch.from_numpy(near_vis).to(dev)[hidden_idx]]] if hid.any() and (~hid).any() else None
            rec["teacher_obs"].append(SM.confusion(sem_gt, ev_label, ((sem_gt >= 1) & (sem_gt <= 18)) & E))
            rec["n_evidence"].append(int(len(ev.flat_index)))
            for s in heads:
                occ, trunk = occupancy_and_trunk(nets[s], x); occ = occ[0]
                rec["occ"][s].append([int((occ & gt_occ & valid).sum()), int((occ & ~gt_occ & valid).sum()), int((~occ & gt_occ & valid).sum())])
                b1, _ = head_labels(heads[s], trunk, ev.flat_index, ev.feature, text)
                reg = build_regions(sem_gt, E, occ)
                # B0 labels are seed-independent, but its regions depend on the seed's occupancy: stored per seed
                for key, lab in ((f"B0_seed{s}", b0), (f"B1_seed{s}", b1)):
                    for r in SEMANTIC_REGIONS:
                        conf[key][r].append(SM.confusion(sem_gt, lab, reg[r]))
                    ssc_pred = torch.where(occ, lab, torch.zeros_like(lab))
                    for r in SSC_REGIONS:
                        conf[key][r].append(SM.confusion(sem_gt, ssc_pred, reg[r]))
                    conf_dist[key].append(np.stack([SM.confusion(sem_gt, lab, reg[PRIMARY] & (dbin == b)) for b in range(nD)]))
                if b0_vis_feat is not None:
                    vis_idx = ev.flat_index[~hid]
                    _, b1_hidden_feat = head_labels(heads[s], trunk, vis_idx, ev.feature[~hid], text, gather=hidden_idx)
                    teacher = ev.feature[hid]
                    rec["heldout"][s].append([int(hid.sum()), float((b1_hidden_feat * teacher).sum(-1).sum()), float((b0_vis_feat * teacher).sum(-1).sum()),
                                              int((classify(b1_hidden_feat, text) == classify(teacher, text)).sum()),
                                              int((classify(b0_vis_feat, text) == classify(teacher, text)).sum())])
            rec["ids"].append(int(t))
            if (k + 1) % 50 == 0:
                print(f"  [eval s{shard}] {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.2f}s each", flush=True)
    out = {"ids": np.asarray(rec["ids"]), "teacher_obs": np.stack(rec["teacher_obs"]), "n_evidence": np.asarray(rec["n_evidence"])}
    for s in heads:
        out[f"occ_seed{s}"] = np.asarray(rec["occ"][s], np.int64)
        out[f"heldout_seed{s}"] = np.asarray(rec["heldout"][s], np.float64)
    for key in keys:
        for r, v in conf[key].items():
            out[f"conf__{key}__{r}"] = np.stack(v)                    # [anchors, 19, 19]
        out[f"confdist__{key}"] = np.stack(conf_dist[key])             # [anchors, distance bins, 19, 19]
    out_path = out_path or os.path.join(paths.source_eval, f"source_eval_stride{stride}_shard{shard}.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **out)
    meta = {"head_checkpoints": {s: os.path.relpath(p, C.REPO_ROOT) for s, p in head_paths.items()}, "stride": stride, "shard": shard, "shards": shards,
            "n_anchors": len(ids), "seconds": time.time() - t0, "firewall": {"opened": len(audit.opened), "violations": audit.violations}}
    json.dump(meta, open(out_path[:-4] + ".json", "w"), indent=1)
    print(f"[eval s{shard}] {len(ids)} anchors in {meta['seconds']/60:.1f} min -> {out_path}", flush=True)
    return meta


__all__ = ["METHODS", "load_head", "head_labels", "evaluate_source", "REGIONS_ROOT"]
