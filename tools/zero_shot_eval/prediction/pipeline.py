"""Canonical predictions for target samples: frozen causal geometry, evidence, B0 and B3 (seeds 0-2). Run once per sample.

Per sample (target frame t, history t-4..t):
    causal clip (``lingbot_occany_comparison/clips/<dataset>/causal``) -> hard mapper -> S, F on the prediction grid
    -> CompletionUNet geometry seed 0 -> occupancy O and trunk -> Trident-H tokens of t-4..t -> evidence (Gate 15 rule)
    -> B0 feature = nearest evidence feature; B3 feature = association head output (evidence voxels keep their own feature)
    -> evaluation grid: occupancy / observation / evidence by the any-subvoxel rule; the label of a native voxel is the readout
       of the normalised mean feature of its predicted-occupied subvoxels (all subvoxels if none is occupied).

The official label volume is read only to list the query voxels (valid & (predicted occupied | GT occupied)); every voxel's
prediction is computed independently of which other voxels are queried.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from core.datasets import grids as CG, sources as S_
from tools.delayed_semantic.gate16.evaluate import load_head
from tools.delayed_semantic.model import EvidenceAssociation
from tools.occany_comparison import config as OC, scoring as SC
from tools.occany_comparison.geometry import WindowClip, build_mapper, hard_channels
from tools.semantic_completion.frozen_geometry import load_frozen_geometry, occupancy_and_trunk
from tools.zero_shot_eval import config as C, samples as ZS
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS
from tools.zero_shot_eval.prediction.evidence import lift_evidence
from tools.zero_shot_eval.prediction.grid import GridSpec, reduce_any, reduce_min, subvoxels, voxel_centres
from tools.zero_shot_eval.prediction.semantics import b3_features, nearest_evidence


@dataclass
class FrozenModels:
    net: torch.nn.Module
    heads: Dict[str, EvidenceAssociation]
    text: torch.Tensor                 # [C, 1024] unit, Trident query features of the frozen phrases
    label_of_channel: torch.Tensor     # [C] official label id of each text channel
    checkpoints: Dict[str, str]


def load_models(dataset: str, device) -> FrozenModels:
    with np.load(C.TEXT_EMBEDDINGS.format(dataset=dataset)) as z:
        text = F.normalize(torch.from_numpy(z["text"]).float(), dim=-1).to(device)
        labels = torch.from_numpy(z["labels"].astype(np.int64)).to(device)
    ck = {f"B3_s{s}": C.b3_checkpoint(s) for s in C.B3_SEEDS}
    return FrozenModels(load_frozen_geometry(C.GEOMETRY_SEED, device), {m: load_head(p, device) for m, p in ck.items()}, text, labels, ck)


def readout(summed_features: torch.Tensor, models: FrozenModels) -> torch.Tensor:
    return models.label_of_channel[(F.normalize(summed_features, dim=-1) @ models.text.T).argmax(-1)]


def camera_intrinsics(sample) -> np.ndarray:
    if sample.K_native is not None:
        return np.asarray(sample.K_native[sample.target_pos], np.float64)
    _, K = S_._sk_calib(C.REPO_ROOT, sample.segment)
    return np.asarray(K, np.float64)


@torch.no_grad()
def predict_sample(sample, dataset: str, models: FrozenModels, device) -> CanonicalPrediction:
    fine, native = GridSpec.of(CG.PREDICTION_GRID[dataset]), GridSpec.of(CG.EVAL_GRID[dataset])
    ratio = int(round(native.voxel_size / fine.voxel_size)); spec = DATASET_SPECS[dataset]
    T = np.asarray(sample.T_cam_to_grid_ours, np.float64)
    clip = WindowClip.load(OC.clip_path(dataset, "causal", sample.sample_id)); assert clip.anchor_slot == 4
    mapper = build_mapper(clip, device); S, F_ = hard_channels(mapper, CG.PREDICTION_GRID[dataset], T); del mapper
    x = torch.stack([S.reshape(fine.dims), torch.zeros(fine.dims, dtype=torch.bool, device=device), F_.reshape(fine.dims)]).float()[None]
    occ, trunk = occupancy_and_trunk(models.net, x); O = occ[0]; trunk_flat = trunk[0].reshape(trunk.shape[1], -1)
    ev = lift_evidence(clip, [C.token_path(dataset, f["key"]) for f in ZS.history_frames(sample)], S, T, fine, device)
    E = ev.dense(fine)
    nearest, dist = nearest_evidence(E, fine)
    label = SC.load_occany(dataset, sample.occany_pkl, native.dims).label.reshape(-1)
    lab = torch.from_numpy(label.astype(np.int64)).to(device)
    occ_n = reduce_any(O, fine, ratio); gt_occ = (lab != 255) & (lab != spec.empty_class)
    q_native = ((lab != 255) & (occ_n | gt_occ)).nonzero(as_tuple=True)[0]
    sub = subvoxels(q_native, native, fine, ratio)
    occ_sub = O[sub]; members = torch.where(occ_sub.any(1, keepdim=True), occ_sub, torch.ones_like(occ_sub))
    mi, mj = members.nonzero(as_tuple=True); member = sub[mi, mj]
    labels: Dict[str, torch.Tensor] = {}
    if len(ev.flat) == 0:                                   # no semantic evidence at all: no identity can be transferred
        for m in ["B0"] + list(models.heads):
            labels[m] = torch.full((len(q_native),), spec.empty_class, dtype=torch.long, device=device)
    else:
        row = torch.full((fine.n_voxels,), -1, dtype=torch.long, device=device); row[ev.flat] = torch.arange(len(ev.flat), device=device)
        in_E = E[member]
        aggregate = lambda feats: readout(torch.zeros(len(q_native), feats.shape[1], device=device).index_add_(0, mi, feats), models)
        labels["B0"] = aggregate(ev.feature[row[torch.from_numpy(nearest).to(device)[member]]])
        q_fine = torch.unique(member[~in_E])
        evd = ev.scorer_inputs(trunk_flat, voxel_centres(fine, ev.flat)); masks = torch.stack([F_, S, O]).reshape(3, *fine.dims)
        for name, head in models.heads.items():
            f = b3_features(head, fine, q_fine, evd, trunk_flat, masks)
            pos = torch.searchsorted(q_fine, member).clamp_(max=max(len(q_fine) - 1, 0))
            own = ev.feature[row[member].clamp(min=0)]
            labels[name] = aggregate(torch.where(in_E[:, None], own, f[pos]) if len(q_fine) else own)
    dist_n = reduce_min(torch.from_numpy(dist).to(device), fine, ratio)
    return CanonicalPrediction(sample.sample_id, dataset, native.dims, native.voxel_size, native.origin, native.frame, T, camera_intrinsics(sample),
                               DATASET_SPECS[dataset].image_width, occ_n.cpu().numpy(), reduce_any(S | F_, fine, ratio).cpu().numpy(),
                               reduce_any(E, fine, ratio).cpu().numpy(), q_native.cpu().numpy(), dist_n[q_native].float().cpu().numpy(),
                               {m: v.to(torch.uint8).cpu().numpy() for m, v in labels.items()},
                               {"n_evidence": int(len(ev.flat)), "clip_scale": clip.scale_m_per_unit, "checkpoints": models.checkpoints,
                                "prediction_grid": fine.name, "ratio": ratio, "history_keys": [f["key"] for f in ZS.history_frames(sample)]})


def run_shard(dataset: str, device: str, shard: int = 0, shards: int = 1, limit: int | None = None) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    models = load_models(dataset, dev)
    samples = ZS.evaluated_samples(dataset)[shard::shards][:limit]
    t0, written = time.time(), 0
    for i, s in enumerate(samples):
        out = C.prediction_path(dataset, s.sample_id)
        if os.path.exists(out):
            continue
        predict_sample(s, dataset, models, dev).save(out); written += 1
        if (i + 1) % 50 == 0:
            print(f"  [{dataset} s{shard}] {i + 1}/{len(samples)} {(time.time() - t0) / max(written, 1):.2f}s each", flush=True)
    return {"dataset": dataset, "shard": shard, "shards": shards, "samples": len(samples), "written": written, "seconds": time.time() - t0}


__all__ = ["FrozenModels", "load_models", "readout", "predict_sample", "run_shard"]
