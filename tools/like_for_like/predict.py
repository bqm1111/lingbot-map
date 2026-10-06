"""Our frozen prediction pipeline under a chosen window protocol.

``predict_sample`` is the body of ``tools/zero_shot_eval/prediction/pipeline.predict_sample`` with two protocol-dependent
lines: which clip is loaded and which frames' tokens are lifted. Everything else -- frozen geometry, evidence lifting,
B0 / B3 readout, grids, reductions -- is the imported frozen code. ``tools/like_for_like/verify.py`` checks that running
this copy with the causal protocol reproduces the frozen stored predictions array for array.
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from core.datasets import grids as CG
from tools.like_for_like import config as LC, frames as LF
from tools.occany_comparison import config as OC, scoring as SC
from tools.occany_comparison.geometry import WindowClip, build_mapper, hard_channels
from tools.semantic_completion.frozen_geometry import occupancy_and_trunk
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction
from tools.zero_shot_eval.prediction.grid import GridSpec, reduce_any, reduce_min, subvoxels, voxel_centres
from tools.like_for_like.evidence import lift_window_evidence
from tools.zero_shot_eval.prediction.pipeline import FrozenModels, camera_intrinsics, load_models, readout
from tools.zero_shot_eval.prediction.semantics import b3_features, nearest_evidence
from tools.zero_shot_eval import config as ZC
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS


def predict_sample(sample, dataset: str, models: FrozenModels, device, protocol: OC.Protocol = LC.PROTOCOL,
                   calibration: Optional[str] = None) -> CanonicalPrediction:
    """One sample under a temporal window protocol.

    ``calibration``: how the finished map is placed in the benchmark grid. Default = the comparison's declared primary
    (``occany_comparison.config.PRIMARY``): SemanticKITTI "shared" (one transform for both methods) and Occ3D "occany"
    (OccAny's own target-camera-to-voxel transform, so both methods place the map identically)."""
    native = GridSpec.of(CG.EVAL_GRID[dataset])
    calibration = calibration or OC.PRIMARY["calibration"][dataset]
    oc_out = SC.load_occany(dataset, sample.occany_pkl, native.dims)
    T = np.asarray(oc_out.T_cam_to_grid if calibration == "occany" else sample.T_cam_to_grid_ours, np.float64)
    window_keys = [sample.frame_keys[j] for j in sample.window(protocol)]
    clip = WindowClip.load(OC.clip_path(dataset, protocol.name, sample.sample_id)); assert clip.anchor_slot == protocol.anchor_slot
    return predict_window(sample, dataset, models, device, clip, window_keys, oc_out.label.reshape(-1), T,
                          {"protocol": protocol.name, "calibration": calibration})


def predict_window(sample, dataset: str, models: FrozenModels, device, clip: WindowClip, window_keys, label: np.ndarray,
                   T: np.ndarray, meta: Dict) -> CanonicalPrediction:
    """The window-independent body: frozen geometry, evidence lifting, B0 / B3 readout, grids, reductions.

    ``clip`` already fixes the window (five temporal frames, or the six surround views); ``window_keys[q]`` names the token
    map of clip slot q; ``label`` is the flat benchmark ground truth whose ignore voxels are excluded from the query set;
    ``T`` places the map in the benchmark grid. ``tools/like_for_like/verify.py`` checks this path against the frozen
    stored predictions array for array."""
    fine, native = GridSpec.of(CG.PREDICTION_GRID[dataset]), GridSpec.of(CG.EVAL_GRID[dataset])
    ratio = int(round(native.voxel_size / fine.voxel_size)); spec = DATASET_SPECS[dataset]
    mapper = build_mapper(clip, device); S, F_ = hard_channels(mapper, CG.PREDICTION_GRID[dataset], T); del mapper
    x = torch.stack([S.reshape(fine.dims), torch.zeros(fine.dims, dtype=torch.bool, device=device), F_.reshape(fine.dims)]).float()[None]
    occ, trunk = occupancy_and_trunk(models.net, x); O = occ[0]; trunk_flat = trunk[0].reshape(trunk.shape[1], -1)
    ev = lift_window_evidence(clip, [ZC.token_path(dataset, k) for k in window_keys], S, T, fine, device, clip.anchor_slot)
    E = ev.dense(fine)
    nearest, dist = nearest_evidence(E, fine)
    lab = torch.from_numpy(np.asarray(label).reshape(-1).astype(np.int64)).to(device)
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
                                "prediction_grid": fine.name, "ratio": ratio, "history_keys": list(window_keys), **meta})


def run_shard(dataset: str, device: str, shard: int = 0, shards: int = 1, limit: Optional[int] = None) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    LC.ensure_dirs()
    models = load_models(dataset, dev)
    samples = LF.samples(dataset)[shard::shards][:limit]
    t0, written = time.time(), 0
    for i, s in enumerate(samples):
        out = LC.prediction_path(dataset, s.sample_id)
        if os.path.exists(out):
            continue
        predict_sample(s, dataset, models, dev).save(out); written += 1
        if (i + 1) % 50 == 0:
            print(f"  [{dataset} s{shard}] {i + 1}/{len(samples)} {(time.time() - t0) / max(written, 1):.2f}s each", flush=True)
    return {"dataset": dataset, "shard": shard, "shards": shards, "samples": len(samples), "written": written, "seconds": time.time() - t0}


__all__ = ["predict_sample", "predict_window", "run_shard"]
