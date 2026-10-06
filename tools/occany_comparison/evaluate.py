"""Score OccAny and the frozen baseline on every OccAny sample, all variants, one shard at a time.

Per sample: OccAny's saved occupancy and stored label -> (independent label check) -> for each protocol
whose window exists: clip -> hard mapper (built once) -> for each benchmark placement: ``[S, 0, F]`` ->
each frozen seed -> occupied := final > 0 -> evaluation grid -> raw / pooled -> TP/FP/FN by OccAny's
``SSCMetrics`` and by a GPU tally. The raw map ``S`` is scored alongside as the *map-only* reference.

Output per shard (``eval/<dataset>_shardKofN.npz``): ``official`` / ``fast`` counts ``[n_samples, n_keys, 3]``
(-1 where a protocol cannot run) plus per-sample metadata. Nothing is selected here; the entry point
refuses to run unless the baseline's source freeze is intact.
"""
from __future__ import annotations

import os
import time
from typing import Dict

import numpy as np
import torch

from core.datasets import grids as G
from core.evaluation.regions import occ_to_eval_grid
from core.model.net import load_checkpoint
from tools.baseline_reconstruction import config as BR
from tools.baseline_reconstruction import freeze as FZ
from tools.baseline_reconstruction.current.evaluate_source import predict
from tools.baseline_reconstruction.current.geometry import network_input
from tools.occany_comparison import config as C
from tools.occany_comparison import scoring as SC
from tools.occany_comparison.geometry import WindowClip, build_mapper, hard_channels
from tools.occany_comparison.samples import build_samples


def evaluate_shard(dataset: str, device: str, shard: int = 0, shards: int = 1, limit: int | None = None) -> Dict:
    frozen = FZ.require_source_freeze()
    dev = torch.device(device); torch.cuda.set_device(dev)
    pgrid, egrid = G.PREDICTION_GRID[dataset], G.EVAL_GRID[dataset]
    edims = tuple(int(d) for d in egrid.dims)
    paths = BR.CurrentPaths()
    ckpts = {s: paths.checkpoint_path(s, C.CHECKPOINT_STEP) for s in C.SEEDS}
    for s, p in ckpts.items():
        assert frozen["sha256"][os.path.relpath(p, C.REPO_ROOT)], p
    nets = {s: load_checkpoint(p, dev) for s, p in ckpts.items()}
    samples = build_samples(dataset)[shard::shards][:limit]
    keys = C.method_keys(dataset); col = {k: i for i, k in enumerate(keys)}
    n = len(samples)
    official = np.full((n, len(keys), 3), -1, np.int64); fast = official.copy()
    meta = {"label_identical": np.zeros(n, bool), "n_keep": np.zeros(n, np.int64), "n_gt": np.zeros(n, np.int64),
            "T_occany_minus_ours_maxabs": np.zeros(n), "T_occany_minus_static_maxabs": np.full(n, np.nan),
            "scale_forward": np.full(n, np.nan), "scale_causal": np.full(n, np.nan)}
    t0 = time.time()
    for i, s in enumerate(samples):
        oc = SC.load_occany(dataset, s.occany_pkl, edims)
        meta["label_identical"][i] = np.array_equal(SC.independent_label(dataset, s.gt_ref), oc.label)
        label = oc.label
        keep = torch.from_numpy(label != C.IGNORE).to(dev)
        gt = keep & torch.from_numpy(label != C.OCCANY[dataset].empty_class).to(dev)
        meta["n_keep"][i], meta["n_gt"][i] = int(keep.sum()), int(gt.sum())

        def score(key: str, occ: torch.Tensor) -> None:
            fast[i, col[key]] = SC.fast_counts(occ, gt, keep)
            official[i, col[key]] = SC.official_counts(dataset, occ.cpu().numpy(), label)

        pred_oc = torch.from_numpy(oc.pred).to(dev)
        score("occany|raw", pred_oc); score("occany|pooled", SC.pool(pred_oc))

        meta["T_occany_minus_ours_maxabs"][i] = float(np.abs(oc.T_cam_to_grid - s.T_cam_to_grid_ours).max())
        if dataset == "semantickitti":
            assert meta["T_occany_minus_ours_maxabs"][i] < 1e-4, (s.sample_id, "SemanticKITTI transforms must be shared")
            placements = {"shared": s.T_cam_to_grid_ours}
        else:
            placements = {"occany": oc.T_cam_to_grid, "ours": s.T_cam_to_grid_ours}
        for pname, protocol in C.PROTOCOLS.items():
            if not s.protocol_ok(protocol):
                continue
            clip = WindowClip.load(C.clip_path(dataset, pname, s.sample_id))
            assert clip.anchor_slot == protocol.anchor_slot
            meta[f"scale_{pname}"][i] = clip.scale_m_per_unit
            mapper = build_mapper(clip, dev)
            for cname in C.CALIBRATIONS[dataset]:
                S_, F_ = hard_channels(mapper, pgrid, placements[cname])
                x = network_input(S_, F_, pgrid.dims)
                volumes = {"maponly": S_}
                for sd in C.SEEDS:
                    volumes[f"seed{sd}"] = predict(nets[sd], x)
                for name, occ in volumes.items():
                    ev = occ_to_eval_grid(occ, dataset).reshape(edims)
                    for post in C.POSTPROCESS:
                        vol = ev if post == "raw" else SC.pool(ev)
                        key = f"maponly|{pname}|{cname}|{post}" if name == "maponly" else f"ours|{pname}|{cname}|{post}|{name}"
                        score(key, vol)
            del mapper
        if (i + 1) % 100 == 0:
            print(f"  [{dataset} s{shard}] {i + 1}/{n} samples, {(time.time() - t0) / (i + 1):.2f}s each", flush=True)

    out = C.shard_path(dataset, shard, shards) if limit is None else C.shard_path(dataset, shard, shards).replace(".npz", f"_limit{limit}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, keys=np.asarray(keys), sample_id=np.asarray([s.sample_id for s in samples]),
                        segment=np.asarray([s.segment for s in samples]), target_pos=np.asarray([s.target_pos for s in samples]),
                        official_anchor=np.asarray([s.official_anchor for s in samples]),
                        forward_ok=np.asarray([s.protocol_ok(C.PROTOCOLS["forward"]) for s in samples]),
                        causal_ok=np.asarray([s.protocol_ok(C.PROTOCOLS["causal"]) for s in samples]),
                        official=official, fast=fast, freeze_frozen_at=frozen["frozen_at"], seconds=time.time() - t0, **meta)
    rec = {"dataset": dataset, "shard": shard, "shards": shards, "n_samples": n, "out": out, "seconds": time.time() - t0,
           "fast_equals_official": bool(np.array_equal(fast, official)), "label_identical": int(meta["label_identical"].sum())}
    print(f"[{dataset} s{shard}] {rec}", flush=True)
    return rec


__all__ = ["evaluate_shard"]
