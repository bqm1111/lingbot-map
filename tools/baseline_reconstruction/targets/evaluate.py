"""Frozen zero-shot evaluation of the current baseline on one target dataset, all seeds in one pass.

Per official anchor: local clip -> hard map (S, F) on the prediction grid (``build_local_geometry``,
anchor camera = identity, ``T_cam_to_grid`` from the adapter) -> ``[S, 0, F]`` -> each seed's
network -> occupied := final log-odds > 0 -> reduced to the evaluation grid -> TP/FP/FN against
the official occupancy inside the official mask, by range bin. The raw map ``S`` is scored the
same way as the *map-only* reference. Nothing is selected or tuned here; the entry point refuses
to run unless the source freeze is present and every frozen file is unchanged.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np
import torch

from core.model.net import load_checkpoint
from tools.baseline_reconstruction import config as C
from tools.baseline_reconstruction import freeze as FZ
from tools.baseline_reconstruction.current.geometry import LocalClip, build_local_geometry, network_input
from tools.baseline_reconstruction.current.evaluate_source import predict
from tools.baseline_reconstruction.metrics import occupancy as M
from tools.baseline_reconstruction.targets.adapters import TargetAdapter

REGIONS = ("full", "H_evidence", "no_H_evidence")


@dataclass
class ZeroShotRecord:
    dataset: str
    protocol: Dict
    seeds: List[int]
    checkpoints: Dict[int, str]
    checkpoint_sha256: Dict[int, str]
    threshold: float
    n_anchors: int
    n_skipped_no_label: int
    n_skipped_no_scale: int
    per_seed: Dict[int, Dict]            # seed -> region summary (counts, iou, precision, recall, by_range)
    map_only: Dict                       # region "full" for the raw map S
    seed_statistics: Dict[str, Dict]     # mean/median/sd of full-region IoU, precision, recall
    freeze_frozen_at: str
    seconds: float


def evaluate_dataset(dataset: str, device: str, paths: C.CurrentPaths = C.CurrentPaths(), limit: int | None = None) -> ZeroShotRecord:
    frozen = FZ.require_source_freeze()
    dev = torch.device(device); torch.cuda.set_device(dev)
    adapter = TargetAdapter(dataset); n_bins = len(C.RANGE_NAMES)
    ckpts = {s: paths.checkpoint_path(s, 6000) for s in C.SEEDS}
    nets = {s: load_checkpoint(p, dev) for s, p in ckpts.items()}
    acc = {s: M.RegionAccumulator(REGIONS, n_bins) for s in C.SEEDS}; acc_map = M.RegionAccumulator(("full",), n_bins)
    per_anchor: List[Dict] = []
    n_done = n_no_label = n_no_scale = 0; t0 = time.time()
    pred_dims = adapter.prediction_grid.dims
    for ref in adapter.anchors():
        if limit and n_done >= limit:
            break
        gt, keep = adapter.official_target(ref, dev)
        if gt is None:
            n_no_label += 1; continue
        clip_path = paths.target_clip_path(dataset, ref.segment, ref.anchor_index)
        clip = LocalClip.load(clip_path) if np.isfinite(float(np.load(clip_path)["scale"])) else None
        if clip is None:
            n_no_scale += 1; continue
        S, F, gstats = build_local_geometry(clip, adapter.prediction_grid, ref.T_cam_to_grid, dev)
        x = network_input(S, F, pred_dims)
        rb = adapter.range_bins(ref, dev)
        evidence = adapter.to_evaluation_grid(S | F)
        masks = {"full": keep, "H_evidence": keep & evidence, "no_H_evidence": keep & ~evidence}
        S_eval = adapter.to_evaluation_grid(S)
        acc_map.add("full", M.counts_by_range(S_eval, gt, keep, rb, n_bins))
        row = {"clip_id": ref.clip_id, "segment": ref.segment, "anchor_index": ref.anchor_index, "scale": clip.scale_m_per_unit,
               "n_keep": int(keep.sum()), "n_gt": int(gt.sum()), "n_rays": gstats["n_rays"], "map_only": M.counts(S_eval, gt, keep).as_dict()}
        for s in C.SEEDS:
            pred = adapter.to_evaluation_grid(predict(nets[s], x))
            for r in REGIONS:
                acc[s].add(r, M.counts_by_range(pred, gt, masks[r], rb, n_bins))
            row[f"seed{s}"] = M.counts(pred, gt, keep).as_dict()
        per_anchor.append(row); n_done += 1
        if n_done % 100 == 0:
            print(f"  [{dataset}] {n_done} anchors, {(time.time()-t0)/n_done:.2f}s each", flush=True)
    stats = {}
    for key in ("iou", "precision", "recall"):
        stats[key] = M.seed_statistics([acc[s].summary()["full"][key] for s in C.SEEDS])
    rec = ZeroShotRecord(dataset=dataset, protocol=asdict(C.TARGET_PROTOCOLS[dataset]), seeds=list(C.SEEDS),
                         checkpoints={s: os.path.relpath(p, C.REPO_ROOT) for s, p in ckpts.items()},
                         checkpoint_sha256={s: frozen["sha256"][os.path.relpath(p, C.REPO_ROOT)] for s, p in ckpts.items()},
                         threshold=C.OCCUPANCY_THRESHOLD, n_anchors=n_done, n_skipped_no_label=n_no_label, n_skipped_no_scale=n_no_scale,
                         per_seed={s: acc[s].summary() for s in C.SEEDS}, map_only=acc_map.summary(), seed_statistics=stats,
                         freeze_frozen_at=frozen["frozen_at"], seconds=time.time() - t0)
    os.makedirs(paths.target_eval, exist_ok=True)
    json.dump(asdict(rec), open(os.path.join(paths.target_eval, f"zero_shot_{dataset}.json"), "w"), indent=1)
    json.dump(per_anchor, open(os.path.join(paths.target_eval, f"per_anchor_{dataset}.json"), "w"))
    seed_ious = " / ".join(f"{100 * acc[s].summary()['full']['iou']:.2f}" for s in C.SEEDS)
    print(f"[{dataset}] {n_done} anchors | SC IoU mean {100*stats['iou']['mean']:.2f} median {100*stats['iou']['median']:.2f} "
          f"(seeds {seed_ious}) | map-only {100*acc_map.summary()['full']['iou']:.2f}", flush=True)
    return rec


__all__ = ["REGIONS", "ZeroShotRecord", "evaluate_dataset"]
