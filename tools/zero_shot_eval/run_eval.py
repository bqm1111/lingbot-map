"""Score every canonical prediction of a dataset with Track P, Track O and OccAny's released geometry (Level A).

Writes ``artifacts/final_zero_shot/scores/scores_<dataset>.npz``: per-sample arrays (stacked in evaluated-sample order),
pooled arrays (summed), sample ids and bootstrap block ids.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Dict

import numpy as np
import torch

from tools.occany_comparison import scoring as SC
from tools.zero_shot_eval import config as C, freeze, samples as ZS
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction
from tools.zero_shot_eval.official.evaluator import score_occupancy_official, score_sample_official
from tools.zero_shot_eval.practical.evaluator import score_occany_geometry, score_sample
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS, METHODS, STATISTICS

SCORE_DIR = os.path.join(C.ARTIFACT_DIR, "scores")


def block_ids(dataset: str, samples) -> np.ndarray:
    if dataset == "semantickitti":
        return np.arange(len(samples)) // STATISTICS.semantickitti_block_samples
    scenes = {s: i for i, s in enumerate(sorted({x.segment for x in samples}))}
    return np.asarray([scenes[x.segment] for x in samples])


def evaluate_dataset(dataset: str, limit: int | None = None) -> str:
    freeze.check("protocol"); freeze.check("evaluators")
    spec = DATASET_SPECS[dataset]; samples = ZS.evaluated_samples(dataset)[:limit]
    per_sample: Dict[str, list] = defaultdict(list); pooled: Dict[str, np.ndarray] = {}
    t0 = time.time()
    for i, s in enumerate(samples):
        pred = CanonicalPrediction.load(C.prediction_path(dataset, s.sample_id))
        oc = SC.load_occany(dataset, s.occany_pkl, pred.dims)
        label = oc.label.reshape(-1)
        res = score_sample(pred, label, spec, list(METHODS))
        res["per_sample"].update(score_sample_official(pred, label, spec, list(METHODS)))
        occany_raw = oc.pred.reshape(-1)
        occany_pooled = SC.pool(torch.from_numpy(oc.pred)).cpu().numpy().reshape(-1)
        for tag, vol in (("raw", occany_raw), ("pooled", occany_pooled)):
            res["per_sample"].update(score_occany_geometry(vol, pred, label, spec, tag))
            res["per_sample"][f"occany_{tag}__official"] = score_occupancy_official(vol, label, spec)
        for k, v in res["per_sample"].items():
            per_sample[k].append(v)
        for k, v in res["pooled"].items():
            pooled[k] = pooled.get(k, 0) + v
        if (i + 1) % 200 == 0:
            print(f"  [score {dataset}] {i + 1}/{len(samples)} {(time.time() - t0) / (i + 1):.2f}s each", flush=True)
    os.makedirs(SCORE_DIR, exist_ok=True)
    out = os.path.join(SCORE_DIR, f"scores_{dataset}{'' if limit is None else f'_limit{limit}'}.npz")
    np.savez_compressed(out, sample_id=np.asarray([s.sample_id for s in samples]), segment=np.asarray([s.segment for s in samples]),
                        block=block_ids(dataset, samples), **{f"ps__{k}": np.stack(v) for k, v in per_sample.items()},
                        **{f"pool__{k}": v for k, v in pooled.items()})
    return out


__all__ = ["SCORE_DIR", "block_ids", "evaluate_dataset"]
