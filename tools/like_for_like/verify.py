"""The protocol-parameterised copy reproduces the frozen pipeline.

Runs ``like_for_like.predict.predict_sample`` with the CAUSAL protocol on a deterministic handful of samples and compares
every array with the stored canonical prediction of the frozen final zero-shot evaluation. Occupancy, observed, evidence,
query index, distances and each method's labels must match exactly; only then is the forward run trustworthy.

    PYTHONPATH=$PWD python -m tools.like_for_like.verify --device cuda:0 --n 4
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch

from tools.like_for_like import config as LC
from tools.like_for_like.predict import predict_sample
from tools.occany_comparison import config as OC
from tools.zero_shot_eval import config as ZC, samples as ZS
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction
from tools.zero_shot_eval.prediction.pipeline import load_models

CAUSAL = OC.PROTOCOLS["causal"]


def compare(a: CanonicalPrediction, b: CanonicalPrediction) -> Dict[str, bool]:
    same = {k: bool(np.array_equal(getattr(a, k), getattr(b, k))) for k in ("occupied", "observed", "evidence", "query_index")}
    # the stored file keeps distances as float16, so compare the same round-trip rather than raw float32
    same["evidence_distance"] = bool(np.array_equal(a.evidence_distance.astype(np.float16), b.evidence_distance.astype(np.float16)))
    same["methods"] = sorted(a.labels) == sorted(b.labels)
    for m in sorted(a.labels):
        same[f"labels__{m}"] = bool(np.array_equal(a.labels[m], b.labels[m]))
    return same


def run(device: str, n: int) -> Dict:
    rec: Dict = {"protocol": "causal", "samples": {}}
    for dataset in ZC.DATASETS:
        dev = torch.device(device); torch.cuda.set_device(dev)
        models = load_models(dataset, dev)
        ss = ZS.evaluated_samples(dataset)
        picks = [ss[int(round(q * (len(ss) - 1)))] for q in np.linspace(0.1, 0.9, n)]
        for s in picks:
            frozen = CanonicalPrediction.load(ZC.prediction_path(dataset, s.sample_id))
            ours = predict_sample(s, dataset, models, dev, protocol=CAUSAL, calibration="ours")   # the frozen evaluation placed the map with our transform
            rec["samples"][f"{dataset}/{s.sample_id}"] = compare(ours, frozen)
    rec["all_equal"] = all(all(v.values()) for v in rec["samples"].values())
    LC.ensure_dirs()
    json.dump(rec, open(os.path.join(LC.ARTIFACT_DIR, "pipeline_equivalence.json"), "w"), indent=1)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--n", type=int, default=4)
    rec = run(ap.parse_args().device, ap.parse_args().n)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
