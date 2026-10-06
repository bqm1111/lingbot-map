"""Score the forward-protocol predictions with the FROZEN evaluators, then report our numbers beside OccAny's published ones.

Track O (official) is the benchmark computation OccAny reports: every valid voxel, no region restriction, OccAny's
``SSCMetrics`` conventions (``canonical/metrics.occany_ssc_miou``; SemanticKITTI ignores the absent ``other`` index,
Occ3D includes ``others``), plus the superclass variant. Track P (our practical region) is scored too, for continuity
with the frozen report. Nothing about the metric or the method changes here; only the sample set and the input window
match OccAny's setting.

    PYTHONPATH=$PWD python -m tools.like_for_like.evaluate --dataset occ3d
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict

import numpy as np

from tools.like_for_like import config as LC, frames as LF, surround as SR
from tools.occany_comparison import scoring as SC
from tools.zero_shot_eval.canonical.metrics import occany_ssc_miou, superclass_confusion
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction
from tools.zero_shot_eval.official.evaluator import score_sample_official
from tools.zero_shot_eval.practical.evaluator import score_sample
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS, METHODS

#: OccAny's ``apply_majority_pooling`` (occany/utils/helpers.py:503, semantic branch, ``update_other_only=False``) transcribed
#: here because importing that module pulls in ``groundingdino``, which cannot load on this machine.
def semantic_majority_pooling(dense_label: np.ndarray, dims, spec, device: str = "cuda:0") -> np.ndarray:
    """Relabel every occupied voxel with the majority class of its 3x3x3 neighbourhood.

    Votes come only from neighbours that are neither empty nor ``other`` (those channels are never built, so ``argmax``
    cannot return them); a voxel with no valid neighbour keeps its own label. Occupancy is untouched, so SC IoU does not
    move -- this is the post-processing behind OccAny's published semantic numbers, applied to OUR labels."""
    import torch
    import torch.nn.functional as F
    lab = torch.from_numpy(np.ascontiguousarray(dense_label.reshape(dims))).to(device)
    voting = [c for c in range(spec.n_classes) if c not in (spec.empty_class, spec.occany_other_class)]
    one_hot = torch.zeros((len(voting),) + tuple(dims), dtype=torch.float32, device=device)
    for i, c in enumerate(voting):
        one_hot[i] = (lab == c).float()
    votes = F.avg_pool3d(one_hot.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
    has_valid = votes.sum(0) > 0
    winner = torch.as_tensor(voting, device=device)[votes.argmax(0)]
    out = torch.where((lab != spec.empty_class) & has_valid, winner, lab)
    return out.to(torch.uint8).cpu().numpy().reshape(-1)


#: OccAny's published semantics (paper Tab. 5/6, reproduced in their README's evaluation table), for the report only.
#: Their nuScenes semantic number exists ONLY for the surround setting: ``occany/datasets/eval_helper.py`` gives nuScenes
#: ``5frames`` a single CAM_FRONT and ``use_surround_label=False`` (rear half of the label ignored), while ``surround``
#: uses six cameras at one timestep and ``use_surround_label=True``. So the Occ3D row is comparable only to our surround
#: setting; the SemanticKITTI row is the single-camera five-frame setting our forward protocol already matches.
OCCANY_PUBLISHED = {"semantickitti": {"ssc_miou": 7.28, "superclass_miou": 13.53, "setting": "1 camera, 5 frames starting at the target"},
                    "occ3d": {"ssc_miou": 6.66, "superclass_miou": 10.32, "setting": "6 surround cameras, single timestep, 360-degree label"}}

#: (prediction source, ground truth) for each evaluated setting. ``forward_surround_label`` rescores the SAME
#: single-camera predictions against the 360-degree label, which separates the label change from the extra cameras.
SETTINGS = {"forward": ("forward", "occany"), "forward_surround_label": ("forward", "surround"), "surround": ("surround", "surround")}


def evaluate_dataset(dataset: str, limit: int | None = None, pooling: str = "none", source: str = "forward",
                     setting: str = "forward") -> Dict:
    """``source='frozen'`` scores the frozen causal predictions instead, which must reproduce the published frozen numbers
    (SemanticKITTI B0 4.32 / B3 4.43; Occ3D 3.71 / 3.80) -- an end-to-end check of this scoring path.

    ``setting`` (Occ3D only) selects one of ``SETTINGS``: which predictions are scored and against which ground truth."""
    spec = DATASET_SPECS[dataset]
    pred_source, gt = SETTINGS[setting]
    if setting != "forward":
        assert dataset == "occ3d" and source == "forward", f"setting {setting!r} is Occ3D-only"
    if source == "frozen":
        from tools.zero_shot_eval import config as ZC, samples as ZS
        samples = ZS.evaluated_samples(dataset)[:limit]; path_of = lambda sid: ZC.prediction_path(dataset, sid)
    elif pred_source == "surround":
        from tools.like_for_like import surround_predict as SP
        samples = SR.samples()[:limit]; path_of = SP.prediction_path
    else:
        samples = LF.samples(dataset)[:limit]; path_of = lambda sid: LC.prediction_path(dataset, sid)
    pooled: Dict[str, np.ndarray] = defaultdict(lambda: 0)
    n, t0 = 0, time.time()
    for i, s in enumerate(samples):
        path = path_of(s.sample_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{dataset}: prediction missing for {s.sample_id} ({path})")
        pred = CanonicalPrediction.load(path)
        label = (SR.surround_label(s) if gt == "surround" else SC.load_occany(dataset, s.occany_pkl, pred.dims).label).reshape(-1)
        if pooling == "semantic":                     # relabel our occupied voxels exactly as OccAny post-processes its own
            pooled_labels = {}
            for m in METHODS:
                dense = np.where(pred.occupied, pred.dense_labels(m, spec.empty_class), spec.empty_class)
                sm = semantic_majority_pooling(dense, pred.dims, spec)
                pooled_labels[m] = sm[pred.query_index].astype(np.uint8)
            pred.labels = pooled_labels
        res = score_sample(pred, label, spec, list(METHODS))
        res["per_sample"].update(score_sample_official(pred, label, spec, list(METHODS)))
        for k, v in res["per_sample"].items():
            pooled[k] = pooled[k] + v
        n += 1
        if (i + 1) % 250 == 0:
            print(f"  [{dataset}] {i + 1}/{len(samples)} {(time.time() - t0) / max(n, 1):.2f}s each", flush=True)
    out = {"dataset": dataset, "protocol": LC.PROTOCOL.name if source == "forward" else "causal (frozen predictions)",
           "source": source, "setting": setting, "ground_truth": gt, "pooling": pooling, "n_samples": n,
           "seconds": time.time() - t0, "methods": {}}
    if setting == "surround":
        out["protocol"] = "surround (6 cameras, single timestep)"
    tp, fp, fn = (float(v) for v in pooled["official_occ"])          # OccAny get_score_completion counts
    out["official_sc_iou"] = 100 * tp / max(tp + fp + fn, 1.0)
    for m in METHODS:                                                # exactly the frozen analysis' computation
        conf = pooled[f"official_conf__{m}"]
        r = occany_ssc_miou(conf, spec.n_classes, spec.empty_class, spec.occany_other_class, spec.occany_ignore_other)
        sc = superclass_confusion(conf, list(spec.superclass_of), len(spec.superclass_names))
        rs = occany_ssc_miou(sc, len(spec.superclass_names), spec.superclass_empty, -1, False)
        out["methods"][m] = {"official_ssc_miou": 100 * r["miou"], "official_superclass_miou": 100 * rs["miou"],
                             "per_class_iou": {spec.class_names[int(i)]: 100 * v for i, v in r["per_class_iou"].items()}}
    tag = ("" if setting == "forward" else f"_{setting}") + ("" if pooling == "none" else f"_{pooling}") \
          + ("" if source == "forward" else "_frozencheck")
    np.savez_compressed(os.path.join(LC.SCORE_DIR, f"pooled_{dataset}{tag}.npz"), **{k: np.asarray(v) for k, v in pooled.items()})
    json.dump(out, open(os.path.join(LC.ARTIFACT_DIR, f"results_{dataset}{tag}.json"), "w"), indent=1, default=float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--limit", type=int)
    ap.add_argument("--pooling", choices=["none", "semantic"], default="none")
    ap.add_argument("--source", choices=["forward", "frozen"], default="forward")
    ap.add_argument("--setting", choices=sorted(SETTINGS), default="forward")
    a = ap.parse_args(); LC.ensure_dirs()
    print(json.dumps(evaluate_dataset(a.dataset, a.limit, a.pooling, a.source, a.setting), indent=1, default=float))


if __name__ == "__main__":
    main()
