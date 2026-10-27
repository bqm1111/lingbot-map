"""IoU and mIoU of our method and OccAny, split by whether the input camera could see the voxel.

One visibility mask per sample (``visibility.regions``) partitions every valid voxel into **observed**
(the camera sees through to it, or terminates on it), **occluded** (in the image, behind a surface) and
**out of view**. Counts are pooled over samples and the metric is computed from the pooled counts, exactly
as OccAny's ``SSCMetrics`` does -- never averaged per frame.

Both post-processings OccAny releases are applied to every method: ``raw`` and ``pooled`` (3x3x3 max
dilation for occupancy, 3x3x3 majority relabel for semantics, which leaves occupancy untouched).

    PYTHONPATH=$PWD python -m tools.observability_split.evaluate --dataset semantickitti
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

from gates.gate8b import pooling as PL
from tools.observability_split import runs as R, visibility as V
from tools.zero_shot_eval.canonical.metrics import occany_ssc_miou, superclass_confusion
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "artifacts", "observability_split")
IGNORE = 255
REGIONS = ("all", "observed", "occluded", "out_of_view")
#: confusion width: one more than the benchmark's class count so OccAny's "other" prediction (id 20 on
#: SemanticKITTI) has a column. ``occany_ssc_miou`` then averages exactly classes 1..19, as their metric does.
NC = {"semantickitti": 21}
OTHER = {"semantickitti": 20}


def _class_counts(conf: np.ndarray) -> np.ndarray:
    """[C, 3] int64 of (tp, fp, fn) per class from a confusion."""
    tp = np.diag(conf).astype(np.int64)
    return np.stack([tp, conf.sum(0) - tp, conf.sum(1) - tp], 1)


def _pool_labels(lab: torch.Tensor, dims, nc: int, other: int, empty: int) -> torch.Tensor:
    return PL.semantic_separate(lab.reshape(dims), nc, other, empty).reshape(-1)


def _method_volumes(dataset: str, sid: str, dims, spec, device) -> Dict[str, Dict[str, torch.Tensor]]:
    """{method_key: {"occ": bool[N], "lab": uint8[N] or None}} -- one entry per scored prediction."""
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for run in R.RUNS[dataset]:
        if run.kind == "ours":
            p = CanonicalPrediction.load(os.path.join(run.path, f"{sid}.npz"))
            occ = torch.from_numpy(p.occupied).to(device)
            out["ours"] = {"occ": occ, "lab": None, "pred": p}
            for m in sorted(p.labels):
                dense = np.where(p.occupied, p.dense_labels(m, spec.empty_class), spec.empty_class)
                out[f"ours_{m}"] = {"occ": None, "lab": torch.from_numpy(dense.astype(np.uint8)).to(device)}
        else:
            label, vol = R.load_occany(run, sid, dims)
            v = torch.from_numpy(vol).to(device)
            out[run.key] = {"occ": v != spec.empty_class, "lab": v if run.semantic else None, "label": label}
    return out


def evaluate(dataset: str, limit: int | None = None, pixel_stride: int = 2, step_m: float = 0.1,
             device: str = "cuda", save_masks: bool = False) -> Dict:
    spec = DATASET_SPECS[dataset]
    nc, other = NC[dataset], OTHER[dataset]
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    sids = R.samples(dataset)[:limit]
    assert sids, "no shared samples"
    sup_map = list(spec.superclass_of) + [spec.superclass_of[0]] * (nc - spec.n_classes)
    ref = CanonicalPrediction.load(os.path.join(R.RUNS[dataset][0].path, f"{sids[0]}.npz"))
    dims, vs, origin = ref.dims, ref.voxel_size, ref.origin
    img = R.IMAGE_SIZE[dataset]

    acc: Dict[str, np.ndarray] = defaultdict(lambda: 0)
    region_voxels: Dict[str, np.ndarray] = defaultdict(lambda: 0)
    per_sample: Dict[str, list] = defaultdict(list)       # [S, ...] counts, for the block bootstrap
    t0 = time.time()
    for i, sid in enumerate(sids):
        p = CanonicalPrediction.load(os.path.join(R.RUNS[dataset][0].path, f"{sid}.npz"))
        vols = _method_volumes(dataset, sid, dims, spec, dev)
        labels = [v["label"] for v in vols.values() if "label" in v]
        for l in labels[1:]:
            assert np.array_equal(l, labels[0]), f"{sid}: OccAny runs disagree on the stored ground truth"
        label = labels[0]
        valid = label != IGNORE
        gt_occ = valid & (label != spec.empty_class)

        reg = V.regions(gt_occ, valid, p.T_cam_to_grid, p.K, img, dims, vs, origin,
                        pixel_stride=pixel_stride, step_m=step_m, device=str(dev))
        masks = {"all": valid, "observed": reg.observed, "occluded": reg.occluded, "out_of_view": reg.out_of_view}
        assert not (reg.observed & reg.occluded).any() and not (reg.observed & reg.out_of_view).any()
        assert (reg.observed | reg.occluded | reg.out_of_view).sum() == valid.sum()

        gt_t = torch.from_numpy(label.astype(np.int64)).to(dev)
        gtocc_t = torch.from_numpy(gt_occ).to(dev)
        idx = {r: torch.from_numpy(np.flatnonzero(m)).to(dev) for r, m in masks.items()}
        for r, m in masks.items():
            region_voxels[f"n__{r}"] = region_voxels[f"n__{r}"] + np.array([m.sum(), (m & gt_occ).sum()], np.int64)

        for key, v in vols.items():
            if v.get("occ") is not None:
                for post, occ in (("raw", v["occ"]), ("pooled", PL.geometry_dilation(v["occ"].reshape(dims)).reshape(-1))):
                    for r, ix in idx.items():
                        po, go = occ[ix], gtocc_t[ix]
                        c = np.array([int((po & go).sum()), int((po & ~go).sum()), int((~po & go).sum())], np.int64)
                        acc[f"occ__{key}__{post}__{r}"] = acc[f"occ__{key}__{post}__{r}"] + c
                        per_sample[f"ps_occ__{key}__{post}__{r}"].append(c)
            if v.get("lab") is not None:
                for post, lab in (("raw", v["lab"].long()),
                                  ("pooled", _pool_labels(v["lab"].long(), dims, nc, other, spec.empty_class).long())):
                    for r, ix in idx.items():
                        c = torch.bincount(gt_t[ix] * nc + lab[ix], minlength=nc * nc).reshape(nc, nc).cpu().numpy()
                        acc[f"conf__{key}__{post}__{r}"] = acc[f"conf__{key}__{post}__{r}"] + c
                        per_sample[f"ps_cls__{key}__{post}__{r}"].append(_class_counts(c))
                        per_sample[f"ps_sup__{key}__{post}__{r}"].append(
                            _class_counts(superclass_confusion(c, sup_map, len(spec.superclass_names))))
        if (i + 1) % 50 == 0:
            print(f"  [{dataset}] {i + 1}/{len(sids)}  {(time.time() - t0) / (i + 1):.2f}s each", flush=True)

    out = {"dataset": dataset, "n_samples": len(sids), "pixel_stride": pixel_stride, "step_m": step_m,
           "seconds": time.time() - t0, "regions": {}, "occupancy": {}, "semantics": {},
           "runs": {r.key: {"window": r.window, "frames": r.frames, "semantic": r.semantic, "note": r.note, "path": r.path}
                    for r in R.RUNS[dataset]}}
    for r in REGIONS:
        n, g = (int(x) for x in region_voxels[f"n__{r}"])
        out["regions"][r] = {"valid_voxels": n, "gt_occupied": g, "gt_occupied_share": g / max(sum(
            int(region_voxels[f'n__{x}'][1]) for x in REGIONS if x != "all"), 1)}
    sup = sup_map
    for k, v in acc.items():
        what, key, post, r = k.split("__")
        if what == "occ":
            tp, fp, fn = (float(x) for x in v)
            out["occupancy"].setdefault(key, {}).setdefault(post, {})[r] = {
                "iou": 100 * tp / max(tp + fp + fn, 1.0), "precision": 100 * tp / max(tp + fp, 1.0),
                "recall": 100 * tp / max(tp + fn, 1.0), "tp": tp, "fp": fp, "fn": fn}
        else:
            m = occany_ssc_miou(v, nc, spec.empty_class, other, True)
            s = occany_ssc_miou(superclass_confusion(v, sup, len(spec.superclass_names)),
                                len(spec.superclass_names), spec.superclass_empty, -1, False)
            out["semantics"].setdefault(key, {}).setdefault(post, {})[r] = {
                "ssc_miou": 100 * m["miou"], "superclass_miou": 100 * s["miou"],
                "per_class_iou": {spec.class_names[int(i)]: 100 * x for i, x in m["per_class_iou"].items()
                                  if int(i) < spec.n_classes}}
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(ARTIFACT_DIR, f"counts_{dataset}.npz"),
                        samples=np.asarray(sids),
                        **{k: np.asarray(v) for k, v in list(acc.items()) + list(region_voxels.items())},
                        **{k: np.stack(v) for k, v in per_sample.items()})
    json.dump(out, open(os.path.join(ARTIFACT_DIR, f"results_{dataset}.json"), "w"), indent=1, default=float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="semantickitti")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pixel-stride", type=int, default=2)
    ap.add_argument("--step-m", type=float, default=0.1)
    a = ap.parse_args()
    out = evaluate(a.dataset, a.limit, a.pixel_stride, a.step_m)
    print(json.dumps({k: v for k, v in out.items() if k != "semantics"}, indent=1, default=float))


if __name__ == "__main__":
    main()
