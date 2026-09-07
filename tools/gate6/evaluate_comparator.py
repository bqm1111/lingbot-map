#!/usr/bin/env python
"""Score the OccAny-aligned comparator with the identical evaluator and masks.

Both teachers are lifted through the same frozen geometry on the same 163 SemanticKITTI
clips, so the only thing that differs is the 2D semantic readout. Reported as a
**comparator**, never as a ranking of OccAny-the-system (§16 of the report explains why
that comparison is not available from this experiment).

    python tools/gate6/evaluate_comparator.py
"""
from __future__ import annotations

import json, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                       # noqa: E402
from gates.gate6 import frames as F, grids, metrics, pipelines, targets, vocab  # noqa: E402

DATASET = "semantickitti"
PRED = "/media/SSD1/MINH_DATASETS/lingbot_gate6/occany_predictions"
CACHE = "/media/SSD1/MINH_DATASETS/lingbot_gate6/occany_semantics"


def score(pred_root, only_clips=None, tag=""):
    v = vocab.load(DATASET)
    G = grids.EVAL_GRID[DATASET]
    n_vox = int(np.prod(G.dims))
    dist, height = metrics.band_masks(G)
    recs = {r.clip_id: r.raw for r in F.read_manifest(DATASET, REPO_ROOT)}
    groups = pipelines.load_groups(DATASET, REPO_ROOT)
    store, named, total_occ = [], 0, 0
    for name in sorted(os.listdir(pred_root)):
        if not name.endswith(".npz"):
            continue
        cid = name[:-4]
        if only_clips is not None and cid not in only_clips:
            continue
        target, keep = targets.semantic_target(DATASET, recs[cid], REPO_ROOT)
        with np.load(os.path.join(pred_root, name)) as z:
            labels = z["labels"].astype(np.int32)
            flat, ch, sup = z["dil_flat"], z["dil_channel"], z["dil_support"]
        pl = np.full(n_vox, v.empty_label, np.int32)
        if len(flat):
            pl[flat.astype(np.int64)] = labels[ch.astype(np.int64)]
        ps = np.zeros(n_vox, np.uint8)
        if len(flat):
            ps[flat.astype(np.int64)] = sup
        named += int((pl[flat.astype(np.int64)] != v.empty_label).sum()) if len(flat) else 0
        total_occ += int(len(flat))
        store.append(metrics.clip_counts(pl.reshape(G.dims), ps.reshape(G.dims), target,
                                         keep, v.labels, v.empty_label, dist, height,
                                         cid, groups.get(cid, "?"), "B-D"))
    agg = metrics.aggregate(store)
    s = metrics.summarize(agg, v.names)
    pc = np.array([[c.btp, c.bfp, c.bfn] for c in store], np.float64)
    den = pc.sum(axis=1)
    s["binary_iou_mean_per_clip"] = float(np.mean(np.where(den > 0, pc[:, 0] / np.maximum(den, 1), 0.0)))
    s["frozen_occupied_voxels"] = total_occ
    s["named_voxels"] = named
    s["named_fraction_of_frozen_occupancy"] = named / max(total_occ, 1)
    s["n_clips"] = len(store)
    s["tag"] = tag
    return s


def main() -> int:
    root = os.path.join(PRED, DATASET)
    clips = {n[:-4] for n in os.listdir(root) if n.endswith(".npz")}
    comp = score(root, clips, "OccAny-aligned Grounded-SAM-2")
    tri_root = json.load(open(os.path.join(
        REPO_ROOT, "artifacts", "gate6",
        f"prediction_manifest_{DATASET}.json")))["prediction_root"]
    tri = score(tri_root, clips, "Trident-H (primary)")
    tri["frozen_occupied_voxels"] = tri["n_pred_occupied"]

    det = []
    for n in sorted(os.listdir(os.path.join(CACHE, DATASET))):
        with np.load(os.path.join(CACHE, DATASET, n)) as z:
            det.append((int(z["n_boxes"]), float((z["label"] > 0).mean())))
    out = {"dataset": DATASET, "n_clips": comp["n_clips"],
           "note": ("comparator only; both teachers lifted through the identical frozen "
                    "geometry on the identical clips. Not a comparison with OccAny as a "
                    "system: its reconstruction, grid and protocol are not reproduced."),
           "grounding_dino_boxes_per_frame_median": float(np.median([d[0] for d in det])),
           "pixels_named_per_frame_median": float(np.median([d[1] for d in det])),
           "comparator": comp, "primary": tri}
    write_json(os.path.join(REPO_ROOT, "artifacts", "gate6",
                            "occany_comparator.json"), out)
    for k in ("comparator", "primary"):
        s = out[k]
        d = s["decomposition"]
        print(f"{s['tag']:34s} binIoU={s['binary_iou_mean_per_clip']:.4f} "
              f"mIoU={s['ssc_miou']:.4f} TPacc={s['tp_conditioned']['top1_accuracy']:.4f} "
              f"TPbal={s['tp_conditioned']['balanced_recall']:.4f} "
              f"miss={d['coverage_miss_fraction']:.4f} name={d['naming_error_fraction']:.4f} "
              f"named={s['named_fraction_of_frozen_occupancy']:.4f}")
    print("gdino boxes/frame median", out["grounding_dino_boxes_per_frame_median"],
          " pixels named/frame median", round(out["pixels_named_per_frame_median"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
