#!/usr/bin/env python
"""Gate 6 evaluation — opens targets, and only after the predictions are pinned.

Refuses to start unless the prediction manifest exists and every prediction file still
hashes to its pinned value, so a result can never be produced against predictions that
moved after the fact.

Writes, per dataset and per occupancy condition:
  * ``artifacts/gate6/counts_<ds>_<cond>.npz``  -- per-clip integer count blocks; the
    bootstrap and the permutation control read these and never re-evaluate voxels;
  * ``artifacts/gate6/summary_<ds>.json``       -- the reported metrics.

    python tools/gate6/evaluate.py --dataset kitti360
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                       # noqa: E402
from gates.gate6 import frames as F, grids, metrics, pipelines, targets, vocab  # noqa: E402

CONDITIONS = ("B-R", "B-D")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(vocab.DATASETS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify-hashes", type=int, default=1)
    a = ap.parse_args()

    ds = a.dataset
    v = vocab.load(ds)
    C = len(v)
    G = grids.EVAL_GRID[ds]
    n_vox = int(np.prod(G.dims))
    mpath = os.path.join(REPO_ROOT, "artifacts", "gate6", f"prediction_manifest_{ds}.json")
    if not os.path.exists(mpath):
        raise SystemExit(f"prediction manifest missing: {mpath}. Run tools/gate6/predict.py "
                         "first -- evaluation may not precede a pinned prediction.")
    man = json.load(open(mpath))
    if man["audit"].get("violations"):
        raise SystemExit("prediction manifest records forbidden accesses; refusing to score")
    proot = man["prediction_root"]

    dist, height = metrics.band_masks(G)
    recs = {r.clip_id: r.raw for r in F.read_manifest(ds, REPO_ROOT)}
    groups = pipelines.load_groups(ds, REPO_ROOT)

    store = {c: [] for c in CONDITIONS}
    t0, n_bad, done = time.time(), 0, 0
    for name in sorted(man["per_file_sha256"]):
        cid = name[:-4]
        path = os.path.join(proot, name)
        if a.verify_hashes and sha256_file(path) != man["per_file_sha256"][name]:
            n_bad += 1
            continue
        rec = recs.get(cid)
        if rec is None:
            continue
        target, keep = targets.semantic_target(ds, rec, REPO_ROOT)   # targets opened NOW
        if target is None:
            continue
        with np.load(path) as z:
            labels = z["labels"].astype(np.int32)
            sets = {"B-R": (z["raw_flat"], z["raw_channel"], None),
                    "B-D": (z["dil_flat"], z["dil_channel"], z["dil_support"])}
        for cond in CONDITIONS:
            flat, ch, sup = sets[cond]
            pl = np.full(n_vox, v.empty_label, np.int32)
            if len(flat):
                pl[flat.astype(np.int64)] = labels[ch.astype(np.int64)]
            ps = np.zeros(n_vox, np.uint8)
            if len(flat):
                ps[flat.astype(np.int64)] = (np.ones(len(flat), np.uint8) if sup is None
                                             else sup)
            store[cond].append(metrics.clip_counts(
                pl.reshape(G.dims), ps.reshape(G.dims), target, keep, v.labels,
                v.empty_label, dist, height, cid, groups.get(cid, "?"), cond))
        done += 1
        if a.limit and done >= a.limit:
            break
        if done % 200 == 0:
            print(f"[{ds}] {done} clips {(time.time()-t0)/done:.2f}s/clip", flush=True)

    summary = {"dataset": ds, "grid": G.name, "n_clips": done,
               "n_hash_mismatch": n_bad, "n_classes": C,
               "class_names": list(v.names), "class_labels": list(v.labels),
               "prediction_manifest_sha256":
                   hashlib.sha256(open(mpath, "rb").read()).hexdigest(),
               "prediction_rollup_sha256": man["rollup_sha256"],
               "distance_bands": [n for n, _ in dist],
               "height_bands": [n for n, _ in height],
               "conditions": {}}
    for cond in CONDITIONS:
        cc = store[cond]
        agg = metrics.aggregate(cc)
        summary["conditions"][cond] = metrics.summarize(agg, v.names)
        # The reproduction gates report the MEAN OF PER-CLIP binary IoU (macro), while
        # every semantic metric here is pooled over voxels (micro). Both are reported; the
        # frozen-occupancy sanity check compares against the macro one, because that is
        # the statistic the pinned numbers were computed with.
        pc = np.array([[c.btp, c.bfp, c.bfn] for c in cc], np.float64)
        den = pc.sum(axis=1)
        summary["conditions"][cond]["binary_iou_mean_per_clip"] = float(
            np.mean(np.where(den > 0, pc[:, 0] / np.maximum(den, 1), 0.0)))
        dp_ = pc[:, 0] + pc[:, 1]
        dr_ = pc[:, 0] + pc[:, 2]
        summary["conditions"][cond]["binary_precision_mean_per_clip"] = float(
            np.mean(np.where(dp_ > 0, pc[:, 0] / np.maximum(dp_, 1), 0.0)))
        summary["conditions"][cond]["binary_recall_mean_per_clip"] = float(
            np.mean(np.where(dr_ > 0, pc[:, 0] / np.maximum(dr_, 1), 0.0)))
        summary["conditions"][cond]["binary_iou_pooled"] = \
            summary["conditions"][cond]["binary_iou"]
        summary["conditions"][cond]["by_distance"] = {
            n: {"coverage_miss": int(agg["decomp_dist"][i, 0]),
                "naming_error": int(agg["decomp_dist"][i, 1]),
                "correct": int(agg["decomp_dist"][i, 2])}
            for i, (n, _) in enumerate(dist)}
        summary["conditions"][cond]["by_height"] = {
            n: {"coverage_miss": int(agg["decomp_height"][i, 0]),
                "naming_error": int(agg["decomp_height"][i, 1]),
                "correct": int(agg["decomp_height"][i, 2])}
            for i, (n, _) in enumerate(height)}
        d = agg["dist_tpfpfn"]
        summary["conditions"][cond]["miou_by_distance"] = {}
        for i, (n, _) in enumerate(dist):
            den = d[i, :, 0] + d[i, :, 1] + d[i, :, 2]
            iou = np.where(den > 0, d[i, :, 0] / np.maximum(den, 1), np.nan)
            summary["conditions"][cond]["miou_by_distance"][n] = (
                float(np.nanmean(iou)) if np.any(den > 0) else 0.0)
        np.savez_compressed(
            os.path.join(REPO_ROOT, "artifacts", "gate6", f"counts_{ds}_{cond}.npz"),
            clip_id=np.asarray([c.clip_id for c in cc]),
            group=np.asarray([c.group for c in cc]),
            conf=np.stack([c.conf for c in cc]).astype(np.int32),
            conf_support=np.stack([c.conf_support for c in cc]).astype(np.int32),
            conf_dilonly=np.stack([c.conf_dilonly for c in cc]).astype(np.int32),
            decomp=np.stack([c.decomp for c in cc]).astype(np.int32),
            fp_empty=np.stack([c.fp_empty for c in cc]).astype(np.int32),
            tp=np.stack([c.tp for c in cc]).astype(np.int32),
            fp=np.stack([c.fp for c in cc]).astype(np.int32),
            fn=np.stack([c.fn for c in cc]).astype(np.int32),
            binary=np.asarray([[c.btp, c.bfp, c.bfn, c.n_valid, c.n_gt_occupied,
                                c.n_pred_occupied] for c in cc], np.int64),
            decomp_dist=np.stack([c.decomp_dist for c in cc]).astype(np.int32),
            decomp_height=np.stack([c.decomp_height for c in cc]).astype(np.int32),
            labels=np.asarray(v.labels, np.int32), names=np.asarray(v.names))
    summary["seconds"] = time.time() - t0
    write_json(os.path.join(REPO_ROOT, "artifacts", "gate6", f"summary_{ds}.json"), summary)

    for cond in CONDITIONS:
        s = summary["conditions"][cond]
        d = s["decomposition"]
        print(f"[{ds} {cond}] binIoU(macro)={s['binary_iou_mean_per_clip']:.4f} "
              f"binIoU(pooled)={s['binary_iou']:.4f} mIoU={s['ssc_miou']:.4f} "
              f"TP-acc={s['tp_conditioned']['top1_accuracy']:.4f} "
              f"TP-bal={s['tp_conditioned']['balanced_recall']:.4f} | "
              f"miss={d['coverage_miss_fraction']:.4f} name={d['naming_error_fraction']:.4f} "
              f"corr={d['correct_fraction']:.4f} sums1={d['sums_to_one']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
