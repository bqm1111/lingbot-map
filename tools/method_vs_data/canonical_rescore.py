#!/usr/bin/env python
"""Canonical-protocol rescore of EVERY available training-source arm, in one pass.

This is a generalization of ``once_arm/cli/score_vs_occany.py`` -- not a second evaluator.
The scoring logic, region definitions, pooling call, block construction and bootstrap are
the same lines of code; the only thing this file adds is that the "ours" arm is no longer
hard-wired to one prediction root, so the KITTI-360, Waymo and ONCE arms are all scored on
the **same anchors in the same pass** and every pair is therefore paired by construction.

Why it exists: ``waymo_arm_vs_occany.md`` §5 records two gaps in the Waymo-vs-OccAny
answer -- no pooled convention and no paired bootstrap -- and states that closing them
needs a scorer that applies the canonical pooling to the stored Waymo volumes. This is
that scorer.

Arms (any subset, ``--arms``):

* ``k360``   -- the frozen KITTI-360 checkpoint, read from the frozen canonical cache;
* ``waymo``  -- the Waymo arm, read from its own prediction root;
* ``once``   -- the ONCE arm, read from its own prediction root;
* ``occany_1f`` / ``occany_5f_fwd`` -- OccAny's released outputs [reproduced].

All of our arms' canonical predictions are produced at ``GEOMETRY_SEED = 0``, so every
"ours" row here is a **single-seed** row. Seed spread is a separate quantity, measured by
the frozen evaluator on its own smaller populations, and the two must not be mixed.

Both post-processing conventions are reported and never mixed: ``raw``, and ``pooled`` =
OccAny's own 3x3x3 max-pool applied to the PREDICTION only, after loading and before the
confusion counts. Ground truth, the valid mask and every region mask are left undilated.

    PYTHONPATH=$PWD python tools/method_vs_data/canonical_rescore.py \\
        --dataset semantickitti --post raw --device cuda:0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch                                                                           # noqa: E402
from gates.gate8b import pooling as PL                                                 # noqa: E402
from tools.method_vs_data.observed_strict import (frame_transforms, sk_pose_chain,      # noqa: E402
                                                  strict_regions)
from tools.method_vs_data.one_frame_score import OCC3D_OCCANY_FORWARD                  # noqa: E402
from tools.observability_split import runs as R, visibility as V                        # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS                            # noqa: E402
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction               # noqa: E402
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS, STATISTICS           # noqa: E402

#: prediction roots for our arms. ``k360`` is the frozen cache; the others are the arms' own
#: roots, which their predict_canonical/predict_semantic scripts wrote and never overwrote.
OURS_ROOTS = {
    "k360": ZC.PREDICTION_ROOT,
    "waymo": "/media/minh/MINH1/lingbot_waymo_arm/zero_shot_predictions",
    "once": "/media/minh/MINH1/lingbot_once_arm/zero_shot_predictions",
}
OCCANY_ARMS = {"occany_1f": "occany_1f_geom", "occany_5f_fwd": "occany_5f_geom"}
IMAGE_SIZE = {"semantickitti": (1226, 370), "occ3d": (1600, 900)}
IGNORE = 255
#: every ordered pair worth a confidence interval, as (a, b, name)
PAIRS = [("waymo", "occany_1f", "waymo_vs_occany_1frame_causal"),
         ("waymo", "occany_5f_fwd", "waymo_vs_occany_5frame_forward"),
         ("waymo", "k360", "waymo_vs_kitti360_same_recipe"),
         ("waymo", "once", "waymo_vs_once_same_recipe"),
         ("once", "occany_1f", "once_vs_occany_1frame_causal"),
         ("once", "occany_5f_fwd", "once_vs_occany_5frame_forward"),
         ("once", "k360", "once_vs_kitti360_same_recipe"),
         ("k360", "occany_1f", "kitti360_vs_occany_1frame_causal"),
         ("k360", "occany_5f_fwd", "kitti360_vs_occany_5frame_forward")]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(arms) -> dict:
    """Code revision, working-tree state and the checkpoint identity behind each arm."""
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ZC.REPO_ROOT,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ZC.REPO_ROOT,
                           capture_output=True, text=True).stdout.strip().splitlines()
    ck = {}
    for arm, d in (("k360", os.path.join(ZC.REPO_ROOT, "artifacts", "baseline_reconstruction",
                                         "checkpoints")),
                   ("waymo", os.path.join(ZC.REPO_ROOT, "artifacts", "waymo_arm", "checkpoints")),
                   ("once", os.path.join(ZC.REPO_ROOT, "artifacts", "method_vs_data",
                                         "once_arm", "checkpoints"))):
        if arm not in arms:
            continue
        p = os.path.join(d, "geo_seed0_step06000.pt")
        ck[arm] = {"path": os.path.relpath(p, ZC.REPO_ROOT) if os.path.exists(p) else None,
                   "sha256": _sha256(p) if os.path.exists(p) else None,
                   "seed_used_for_canonical_predictions": ZC.GEOMETRY_SEED}
    return {"git_rev": rev, "working_tree_modified": dirty, "checkpoints": ck,
            "prediction_roots": {a: OURS_ROOTS[a] for a in arms if a in OURS_ROOTS},
            "geometry_seed": ZC.GEOMETRY_SEED,
            "tool": "tools/method_vs_data/canonical_rescore.py"}


def score(dataset: str, arms, limit=None, post="raw", device="cuda:0") -> dict:
    spec = DATASET_SPECS[dataset]
    runs = {r.key: r for r in R.RUNS[dataset]} if dataset in R.RUNS else {}
    if dataset == "occ3d" and os.path.isdir(OCC3D_OCCANY_FORWARD):
        runs["occany_5f_geom"] = R.Run("occany_5f_geom", "occany", "forward",
                                       OCC3D_OCCANY_FORWARD, False, 5,
                                       "OccAny 5-frame FORWARD nuScenes preset [reproduced]")
    sids = R.samples(dataset)[:limit] if dataset in R.RUNS else \
        [s.sample_id for s in ZS.evaluated_samples(dataset)[:limit]]
    by_sid = {s.sample_id: s for s in ZS.evaluated_samples(dataset)}
    sids = [s for s in sids if s in by_sid]

    # anchor-identity gate: keep only anchors EVERY requested arm predicted, so the
    # bootstrap is paired and no two rows describe different populations.
    ours = [a for a in arms if a in OURS_ROOTS]
    missing = {a: [s for s in sids
                   if not os.path.exists(os.path.join(OURS_ROOTS[a], dataset, f"{s}.npz"))]
               for a in ours}
    drop = set().union(*missing.values()) if missing else set()
    sids = [s for s in sids if s not in drop]
    assert sids, "no anchor has a prediction from every requested arm"

    ref = CanonicalPrediction.load(os.path.join(OURS_ROOTS[ours[0]], dataset, f"{sids[0]}.npz"))
    dims, vs, origin = ref.dims, ref.voxel_size, ref.origin
    W, H = IMAGE_SIZE[dataset]
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    arms = [a for a in arms if a in OURS_ROOTS or OCCANY_ARMS.get(a) in runs]
    if dataset == "semantickitti":
        poses, cam_to_velo = sk_pose_chain(ZC.REPO_ROOT, "08")

    acc, per, comp = defaultdict(lambda: 0), defaultdict(list), defaultdict(lambda: 0)
    segs, used, t0 = [], [], time.time()
    for i, sid in enumerate(sids):
        s = by_sid[sid]
        vols = {a: CanonicalPrediction.load(os.path.join(OURS_ROOTS[a], dataset, f"{sid}.npz"))
                for a in ours}
        a0 = vols[ours[0]]
        for a, p in vols.items():
            assert np.allclose(p.T_cam_to_grid, a0.T_cam_to_grid, atol=1e-6), \
                f"{sid}: arm {a} disagrees on the anchor transform"
            assert tuple(p.dims) == tuple(dims) and p.occupied.shape == a0.occupied.shape, \
                f"{sid}: arm {a} has a different grid"
            assert np.isfinite(p.T_cam_to_grid).all() and np.isfinite(p.K).all(), f"{sid}/{a}"
        if runs:
            _lk = "occany_1f_geom" if "occany_1f_geom" in runs else sorted(runs)[0]
            label, _ = R.load_occany(runs[_lk], sid, dims)
        else:
            from tools.occany_comparison import scoring as SC
            label = SC.load_occany(dataset, s.occany_pkl, dims).label.reshape(-1)
        valid = label != IGNORE
        gt_occ = valid & (label != spec.empty_class)

        masks = {"all": valid}
        if dataset == "semantickitti":
            natives = [int(k.split("_")[1]) for k in (f["key"] for f in ZS.history_frames(s))]
            Ts = frame_transforms(poses, cam_to_velo, natives, natives[-1])
            assert np.allclose(Ts[-1], a0.T_cam_to_grid, atol=1e-6), sid
            m, _ = strict_regions(gt_occ, valid, Ts, a0.K, (W, H), dims, vs, origin, dev,
                                  2, 0.1)
            masks.update({k: v for k, v in m.items() if k != "all"})
        fov = V.frustum(a0.T_cam_to_grid, a0.K, (W, H), dims, vs, origin)
        masks["in_frustum"] = valid & fov
        masks["out_frustum"] = valid & ~fov
        masks["A0fixed_observed"] = valid & a0.observed
        masks["A0fixed_completed"] = valid & ~a0.observed
        for k, mm in masks.items():
            comp[f"{k}__valid"] += int(mm.sum())
            comp[f"{k}__gt"] += int((mm & gt_occ).sum())

        for arm in arms:
            if arm in OURS_ROOTS:
                occ = vols[arm].occupied
            else:
                _l, vol = R.load_occany(runs[OCCANY_ARMS[arm]], sid, dims)
                occ = (vol != spec.empty_class) & (vol != IGNORE)
            if post == "pooled":
                occ = PL.geometry_dilation(
                    torch.from_numpy(occ).reshape(dims)).reshape(-1).numpy()
            for k, mm in masks.items():
                c = np.array([int((occ & gt_occ & mm).sum()), int((occ & ~gt_occ & mm).sum()),
                              int((~occ & gt_occ & mm).sum())], np.int64)
                acc[f"{arm}__{k}"] = acc[f"{arm}__{k}"] + c
                per[f"{arm}__{k}"].append(c)
        segs.append(s.segment); used.append(sid)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(sids)}  {(time.time() - t0) / (i + 1):.2f}s each", flush=True)

    def sc(c):
        tp, fp, fn = (float(x) for x in c)
        return {"iou": 100 * tp / max(tp + fp + fn, 1.0),
                "precision": 100 * tp / max(tp + fp, 1.0),
                "recall": 100 * tp / max(tp + fn, 1.0),
                "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    if dataset == "occ3d":
        by = {}
        for i, sg in enumerate(segs):
            by.setdefault(sg, []).append(i)
        blocks = [v for _, v in sorted(by.items())]
    else:
        b = STATISTICS.semantickitti_block_samples
        blocks = [list(range(a, min(a + b, len(used)))) for a in range(0, len(used), b)]
    rng = np.random.default_rng(STATISTICS.seed)
    picks = [np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
             for _ in range(STATISTICS.resamples)]
    stack = {k: np.stack(v) for k, v in per.items()}
    iou_of = lambda c: 100 * c[0] / max(c.sum(), 1)                            # noqa: E731
    regions = list(dict.fromkeys(k.split("__", 1)[1] for k in acc))
    out = {"provenance": provenance(arms), "dataset": dataset, "n_samples": len(used),
           "postprocess": post, "arms": arms,
           "n_missing_predictions": {a: len(v) for a, v in missing.items()},
           "anchors_dropped": sorted(drop),
           "statistics": {"blocks": len(blocks), "resamples": STATISTICS.resamples,
                          "bootstrap_seed": STATISTICS.seed,
                          "block_rule": ("one per scene" if dataset == "occ3d"
                                         else f"{STATISTICS.semantickitti_block_samples} contiguous anchors"),
                          "pairing": "every arm resampled on the same anchors, one pass",
                          "block_unit": ("nuScenes scene" if dataset == "occ3d"
                                         else "contiguous run of 25 sequence-08 anchors"),
                          "uncertainty_covered": "evaluation-sample only; NOT training-seed spread",
                          "global_iou_rule": "recomputed from summed TP/FP/FN, not averaged per-anchor"},
           "composition": {}, "iou": {}, "contrasts": {}}
    tv, tg = comp["all__valid"], comp["all__gt"]
    for k in regions:
        out["composition"][k] = {"valid_voxels": comp[f"{k}__valid"],
                                 "gt_occupied": comp[f"{k}__gt"],
                                 "share_of_valid": comp[f"{k}__valid"] / max(tv, 1),
                                 "share_of_gt_occupied": comp[f"{k}__gt"] / max(tg, 1)}
    for k in regions:
        out["iou"][k] = {arm: sc(acc[f"{arm}__{k}"]) for arm in arms}

    def ci(a, b, k):
        d = np.array([iou_of(stack[f"{a}__{k}"][p].sum(0)) - iou_of(stack[f"{b}__{k}"][p].sum(0))
                      for p in picks])
        lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        return {"delta": out["iou"][k][a]["iou"] - out["iou"][k][b]["iou"],
                "ci_low": lo, "ci_high": hi, "significant_at_95": bool(lo > 0 or hi < 0)}

    for a, b, name in PAIRS:
        if a in arms and b in arms:
            out["contrasts"][name] = {"a": a, "b": b, **{k: ci(a, b, k) for k in regions}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["semantickitti", "occ3d"])
    ap.add_argument("--post", default="raw", choices=["raw", "pooled"])
    ap.add_argument("--arms", nargs="+",
                    default=["k360", "waymo", "once", "occany_1f", "occany_5f_fwd"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = score(a.dataset, a.arms, a.limit, a.post, a.device)
    out = a.out or os.path.join(ZC.REPO_ROOT, "artifacts", "method_vs_data",
                                "waymo_canonical_rescore",
                                f"canonical_{a.dataset}_{a.post}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(r, open(out, "w"), indent=1, default=float)
    print(f"\n{a.dataset} / {a.post}: {r['n_samples']} anchors, arms {r['arms']}")
    for arm in r["arms"]:
        v = r["iou"]["all"][arm]
        print(f"  {arm:14s} IoU {v['iou']:6.2f}  P {v['precision']:6.2f}  R {v['recall']:6.2f}"
              f"  tp {v['tp']:>10d} fp {v['fp']:>10d} fn {v['fn']:>10d}")
    for name, c in r["contrasts"].items():
        x = c["all"]
        print(f"  {name:38s} {x['delta']:+6.2f} [{x['ci_low']:+.2f}, {x['ci_high']:+.2f}]"
              f"{' *' if x['significant_at_95'] else ''}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
