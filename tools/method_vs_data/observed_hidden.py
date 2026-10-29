#!/usr/bin/env python
"""Where the accuracy lives: observed surface vs remembered surface vs genuinely hidden geometry.

This is ``canonical_rescore.py`` with one change to the regions and one added arm. It is not a
second evaluator: the confusion counts, the pooling call, the block definitions and the bootstrap
are the same lines, and every arm is scored in the same pass on the same anchors.

Regions. The existing strict construction (``observed_strict.py``) splits the valid volume by
whether a ray from one of the FIVE PERMITTED CAUSAL FRAMES reaches the voxel, using the
ground-truth occupancy and the ground-truth poses -- never a method's own reconstruction. The
brief asks for its ``never_observed`` leg to be cut by the anchor frustum, giving a four-way
partition of every valid voxel:

    anchor_observed              reached from the anchor camera
    earlier_only                 reached from an earlier permitted frame but not the anchor
    never_observed_in_frustum    reached from none of the five, inside the anchor frustum
    never_observed_out_frustum   reached from none of the five, outside the anchor frustum

The union ``never_observed`` is kept as its own row. Two other families are reported beside these
and are NOT the same partition: the frustum family (``in_frustum`` / ``out_frustum``) and the
hard-map evidence family (``A0fixed_observed`` / ``A0fixed_completed``). In-frustum is not
observed; out-of-frustum is not never-observed; hard-map known/unknown is not GT visibility.

Scope of the masks. They describe visibility under OUR five causal frames. OccAny's five-frame
configuration reads t..t+40, so a voxel that is never-observed here may be plainly visible to it.
That is a property of the comparison, stated, not corrected.

Arms. ``k360`` / ``waymo`` / ``once`` are our three source heads at ``GEOMETRY_SEED = 0``;
``map_only`` is the same pipeline with the completion head removed (``map_only_predictions.py``);
``occany_1f`` and ``occany_5f_fwd`` are OccAny's released outputs [reproduced].

Both conventions are computed in the SAME pass -- raw, and OccAny's 3x3x3 max-pool applied to the
prediction only, after loading and before the confusion counts. No mask and no ground truth is
ever dilated.

    PYTHONPATH=$PWD python tools/method_vs_data/observed_hidden.py \\
        --dataset semantickitti --device cuda:3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch                                                                           # noqa: E402
from core.datasets import nuscenes as NU, sources as S_                                # noqa: E402
from gates.gate8b import pooling as PL                                                 # noqa: E402
from tools.method_vs_data.canonical_rescore import OCCANY_ARMS, OURS_ROOTS, provenance  # noqa: E402
from tools.method_vs_data.map_only_predictions import load_map_only                     # noqa: E402
from tools.method_vs_data.observed_strict import frame_transforms, sk_pose_chain        # noqa: E402
from tools.method_vs_data.one_frame_score import OCC3D_OCCANY_FORWARD                   # noqa: E402
from tools.observability_split import runs as R, visibility as V                        # noqa: E402
from tools.occany_comparison import config as OC                                        # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS                            # noqa: E402
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction               # noqa: E402
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS, STATISTICS           # noqa: E402

IMAGE_SIZE = {"semantickitti": (1226, 370), "occ3d": (1600, 900)}
IGNORE = 255
POSTS = ("raw", "pooled")
#: the four mutually exclusive regions that must sum to the valid volume
PARTITION = ("anchor_observed", "earlier_only",
             "never_observed_in_frustum", "never_observed_out_frustum")
#: every region reported, partition first
REGIONS = ("all",) + PARTITION + ("never_observed", "in_frustum", "out_frustum",
                                  "A0fixed_observed", "A0fixed_completed")
ARMS_DEFAULT = ["k360", "waymo", "once", "map_only", "occany_1f", "occany_5f_fwd"]
#: (a, b, name): a is ours, b is the comparator
PAIRS = [(o, c, f"{o}_vs_{c}") for o in ("k360", "waymo", "once")
         for c in ("occany_1f", "occany_5f_fwd", "map_only")] + \
        [("k360", "waymo", "k360_vs_waymo"), ("k360", "once", "k360_vs_once"),
         ("map_only", "occany_1f", "map_only_vs_occany_1f"),
         ("map_only", "occany_5f_fwd", "map_only_vs_occany_5f_fwd")]


def occ3d_chain(ann, scene_cache, sample):
    """camera(j)->grid for the five permitted frames; grid = the anchor keyframe's ego frame.

    ``inv(T_ego_keyframe(anchor)->world) @ T_ego_cam(j)->world @ T_camera_to_ego_cam(j)`` -- the
    same single-timestamp chain ``core/datasets/nuscenes.Frame.T_camera_to_ego`` applies, which is
    why the anchor slot must collapse to the stored ``T_cam_to_grid``. Asserted per sample.
    """
    if sample.segment not in scene_cache:
        scene_cache[sample.segment] = NU.scene_frames(ann, sample.segment, "CAM_FRONT",
                                                      OC.NUSCENES_ROOT)
    frs = scene_cache[sample.segment]
    idx = sample.window(OC.PROTOCOLS["causal"])
    A = np.linalg.inv(np.asarray(frs[sample.target_pos].T_ego_keyframe_to_world, np.float64))
    Ts = [A @ np.asarray(frs[j].T_ego_cam_to_world, np.float64)
          @ np.asarray(frs[j].T_camera_to_ego_cam, np.float64) for j in idx]
    Ks = [np.asarray(frs[j].K, np.float64) for j in idx]
    return Ts, Ks


def strict_four(gt_occ, valid, Ts, Ks, img, dims, vs, origin, dev, stride, step):
    """The four-way partition plus the preserved ``never_observed`` union and the frustum family."""
    seen = [V.observed_from_camera(gt_occ, T, K, img, dims, vs, origin,
                                   pixel_stride=stride, step_m=step, device=str(dev))
            for T, K in zip(Ts, Ks)]
    anchor = seen[-1]
    union = np.zeros_like(anchor)
    for s in seen:
        union |= s
    fov = V.frustum(Ts[-1], Ks[-1], img, dims, vs, origin)
    never = valid & ~union
    m = {"all": valid,
         "anchor_observed": valid & anchor,
         "earlier_only": valid & union & ~anchor,
         "never_observed_in_frustum": never & fov,
         "never_observed_out_frustum": never & ~fov,
         "never_observed": never,
         "in_frustum": valid & fov,
         "out_frustum": valid & ~fov}
    p = [m[k] for k in PARTITION]
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            assert not (p[i] & p[j]).any(), f"{PARTITION[i]} overlaps {PARTITION[j]}"
    assert int(sum(int(x.sum()) for x in p)) == int(valid.sum()), "partition misses valid voxels"
    assert np.array_equal(m["never_observed"],
                          m["never_observed_in_frustum"] | m["never_observed_out_frustum"])
    return m, seen, fov


def score(dataset: str, arms, limit=None, device="cuda:3", stride=2, step=0.1) -> dict:
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

    ours = [a for a in arms if a in OURS_ROOTS]
    missing = {a: [s for s in sids
                   if not os.path.exists(os.path.join(OURS_ROOTS[a], dataset, f"{s}.npz"))]
               for a in ours}
    if "map_only" in arms:
        from tools.method_vs_data.map_only_predictions import cache_path
        missing["map_only"] = [s for s in sids if not os.path.exists(cache_path(dataset, s))]
    drop = set().union(*missing.values()) if missing else set()
    sids = [s for s in sids if s not in drop]
    assert sids, "no anchor has a prediction from every requested arm"
    arms = [a for a in arms if a in OURS_ROOTS or a == "map_only"
            or OCCANY_ARMS.get(a) in runs]

    ref = CanonicalPrediction.load(os.path.join(OURS_ROOTS[ours[0]], dataset, f"{sids[0]}.npz"))
    dims, vs, origin = ref.dims, ref.voxel_size, ref.origin
    img = IMAGE_SIZE[dataset]
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if dataset == "semantickitti":
        poses, cam_to_velo = sk_pose_chain(ZC.REPO_ROOT, "08")
    else:
        ann, scene_cache = NU.load_annotations(S_.OCC3D_ROOT), {}

    acc = {p: defaultdict(lambda: 0) for p in POSTS}
    per = {p: defaultdict(list) for p in POSTS}
    comp, frame_seen = defaultdict(lambda: 0), defaultdict(lambda: 0)
    segs, used, t0 = [], [], time.time()
    for i, sid in enumerate(sids):
        s = by_sid[sid]
        vols = {a: CanonicalPrediction.load(os.path.join(OURS_ROOTS[a], dataset, f"{sid}.npz"))
                for a in ours}
        a0 = vols[ours[0]]
        for a, p in vols.items():
            assert np.allclose(p.T_cam_to_grid, a0.T_cam_to_grid, atol=1e-6), (sid, a)
            assert tuple(p.dims) == tuple(dims) and p.occupied.shape == a0.occupied.shape, (sid, a)
        occs = {a: vols[a].occupied for a in ours}
        if "map_only" in arms:
            mo, T_mo = load_map_only(dataset, sid)
            assert np.allclose(T_mo, a0.T_cam_to_grid, atol=1e-6), (sid, "map_only transform")
            occs["map_only"] = mo

        _lk = "occany_1f_geom" if "occany_1f_geom" in runs else sorted(runs)[0]
        label, _ = R.load_occany(runs[_lk], sid, dims)
        valid = label != IGNORE
        gt_occ = valid & (label != spec.empty_class)

        if dataset == "semantickitti":
            natives = [int(k.split("_")[1]) for k in (f["key"] for f in ZS.history_frames(s))]
            Ts = frame_transforms(poses, cam_to_velo, natives, natives[-1])
            Ks = [a0.K] * len(Ts)
        else:
            Ts, Ks = occ3d_chain(ann, scene_cache, s)
        assert np.allclose(Ts[-1], a0.T_cam_to_grid, atol=1e-6), (sid, "anchor transform gate")
        masks, seen, _fov = strict_four(gt_occ, valid, Ts, Ks, img, dims, vs, origin,
                                        dev, stride, step)
        masks["A0fixed_observed"] = valid & a0.observed
        masks["A0fixed_completed"] = valid & ~a0.observed
        for k in REGIONS:
            comp[f"{k}__valid"] += int(masks[k].sum())
            comp[f"{k}__gt"] += int((masks[k] & gt_occ).sum())
        for j, sv in enumerate(seen):
            frame_seen[j] += int((valid & sv & gt_occ).sum())

        for arm in arms:
            if arm in occs:
                base = occs[arm]
            else:
                _l, vol = R.load_occany(runs[OCCANY_ARMS[arm]], sid, dims)
                base = (vol != spec.empty_class) & (vol != IGNORE)
            for post in POSTS:
                occ = base if post == "raw" else PL.geometry_dilation(
                    torch.from_numpy(base).reshape(dims)).reshape(-1).numpy()
                tot = np.zeros(3, np.int64)
                for k in REGIONS:
                    mm = masks[k]
                    c = np.array([int((occ & gt_occ & mm).sum()),
                                  int((occ & ~gt_occ & mm).sum()),
                                  int((~occ & gt_occ & mm).sum())], np.int64)
                    acc[post][f"{arm}__{k}"] = acc[post][f"{arm}__{k}"] + c
                    per[post][f"{arm}__{k}"].append(c)
                    if k in PARTITION:
                        tot = tot + c
                # the four regions must reconstruct the global counts exactly
                assert np.array_equal(tot, np.array(
                    [int((occ & gt_occ & valid).sum()), int((occ & ~gt_occ & valid).sum()),
                     int((~occ & gt_occ & valid).sum())], np.int64)), (sid, arm, post)
        segs.append(s.segment)
        used.append(sid)
        if (i + 1) % 25 == 0:
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
    iou_of = lambda c: 100 * c[0] / max(c.sum(), 1)                            # noqa: E731

    tv, tg = comp["all__valid"], comp["all__gt"]
    out = {"provenance": provenance(ours), "dataset": dataset, "n_samples": len(used),
           "arms": arms, "regions": list(REGIONS), "partition": list(PARTITION),
           "visibility": {"source": "GT occupancy + GT poses of the five permitted causal frames",
                          "pixel_stride": stride, "step_m": step, "max_range_m": 80.0,
                          "rule": "every voxel a ray enters up to and including the first "
                                  "GT-occupied voxel is observed",
                          "window_scope": "our five causal frames; OccAny's 5-frame arm reads "
                                          "t..t+40 and is NOT limited by these masks"},
           "n_missing_predictions": {a: len(v) for a, v in missing.items()},
           "anchors_dropped": sorted(drop),
           "statistics": {"blocks": len(blocks), "resamples": STATISTICS.resamples,
                          "bootstrap_seed": STATISTICS.seed,
                          "block_rule": ("one per scene" if dataset == "occ3d"
                                         else f"{STATISTICS.semantickitti_block_samples} contiguous anchors"),
                          "pairing": "every arm resampled on the same anchors, one pass",
                          "uncertainty_covered": "evaluation-sample only; NOT training-seed spread",
                          "global_iou_rule": "recomputed from summed TP/FP/FN per resample"},
           "composition": {k: {"valid_voxels": comp[f"{k}__valid"],
                               "gt_occupied": comp[f"{k}__gt"],
                               "share_of_valid": comp[f"{k}__valid"] / max(tv, 1),
                               "share_of_gt_occupied": comp[f"{k}__gt"] / max(tg, 1)}
                           for k in REGIONS},
           "per_frame_observed_gt_occupied": {
               f"frame_{j}": {"gt_occupied_seen": frame_seen[j],
                              "share_of_gt_occupied": frame_seen[j] / max(tg, 1)}
               for j in sorted(frame_seen)},
           "iou": {}, "contrasts": {}}
    for post in POSTS:
        stack = {k: np.stack(v) for k, v in per[post].items()}
        out["iou"][post] = {k: {a: sc(acc[post][f"{a}__{k}"]) for a in arms} for k in REGIONS}

        def ci(a, b, k):
            d = np.array([iou_of(stack[f"{a}__{k}"][p].sum(0))
                          - iou_of(stack[f"{b}__{k}"][p].sum(0)) for p in picks])
            lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
            return {"delta": out["iou"][post][k][a]["iou"] - out["iou"][post][k][b]["iou"],
                    "ci_low": lo, "ci_high": hi, "significant_at_95": bool(lo > 0 or hi < 0)}

        out["contrasts"][post] = {name: {"a": a, "b": b, **{k: ci(a, b, k) for k in REGIONS}}
                                  for a, b, name in PAIRS if a in arms and b in arms}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["semantickitti", "occ3d"])
    ap.add_argument("--arms", nargs="+", default=ARMS_DEFAULT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--pixel-stride", type=int, default=2)
    ap.add_argument("--step-m", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = score(a.dataset, a.arms, a.limit, a.device, a.pixel_stride, a.step_m)
    out = a.out or os.path.join(ZC.REPO_ROOT, "artifacts", "method_vs_data", "observed_hidden",
                                f"regional_{a.dataset}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(r, open(out, "w"), indent=1, default=float)
    print(f"\n{a.dataset}: {r['n_samples']} anchors, arms {r['arms']}")
    print(f"  {'region':28s} {'%valid':>7s} {'%gtocc':>7s}")
    for k in REGIONS:
        c = r["composition"][k]
        print(f"  {k:28s} {c['share_of_valid']*100:6.2f}% {c['share_of_gt_occupied']*100:6.2f}%")
    for post in POSTS:
        print(f"\n  --- {post} IoU")
        print("  " + f"{'region':28s}" + "".join(f"{a_:>12s}" for a_ in r["arms"]))
        for k in REGIONS:
            print(f"  {k:28s}" + "".join(f"{r['iou'][post][k][a_]['iou']:12.2f}"
                                         for a_ in r["arms"]))
        for name, c in r["contrasts"][post].items():
            x = c["all"]
            print(f"  {name:34s} all {x['delta']:+7.2f} [{x['ci_low']:+.2f}, {x['ci_high']:+.2f}]"
                  f"{' *' if x['significant_at_95'] else ''}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
