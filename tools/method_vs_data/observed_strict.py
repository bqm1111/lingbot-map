#!/usr/bin/env python
"""Brief §9 — accumulated reconstruction vs genuine completion, on identical populations.

The existing `tools/observability_split` split is an **anchor-frame** visibility split:
observed / occluded / out-of-view, all judged from the target frame's camera. The brief warns
that this is not the distinction the method claim needs, and it is right: a voxel outside the
anchor frustum may have been seen perfectly well four frames earlier, and calling that
"completion of unobserved space" would be wrong.

This tool computes the stricter partition the brief asks for, over every valid voxel:

  anchor_observed   a ray from the ANCHOR camera reaches it, up to and including the first
                    GT-occupied voxel it meets.
  earlier_only      not anchor-observed, but reached the same way from one of the four
                    EARLIER causal frames. This is accumulated reconstruction: the permitted
                    input did observe it, just not at t.
  never_observed    reached from none of the five permitted frames. Only here is a correct
                    prediction genuine completion of unobserved space.

Both methods are scored on identical populations, and the populations are built from the
**ground-truth occupancy and the ground-truth poses** of the five permitted frames -- not
from either method's reconstruction or pose estimate. So the split is a property of the
permitted input, which is what makes it usable as a common denominator.

Pose chain, with a gate. The grid frame is the anchor's velodyne frame, so

    T_cam(j)->grid = inv(M_anchor) @ M_j @ cam_to_velo,    M_n = P_cam2_c2w(n) @ inv(cam_to_velo)

and for the anchor this must collapse to ``cam_to_velo`` exactly, which is also the
``T_cam_to_grid`` stored in the canonical prediction. That identity is asserted per sample;
if the chain were wrong, the split would be silently wrong, so it is checked rather than
trusted.

    PYTHONPATH=$PWD python tools/method_vs_data/observed_strict.py --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.datasets import sources as S_                                                 # noqa: E402
from core.datasets.kitti_odometry import load_poses_cam2_c2w, parse_calibration         # noqa: E402
from tools.observability_split import runs as R, visibility as V                        # noqa: E402
from gates.gate8b import pooling as PL                                                  # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS                            # noqa: E402
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction               # noqa: E402
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS, STATISTICS           # noqa: E402

REGIONS = ("all", "anchor_observed", "earlier_only", "never_observed")
#: scored methods: our causal system and the only causal OccAny run that exists locally.
#: OccAny's 5-frame mode consumes frames t..t+40, so it is not a causal comparator.
METHODS = ("ours", "occany_1f_geom")
IGNORE = 255


def sk_pose_chain(repo_root: str, seq: str):
    """``(poses_cam2_to_world[N,4,4], cam_to_velo[4,4])`` for one SemanticKITTI sequence."""
    root = os.path.join(repo_root, "data/kitti/dataset")
    calib = parse_calibration(os.path.join(root, "sequences", seq, "calib.txt"))
    poses = load_poses_cam2_c2w(os.path.join(root, "sequences", seq, "poses.txt"), calib)
    cam_to_velo, _K = S_._sk_calib(repo_root, seq)
    return poses, np.asarray(cam_to_velo, np.float64)


def frame_transforms(poses, cam_to_velo, natives, anchor_native):
    """camera->grid for each permitted frame; grid = the anchor's velodyne frame."""
    inv_ctv = np.linalg.inv(cam_to_velo)
    M = {n: poses[n] @ inv_ctv for n in natives}                 # velodyne(n) -> world
    M_a_inv = np.linalg.inv(M[anchor_native])
    return [M_a_inv @ M[n] @ cam_to_velo for n in natives]


def strict_regions(gt_occ, valid, Ts, K, img, dims, vs, origin, dev, stride, step):
    seen = []
    for T in Ts:
        seen.append(V.observed_from_camera(gt_occ, T, K, img, dims, vs, origin,
                                           pixel_stride=stride, step_m=step, device=str(dev)))
    anchor = seen[-1]                                            # the anchor is the last slot
    union = np.zeros_like(anchor)
    for s in seen:
        union |= s
    a = valid & anchor
    e = valid & union & ~anchor
    n = valid & ~union
    assert not (a & e).any() and not (a & n).any() and not (e & n).any()
    assert int(a.sum() + e.sum() + n.sum()) == int(valid.sum())
    return {"all": valid, "anchor_observed": a, "earlier_only": e, "never_observed": n}, seen


def run(limit=None, stride=2, step=0.1, device="cuda:0", post="raw"):
    ds = "semantickitti"
    spec = DATASET_SPECS[ds]
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    sids = R.samples(ds)[:limit]
    img = R.IMAGE_SIZE[ds]
    ref = CanonicalPrediction.load(os.path.join(R.RUNS[ds][0].path, f"{sids[0]}.npz"))
    dims, vs, origin = ref.dims, ref.voxel_size, ref.origin
    poses, cam_to_velo = sk_pose_chain(ZC.REPO_ROOT, "08")
    by_sid = {s.sample_id: s for s in ZS.evaluated_samples(ds)}
    runs = {r.key: r for r in R.RUNS[ds]}

    acc = defaultdict(lambda: 0)
    per = defaultdict(list)
    comp = defaultdict(lambda: 0)
    t0 = time.time()
    for i, sid in enumerate(sids):
        p = CanonicalPrediction.load(os.path.join(runs["ours"].path, f"{sid}.npz"))
        label, _ = R.load_occany(runs["occany_1f_geom"], sid, dims)
        valid = label != IGNORE
        gt_occ = valid & (label != spec.empty_class)

        s = by_sid[sid]
        natives = [int(k.split("_")[1]) for k in (f["key"] for f in ZS.history_frames(s))]
        Ts = frame_transforms(poses, cam_to_velo, natives, natives[-1])
        assert np.allclose(Ts[-1], cam_to_velo, atol=1e-9), sid
        assert np.allclose(Ts[-1], p.T_cam_to_grid, atol=1e-6), (sid, "anchor transform gate")

        masks, seen = strict_regions(gt_occ, valid, Ts, p.K, img, dims, vs, origin,
                                     dev, stride, step)
        for r, m in masks.items():
            comp[f"{r}__valid"] = comp[f"{r}__valid"] + int(m.sum())
            comp[f"{r}__gt_occ"] = comp[f"{r}__gt_occ"] + int((m & gt_occ).sum())
        for j, sv in enumerate(seen):
            comp[f"frame{j}__observed_gt_occ"] = comp[f"frame{j}__observed_gt_occ"] + \
                int((valid & sv & gt_occ).sum())

        for key in METHODS:
            r_ = runs[key]
            if r_.kind == "ours":
                occ = p.occupied
            else:
                _lab, vol = R.load_occany(r_, sid, dims)
                occ = (vol != spec.empty_class) & (vol != IGNORE)
            if post == "pooled":
                occ = PL.geometry_dilation(torch.from_numpy(occ).reshape(dims)).reshape(-1).numpy()
            for r, m in masks.items():
                c = np.array([int((occ & gt_occ & m).sum()), int((occ & ~gt_occ & m).sum()),
                              int((~occ & gt_occ & m).sum())], np.int64)
                acc[f"{key}__{r}"] = acc[f"{key}__{r}"] + c
                per[f"{key}__{r}"].append(c)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sids)}  {(time.time() - t0) / (i + 1):.2f}s each", flush=True)

    def sc(c):
        tp, fp, fn = (float(x) for x in c)
        return {"iou": 100 * tp / max(tp + fp + fn, 1.0),
                "precision": 100 * tp / max(tp + fp, 1.0),
                "recall": 100 * tp / max(tp + fn, 1.0),
                "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    n = len(sids)
    blocks = [list(range(a, min(a + STATISTICS.semantickitti_block_samples, n)))
              for a in range(0, n, STATISTICS.semantickitti_block_samples)]
    rng = np.random.default_rng(STATISTICS.seed)
    picks = [np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
             for _ in range(STATISTICS.resamples)]
    stack = {k: np.stack(v) for k, v in per.items()}

    out = {"dataset": ds, "n_samples": n, "postprocess": post,
           "pixel_stride": stride, "step_m": step,
           "methods": {k: runs[k].note for k in METHODS},
           "windows": {k: runs[k].window for k in METHODS},
           "composition": {}, "per_frame_observed_gt_occupied": {},
           "iou": {}, "delta": {},
           "statistics": {"blocks": len(blocks), "resamples": STATISTICS.resamples,
                          "bootstrap_seed": STATISTICS.seed,
                          "block_rule": f"{STATISTICS.semantickitti_block_samples} contiguous anchors",
                          "paired": "both methods resampled on the same anchors"}}
    tot_v = comp["all__valid"]; tot_o = comp["all__gt_occ"]
    for r in REGIONS:
        out["composition"][r] = {
            "valid_voxels": comp[f"{r}__valid"], "gt_occupied": comp[f"{r}__gt_occ"],
            "share_of_valid": comp[f"{r}__valid"] / max(tot_v, 1),
            "share_of_gt_occupied": comp[f"{r}__gt_occ"] / max(tot_o, 1)}
    for j in range(5):
        out["per_frame_observed_gt_occupied"][f"frame_{j}"] = {
            "gt_occupied_seen": comp[f"frame{j}__observed_gt_occ"],
            "share_of_gt_occupied": comp[f"frame{j}__observed_gt_occ"] / max(tot_o, 1)}
    for r in REGIONS:
        out["iou"][r] = {k: sc(acc[f"{k}__{r}"]) for k in METHODS}
        iou = lambda c: 100 * c[0] / max(c.sum(), 1)
        d = np.array([iou(stack[f"ours__{r}"][p_].sum(0))
                      - iou(stack[f"occany_1f_geom__{r}"][p_].sum(0)) for p_ in picks])
        lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        out["delta"][r] = {"ours_minus_occany": out["iou"][r]["ours"]["iou"]
                           - out["iou"][r]["occany_1f_geom"]["iou"],
                           "ci_low": lo, "ci_high": hi,
                           "significant_at_95": bool(lo > 0 or hi < 0)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pixel-stride", type=int, default=2)
    ap.add_argument("--step-m", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--postprocess", default="raw", choices=["raw", "pooled"])
    ap.add_argument("--out", default="artifacts/method_vs_data/results/observed_strict.json")
    a = ap.parse_args()
    r = run(a.limit, a.pixel_stride, a.step_m, a.device, a.postprocess)
    print(f"\n=== SemanticKITTI, {r['n_samples']} anchors, {r['postprocess']}")
    print(f"  {'region':17s} {'% of valid':>11s} {'% of GT occ':>12s}")
    for k, v in r["composition"].items():
        print(f"  {k:17s} {v['share_of_valid']*100:10.2f}% {v['share_of_gt_occupied']*100:11.2f}%")
    print("  GT-occupied surface seen, per permitted frame (frame_4 = anchor):")
    for k, v in r["per_frame_observed_gt_occupied"].items():
        print(f"    {k}: {v['share_of_gt_occupied']*100:5.2f}%")
    print(f"  {'region':17s} {'ours':>8s} {'OccAny1f':>9s} {'delta':>8s} {'95% CI':>20s}")
    for k in REGIONS:
        i = r["iou"][k]; d = r["delta"][k]
        print(f"  {k:17s} {i['ours']['iou']:8.2f} {i['occany_1f_geom']['iou']:9.2f} "
              f"{d['ours_minus_occany']:+8.2f} "
              f"[{d['ci_low']:+.2f},{d['ci_high']:+.2f}]"
              f"{'*' if d['significant_at_95'] else ' '}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(r, open(a.out, "w"), indent=1, default=float)
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
