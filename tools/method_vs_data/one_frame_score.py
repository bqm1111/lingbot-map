#!/usr/bin/env python
"""Phase A — score every one-frame / five-frame arm on identical populations.

Arms, all scored by one function in one pass so no column can differ by scorer:

  A0  five_frame_unet   the deployed system -- read from the EXISTING canonical predictions,
                        so this arm is anchored to the already-published 26.396948
  A1  one_frame_unet    one causal frame, same frozen CompletionUNet
  A2  one_frame_map     one causal frame, hard map only
      five_frame_map    five frames, hard map only (the published map-only floor)
  A3  occany_1f         OccAny's 1-frame causal preset [reproduced]
      occany_5f_fwd     OccAny's 5-frame FORWARD preset (t..t+40) -- not causal
  D   one_frame_unet_scale5   labelled diagnostic, five-frame scale forced into A1

Populations. The brief requires that paired contrasts use the SAME benchmark-defined strata
for every arm, and that the one-frame arm's own visibility also be reported so a reader does
not mistake changed input coverage for a changed evaluation stratum. Both are given:

  fixed strata (SemanticKITTI): anchor_observed / earlier_only / never_observed, from a GT
      raycast over the five permitted frames' GT poses -- identical for every arm.
  one-frame-own strata: anchor_observed / own_never_observed, where
      own_never_observed = earlier_only UNION never_observed. For a one-frame input there are
      no earlier frames, so this IS its true visibility split, and it is derived from the same
      masks rather than recomputed, which keeps the two views consistent by construction.
  frustum: in / out of the anchor camera's image rectangle -- geometric, arm-independent,
      available on both benchmarks.
  A0-fixed observed/completed (Occ3D): the FIVE-frame map's observed mask, held fixed across
      arms. Labelled as A0-fixed because it is not arm-independent in origin, only in use.

``never_observed`` is never redefined between arms.

Statistics: paired block bootstrap, 25 contiguous anchors on SemanticKITTI / one block per
scene on Occ3D, 2,000 resamples, seed 0; every arm resampled on the same anchors.

    PYTHONPATH=$PWD python tools/method_vs_data/one_frame_score.py --dataset semantickitti
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
from gates.gate8b import pooling as PL                                                  # noqa: E402
import torch                                                                            # noqa: E402
from tools.method_vs_data.observed_strict import (frame_transforms, sk_pose_chain,       # noqa: E402
                                                  strict_regions)
from tools.method_vs_data.one_frame_predict import out_path                             # noqa: E402
from tools.observability_split import runs as R, visibility as V                        # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS                            # noqa: E402
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction               # noqa: E402
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS, STATISTICS           # noqa: E402

OURS_ARMS = ("one_frame_unet", "one_frame_map", "five_frame_map", "one_frame_unet_scale5")
#: ``observability_split.runs`` registers OccAny runs for SemanticKITTI only. Occ3D has one
#: OccAny geometry run on disk -- the 5-frame FORWARD nuScenes preset, which is the comparator
#: ``tools/occany_comparison`` already uses for its headline Occ3D number. It is wired in here
#: so the Occ3D contrast gets a PAIRED interval instead of a cross-tool number comparison.
#: Note it is doubly off-setting against our causal one-frame arm: five frames, and future ones.
OCC3D_OCCANY_FORWARD = ("/media/SSD1/MINH_DATASETS/occany_out/ssc_voxel_pred/"
                        "OccAny_5frames_nuscenes512_rot60_vpi10_fwd3_sTrans2")
OCCANY_ARMS = {"occany_1f": "occany_1f_geom", "occany_5f_fwd": "occany_5f_geom"}
IMAGE_SIZE = {"semantickitti": (1226, 370), "occ3d": (1600, 900)}
IGNORE = 255


def _unpack(path: str, key: str, n: int) -> np.ndarray:
    with np.load(path) as z:
        return np.unpackbits(z[key])[:n].astype(bool)


def score(dataset: str, limit=None, post="raw", strict=True, device="cuda:0") -> dict:
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
    ref = CanonicalPrediction.load(os.path.join(ZC.PREDICTION_ROOT, dataset, f"{sids[0]}.npz"))
    dims, vs, origin = ref.dims, ref.voxel_size, ref.origin
    n = int(np.prod(dims))
    W, H = IMAGE_SIZE[dataset]
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    arms = ["five_frame_unet"] + [a for a in OURS_ARMS
                                  if os.path.isdir(os.path.join(
                                      os.path.dirname(os.path.dirname(
                                          out_path(a, dataset, "x"))), dataset))]
    arms += [k for k, v in OCCANY_ARMS.items() if v in runs]

    # Keep only anchors every ours-arm produced, so the bootstrap stays paired. A one-frame
    # clip with a non-finite MoGe-B scale has no metric scale at all (no median to fall back
    # on with a single candidate), so such anchors are absent from the one-frame arms; they
    # are dropped from EVERY arm and reported rather than silently filled.
    ours = [a for a in arms if a in OURS_ARMS]
    excluded = sorted({sid for sid in sids
                       for a in ours if not os.path.exists(out_path(a, dataset, sid))})
    if excluded:
        sids = [s for s in sids if s not in set(excluded)]

    if strict and dataset == "semantickitti":
        poses, cam_to_velo = sk_pose_chain(ZC.REPO_ROOT, "08")
    acc = defaultdict(lambda: 0)
    per = defaultdict(list)
    comp = defaultdict(lambda: 0)
    segs, used = [], []
    t0 = time.time()
    for i, sid in enumerate(sids):
        s = by_sid[sid]
        a0 = CanonicalPrediction.load(os.path.join(ZC.PREDICTION_ROOT, dataset, f"{sid}.npz"))
        if runs:
            _lk = "occany_1f_geom" if "occany_1f_geom" in runs else sorted(runs)[0]
            label, _ = R.load_occany(runs[_lk], sid, dims)
        else:
            from tools.occany_comparison import scoring as SC
            label = SC.load_occany(dataset, s.occany_pkl, dims).label.reshape(-1)
        valid = label != IGNORE
        gt_occ = valid & (label != spec.empty_class)

        masks = {"all": valid}
        if strict and dataset == "semantickitti":
            natives = [int(k.split("_")[1]) for k in (f["key"] for f in ZS.history_frames(s))]
            Ts = frame_transforms(poses, cam_to_velo, natives, natives[-1])
            assert np.allclose(Ts[-1], a0.T_cam_to_grid, atol=1e-6), sid
            m, _ = strict_regions(gt_occ, valid, Ts, a0.K, (W, H), dims, vs, origin, dev, 2, 0.1)
            masks.update({k: v for k, v in m.items() if k != "all"})
            masks["own_never_observed"] = m["earlier_only"] | m["never_observed"]
        fov = V.frustum(a0.T_cam_to_grid, a0.K, (W, H), dims, vs, origin)
        masks["in_frustum"] = valid & fov
        masks["out_frustum"] = valid & ~fov
        masks["A0fixed_observed"] = valid & a0.observed
        masks["A0fixed_completed"] = valid & ~a0.observed
        for k, mm in masks.items():
            comp[f"{k}__valid"] = comp[f"{k}__valid"] + int(mm.sum())
            comp[f"{k}__gt"] = comp[f"{k}__gt"] + int((mm & gt_occ).sum())

        for arm in arms:
            if arm == "five_frame_unet":
                occ = a0.occupied
            elif arm in OCCANY_ARMS:
                _l, vol = R.load_occany(runs[OCCANY_ARMS[arm]], sid, dims)
                occ = (vol != spec.empty_class) & (vol != IGNORE)
            else:
                p = out_path(arm, dataset, sid)
                if not os.path.exists(p):
                    raise FileNotFoundError(p)
                occ = _unpack(p, "occupied", n)
            if post == "pooled":
                occ = PL.geometry_dilation(torch.from_numpy(occ).reshape(dims)).reshape(-1).numpy()
            for k, mm in masks.items():
                c = np.array([int((occ & gt_occ & mm).sum()), int((occ & ~gt_occ & mm).sum()),
                              int((~occ & gt_occ & mm).sum())], np.int64)
                acc[f"{arm}__{k}"] = acc[f"{arm}__{k}"] + c
                per[f"{arm}__{k}"].append(c)
        # each ours-arm's own observed coverage, for the record
        for arm in [a for a in arms if a in OURS_ARMS]:
            obs = _unpack(out_path(arm, dataset, sid), "observed", n)
            comp[f"{arm}__own_observed_valid"] = comp[f"{arm}__own_observed_valid"] + int((valid & obs).sum())
            comp[f"{arm}__own_observed_gt"] = comp[f"{arm}__own_observed_gt"] + int((valid & obs & gt_occ).sum())
        comp["A0__own_observed_valid"] = comp["A0__own_observed_valid"] + int((valid & a0.observed).sum())
        comp["A0__own_observed_gt"] = comp["A0__own_observed_gt"] + int((valid & a0.observed & gt_occ).sum())
        segs.append(s.segment); used.append(sid)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sids)}  {(time.time() - t0) / (i + 1):.2f}s each", flush=True)

    def sc(c):
        tp, fp, fn = (float(x) for x in c)
        return {"iou": 100 * tp / max(tp + fp + fn, 1.0),
                "precision": 100 * tp / max(tp + fp, 1.0),
                "recall": 100 * tp / max(tp + fn, 1.0)}

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
    iou_of = lambda c: 100 * c[0] / max(c.sum(), 1)

    regions = list(dict.fromkeys(k.split("__", 1)[1] for k in acc))
    out = {"dataset": dataset, "n_samples": len(used), "postprocess": post,
           "arms": arms, "n_excluded": len(excluded), "excluded_samples": excluded[:20],
           "exclusion_reason": ("anchors without a finite one-frame MoGe-B metric scale, "
                                "dropped from every arm to keep the bootstrap paired"), "composition": {}, "coverage": {}, "iou": {}, "contrasts": {},
           "statistics": {"blocks": len(blocks), "resamples": STATISTICS.resamples,
                          "bootstrap_seed": STATISTICS.seed,
                          "block_rule": ("one per scene" if dataset == "occ3d"
                                         else f"{STATISTICS.semantickitti_block_samples} contiguous anchors"),
                          "paired": "all arms resampled on the same anchors"}}
    tv, tg = comp["all__valid"], comp["all__gt"]
    for k in regions:
        out["composition"][k] = {"valid_voxels": comp[f"{k}__valid"],
                                 "gt_occupied": comp[f"{k}__gt"],
                                 "share_of_valid": comp[f"{k}__valid"] / max(tv, 1),
                                 "share_of_gt_occupied": comp[f"{k}__gt"] / max(tg, 1)}
    for tag in ["A0"] + [a for a in arms if a in OURS_ARMS]:
        out["coverage"][tag] = {
            "own_observed_share_of_valid": comp[f"{tag}__own_observed_valid"] / max(tv, 1),
            "own_observed_share_of_gt_occupied": comp[f"{tag}__own_observed_gt"] / max(tg, 1)}
    for k in regions:
        out["iou"][k] = {arm: sc(acc[f"{arm}__{k}"]) for arm in arms}

    def ci(a, b, k):
        d = np.array([iou_of(stack[f"{a}__{k}"][p].sum(0)) - iou_of(stack[f"{b}__{k}"][p].sum(0))
                      for p in picks])
        lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        return {"delta": out["iou"][k][a]["iou"] - out["iou"][k][b]["iou"],
                "ci_low": lo, "ci_high": hi, "significant_at_95": bool(lo > 0 or hi < 0)}

    pairs = [("one_frame_unet", "five_frame_unet", "C1_remove_temporal_input"),
             ("one_frame_unet", "one_frame_map", "C2_completion_over_one_frame_map"),
             ("one_frame_unet", "occany_1f", "C3_ours_1f_vs_occany_1f"),
             ("five_frame_unet", "occany_1f", "C0_ours_5f_vs_occany_1f"),
             ("one_frame_unet_scale5", "one_frame_unet", "D_scale_only"),
             ("one_frame_unet", "occany_5f_fwd", "C3b_ours_1f_vs_occany_5f_forward"),
             ("five_frame_unet", "occany_5f_fwd", "C0b_ours_5f_vs_occany_5f_forward")]
    for a, b, name in pairs:
        if a in arms and b in arms:
            out["contrasts"][name] = {"a": a, "b": b,
                                      **{k: ci(a, b, k) for k in regions}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["semantickitti", "occ3d"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--postprocess", default="raw", choices=["raw", "pooled"])
    ap.add_argument("--no-strict", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = score(a.dataset, a.limit, a.postprocess, not a.no_strict, a.device)
    print(f"\n=== {a.dataset}  {r['n_samples']} anchors, {r['postprocess']}, "
          f"{r['statistics']['blocks']} blocks")
    print("  own observed coverage (share of GT occupied):")
    for k, v in r["coverage"].items():
        print(f"    {k:22s} {v['own_observed_share_of_gt_occupied']*100:6.2f}%  "
              f"(valid {v['own_observed_share_of_valid']*100:6.2f}%)")
    key = [k for k in ("all", "anchor_observed", "earlier_only", "never_observed",
                       "own_never_observed", "in_frustum", "out_frustum",
                       "A0fixed_observed", "A0fixed_completed") if k in r["iou"]]
    hdr = "  " + "region".ljust(20) + "".join(f"{x[:13]:>14s}" for x in r["arms"])
    print(hdr)
    for k in key:
        print("  " + k.ljust(20) + "".join(f"{r['iou'][k][x]['iou']:14.2f}" for x in r["arms"]))
    print("\n  contrasts on 'all':")
    for name, c in r["contrasts"].items():
        v = c["all"]
        print(f"    {name:34s} {c['a'][:20]:20s} - {c['b'][:16]:16s} "
              f"{v['delta']:+7.2f} [{v['ci_low']:+.2f},{v['ci_high']:+.2f}]"
              f"{'*' if v['significant_at_95'] else ''}")
    d = os.path.join(ZC.REPO_ROOT, "artifacts", "method_vs_data", "results")
    os.makedirs(d, exist_ok=True)
    p = a.out or os.path.join(d, f"one_frame_{a.dataset}_{a.postprocess}.json")
    json.dump(r, open(p, "w"), indent=1, default=float)
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
