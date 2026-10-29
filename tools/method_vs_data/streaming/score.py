#!/usr/bin/env python
"""Score every source head and seed, under both inference modes, on identical anchors.

One pass, one scorer, one set of masks. For each anchor the pass loads the mode-A map and
the mode-B map, runs all nine completion checkpoints (three sources x three seeds) on BOTH,
adds the map-only arm for both modes, adds OccAny's released causal one-frame and forward
five-frame outputs as fixed comparators, and accumulates confusion counts per region under
BOTH post-processing conventions. Every arm therefore sees the same anchors, the same ground
truth, the same valid mask and the same region masks, so every contrast is paired by
construction and the paired block bootstrap needs no realignment.

The scoring logic is the canonical one, imported not restated: OccAny's stored ``voxel_label``
as ground truth for all arms, ``label != 255`` valid, ``label != empty_class`` occupied,
``gates.gate8b.pooling.geometry_dilation`` applied to the PREDICTION ONLY for the pooled
convention, ground truth and every mask left undilated, global IoU recomputed from summed
TP/FP/FN, and the paired block bootstrap with 2,000 resamples at seed 0.

Blocks. Occ3D: one block per nuScenes scene, 150 of them, which are 150 genuinely
independent streams. SemanticKITTI: the canonical 25-contiguous-anchor blocks -- but
SemanticKITTI is **one** physical stream, so those blocks are segments of a single stream
and not independent streams. Both are reported, and the SemanticKITTI intervals carry that
limitation explicitly; ``--sk-stream-sensitivity`` additionally reports how the interval
widens as the block length grows, which is the sensitivity check the protocol asks for when
the established blocks are shorter than the correlation length of a persistent stream.

    PYTHONPATH=$PWD python tools/method_vs_data/streaming/score.py \\
        --dataset semantickitti --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch                                                                            # noqa: E402
from gates.gate8b import pooling as PL                                                  # noqa: E402
from tools.method_vs_data.one_frame_score import OCC3D_OCCANY_FORWARD                   # noqa: E402
from tools.method_vs_data.streaming import (build_maps as BM, config as SV, counting as CT,  # noqa: E402
                                            heads as HD, streams as ST)
from tools.observability_split import runs as R, visibility as V                         # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS                             # noqa: E402
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS, STATISTICS            # noqa: E402

IGNORE = 255
OCCANY_ARMS = {"occany_1f": "occany_1f_geom", "occany_5f_fwd": "occany_5f_geom"}
POSTS = ("raw", "pooled")
#: regions here are the cheap, mask-only ones. Visibility regions that need ray casting are
#: produced by tools/method_vs_data/streaming/visibility.py, which owns the five history-aware
#: regions and the recent-window ones, so no ray is cast twice.
REGIONS = ("all", "in_frustum", "out_frustum")


def arm_names(modes: Sequence[str], sources: Sequence[str], seeds: Sequence[int],
              occany: bool = True) -> List[str]:
    out = []
    for m in modes:
        for src in sources:
            out += [f"{m}:{src}_s{s}" for s in seeds]
        out.append(f"{m}:map_only")
    if occany:
        out += list(OCCANY_ARMS)
    return out


def provenance(modes, sources, seeds) -> dict:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SV.REPO_ROOT,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=SV.REPO_ROOT,
                           capture_output=True, text=True).stdout.strip().splitlines()
    return {"git_rev": rev, "working_tree_modified": dirty,
            "tool": "tools/method_vs_data/streaming/score.py",
            "checkpoints": {k: v for k, v in HD.checkpoint_manifest().items()
                            if any(k.startswith(s) for s in sources)},
            "modes": {m: SV.MODE_NAMES[m] for m in modes},
            "map_cache_root": SV.CACHE_ROOT,
            "lingbot_checkpoint": "checkpoints/lingbot-map/204754b/lingbot-map.pt",
            "moge_revision": "39c4d5e957afe587e04eec59dc2bcc3be5ecd968",
            "exploratory_sources": list(SV.EXPLORATORY),
            "seed_note": ("per-seed rows are kept separate; a mean/spread over the three "
                          "training seeds is a different quantity from the evaluation-sample "
                          "confidence interval and the two are never combined")}


def score(dataset: str, modes: Sequence[str], sources: Sequence[str], seeds: Sequence[int],
          device: str, limit: Optional[int] = None, occany: bool = True,
          progress: int = 200) -> dict:
    spec = DATASET_SPECS[dataset]
    dev = torch.device(device)
    torch.cuda.set_device(dev)
    rd = HD.Readout(dataset, dev)
    counter = CT.RegionCounter(dev, rd.native.dims)
    nets = {(src, s): HD.load_head(src, s, dev) for src in sources for s in seeds}

    runs = {r.key: r for r in R.RUNS[dataset]} if dataset in R.RUNS else {}
    if dataset == "occ3d" and os.path.isdir(OCC3D_OCCANY_FORWARD):
        runs["occany_5f_geom"] = R.Run("occany_5f_geom", "occany", "forward",
                                       OCC3D_OCCANY_FORWARD, False, 5,
                                       "OccAny 5-frame FORWARD nuScenes preset [reproduced]")
    arms = arm_names(modes, sources, seeds, occany and bool(runs))
    # only the OccAny configurations that actually exist for this benchmark: nuScenes was
    # released/reproduced in the five-frame FORWARD preset only, so Occ3D has no causal
    # one-frame comparator and no table claims one.
    arms = [a for a in arms if ":" in a or OCCANY_ARMS[a] in runs]

    by_sid = {s.sample_id: s for s in ZS.evaluated_samples(dataset)}
    sids = sorted(by_sid)
    # population gate: every mode must have a cached map for the anchor, or no row describes it
    miss = {m: [s for s in sids if not os.path.exists(SV.map_cache_path(dataset, m, s))]
            for m in modes}
    drop = sorted(set().union(*miss.values())) if miss else []
    sids = [s for s in sids if s not in set(drop)][:limit]
    assert sids, "no anchor has a cached map in every requested mode"

    acc: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(3, np.int64))
    per: Dict[str, List[np.ndarray]] = defaultdict(list)
    comp: Dict[str, int] = defaultdict(int)
    scales: Dict[str, List] = defaultdict(list)
    segs, used, t0 = [], [], time.time()

    for i, sid in enumerate(sids):
        s = by_sid[sid]
        maps = {}
        for m in modes:
            S_, F_, meta = BM.load_map(dataset, m, sid)
            maps[m] = (S_, F_, meta)
            scales[m].append({"sample_id": sid, "segment": s.segment,
                              "scale": meta["scale_m_per_unit"],
                              "n_occupied": meta["n_occupied"], "n_free": meta["n_free"],
                              "target_pos": meta["target_pos"]})
        lk = "occany_1f_geom" if "occany_1f_geom" in runs else (sorted(runs)[0] if runs else None)
        if lk:
            label, _ = R.load_occany(runs[lk], sid, rd.native.dims)
        else:
            from tools.occany_comparison import scoring as OSC
            label = OSC.load_occany(dataset, s.occany_pkl, rd.native.dims).label.reshape(-1)
        valid = label != IGNORE
        gt_occ = valid & (label != spec.empty_class)
        T = np.asarray(s.T_cam_to_grid_ours, np.float64)
        from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction
        K = CanonicalPrediction.load(ZC.prediction_path(dataset, sid)).K
        fov = V.frustum(T, K, SV.IMAGE_SIZE[dataset], rd.native.dims,
                        rd.native.voxel_size, rd.native.origin)
        masks = {"all": valid, "in_frustum": valid & fov, "out_frustum": valid & ~fov}
        counter.set_anchor(gt_occ, masks)
        for k, (nv, ng) in counter.support(REGIONS).items():
            comp[f"{k}__valid"] += nv
            comp[f"{k}__gt"] += ng

        preds: Dict[str, np.ndarray] = {}
        for m in modes:
            S_, F_, _ = maps[m]
            for (src, sd), net in nets.items():
                preds[f"{m}:{src}_s{sd}"] = rd.completion(net, S_, F_)
            preds[f"{m}:map_only"] = rd.map_only(S_)
        for a, key in OCCANY_ARMS.items():
            if a in arms:
                _l, vol = R.load_occany(runs[key], sid, rd.native.dims)
                preds[a] = (vol != spec.empty_class) & (vol != IGNORE)

        for arm in arms:
            for post in POSTS:
                cc = counter.counts(preds[arm], post, REGIONS)
                for k, c in cc.items():
                    acc[f"{arm}|{post}|{k}"] = acc[f"{arm}|{post}|{k}"] + c
                    per[f"{arm}|{post}|{k}"].append(c)
        segs.append(s.segment); used.append(sid)
        if progress and (i + 1) % progress == 0:
            print(f"  {i+1}/{len(sids)}  {(time.time()-t0)/(i+1):.2f}s each", flush=True)

    # ---- aggregation
    def sc(c):
        tp, fp, fn = (float(x) for x in c)
        return {"iou": 100 * tp / max(tp + fp + fn, 1.0),
                "precision": 100 * tp / max(tp + fp, 1.0),
                "recall": 100 * tp / max(tp + fn, 1.0),
                "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    blocks = _blocks(dataset, segs, used, STATISTICS.semantickitti_block_samples)
    stack = {k: np.stack(v) for k, v in per.items()}
    # One resample-count matrix W [n_resamples, n_anchors] instead of re-summing per pair:
    # row r counts how many times each anchor was drawn in resample r, so W @ counts gives
    # every arm's resampled TP/FP/FN in one matmul. Identical draws to the canonical loop --
    # same rng, same seed, same block order -- just not recomputed 348 times.
    W = _resample_matrix(blocks, len(used), STATISTICS.resamples, STATISTICS.seed)
    boot = {k: (W @ v.astype(np.int64)) for k, v in stack.items()}
    iou_of = lambda c: 100 * c[0] / max(c.sum(), 1)                            # noqa: E731

    out = {"provenance": provenance(modes, sources, seeds), "dataset": dataset,
           "modes": list(modes), "sources": list(sources), "seeds": list(seeds),
           "arms": arms, "n_anchors": len(used),
           "canonical_population": len(by_sid),
           "anchors_dropped": drop,
           "n_missing_maps_per_mode": {m: len(v) for m, v in miss.items()},
           "statistics": {"blocks": len(blocks), "resamples": STATISTICS.resamples,
                          "bootstrap_seed": STATISTICS.seed,
                          "block_rule": ("one per nuScenes scene" if dataset == "occ3d"
                                         else f"{STATISTICS.semantickitti_block_samples} contiguous anchors"),
                          "n_independent_streams": (len(set(segs))),
                          "independence_caveat": (
                              "SemanticKITTI is ONE physical stream: its blocks are segments "
                              "of that stream, not independent streams, so its intervals "
                              "understate between-stream variation. Occ3D's 150 blocks are "
                              "150 independent scenes."
                              if dataset == "semantickitti" else
                              "one block per independent validation scene"),
                          "uncertainty_covered": "evaluation sample only; NOT training-seed spread",
                          "global_iou_rule": "recomputed from summed TP/FP/FN per resample"},
           "composition": {k: {"valid_voxels": comp[f"{k}__valid"],
                               "gt_occupied": comp[f"{k}__gt"],
                               "share_of_valid": comp[f"{k}__valid"] / max(comp["all__valid"], 1),
                               "share_of_gt_occupied": comp[f"{k}__gt"] / max(comp["all__gt"], 1)}
                           for k in REGIONS},
           "iou": {post: {k: {arm: sc(acc[f"{arm}|{post}|{k}"]) for arm in arms}
                          for k in REGIONS} for post in POSTS},
           "scales": {m: scales[m] for m in modes},
           "contrasts": {}}

    def ci(a, b, post, k):
        ca, cb = boot[f"{a}|{post}|{k}"], boot[f"{b}|{post}|{k}"]
        d = (100 * ca[:, 0] / np.maximum(ca.sum(1), 1)
             - 100 * cb[:, 0] / np.maximum(cb.sum(1), 1))
        lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        return {"delta": out["iou"][post][k][a]["iou"] - out["iou"][post][k][b]["iou"],
                "ci_low": lo, "ci_high": hi, "significant_at_95": bool(lo > 0 or hi < 0)}

    pairs = _pairs(modes, sources, seeds, arms)
    for name, (a, b, why) in pairs.items():
        out["contrasts"][name] = {"a": a, "b": b, "question": why,
                                  **{post: {k: ci(a, b, post, k) for k in REGIONS}
                                     for post in POSTS}}
    out["per_anchor_counts_available"] = True
    out["_per"] = {k: np.stack(v).tolist() for k, v in per.items()}
    out["_anchors"] = used
    out["_segments"] = segs
    return out


def _resample_matrix(blocks, n: int, resamples: int, seed: int) -> np.ndarray:
    """``W[r, i]`` = how many times anchor ``i`` is drawn in block-bootstrap resample ``r``."""
    rng = np.random.default_rng(seed)
    W = np.zeros((resamples, n), np.int64)
    for r in range(resamples):
        for j in rng.integers(0, len(blocks), len(blocks)):
            np.add.at(W[r], blocks[j], 1)
    return W


def _blocks(dataset, segs, used, b):
    if dataset == "occ3d":
        by = {}
        for i, sg in enumerate(segs):
            by.setdefault(sg, []).append(i)
        return [v for _, v in sorted(by.items())]
    return [list(range(a, min(a + b, len(used)))) for a in range(0, len(used), b)]


def _pairs(modes, sources, seeds, arms) -> Dict[str, tuple]:
    p: Dict[str, tuple] = {}
    # (1) streaming minus clip-reset, within each source and seed
    if "A" in modes and "B" in modes:
        for src in sources:
            for s in seeds:
                p[f"streaming_minus_clipreset__{src}_s{s}"] = (
                    f"B:{src}_s{s}", f"A:{src}_s{s}",
                    "does this source's head survive persistent state with startup-only scale?")
        p["streaming_minus_clipreset__map_only"] = (
            "B:map_only", "A:map_only",
            "does the upstream reconstruction itself change under the streaming mode?")
    # (2) source differences under each fixed mode
    for m in modes:
        for a, b in (("k360", "waymo"), ("k360", "once"), ("waymo", "once")):
            if a in sources and b in sources:
                for s in seeds:
                    p[f"source__{a}_vs_{b}__mode{m}_s{s}"] = (
                        f"{m}:{a}_s{s}", f"{m}:{b}_s{s}",
                        f"source ranking under mode {m}, held at one inference mode")
    # (3) completion minus map-only, under each mode
    for m in modes:
        for src in sources:
            for s in seeds:
                p[f"completion_minus_maponly__{src}_s{s}__mode{m}"] = (
                    f"{m}:{src}_s{s}", f"{m}:map_only",
                    f"what the learned head adds over the raw map under mode {m}")
    # (4) against the fixed OccAny comparators
    for m in modes:
        for src in sources:
            for oc in ("occany_1f", "occany_5f_fwd"):
                if oc in arms:
                    p[f"vs_{oc}__{src}_s0__mode{m}"] = (
                        f"{m}:{src}_s0", oc,
                        f"mode {m} against OccAny's {'causal one-frame' if oc.endswith('1f') else 'forward five-frame'} configuration")
    return p


def sk_stream_sensitivity(rec: dict, pairs: Sequence[str], block_lengths=(25, 50, 100, 200, 400, 801)) -> dict:
    """How much do SemanticKITTI's intervals widen as the block grows toward the whole stream?

    SemanticKITTI is one stream, so the 25-anchor block cannot see correlation longer than
    ~12 s. Re-running the same paired bootstrap at growing block lengths shows how much of
    the interval is an artefact of a block shorter than the stream's correlation length. The
    801-anchor row is the degenerate single-block case and is reported as such.
    """
    per = {k: np.asarray(v, np.int64) for k, v in rec["_per"].items()}
    n = rec["n_anchors"]
    out = {}
    for name in pairs:
        a, b = rec["contrasts"][name]["a"], rec["contrasts"][name]["b"]
        rows = {}
        for L in block_lengths:
            blocks = [list(range(x, min(x + L, n))) for x in range(0, n, L)]
            W = _resample_matrix(blocks, n, STATISTICS.resamples, STATISTICS.seed)
            ca = W @ per[f"{a}|pooled|all"]
            cb = W @ per[f"{b}|pooled|all"]
            d = (100 * ca[:, 0] / np.maximum(ca.sum(1), 1)
                 - 100 * cb[:, 0] / np.maximum(cb.sum(1), 1))
            lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
            rows[str(L)] = {"n_blocks": len(blocks), "ci_low": lo, "ci_high": hi,
                            "width": hi - lo,
                            "significant_at_95": bool(lo > 0 or hi < 0)}
        out[name] = rows
    return {"pooled_all_region": out,
            "note": ("one physical stream, so no block length makes these intervals "
                     "between-stream intervals. The widening shows how much the canonical "
                     "25-anchor block understates within-stream correlation; the 801 row is "
                     "a single block and its interval is degenerate.")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(SV.DATASETS))
    ap.add_argument("--modes", nargs="+", default=list(SV.MODES))
    ap.add_argument("--sources", nargs="+", default=list(SV.SOURCES))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SV.SEEDS))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-occany", action="store_true")
    ap.add_argument("--sk-stream-sensitivity", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    SV.ensure_dirs()
    r = score(a.dataset, a.modes, a.sources, a.seeds, a.device, a.limit, not a.no_occany)
    if a.sk_stream_sensitivity and a.dataset == "semantickitti":
        want = [k for k in r["contrasts"] if k.startswith("streaming_minus_clipreset")]
        r["sk_stream_sensitivity"] = sk_stream_sensitivity(r, want)
    out = a.out or os.path.join(SV.SCORE_DIR, f"streaming_{a.dataset}.json")
    per = r.pop("_per")
    anchors, segments = r.pop("_anchors"), r.pop("_segments")
    json.dump(r, open(out, "w"), indent=1, default=float)
    np.savez_compressed(out.replace(".json", "_per_anchor.npz"),
                        anchors=np.asarray(anchors), segments=np.asarray(segments),
                        **{k: np.asarray(v, np.int64) for k, v in per.items()})
    print(f"\n{a.dataset}: {r['n_anchors']} anchors, {len(r['arms'])} arms")
    for post in POSTS:
        print(f"  --- {post} IoU (all)")
        for arm in r["arms"]:
            print(f"    {arm:22s} {r['iou'][post]['all'][arm]['iou']:6.2f}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
