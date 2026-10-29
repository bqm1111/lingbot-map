#!/usr/bin/env python
"""History-aware visibility: what a PERSISTENT-state system has and has not seen.

The five-frame masks every earlier report used describe visibility inside a five-frame
causal window. Under mode B the model has processed every earlier frame of the segment, so
those masks can no longer be read as "never observed by the system": a voxel hidden in the
recent five frames may have been in plain view thirty seconds ago. Both families are
therefore produced here, side by side, and labelled for what they are.

Recent-window regions (the historical ones, kept for comparison, explicitly labelled):
    ``recent_anchor_observed``, ``recent_earlier_only``, ``recent_never_in_frustum``,
    ``recent_never_out_frustum``, and the union ``recent_never_observed``.

History-aware regions over EVERY frame processed since the segment reset, which is the
whole stream prefix ``[0, anchor]`` -- the five mutually exclusive regions the protocol asks
for, in this order of precedence:
    1 ``current_frame``        -- observed in the anchor frame itself
    2 ``recent_other_four``    -- observed in the other four mapping frames, not the current
    3 ``earlier_in_stream``    -- observed earlier in the processed stream, outside those five
    4 ``never_in_frustum``     -- never observed in the processed stream, inside the anchor frustum
    5 ``never_out_frustum``    -- never observed in the processed stream, outside it
plus the preserved union ``never_observed_stream`` = 4 | 5.

Construction. One rule, the validated one, applied to more cameras: rays through every
second pixel, 0.1 m steps, 80 m maximum range, and every voxel a ray enters up to and
including the first GT-occupied voxel counts as observed (``tools/observability_split
/visibility.observed_from_camera``). Nothing is approximated and no proxy is introduced: a
frame is either ray-cast or provably unable to contribute.

    *Exact pruning.* A camera farther from the grid centre than ``80 m + half the grid
    diagonal`` cannot reach ANY voxel of the grid within the 80 m ray budget, so skipping it
    changes no mask. That is a geometric fact, not an approximation, and the number of
    frames actually cast and the number pruned are both recorded per anchor.

    *Cache eviction is irrelevant here.* LingBot's KV cache is a 64-block sliding window, so
    frames older than that are evicted from the explicit cache while still having shaped the
    retained state. These masks are over every frame the stream PROCESSED, not over cache
    residency, which is the conservative choice: a voxel is called never-observed only if no
    processed frame could see it.

Free space, unknown labels, ray termination and incomplete coverage are handled exactly as
the frozen construction handles them, and §5 of the report states each one.

    PYTHONPATH=$PWD python tools/method_vs_data/streaming/visibility.py \\
        --dataset semantickitti --device cuda:2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch                                                                            # noqa: E402
from gates.gate8b import pooling as PL                                                  # noqa: E402
from tools.method_vs_data.observed_strict import frame_transforms, sk_pose_chain         # noqa: E402
from tools.method_vs_data.one_frame_score import OCC3D_OCCANY_FORWARD                   # noqa: E402
from tools.method_vs_data.streaming import (build_maps as BM, config as SV, counting as CT,  # noqa: E402
                                            heads as HD, score as SO, streams as ST)
from tools.observability_split import runs as R, visibility as V                         # noqa: E402
from tools.zero_shot_eval import config as ZC, samples as ZS                             # noqa: E402
from tools.zero_shot_eval.canonical.prediction import CanonicalPrediction                # noqa: E402
from tools.zero_shot_eval.practical.protocol import DATASET_SPECS, STATISTICS            # noqa: E402

IGNORE = 255
MAX_RANGE_M = 80.0
PIXEL_STRIDE = 2
STEP_M = 0.1
HISTORY = ("current_frame", "recent_other_four", "earlier_in_stream",
           "never_in_frustum", "never_out_frustum")
RECENT = ("recent_anchor_observed", "recent_earlier_only",
          "recent_never_in_frustum", "recent_never_out_frustum")
REGIONS = (("all",) + HISTORY + ("never_observed_stream",) + RECENT
           + ("recent_never_observed", "in_frustum", "out_frustum"))
#: the arms the regional tables carry. The ray casting dominates, so arms are nearly free;
#: this set is the one the questions of section 5 need.
ARMS_DEFAULT = ["A:k360_s0", "B:k360_s0", "A:waymo_s0", "B:waymo_s0",
                "A:once_s0", "B:once_s0", "A:map_only", "B:map_only",
                "occany_1f", "occany_5f_fwd"]


def grid_prune_radius(dims, voxel_size) -> float:
    half = 0.5 * np.asarray(dims, np.float64) * float(voxel_size)
    return MAX_RANGE_M + float(np.linalg.norm(half))


def grid_centre(dims, voxel_size, origin) -> np.ndarray:
    return (np.asarray(origin, np.float64)
            + 0.5 * np.asarray(dims, np.float64) * float(voxel_size))


def occ3d_transforms(ann, cache: Dict, segment: str, positions: Sequence[int],
                     anchor_pos: int):
    from core.datasets import nuscenes as NU
    from tools.occany_comparison import config as OC
    if segment not in cache:
        cache[segment] = NU.scene_frames(ann, segment, "CAM_FRONT", OC.NUSCENES_ROOT)
    frs = cache[segment]
    A = np.linalg.inv(np.asarray(frs[anchor_pos].T_ego_keyframe_to_world, np.float64))
    Ts = [A @ np.asarray(frs[j].T_ego_cam_to_world, np.float64)
          @ np.asarray(frs[j].T_camera_to_ego_cam, np.float64) for j in positions]
    Ks = [np.asarray(frs[j].K, np.float64) for j in positions]
    return Ts, Ks


def regions_for_anchor(gt_occ, valid, Ts_all, Ks_all, positions, recent_positions,
                       anchor_pos, img, dims, vs, origin, dev) -> Dict[str, np.ndarray]:
    """The five history regions, the four recent-window ones, and the frustum cut."""
    centre = grid_centre(dims, vs, origin)
    radius = grid_prune_radius(dims, vs)
    seen: Dict[int, np.ndarray] = {}
    n_cast = n_pruned = 0
    for j, T, K in zip(positions, Ts_all, Ks_all):
        if np.linalg.norm(np.asarray(T, np.float64)[:3, 3] - centre) > radius:
            n_pruned += 1
            continue                      # provably cannot reach any voxel within 80 m
        seen[j] = V.observed_from_camera(gt_occ, T, K, img, dims, vs, origin,
                                         pixel_stride=PIXEL_STRIDE, step_m=STEP_M,
                                         max_range_m=MAX_RANGE_M, device=str(dev))
        n_cast += 1
    zero = np.zeros(int(np.prod(dims)), bool)
    cur = seen.get(anchor_pos, zero)
    other4 = zero.copy()
    for j in recent_positions:
        if j != anchor_pos and j in seen:
            other4 |= seen[j]
    earlier = zero.copy()
    for j, m in seen.items():
        if j not in recent_positions:
            earlier |= m
    fov = V.frustum(Ts_all[list(positions).index(anchor_pos)],
                    Ks_all[list(positions).index(anchor_pos)], img, dims, vs, origin)

    # --- history-aware partition, by precedence
    m: Dict[str, np.ndarray] = {}
    m["current_frame"] = valid & cur
    m["recent_other_four"] = valid & other4 & ~cur
    m["earlier_in_stream"] = valid & earlier & ~cur & ~other4
    never = valid & ~cur & ~other4 & ~earlier
    m["never_in_frustum"] = never & fov
    m["never_out_frustum"] = never & ~fov
    m["never_observed_stream"] = never

    # --- recent-window-only partition (the historical masks), explicitly labelled
    rec_union = cur | other4
    rn = valid & ~rec_union
    m["recent_anchor_observed"] = valid & cur
    m["recent_earlier_only"] = valid & other4 & ~cur
    m["recent_never_in_frustum"] = rn & fov
    m["recent_never_out_frustum"] = rn & ~fov
    m["recent_never_observed"] = rn
    m["in_frustum"] = valid & fov
    m["out_frustum"] = valid & ~fov
    m["all"] = valid

    # --- gates: both families partition the valid volume exactly
    for fam in (HISTORY, RECENT):
        parts = [m[k] for k in fam]
        for i in range(len(parts)):
            for j2 in range(i + 1, len(parts)):
                assert not (parts[i] & parts[j2]).any(), f"{fam[i]} overlaps {fam[j2]}"
        assert sum(int(x.sum()) for x in parts) == int(valid.sum()), \
            f"{fam} does not partition the valid volume"
    assert np.array_equal(m["never_observed_stream"],
                          m["never_in_frustum"] | m["never_out_frustum"])
    assert np.array_equal(m["recent_never_observed"],
                          m["recent_never_in_frustum"] | m["recent_never_out_frustum"])
    # a voxel never seen in the whole stream is certainly never seen in the recent window
    assert not (m["never_observed_stream"] & ~m["recent_never_observed"]).any(), \
        "stream-never-observed must be a subset of recent-never-observed"
    return m, {"n_frames_cast": n_cast, "n_frames_pruned": n_pruned,
               "n_frames_in_prefix": len(positions)}


def run(dataset: str, device: str, arms: Sequence[str] = ARMS_DEFAULT,
        limit: Optional[int] = None, progress: int = 25) -> dict:
    spec = DATASET_SPECS[dataset]
    dev = torch.device(device)
    torch.cuda.set_device(dev)
    rd = HD.Readout(dataset, dev)
    counter = CT.RegionCounter(dev, rd.native.dims)
    need = sorted({a.split(":")[1] for a in arms if ":" in a} - {"map_only"})
    nets = {h: HD.load_head(h.split("_s")[0], int(h.split("_s")[1]), dev) for h in need}
    modes = sorted({a.split(":")[0] for a in arms if ":" in a})

    runs = {r.key: r for r in R.RUNS[dataset]} if dataset in R.RUNS else {}
    if dataset == "occ3d" and os.path.isdir(OCC3D_OCCANY_FORWARD):
        runs["occany_5f_geom"] = R.Run("occany_5f_geom", "occany", "forward",
                                       OCC3D_OCCANY_FORWARD, False, 5, "OccAny 5f fwd [reproduced]")
    arms = [a for a in arms if ":" in a or SO.OCCANY_ARMS.get(a) in runs]

    by_sid = {s.sample_id: s for s in ZS.evaluated_samples(dataset)}
    segs_all = {s.name: s for s in ST.build(dataset)}
    sids = [s for s in sorted(by_sid)
            if all(os.path.exists(SV.map_cache_path(dataset, m, s)) for m in modes)][:limit]
    assert sids
    img = SV.IMAGE_SIZE[dataset]
    dims, vs, origin = rd.native.dims, rd.native.voxel_size, rd.native.origin
    if dataset == "semantickitti":
        poses, cam_to_velo = sk_pose_chain(SV.REPO_ROOT, "08")
    else:
        from core.datasets import nuscenes as NU, sources as S
        ann = NU.load_annotations(S.OCC3D_ROOT)
        scene_cache: Dict[str, List] = {}

    acc = defaultdict(lambda: np.zeros(3, np.int64))
    per = defaultdict(list)
    comp = defaultdict(int)
    cast_stats, segments, used, t0 = [], [], [], time.time()
    for i, sid in enumerate(sids):
        s = by_sid[sid]
        seg = segs_all[s.segment]
        p = s.target_pos
        prefix = list(range(0, p + 1))                 # every frame processed since the reset
        recent = [p + o for o in ST.CAUSAL.offsets]
        lk = "occany_1f_geom" if "occany_1f_geom" in runs else sorted(runs)[0]
        label, _ = R.load_occany(runs[lk], sid, dims)
        valid = label != IGNORE
        gt_occ = valid & (label != spec.empty_class)
        if dataset == "semantickitti":
            natives = [int(seg.frame_keys[j].split("_")[1]) for j in prefix]
            Ts = frame_transforms(poses, cam_to_velo, natives, natives[p])
            K0 = CanonicalPrediction.load(ZC.prediction_path(dataset, sid)).K
            Ks = [K0] * len(prefix)
            assert np.allclose(Ts[p], np.asarray(s.T_cam_to_grid_ours, np.float64), atol=1e-6), sid
        else:
            Ts, Ks = occ3d_transforms(ann, scene_cache, s.segment, prefix, p)
            assert np.allclose(Ts[p], np.asarray(s.T_cam_to_grid_ours, np.float64), atol=1e-5), sid
        masks, cs = regions_for_anchor(gt_occ, valid, Ts, Ks, prefix, recent, p,
                                       img, dims, vs, origin, dev)
        cs["sample_id"] = sid
        cs["time_s_in_segment"] = float(seg.times_s[p])
        cast_stats.append(cs)
        counter.set_anchor(gt_occ, masks)
        for k, (nv, ng) in counter.support(REGIONS).items():
            comp[f"{k}__valid"] += nv
            comp[f"{k}__gt"] += ng

        preds = {}
        for m in modes:
            S_, F_, _ = BM.load_map(dataset, m, sid)
            for h, net in nets.items():
                if f"{m}:{h}" in arms:
                    preds[f"{m}:{h}"] = rd.completion(net, S_, F_)
            if f"{m}:map_only" in arms:
                preds[f"{m}:map_only"] = rd.map_only(S_)
        for a, key in SO.OCCANY_ARMS.items():
            if a in arms:
                _l, vol = R.load_occany(runs[key], sid, dims)
                preds[a] = (vol != spec.empty_class) & (vol != IGNORE)

        for arm in arms:
            for post in SO.POSTS:
                cc = counter.counts(preds[arm], post, REGIONS)
                tot = np.zeros(3, np.int64)
                for k in REGIONS:
                    acc[f"{arm}|{post}|{k}"] = acc[f"{arm}|{post}|{k}"] + cc[k]
                    per[f"{arm}|{post}|{k}"].append(cc[k])
                    if k in HISTORY:
                        tot = tot + cc[k]
                # the five history regions must reconstruct the global confusion counts
                assert np.array_equal(tot, cc["all"]), (sid, arm, post)
                # ... and so must the four recent-window ones
                assert np.array_equal(sum(cc[k] for k in RECENT), cc["all"]), (sid, arm, post)
        segments.append(s.segment); used.append(sid)
        if progress and (i + 1) % progress == 0:
            print(f"  {i+1}/{len(sids)}  {(time.time()-t0)/(i+1):.2f}s each  "
                  f"cast {cs['n_frames_cast']}/{cs['n_frames_in_prefix']}", flush=True)

    def sc(c):
        tp, fp, fn = (float(x) for x in c)
        return {"iou": 100 * tp / max(tp + fp + fn, 1.0),
                "precision": 100 * tp / max(tp + fp, 1.0),
                "recall": 100 * tp / max(tp + fn, 1.0),
                "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    blocks = SO._blocks(dataset, segments, used, STATISTICS.semantickitti_block_samples)
    W = SO._resample_matrix(blocks, len(used), STATISTICS.resamples, STATISTICS.seed)
    boot = {k: (W @ np.stack(v).astype(np.int64)) for k, v in per.items()}
    nc = np.asarray([c["n_frames_cast"] for c in cast_stats])
    npru = np.asarray([c["n_frames_pruned"] for c in cast_stats])
    out = {"provenance": {**SO.provenance(modes, SV.SOURCES, (0,)),
                          "tool": "tools/method_vs_data/streaming/visibility.py",
                          "construction": {"pixel_stride": PIXEL_STRIDE, "step_m": STEP_M,
                                           "max_range_m": MAX_RANGE_M,
                                           "rule": ("every voxel a ray enters up to and "
                                                    "including the first GT-occupied voxel "
                                                    "is observed"),
                                           "prune_radius_m": grid_prune_radius(dims, vs),
                                           "prune_is_exact": True}},
           "dataset": dataset, "arms": arms, "n_anchors": len(used),
           "regions_history_aware": list(HISTORY),
           "regions_recent_window_only": list(RECENT),
           "ray_casting": {"frames_cast": {"mean": float(nc.mean()), "min": int(nc.min()),
                                           "max": int(nc.max()), "total": int(nc.sum())},
                           "frames_pruned": {"mean": float(npru.mean()),
                                             "total": int(npru.sum())},
                           "prefix_length": {"mean": float(np.mean(
                               [c["n_frames_in_prefix"] for c in cast_stats])),
                               "max": int(np.max([c["n_frames_in_prefix"] for c in cast_stats]))}},
           "statistics": {"blocks": len(blocks), "resamples": STATISTICS.resamples,
                          "bootstrap_seed": STATISTICS.seed,
                          "n_independent_streams": len(set(segments))},
           "composition": {k: {"valid_voxels": comp[f"{k}__valid"],
                               "gt_occupied": comp[f"{k}__gt"],
                               "share_of_valid": comp[f"{k}__valid"] / max(comp["all__valid"], 1),
                               "share_of_gt_occupied": comp[f"{k}__gt"] / max(comp["all__gt"], 1)}
                           for k in REGIONS},
           "iou": {post: {k: {a: sc(acc[f"{a}|{post}|{k}"]) for a in arms} for k in REGIONS}
                   for post in SO.POSTS},
           "contrasts": {}}

    def ci(a, b, post, k):
        ca, cb = boot[f"{a}|{post}|{k}"], boot[f"{b}|{post}|{k}"]
        d = (100 * ca[:, 0] / np.maximum(ca.sum(1), 1)
             - 100 * cb[:, 0] / np.maximum(cb.sum(1), 1))
        lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        return {"delta": out["iou"][post][k][a]["iou"] - out["iou"][post][k][b]["iou"],
                "ci_low": lo, "ci_high": hi, "significant_at_95": bool(lo > 0 or hi < 0)}

    pairs = {}
    for h in ("k360_s0", "waymo_s0", "once_s0", "map_only"):
        if f"A:{h}" in arms and f"B:{h}" in arms:
            pairs[f"streaming_minus_clipreset__{h}"] = (f"B:{h}", f"A:{h}")
    for m in modes:
        for h in ("k360_s0", "waymo_s0", "once_s0"):
            if f"{m}:{h}" in arms and f"{m}:map_only" in arms:
                pairs[f"completion_minus_maponly__{h}__mode{m}"] = (f"{m}:{h}", f"{m}:map_only")
            for oc in ("occany_1f", "occany_5f_fwd"):
                if f"{m}:{h}" in arms and oc in arms:
                    pairs[f"vs_{oc}__{h}__mode{m}"] = (f"{m}:{h}", oc)
    for name, (a, b) in pairs.items():
        out["contrasts"][name] = {"a": a, "b": b,
                                  **{post: {k: ci(a, b, post, k) for k in REGIONS}
                                     for post in SO.POSTS}}
    out["_per"] = {k: np.stack(v).tolist() for k, v in per.items()}
    out["_anchors"], out["_segments"] = used, segments
    out["_cast"] = cast_stats
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(SV.DATASETS))
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--arms", nargs="+", default=list(ARMS_DEFAULT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    SV.ensure_dirs()
    r = run(a.dataset, a.device, a.arms, a.limit)
    out = a.out or os.path.join(SV.VISIBILITY_DIR, f"regional_{a.dataset}.json")
    per = r.pop("_per"); anchors = r.pop("_anchors"); segments = r.pop("_segments")
    cast = r.pop("_cast")
    json.dump(r, open(out, "w"), indent=1, default=float)
    np.savez_compressed(out.replace(".json", "_per_anchor.npz"),
                        anchors=np.asarray(anchors), segments=np.asarray(segments),
                        **{k: np.asarray(v, np.int64) for k, v in per.items()})
    json.dump(cast, open(out.replace(".json", "_raycast.json"), "w"), indent=1, default=float)
    print(f"\n{a.dataset}: {r['n_anchors']} anchors, arms {r['arms']}")
    print("  region composition (%valid / %gt_occ)")
    for k in REGIONS:
        c = r["composition"][k]
        print(f"    {k:28s} {100*c['share_of_valid']:6.2f}% {100*c['share_of_gt_occupied']:6.2f}%")
    for post in SO.POSTS:
        print(f"  --- {post} IoU")
        for k in REGIONS:
            print(f"    {k:28s} " + "".join(f"{r['iou'][post][k][a]['iou']:8.2f}" for a in r["arms"]))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
