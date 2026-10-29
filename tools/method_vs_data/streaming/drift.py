#!/usr/bin/env python
"""Scale and geometry drift along a persistent stream — diagnostics only.

Two independent readings, both computed OUTSIDE the inference loop and neither of which can
reach a prediction, a pose, a map or a scale:

**(1) Ground-truth odometry reference.** The predicted camera motion between two consecutive
frames, multiplied by the gauge in force, against the published ego/odometry motion over the
same pair. A ratio of 1 means the scaled reconstruction moves the right number of metres.
Translation magnitude between consecutive frames is invariant to the frame the poses are
expressed in, so the predicted canonical chain and the world-frame ground truth are directly
comparable without aligning them. Ground truth is read here and nowhere else; the mode-B maps
were already built and are not recomputed.

**(2) MoGe-2 consistency check.** The per-frame metric candidate that the G-A estimator would
have produced at a later frame, against the scalar actually frozen at startup. This is a
consistency check between two monocular estimates and is **not** a ground-truth scale: it
says whether the gauge the system holds still matches what the same estimator would say now.

Both are reported on a fixed diagnostic subset spanning the early, middle and late thirds of
each stream, and the extra MoGe calls here are isolated from the primary inference loop and
from every timing figure.

    PYTHONPATH=$PWD python tools/method_vs_data/streaming/drift.py \\
        --dataset semantickitti --device cuda:2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch                                                                            # noqa: E402
from gates.gate7b import replay as RP, scale as SC                                       # noqa: E402
from tools.method_vs_data.streaming import (build_maps as BM, config as SV, streams as ST,  # noqa: E402
                                            trace as TR)
from tools.occany_comparison import config as OC                                         # noqa: E402

CAUSAL = OC.PROTOCOLS["causal"]
#: MoGe consistency probes per stream, spread over the early, middle and late thirds
N_PROBES = 9


def gt_positions(dataset: str, seg) -> np.ndarray:
    """Published ego/camera world positions per stream frame. DIAGNOSTIC USE ONLY."""
    if dataset == "semantickitti":
        from tools.method_vs_data.observed_strict import sk_pose_chain
        poses, _cam_to_velo = sk_pose_chain(SV.REPO_ROOT, "08")
        nat = [int(k.split("_")[1]) for k in seg.frame_keys]
        return np.stack([np.asarray(poses[n], np.float64)[:3, 3] for n in nat])
    from core.datasets import nuscenes as NU, sources as S
    ann = NU.load_annotations(S.OCC3D_ROOT)
    frs = NU.scene_frames(ann, seg.name, "CAM_FRONT", OC.NUSCENES_ROOT)
    return np.stack([np.asarray(f.T_ego_cam_to_world, np.float64)[:3, 3] for f in frs])


def clipreset_motion_ratio(dataset: str, seg, gt: np.ndarray) -> dict:
    """The same ratio for the CLIP-RESET reference, from its frozen clip cache. No GPU.

    Each clip carries its own five predicted poses and its own five-frame scale, so the
    reference's ratio is computed from the last two slots of each clip. Comparing the two
    curves separates the two things mode B changes at once: if the reference's ratio stays
    flat where mode B's drifts, the drift belongs to the persistent pose chain and the stale
    gauge rather than to the estimator or the benchmark's own odometry.
    """
    d_gt = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    rows = []
    for sid, p_ in seg.anchors:
        path = OC.clip_path(dataset, "causal", sid)
        if not os.path.exists(path):
            continue
        with np.load(path) as z:
            P = z["pred_pose_c2w"].astype(np.float64)
            sc = float(z["scale"])
        a = CAUSAL.anchor_slot
        dp = float(np.linalg.norm(P[a][:3, 3] - P[a - 1][:3, 3]) * sc)
        g = float(d_gt[p_ - 1]) if 0 < p_ <= len(d_gt) else float("nan")
        rows.append({"sample_id": sid, "position": int(p_),
                     "time_s": float(seg.times_s[p_]), "scale": sc,
                     "ratio": (dp / g) if (np.isfinite(g) and g > 0.25) else None})
    v = np.asarray([r["ratio"] for r in rows if r["ratio"] is not None], np.float64)
    if v.size == 0:
        return {"rows": rows, "whole_stream": None}
    t = np.asarray([r["time_s"] for r in rows if r["ratio"] is not None])
    o = np.argsort(t)
    # a short scene can leave a third empty (Occ3D scenes are ~20 s and the g > 0.25 m gate
    # drops near-stationary steps), so an empty bucket is reported as None, never as a
    # percentile of nothing
    summ = lambda ix: ({"n": int(len(ix)), "median": float(np.median(v[ix])),   # noqa: E731
                        "p05": float(np.percentile(v[ix], 5)),
                        "p95": float(np.percentile(v[ix], 95))} if len(ix) else None)
    thirds = np.array_split(o, 3) if len(o) >= 3 else [o, np.array([], int), np.array([], int)]
    return {"rows": rows, "whole_stream": summ(o), "early_third": summ(thirds[0]),
            "middle_third": summ(thirds[1]), "late_third": summ(thirds[2]),
            "scale_distinct_values": int(len(set(np.round([r["scale"] for r in rows], 10)))),
            "note": ("the reference re-estimates its gauge at every anchor, so a flat curve "
                     "here against a drifting mode-B curve localises the drift to the "
                     "persistent state and the frozen gauge")}


@torch.no_grad()
def one_stream(model, dataset: str, seg, dev, probes: int = N_PROBES) -> dict:
    n = len(seg)
    images = RP.load_images(list(seg.frame_paths))
    k = RP.keyframe_interval_for(n)
    with TR.StateTrace(model) as tr:
        out = RP.replay_segment(model, images, k, dev)
    assert not tr.mid_stream_resets()
    depth = out["pred_depth"].astype(np.float32)
    conf = out["pred_depth_conf"].astype(np.float32)
    pose = out["pred_pose_c2w"].astype(np.float64)

    # the gauge the scored mode-B maps actually used
    cands = []
    for p in range(min(SV.N_INIT_FRAMES, n)):
        md, mm, _f = BM.read_moge(dataset, seg.frame_keys[p])
        cands.append(SC.frame_candidate(md, mm, depth[p], conf[p])["log_s"])
    fin = [x for x in cands if np.isfinite(x)]
    s_frozen = float(np.exp(np.median(fin))) if fin else float("nan")

    # (1) scaled predicted motion against the published motion
    gt = gt_positions(dataset, seg)
    pred_t = pose[:, :3, 3]
    d_pred = np.linalg.norm(np.diff(pred_t, axis=0), axis=1) * s_frozen
    d_gt = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    ok = d_gt > 0.25                                   # a near-stationary pair carries no ratio
    ratio = np.full(d_gt.shape, np.nan)
    ratio[ok] = d_pred[ok] / d_gt[ok]
    thirds = np.array_split(np.arange(len(ratio)), 3)

    def summ(idx):
        v = ratio[idx]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        return {"n": int(v.size), "median": float(np.median(v)),
                "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
                "mean_abs_log": float(np.mean(np.abs(np.log(v))))}

    # (2) MoGe consistency at later frames -- isolated from inference
    probe_pos = sorted(set(np.linspace(SV.N_INIT_FRAMES, n - 1, probes).astype(int).tolist()))
    probe = []
    for p in probe_pos:
        md, mm, _f = BM.read_moge(dataset, seg.frame_keys[p])
        c = SC.frame_candidate(md, mm, depth[p], conf[p])
        probe.append({"position": int(p), "time_s": float(seg.times_s[p]),
                      "log_s_now": c["log_s"], "n_valid": c["n_valid"],
                      "scale_now": float(np.exp(c["log_s"])) if np.isfinite(c["log_s"]) else None,
                      "log_difference_from_frozen": (float(c["log_s"] - np.log(s_frozen))
                                                     if np.isfinite(c["log_s"]) else None)})
    ld = [x["log_difference_from_frozen"] for x in probe
          if x["log_difference_from_frozen"] is not None]

    # map coverage along the stream, read from the maps already built (no recomputation)
    cov = []
    for sid, p in seg.anchors:
        pth = SV.map_cache_path(dataset, "B", sid)
        if os.path.exists(pth):
            _S, _F, m = BM.load_map(dataset, "B", sid)
            cov.append({"sample_id": sid, "position": int(p),
                        "time_s": float(seg.times_s[p]),
                        "n_occupied": m["n_occupied"], "n_free": m["n_free"]})
    covA = []
    for sid, p in seg.anchors:
        pth = SV.map_cache_path(dataset, "A", sid)
        if os.path.exists(pth):
            _S, _F, m = BM.load_map(dataset, "A", sid)
            covA.append({"sample_id": sid, "n_occupied": m["n_occupied"],
                         "n_free": m["n_free"], "scale": m["scale_m_per_unit"]})
    return {
        "segment": seg.name, "n_frames": n, "duration_s": float(seg.times_s[-1]),
        "startup_scale": s_frozen,
        "scaled_motion_vs_published_motion": {
            "whole_stream": summ(np.arange(len(ratio))),
            "early_third": summ(thirds[0]), "middle_third": summ(thirds[1]),
            "late_third": summ(thirds[2]),
            "pairs_excluded_below_0p25m": int((~ok).sum()),
            "note": ("a ratio of 1 means the frozen gauge moves the reconstruction the right "
                     "number of metres; ground truth is read for this diagnostic only")},
        "moge_consistency_probe": {
            "probes": probe,
            "log_difference_from_frozen": ({"median": float(np.median(ld)),
                                            "min": float(np.min(ld)), "max": float(np.max(ld)),
                                            "max_abs": float(np.max(np.abs(ld)))} if ld else None),
            "note": ("a consistency check between two monocular estimates, NOT a "
                     "ground-truth scale; these MoGe calls are outside the inference loop "
                     "and outside every timing figure")},
        "clip_reset_motion_ratio": clipreset_motion_ratio(dataset, seg, gt),
        "map_coverage_mode_B": cov,
        "map_coverage_mode_A": covA,
        "state_trace": tr.summary(),
    }


def run(dataset: str, device: str, limit: Optional[int] = None) -> dict:
    dev = torch.device(device)
    torch.cuda.set_device(dev)
    model, info = RP.build_model(dev)
    segs = ST.build(dataset)[:limit]
    rows, t0 = [], time.time()
    for i, seg in enumerate(segs):
        rows.append(one_stream(model, dataset, seg, dev))
        print(f"  [{i+1}/{len(segs)}] {seg.name}: ratio median "
              f"{rows[-1]['scaled_motion_vs_published_motion']['whole_stream']['median']:.4f}"
              if rows[-1]["scaled_motion_vs_published_motion"]["whole_stream"] else
              f"  [{i+1}/{len(segs)}] {seg.name}: no usable pair", flush=True)
        torch.cuda.empty_cache()
    med = [r["scaled_motion_vs_published_motion"]["whole_stream"]["median"] for r in rows
           if r["scaled_motion_vs_published_motion"]["whole_stream"]]
    early = [r["scaled_motion_vs_published_motion"]["early_third"]["median"] for r in rows
             if r["scaled_motion_vs_published_motion"]["early_third"]]
    late = [r["scaled_motion_vs_published_motion"]["late_third"]["median"] for r in rows
            if r["scaled_motion_vs_published_motion"]["late_third"]]
    mx = [r["moge_consistency_probe"]["log_difference_from_frozen"]["max_abs"] for r in rows
          if r["moge_consistency_probe"]["log_difference_from_frozen"]]
    cr = [r["clip_reset_motion_ratio"] for r in rows if r["clip_reset_motion_ratio"]["whole_stream"]]
    cr_e = [x["early_third"]["median"] for x in cr if x["early_third"]]
    cr_l = [x["late_third"]["median"] for x in cr if x["late_third"]]
    return {"dataset": dataset, "n_segments": len(rows), "lingbot_load": info,
            "seconds": time.time() - t0,
            "summary": {
                "scale_ratio_median_over_streams": float(np.median(med)) if med else None,
                "scale_ratio_early_third_median": float(np.median(early)) if early else None,
                "scale_ratio_late_third_median": float(np.median(late)) if late else None,
                "early_to_late_median_shift": (float(np.median(late) - np.median(early))
                                               if early and late else None),
                "moge_log_difference_max_abs_median_over_streams": (float(np.median(mx))
                                                                    if mx else None),
                "clip_reset_ratio_early_third_median": (float(np.median(cr_e)) if cr_e else None),
                "clip_reset_ratio_late_third_median": (float(np.median(cr_l)) if cr_l else None),
                "clip_reset_early_to_late_median_shift": (
                    float(np.median(cr_l) - np.median(cr_e)) if cr_e and cr_l else None)},
            "streams": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(SV.DATASETS))
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    SV.ensure_dirs()
    r = run(a.dataset, a.device, a.limit)
    out = a.out or os.path.join(SV.SCORE_DIR, f"drift_{a.dataset}.json")
    json.dump(r, open(out, "w"), indent=1, default=float)
    print(json.dumps(r["summary"], indent=1, default=float))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
