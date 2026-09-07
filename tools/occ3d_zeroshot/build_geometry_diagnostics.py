#!/usr/bin/env python
"""Gate 4.1 step 1 — per-clip scale, pose and depth-shape diagnostics for all 1 182 clips.

Computes, once, everything the factorization needs that does not depend on occupancy:

  * the per-clip LiDAR-measured oracle scale (robust weighted-median log ratio at the
    anchor frame) and the number of valid LiDAR correspondences behind it;
  * the frozen C3 learned scale and the frozen constant s0;
  * depth-shape metrics of the frozen LingBot depth under each of the three scales;
  * LingBot-vs-ground-truth relative pose errors, by temporal offset.

LiDAR and nuScenes pose metadata are read here as **diagnostic ground truth**. No Occ3D
label, mask or occupancy array is opened anywhere in this file.

    python tools/occ3d_zeroshot/build_geometry_diagnostics.py
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from gates.scale_gate.kitti import Preprocess
from gates.scale_gate.scale import weighted_median
from lidar_scale_diagnostic import build_lidar_index, lidar_depth_on_lattice   # noqa: E402
from occ3d_zeroshot.factorization import (
    depth_metrics, gt_camera_to_world, pose_errors,
)
from occ3d_zeroshot.nuscenes_adapter import load_annotations, scene_frames
from occ3d_zeroshot.pipeline import clip_scale, load_depth_head


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/occ3d_zeroshot/gate41_factorization.yaml")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    g4 = load_config(cfg.experiment.gate4_config)
    fv = g4.frozen_values
    dev = torch.device(g4.lingbot.device if torch.cuda.is_available() else "cpu")
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    os.makedirs(art, exist_ok=True)
    ns_root, occ3d, cam = g4.data.nuscenes_root, g4.data.occ3d_root, g4.data.camera

    idx = build_lidar_index(ns_root, os.path.join(
        REPO_ROOT, g4.experiment.output_dir, "nuscenes_lidar_index.json"))
    ann = load_annotations(occ3d)
    head, hck = load_depth_head(os.path.join(REPO_ROOT, g4.frozen.depth_head), dev)
    assert not any(p.requires_grad for p in head.parameters())
    pre = Preprocess.build((900, 1600), int(fv.lingbot_image_size),
                           int(fv.lingbot_patch_size))

    recs = [json.loads(l) for l in
            open(os.path.join(REPO_ROOT, "manifests", "occ3d_zeroshot", "val.jsonl"))]
    if a.limit:
        recs = recs[: a.limit]
    min_corr = int(cfg.oracle_scale.min_correspondences)

    fmap_cache, rows, pose_rows = {}, [], []
    t0, n_fail = time.time(), 0
    for i, r in enumerate(recs):
        cp = os.path.join(g4.data.cache_root, r["clip_id"] + ".npz")
        tok = r["sample_tokens"][-1]
        if not os.path.exists(cp):
            continue
        if r["scene"] not in fmap_cache:
            fmap_cache[r["scene"]] = {f.token: f for f in
                                      scene_frames(ann, r["scene"], cam, ns_root)}
        fmap = fmap_cache[r["scene"]]
        frames = [fmap[t] for t in r["sample_tokens"]]
        d = np.load(cp)
        dep = d["pred_depth"].astype(np.float32)
        conf = d["pred_depth_conf"].astype(np.float32)
        pose_pred = d["pred_pose_c2w"].astype(np.float64)
        anchor = dep.shape[0] - 1

        a_clip, s_c3, _, sat = clip_scale(
            head, hck, dep, conf, float(fv.s0), float(fv.confidence_threshold),
            float(fv.min_depth_m), float(fv.max_depth_m), dev)

        gt_depth, gt_valid, n_corr, s_oracle = None, None, 0, float("nan")
        if tok in idx:
            gt_depth, gt_valid = lidar_depth_on_lattice(ns_root, idx[tok],
                                                        frames[anchor], pre)
        if gt_depth is not None:
            anchor_dep = dep[anchor].astype(np.float64)
            m = (gt_valid & (gt_depth > 0) & np.isfinite(anchor_dep) & (anchor_dep > 0)
                 & (conf[anchor] >= float(fv.confidence_threshold)))
            n_corr = int(m.sum())
            if n_corr >= min_corr:
                lr = np.log(gt_depth[m]) - np.log(anchor_dep[m])
                s_oracle = float(np.exp(weighted_median(
                    lr, conf[anchor][m].astype(np.float64))))
        ok = bool(np.isfinite(s_oracle) and s_oracle > 0)
        if not ok:
            n_fail += 1

        row = {"clip_id": r["clip_id"], "scene": r["scene"], "anchor_token": tok,
               "n_lidar_correspondences": n_corr, "oracle_scale_ok": int(ok),
               "s0": float(fv.s0), "s_c3": s_c3, "s_oracle": s_oracle,
               "a_clip": a_clip, "residual_saturation": sat}
        if ok:
            row.update({
                "abs_log_err_const": abs(np.log(float(fv.s0)) - np.log(s_oracle)),
                "abs_log_err_c3": abs(np.log(s_c3) - np.log(s_oracle)),
                "rel_err_const": abs(float(fv.s0) / s_oracle - 1.0),
                "rel_err_c3": abs(s_c3 / s_oracle - 1.0)})
        else:
            row.update({k: float("nan") for k in
                        ("abs_log_err_const", "abs_log_err_c3",
                         "rel_err_const", "rel_err_c3")})
        # depth shape under each scale, at LiDAR-valid pixels of the anchor frame
        if gt_depth is not None and n_corr > 0:
            vm = gt_valid & (conf[anchor] >= float(fv.confidence_threshold))
            for tag, s in (("const", float(fv.s0)), ("c3", s_c3),
                           ("oracle", s_oracle if ok else float("nan"))):
                dm = depth_metrics(dep[anchor], gt_depth, vm, s) if np.isfinite(s) else \
                     {k: float("nan") for k in ("abs_rel", "median_abs_rel",
                                                "median_abs_log", "rmse_m", "delta1")} | \
                     {"n_valid": 0}
                row.update({f"depth_{tag}_{k}": v for k, v in dm.items()})
        rows.append(row)

        c2w = gt_camera_to_world(frames)
        for tag, s in (("const", float(fv.s0)),
                       ("oracle", s_oracle if ok else float("nan"))):
            if not np.isfinite(s):
                continue
            for pe in pose_errors(pose_pred, c2w, anchor, s):
                pose_rows.append({"clip_id": r["clip_id"], "scene": r["scene"],
                                  "scale_tag": tag, **pe})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(art, "geometry_per_clip.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(art, "pose_errors_per_frame.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pose_rows[0])); w.writeheader()
        w.writerows(pose_rows)

    ok_rows = [r for r in rows if r["oracle_scale_ok"]]
    A = lambda k, src=ok_rows: np.array([x[k] for x in src], dtype=np.float64)
    q = lambda v, p: float(np.quantile(v[np.isfinite(v)], p))
    summ = {
        "n_clips": len(rows), "n_oracle_ok": len(ok_rows), "n_oracle_failed": n_fail,
        "min_correspondences": min_corr,
        "failure_policy": cfg.oracle_scale.on_failure,
        "failed_clip_ids": [r["clip_id"] for r in rows if not r["oracle_scale_ok"]],
        "correspondences": {"median": float(np.median(A("n_lidar_correspondences", rows))),
                            "min": int(A("n_lidar_correspondences", rows).min()),
                            "p05": q(A("n_lidar_correspondences", rows), .05)},
        "scale": {
            "s0": float(fv.s0),
            "s_oracle": {"median": float(np.median(A("s_oracle"))),
                         "p05": q(A("s_oracle"), .05), "p95": q(A("s_oracle"), .95)},
            "s_c3": {"median": float(np.median(A("s_c3"))),
                     "p05": q(A("s_c3"), .05), "p95": q(A("s_c3"), .95)},
            "median_abs_log_err": {"const": float(np.median(A("abs_log_err_const"))),
                                   "c3": float(np.median(A("abs_log_err_c3")))},
            "median_rel_err": {"const": float(np.median(A("rel_err_const"))),
                               "c3": float(np.median(A("rel_err_c3")))},
            "ratio_const_over_oracle": float(np.median(float(fv.s0) / A("s_oracle"))),
            "ratio_c3_over_oracle": float(np.median(A("s_c3") / A("s_oracle"))),
        },
        "depth_shape": {
            tag: {k: float(np.nanmedian(A(f"depth_{tag}_{k}")))
                  for k in ("abs_rel", "median_abs_rel", "median_abs_log", "rmse_m", "delta1")}
            for tag in ("const", "c3", "oracle")},
        "pose": {},
    }
    for tag in ("const", "oracle"):
        sub = [p for p in pose_rows if p["scale_tag"] == tag]
        if not sub:
            continue
        P = lambda k, s=sub: np.array([x[k] for x in s], dtype=np.float64)
        summ["pose"][tag] = {
            "n_frame_pairs": len(sub),
            "rot_err_deg": {"median": float(np.nanmedian(P("rot_err_deg"))),
                            "p95": q(P("rot_err_deg"), .95)},
            "trans_err_m": {"median": float(np.nanmedian(P("trans_err_m"))),
                            "p95": q(P("trans_err_m"), .95)},
            "trans_mag_ratio": {"median": float(np.nanmedian(P("trans_mag_ratio"))),
                                "p05": q(P("trans_mag_ratio"), .05),
                                "p95": q(P("trans_mag_ratio"), .95)},
            "trans_dir_err_deg": {"median": float(np.nanmedian(P("trans_dir_err_deg"))),
                                  "p95": q(P("trans_dir_err_deg"), .95)},
            "by_offset": {str(o): {
                "rot_err_deg": float(np.nanmedian(P("rot_err_deg", [x for x in sub if x["offset"] == o]))),
                "trans_err_m": float(np.nanmedian(P("trans_err_m", [x for x in sub if x["offset"] == o]))),
                "trans_mag_gt_m": float(np.nanmedian(P("trans_mag_gt_m", [x for x in sub if x["offset"] == o]))),
                "trans_mag_ratio": float(np.nanmedian(P("trans_mag_ratio", [x for x in sub if x["offset"] == o]))),
                "trans_dir_err_deg": float(np.nanmedian(P("trans_dir_err_deg", [x for x in sub if x["offset"] == o]))),
            } for o in sorted({x["offset"] for x in sub})},
        }
    write_json(os.path.join(art, "geometry_diagnostics.json"), summ)

    print(f"\n{len(rows)} clips, oracle scale OK for {len(ok_rows)}, failed {n_fail}")
    print(f"  correspondences median {summ['correspondences']['median']:.0f}, "
          f"min {summ['correspondences']['min']}")
    s = summ["scale"]
    print(f"  s_oracle median {s['s_oracle']['median']:.3f}  s0 {s['s0']}  "
          f"s_c3 median {s['s_c3']['median']:.3f}")
    print(f"  median rel scale error: const {s['median_rel_err']['const']:.4f}  "
          f"c3 {s['median_rel_err']['c3']:.4f}")
    print(f"  depth AbsRel (median over clips): "
          + "  ".join(f"{t} {summ['depth_shape'][t]['abs_rel']:.4f}"
                      for t in ("const", "c3", "oracle")))
    for tag in summ["pose"]:
        p = summ["pose"][tag]
        print(f"  pose[{tag}]: rot {p['rot_err_deg']['median']:.3f} deg, "
              f"trans {p['trans_err_m']['median']:.3f} m, "
              f"mag ratio {p['trans_mag_ratio']['median']:.3f}, "
              f"dir {p['trans_dir_err_deg']['median']:.2f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
