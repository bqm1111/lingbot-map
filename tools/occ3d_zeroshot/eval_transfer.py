#!/usr/bin/env python
"""Gate 4 step 3 — the single frozen zero-shot evaluation on Occ3D-nuScenes val.

Seven configurations, all built on the canonical 0.2 m grid and converted to the official
0.4 m grid by the identical frozen any-subvoxel rule:

    T0 C0 constant-scale fusion          T4 C3 + frozen A_occ_only  (seed 0)
    T1 C3 learned clip-scale fusion      T5 C0 + frozen A_c0_corrector (seed 0)
    T2 C3 + frozen V1 dilate_r2          T6 C3 + frozen full V3 (seed 0)
    T3 C3 + frozen V2 fill_r1_k3

Occ3D labels and masks are opened by the metric code only, after every prediction for the
clip is already fixed. No optimiser is constructed and every parameter is frozen.

    python tools/occ3d_zeroshot/eval_transfer.py
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from gates.scale_gate.scale import bootstrap_ci
from gates.voxel_gate.controls import control
from occ3d_zeroshot.grid import (
    CANONICAL, NATIVE, canonical_to_native, native_binary_target, native_distance_bands,
    points_to_canonical,
)
from occ3d_zeroshot.pipeline import (
    canonical_features, clip_scale, fuse_to_ego, load_corrector, load_depth_head,
    region_from, run_corrector,
)

ORDER = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "REF"]
LABEL = {"T0": "C0 constant scale", "T1": "C3 learned clip scale",
         "T2": "C3 + V1 dilate_r2", "T3": "C3 + V2 fill_r1_k3",
         "T4": "C3 + A_occ_only", "T5": "C0 + A_c0_corrector",
         "T6": "C3 + full V3 seed 0",
         "REF": "in-band oracle (GT within R_infer) -- diagnostic reference, not a method"}
CONTRASTS = [("T0", "T1"), ("T1", "T2"), ("T1", "T3"), ("T1", "T4"), ("T1", "T6"),
             ("T2", "T6"), ("T3", "T6"), ("T4", "T6"), ("T5", "T6")]
# REF is a reference row, never a contrast partner and never a decision input.


def scores(pred: torch.Tensor, gt: torch.Tensor, keep: torch.Tensor):
    p, t = pred & keep, gt & keep
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    den = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "iou": tp / den if den else 0.0,
            "n_pred_occupied": int(p.sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/occ3d_zeroshot/frozen_transfer.yaml")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    fv = cfg.frozen_values
    dev = torch.device(cfg.lingbot.device if torch.cuda.is_available() else "cpu")
    art = os.path.join(REPO_ROOT, cfg.experiment.output_dir)
    out_dir = os.path.join(art, "eval"); os.makedirs(out_dir, exist_ok=True)

    r_vox = int(round(float(fv.radius_m) / float(fv.canonical_voxel_size)))
    assert r_vox == 3 and abs(r_vox * float(fv.canonical_voxel_size) - 0.6) < 1e-9, \
        "physical correction radius must stay at exactly 0.6 m"

    head, hck = load_depth_head(os.path.join(REPO_ROOT, cfg.frozen.depth_head), dev)
    models = {k: load_corrector(os.path.join(REPO_ROOT, getattr(cfg.frozen, k)), dev)
              for k in ("full_s0", "occ_only_s0", "c0_corrector_s0")}
    for _, (m, _) in models.items():
        assert not any(p.requires_grad for p in m.parameters())
    assert not any(p.requires_grad for p in head.parameters())
    for k, (_, ck) in models.items():
        assert int(ck["radius"]) == 3 and float(ck["threshold"]) == float(fv.tau), \
            f"{k}: frozen radius/threshold changed"
    print(f"radius {r_vox} voxels x {fv.canonical_voxel_size} m = {fv.radius_m} m   "
          f"tau {fv.tau}   s0 {fv.s0}")

    recs = [json.loads(l) for l in
            open(os.path.join(REPO_ROOT, "manifests", "occ3d_zeroshot", "val.jsonl"))]
    if a.limit:
        recs = recs[: a.limit]
    bands = {k: torch.from_numpy(m).to(dev)
             for k, m in native_distance_bands(
                 [tuple(b) for b in cfg.eval.distance_bands_m]).items()}
    v1_kw = dict(cfg.frozen_values.v1_control)
    v2_kw = dict(cfg.frozen_values.v2_control)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    rows, band_rows, vis_rows, scale_rows = [], [], [], []
    t_inf, t0 = 0.0, time.time()

    for i, r in enumerate(recs):
        cp = os.path.join(cfg.data.cache_root, r["clip_id"] + ".npz")
        gp = os.path.join(cfg.data.occ3d_root, r["anchor_gt_path"])
        if not (os.path.exists(cp) and os.path.exists(gp)):
            continue
        d = np.load(cp)
        dep = d["pred_depth"].astype(np.float32)
        conf = d["pred_depth_conf"].astype(np.float32)
        K = d["pred_K"].astype(np.float64)
        pose = d["pred_pose_c2w"].astype(np.float64)
        Tce = d["T_camera_to_ego"][-1].astype(np.float64)

        a_clip, s_learned, support, sat = clip_scale(
            head, hck, dep, conf, float(fv.s0), float(fv.confidence_threshold),
            float(fv.min_depth_m), float(fv.max_depth_m), dev)
        if not np.isfinite(s_learned):
            continue
        scale_rows.append({"clip_id": r["clip_id"], "scene": r["scene"],
                           "a_clip": a_clip, "s_learned": s_learned,
                           "s0": float(fv.s0), "saturation": sat})

        # ---- frozen fusion, canonical 0.2 m ---------------------------------- #
        geom = {}
        for nm, s in (("c0", float(fv.s0)), ("c3", s_learned)):
            pe, fr, cf, dp = fuse_to_ego(dep, conf, K, pose, Tce, s,
                                         float(fv.confidence_threshold),
                                         float(fv.min_depth_m), float(fv.max_depth_m))
            feat5, sp = canonical_features(pe, fr, cf, dp, dev)
            occ = feat5[0] > 0
            _, keep_in = points_to_canonical(pe) if len(pe) else (None, np.zeros(0, bool))
            geom[nm] = {"feat5": feat5, "occ": occ, "R": region_from(occ, r_vox),
                        "n_points": len(pe),
                        "in_grid": float(keep_in.mean()) if len(pe) else 0.0}

        # ---- the seven configurations, canonical -> native -------------------- #
        canon = {"T0": geom["c0"]["occ"], "T1": geom["c3"]["occ"],
                 "T2": control(geom["c3"]["occ"], geom["c3"]["R"], **v1_kw),
                 "T3": control(geom["c3"]["occ"], geom["c3"]["R"], **v2_kw)}
        for tag, key, g, occ_only in (("T4", "occ_only_s0", "c3", True),
                                      ("T5", "c0_corrector_s0", "c0", False),
                                      ("T6", "full_s0", "c3", False)):
            m, ck = models[key]
            ts = time.time()
            canon[tag] = run_corrector(m, geom[g]["feat5"], geom[g]["R"], ck["norm"],
                                       float(ck["threshold"]), occ_only)
            t_inf += time.time() - ts
        # ---- labels opened only now, by the metric code ----------------------- #
        # Every prediction above is already fixed; nothing below can influence it.
        lab = np.load(gp)
        gt_np, keep_np = native_binary_target(
            lab, bool(cfg.eval.apply_camera_mask), bool(cfg.eval.apply_lidar_mask),
            int(cfg.eval.single_camera_x_cut))
        gt = torch.from_numpy(gt_np).to(dev)
        keep = torch.from_numpy(keep_np).to(dev)
        # Diagnostic reference only: the best any band-restricted method could score.
        gt_canon = gt.repeat_interleave(2, 0).repeat_interleave(2, 1).repeat_interleave(2, 2)
        canon["REF"] = gt_canon & geom["c3"]["R"]
        native = {k: canonical_to_native(v) for k, v in canon.items()}
        cam_vis = torch.from_numpy(np.asarray(lab["mask_camera"]).astype(bool)).to(dev)
        # Diagnostic partition only. Under the official setting `apply_camera_mask=True`
        # every evaluable voxel is camera-visible by construction, so the visible/
        # non-visible split is degenerate there. This second, clearly-separate scoring
        # applies ONLY the single-camera x-cut, so the non-visible half exists.
        gtx_np, keepx_np = native_binary_target(
            lab, False, False, int(cfg.eval.single_camera_x_cut))
        gtx = torch.from_numpy(gtx_np).to(dev)
        keepx = torch.from_numpy(keepx_np).to(dev)

        base_n, base_c = native["T1"], canon["T1"]
        for tag in ORDER:
            pn, pc = native[tag], canon[tag]
            sc = scores(pn, gt, keep)
            rows.append({
                "clip_id": r["clip_id"], "scene": r["scene"], "config": tag, **sc,
                "canonical_occupied": int(pc.sum()),
                "native_occupied": int(pn.sum()),
                "added_vs_T1_native": int((pn & ~base_n & keep).sum()),
                "removed_vs_T1_native": int((~pn & base_n & keep).sum()),
                "added_vs_T1_canonical": int((pc & ~base_c).sum()),
                "removed_vs_T1_canonical": int((~pc & base_c).sum()),
                "frac_pred_in_eval_mask": (float((pn & keep).sum()) /
                                           max(float(pn.sum()), 1.0)),
                "n_fused_points": geom["c0" if tag == "T0" or tag == "T5" else "c3"]["n_points"],
                "frac_points_in_grid": geom["c0" if tag in ("T0", "T5") else "c3"]["in_grid"],
                "n_eval_voxels": int(keep.sum()),
                "n_gt_occupied": int((gt & keep).sum()),
            })
            for bn, bm in bands.items():
                band_rows.append({"clip_id": r["clip_id"], "config": tag, "band": bn,
                                  **scores(pn & bm, gt & bm, keep)})
            vis_rows.append({
                "clip_id": r["clip_id"], "config": tag,
                # official setting (camera mask applied to the label)
                **{f"official_{k}": v for k, v in scores(pn, gt, keep).items()},
                # diagnostic: x-cut only, partitioned by the official mask_camera
                **{f"vis_{k}": v for k, v in
                   scores(pn & cam_vis, gtx & cam_vis, keepx).items()},
                **{f"nonvis_{k}": v for k, v in
                   scores(pn & ~cam_vis, gtx & ~cam_vis, keepx).items()},
                "frac_pred_camera_visible": (float((pn & cam_vis).sum()) /
                                             max(float(pn.sum()), 1.0))})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)

    # ---- aggregate ---------------------------------------------------------- #
    by = {}
    for x in rows:
        by.setdefault(x["config"], {})[x["clip_id"]] = x
    n_clips = len(by["T1"])
    mean = lambda t, k: float(np.mean([v[k] for v in by[t].values()]))
    agg = {t: {"label": LABEL[t], "n_clips": len(by[t]),
               **{k: mean(t, k) for k in
                  ("iou", "precision", "recall", "tp", "fp", "fn", "n_pred_occupied",
                   "canonical_occupied", "native_occupied", "added_vs_T1_native",
                   "removed_vs_T1_native", "added_vs_T1_canonical",
                   "removed_vs_T1_canonical", "frac_pred_in_eval_mask",
                   "n_fused_points", "frac_points_in_grid", "n_eval_voxels",
                   "n_gt_occupied")},
               **{f"iou_{q}": float(np.quantile([v["iou"] for v in by[t].values()], p))
                  for q, p in (("p25", .25), ("median", .5), ("p75", .75))}}
           for t in ORDER}

    scene_of = {x["clip_id"]: x["scene"] for x in rows}
    scenes = sorted({s for s in scene_of.values()})
    per_scene = {t: {s: float(np.mean([v["iou"] for cid, v in by[t].items()
                                       if scene_of[cid] == s])) for s in scenes}
                 for t in ORDER}

    def paired(x, y):
        d = [per_scene[y][s] - per_scene[x][s] for s in scenes]
        ci = bootstrap_ci(d, int(cfg.eval.bootstrap_n), int(cfg.eval.bootstrap_seed))
        ci.update({"scenes_improved": int(np.sum([v > 0 for v in d])),
                   "n_scenes": len(d),
                   "scene_improved_fraction": float(np.mean([v > 0 for v in d]))})
        return ci
    contrasts = {f"{x}->{y}": paired(x, y) for x, y in CONTRASTS}

    bb = {}
    for x in band_rows:
        bb.setdefault(x["config"], {}).setdefault(x["band"], []).append(x)
    banded = {t: {b: {k: float(np.mean([q[k] for q in v])) for k in
                      ("iou", "precision", "recall", "n_pred_occupied")}
                  for b, v in dd.items()} for t, dd in bb.items()}
    vv = {}
    for x in vis_rows:
        vv.setdefault(x["config"], []).append(x)
    vis = {t: {k: float(np.mean([q[k] for q in v])) for k in v[0]
               if k not in ("clip_id", "config")} for t, v in vv.items()}

    S = lambda k: np.array([x[k] for x in scale_rows])
    scale_stats = {
        "n_clips": len(scale_rows),
        "a_clip": {"median": float(np.median(S("a_clip"))),
                   "p05": float(np.quantile(S("a_clip"), .05)),
                   "p95": float(np.quantile(S("a_clip"), .95)),
                   "min": float(S("a_clip").min()), "max": float(S("a_clip").max())},
        "s_learned": {"median": float(np.median(S("s_learned"))),
                      "p05": float(np.quantile(S("s_learned"), .05)),
                      "p95": float(np.quantile(S("s_learned"), .95))},
        "s0": float(fv.s0),
        "residual_saturation_fraction": {"median": float(np.median(S("saturation"))),
                                         "max": float(S("saturation").max())},
    }

    for nm, data in (("per_clip", rows), ("per_band", band_rows),
                     ("visible_split", vis_rows), ("scale", scale_rows)):
        with open(os.path.join(out_dir, f"{nm}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    write_json(os.path.join(out_dir, "per_scene_iou.json"), per_scene)

    peak = (torch.cuda.max_memory_allocated(dev) / 2**30) if dev.type == "cuda" else 0.0
    write_json(os.path.join(out_dir, "summary.json"), {
        "n_clips": n_clips, "n_scenes": len(scenes), "camera": cfg.data.camera,
        "radius_voxels": r_vox, "radius_m": float(fv.radius_m),
        "canonical_grid": {"dims": list(CANONICAL.dims), "voxel_size": CANONICAL.voxel_size,
                           "origin": list(CANONICAL.origin)},
        "native_grid": {"dims": list(NATIVE.dims), "voxel_size": NATIVE.voxel_size,
                        "origin": list(NATIVE.origin)},
        "conversion_rule": "native occupied iff ANY of the 8 canonical 0.2 m subvoxels occupied",
        "eval_masks": {"apply_camera_mask": bool(cfg.eval.apply_camera_mask),
                       "apply_lidar_mask": bool(cfg.eval.apply_lidar_mask),
                       "single_camera_x_cut": int(cfg.eval.single_camera_x_cut)},
        "aggregate": agg, "contrasts": contrasts, "by_distance": banded,
        "visible_split": vis, "scale_stats": scale_stats,
        "inference_seconds_total": t_inf,
        "inference_seconds_per_clip": t_inf / max(n_clips, 1),
        "peak_gpu_gib": peak, "elapsed_s": time.time() - t0})

    print(f"\n{n_clips} clips / {len(scenes)} scenes")
    print(f"{'cfg':4s} {'IoU':>8} {'P':>7} {'R':>7} {'native occ':>11} {'canon occ':>10} "
          f"{'TP':>8} {'FP':>9} {'FN':>8} {'in-mask':>8}")
    for t in ORDER:
        s = agg[t]
        print(f"{t:4s} {s['iou']:8.4f} {s['precision']:7.3f} {s['recall']:7.3f} "
              f"{s['native_occupied']:11.0f} {s['canonical_occupied']:10.0f} "
              f"{s['tp']:8.0f} {s['fp']:9.0f} {s['fn']:8.0f} "
              f"{s['frac_pred_in_eval_mask']:8.3f}")
    print(f"\n{'contrast':12s} {'dIoU':>9}  95% CI               scenes improved")
    for k, v in contrasts.items():
        print(f"{k:12s} {v['mean']:+9.4f}  [{v['lo']:+.4f},{v['hi']:+.4f}]"
              f"{'*' if v['excludes_zero'] else ' '} {v['scenes_improved']:4d}/{v['n_scenes']}")
    print(f"\nscale: s0 {fv.s0}  s_learned median {scale_stats['s_learned']['median']:.2f} "
          f"[{scale_stats['s_learned']['p05']:.2f}, {scale_stats['s_learned']['p95']:.2f}]  "
          f"a_clip median {scale_stats['a_clip']['median']:+.3f}  "
          f"saturation {scale_stats['residual_saturation_fraction']['median']:.4f}")
    print(f"peak GPU {peak:.2f} GiB, inference {1000*t_inf/max(n_clips,1):.0f} ms/clip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
