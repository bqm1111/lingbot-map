#!/usr/bin/env python
"""Gate 5.2 step 4 — label-free preflight on 10 clips spread across sequence 0006.

Runs before any ``.label``, ``.invalid`` or LiDAR file is opened, and proves it: the whole
body executes inside :class:`sscbench_kitti360.audit.FileAudit`, which raises if a
forbidden path is touched.

    python tools/gate5_2/preflight.py
"""
from __future__ import annotations

import argparse, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                                "depth_gate")))

from decompose_residual import scaled_relative_pose
from gates.scale_gate.config import REPO_ROOT, load_config, write_json
from gates.scale_gate.kitti import Preprocess, read_manifest
from moge_gauge.calibrated import CalibratedMoGe
from moge_gauge.calibration import calibrated_fov_x_deg
from moge_gauge.estimator import estimate_clip_scale, frame_valid
from sscbench_kitti360.adapter import SEQUENCE, load_cam0_to_world, parse_calibration
from sscbench_kitti360.audit import FileAudit


def as4x4(p):
    if p.shape[-2:] == (4, 4):
        return p.astype(np.float64)
    out = np.tile(np.eye(4), (len(p), 1, 1))
    out[:, :3, :4] = p
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gate5_2/kitti360_transfer.yaml")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    cfg = load_config(a.config)
    g5 = load_config(cfg.experiment.gate5_config)
    scfg = load_config("configs/scale_gate/semantickitti.yaml")
    est = g5.estimator
    dev = torch.device(cfg.lingbot.device if torch.cuda.is_available() else "cpu")
    conf_thr = float(scfg.lingbot.confidence_threshold)

    recs = read_manifest(os.path.join(REPO_ROOT, "manifests", "gate5_2", "val.jsonl"))
    recs = [r for r in recs
            if os.path.exists(os.path.join(cfg.lingbot.cache_root, r["clip_id"] + ".npz"))]
    pick = [recs[int(round(i * (len(recs) - 1) / (a.n - 1)))] for i in range(a.n)]
    print(f"{len(recs)} cached clips; preflight on {len(pick)} spread across the drive")

    results, failures = [], []
    with FileAudit() as audit:
        from lingbot_map.utils.load_fn import load_and_preprocess_images
        calib = parse_calibration(os.path.join(cfg.dataset.root, "calibration"))
        pre = Preprocess.build(calib.native_hw, int(cfg.lingbot.inference_resolution),
                               int(cfg.lingbot.patch_size))
        Kp = pre.scale_intrinsics(calib.K)
        fov_cal = calibrated_fov_x_deg(float(Kp[0, 0]), int(pre.proc_hw[1]))
        c2w_gt = load_cam0_to_world(os.path.join(cfg.dataset.root, "data_poses", SEQUENCE,
                                                 "cam0_to_world.txt"))
        moge = CalibratedMoGe(dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo,
                              revision=g5.moge.hf_revision)
        root = os.path.join(cfg.dataset.kitti360_root, "data_2d_raw", SEQUENCE)
        t0 = time.time()

        for rec in pick:
            cid = rec["clip_id"]
            chk = {}
            nat = list(rec["native_frames"])
            ts = list(rec["timestamps_s"])
            chk["five_frames"] = len(nat) == 5 == len(set(nat))
            chk["chronological"] = all(nat[i] < nat[i + 1] for i in range(4)) and \
                all(ts[i] < ts[i + 1] for i in range(4))
            chk["anchor_is_last"] = nat[-1] == max(nat)
            chk["no_future_frame"] = max(nat) == nat[-1] and \
                nat[-1] == int(rec["native_frames"][-1])
            span = ts[-1] - ts[0]
            chk["span_about_two_seconds"] = 1.8 <= span <= 2.4

            d = np.load(os.path.join(cfg.lingbot.cache_root, cid + ".npz"),
                        allow_pickle=False)
            dep = d["pred_depth"].astype(np.float32)
            conf = d["pred_depth_conf"].astype(np.float32)
            pose = as4x4(d["pred_pose_c2w"])
            rgb = load_and_preprocess_images(
                [os.path.join(root, q) for q in rec["image_paths"]], mode="crop",
                image_size=int(cfg.lingbot.inference_resolution),
                patch_size=int(cfg.lingbot.patch_size)).numpy()

            chk["rgb_in_unit_range"] = bool(rgb.min() >= -1e-6 and rgb.max() <= 1 + 1e-6)
            chk["intrinsics_transformed"] = bool(
                np.allclose(Kp[0, 0], calib.K[0, 0] * pre.sx) and
                np.allclose(Kp[1, 1], calib.K[1, 1] * pre.sy) and
                np.allclose(Kp[0, 2], calib.K[0, 2] * pre.sx) and
                np.allclose(Kp[1, 2], calib.K[1, 2] * pre.sy))
            chk["lattice_matches"] = tuple(rgb.shape[-2:]) == tuple(pre.proc_hw) == \
                tuple(dep.shape[-2:])

            # pose direction: predicted vs ground-truth relative translation, anchor-last
            gt = np.stack([c2w_gt[f] for f in nat])
            cos = []
            for f in range(4):
                tp = scaled_relative_pose(pose, f, 4, 1.0)[:3, 3]
                tg = (np.linalg.inv(gt[4]) @ gt[f])[:3, 3]
                if np.linalg.norm(tp) > 1e-9 and np.linalg.norm(tg) > 1e-9:
                    cos.append(float(tp @ tg / (np.linalg.norm(tp) * np.linalg.norm(tg))))
            chk["pose_direction_positive"] = bool(cos and float(np.median(cos)) > 0.9)

            # pure-similarity fusion: scaling depth and translations scales the cloud
            s = 3.7
            P1 = scaled_relative_pose(pose, 0, 4, 1.0)
            Ps = scaled_relative_pose(pose, 0, 4, s)
            chk["rotation_unscaled"] = bool(np.allclose(P1[:3, :3], Ps[:3, :3], atol=1e-12))
            chk["translation_scaled"] = bool(np.allclose(Ps[:3, 3], s * P1[:3, 3],
                                                         rtol=1e-9, atol=1e-12))

            md = np.empty(dep.shape, np.float32)
            mm = np.empty(dep.shape, bool)
            fovs_i, fovs_c, per_frame_s = [], [], []
            for t in range(5):
                oi = moge.infer_calibrated(rgb[t], fov_x_deg=None)
                oc = moge.infer_calibrated(rgb[t], fov_x_deg=fov_cal)
                fovs_i.append(oi.fov_x_deg); fovs_c.append(oc.fov_x_deg)
                md[t], mm[t] = oc.depth_z, oc.mask
                fin = np.isfinite(oc.depth_z) & mm[t]
                if not (fin & (oc.depth_z > 0)).sum() == fin.sum():
                    failures.append(f"{cid} f{t}: non-positive teacher depth inside mask")
                v = frame_valid(oc.depth_z, oc.mask, dep[t], conf[t], conf_thr,
                                float(est.min_depth_m), float(est.max_depth_m))
                per_frame_s.append(float(np.exp(np.median(
                    np.log(oc.depth_z[v]) - np.log(dep[t][v])))) if v.any() else np.nan)
            chk["teacher_depth_finite_positive"] = bool(
                np.isfinite(md[mm]).all() and (md[mm] > 0).all())
            chk["teacher_lattice_matches"] = md.shape == dep.shape
            chk["fov_calibrated_plausible"] = 1.0 < fov_cal < 179.0

            r = estimate_clip_scale(md, mm, dep, conf, float(est.conf_threshold),
                                    float(est.min_depth_m), float(est.max_depth_m),
                                    int(est.min_valid_pixels_per_clip))
            chk["one_scalar_per_clip"] = bool(np.isscalar(r["s_moge"]) or
                                              np.asarray(r["s_moge"]).ndim == 0)
            chk["enough_valid_pixels"] = bool(r["ok"])
            results.append({
                "clip_id": cid, "anchor": int(rec["anchor"]), "native_frames": nat,
                "span_s": round(span, 4), "checks": chk,
                "fov_moge_inferred_deg": float(np.median(fovs_i)),
                "fov_calibrated_deg": fov_cal,
                "fov_out_when_supplied_deg": float(np.median(fovs_c)),
                "fov_difference_deg": float(np.median(fovs_i)) - fov_cal,
                "moge_mask_fraction": float(mm.mean()),
                "n_valid_scale_px": int(r["n_valid_total"]),
                "per_frame_implied_scale": [round(x, 5) for x in per_frame_s],
                "within_clip_log_scale_std": r["per_frame_log_scale_std"],
                "s_moge_calibrated": r["s_moge"],
                "pose_direction_cosine_median": float(np.median(cos)) if cos else None,
            })
            failures += [f"{cid}: {k}" for k, v in chk.items() if not v]

        elapsed = time.time() - t0

    summary = {"n_clips": len(results), "elapsed_s": elapsed,
               "file_audit": audit.summary(),
               "no_target_or_lidar_opened": not audit.violations,
               "failures": failures, "clips": results,
               "aggregate": {
                   "span_s_median": float(np.median([r["span_s"] for r in results])),
                   "fov_calibrated_deg": results[0]["fov_calibrated_deg"],
                   "fov_moge_inferred_deg_median": float(
                       np.median([r["fov_moge_inferred_deg"] for r in results])),
                   "fov_difference_deg_median": float(
                       np.median([r["fov_difference_deg"] for r in results])),
                   "moge_mask_fraction_median": float(
                       np.median([r["moge_mask_fraction"] for r in results])),
                   "n_valid_scale_px_min": int(
                       min(r["n_valid_scale_px"] for r in results)),
                   "within_clip_log_scale_std_median": float(np.median(
                       [r["within_clip_log_scale_std"] for r in results])),
               }}
    write_json(os.path.join(REPO_ROOT, cfg.experiment.output_dir, "preflight.json"), summary)
    print(f"\nfile audit: {audit.summary()['n_unique']} unique paths, "
          f"{len(audit.violations)} forbidden")
    for k, v in summary["aggregate"].items():
        print(f"  {k}: {v}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("\npreflight PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
