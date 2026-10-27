#!/usr/bin/env python
"""ONCE distortion gate — brief step 7, resolved geometrically rather than by reading.

The ambiguity: ONCE's documentation says *"the distortions are removed to enhance the quality
of the images"*, yet every sequence ships a non-zero 1x7 `distortion` vector and OccAny's
reader calls `cv2.undistort` on the loaded JPEG. If the shipped images are already rectified,
that second undistortion would corrupt the geometry our hard map depends on.

Two measurements, both on real ONCE images and point clouds.

**M1 — does the choice even matter?** For every LiDAR point that lands in cam01, project it
with the plain pinhole `K` and again with the published distortion applied, and convert the
pixel disagreement into a metric offset at that point's own depth,
``d_metric ~= d_pixel * Z / fx``. This is the criterion the Waymo T2 gate used: a disagreement
whose p99 stays under one voxel (0.20 m) cannot change an occupancy target. If it passes,
the ambiguity is immaterial to us and needs no further resolution.

**M2 — which projection actually aligns with the image?** LiDAR depth discontinuities fall on
object silhouettes, so they should coincide with image gradients. For each variant, score the
mean image-gradient magnitude sampled at projected depth-edge points; the better-aligned
variant scores higher. Reported as a direction indicator, with its own weakness stated: the
two projections differ by so little in the image centre that the test only has power near
the periphery, where distortion is largest.

    PYTHONPATH=$PWD python tools/method_vs_data/once_distortion_gate.py --data-root <root>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.method_vs_data.once_reader import OnceSequence, sequences                    # noqa: E402

ONE_VOXEL_M = 0.20
CAM = "cam01"


def project(pts_cam: np.ndarray, K: np.ndarray, dist: np.ndarray | None) -> np.ndarray:
    """[N,2] pixels. ``dist=None`` is the plain pinhole model."""
    z = pts_cam[:, 2]
    if dist is None:
        u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
        return np.stack([u, v], 1)
    out, _ = cv2.projectPoints(pts_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                               K, dist.reshape(1, -1))
    return out.reshape(-1, 2)


def one_frame(seq: OnceSequence, nf: int, W: int, H: int) -> dict | None:
    img = cv2.imread(seq.image_path(nf, CAM), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    H0, W0 = img.shape[:2]
    K = seq.K(CAM)
    dist = seq.distortion(CAM)
    # LiDAR (our rotated frame) -> camera: invert cam_to_velo, both in our convention
    T_c2v = seq.cam_to_velo(CAM)
    T_v2c = np.linalg.inv(T_c2v)
    p = seq.read_velodyne(nf)
    pc = p @ T_v2c[:3, :3].T + T_v2c[:3, 3]
    keep = pc[:, 2] > 1.0
    pc = pc[keep]
    uv0 = project(pc, K, None)
    uv1 = project(pc, K, dist)
    inside = ((uv0[:, 0] >= 0) & (uv0[:, 0] < W0) & (uv0[:, 1] >= 0) & (uv0[:, 1] < H0))
    if inside.sum() < 1000:
        return None
    pc, uv0, uv1 = pc[inside], uv0[inside], uv1[inside]
    dpix = np.linalg.norm(uv0 - uv1, axis=1)
    dmet = dpix * pc[:, 2] / K[0, 0]
    r = np.linalg.norm(uv0 - np.array([K[0, 2], K[1, 2]]), axis=1)

    # M2: image gradient sampled at projected depth-edge points
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx * gx + gy * gy)
    order = np.argsort(uv0[:, 0])
    z = pc[:, 2]
    # depth-edge points: large |dZ| between neighbours in image-x order
    dz = np.abs(np.diff(z[order]))
    edge = order[:-1][dz > max(np.percentile(dz, 97), 1.0)]
    scores = {}
    for tag, uv in (("pinhole", uv0), ("undistorted_again", uv1)):
        xi = np.clip(uv[edge, 0].astype(int), 0, W0 - 1)
        yi = np.clip(uv[edge, 1].astype(int), 0, H0 - 1)
        scores[tag] = float(gm[yi, xi].mean()) if len(edge) else float("nan")
    return {"image_hw": [H0, W0], "n_points_in_image": int(len(pc)),
            "dpix": {"p50": float(np.percentile(dpix, 50)),
                     "p99": float(np.percentile(dpix, 99)), "max": float(dpix.max())},
            "dmetric_m": {"p50": float(np.percentile(dmet, 50)),
                          "p99": float(np.percentile(dmet, 99)), "max": float(dmet.max())},
            "dpix_by_radius": {"centre_r<300": float(np.median(dpix[r < 300])) if (r < 300).any() else None,
                               "mid_300_700": float(np.median(dpix[(r >= 300) & (r < 700)])) if ((r >= 300) & (r < 700)).any() else None,
                               "edge_r>=700": float(np.median(dpix[r >= 700])) if (r >= 700).any() else None},
            "n_depth_edges": int(len(edge)), "gradient_score": scores}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--n-frames", type=int, default=6, help="per sequence")
    ap.add_argument("--out", default="artifacts/method_vs_data/results/once_distortion_gate.json")
    a = ap.parse_args()
    res = {"camera": CAM, "one_voxel_m": ONE_VOXEL_M, "sequences": {}}
    allm, allp, g_pin, g_und = [], [], [], []
    for s in sequences(a.data_root):
        seq = OnceSequence(a.data_root, s)
        have = [i for i in sorted(seq.T_sensor_to_world)
                if seq.has_cloud(i) and os.path.exists(seq.image_path(i, CAM))]
        if not have:
            continue
        picks = have[:: max(len(have) // a.n_frames, 1)][:a.n_frames]
        recs = [r for r in (one_frame(seq, i, 1920, 1020) for i in picks) if r]
        if not recs:
            continue
        res["sequences"][s] = {"n_frames": len(recs), "frames": recs,
                               "distortion": [round(float(v), 6) for v in seq.distortion(CAM)],
                               "K_fx_fy_cx_cy": [round(float(seq.K(CAM)[0, 0]), 2),
                                                 round(float(seq.K(CAM)[1, 1]), 2),
                                                 round(float(seq.K(CAM)[0, 2]), 2),
                                                 round(float(seq.K(CAM)[1, 2]), 2)]}
        allm += [r["dmetric_m"]["p99"] for r in recs]
        allp += [r["dpix"]["p99"] for r in recs]
        g_pin += [r["gradient_score"]["pinhole"] for r in recs]
        g_und += [r["gradient_score"]["undistorted_again"] for r in recs]
        print(f"  {s}: dpix p99 {np.median([r['dpix']['p99'] for r in recs]):7.2f} px | "
              f"dmetric p99 {np.median([r['dmetric_m']['p99'] for r in recs]):6.3f} m | "
              f"grad pinhole {np.mean([r['gradient_score']['pinhole'] for r in recs]):7.2f} "
              f"vs re-undistorted {np.mean([r['gradient_score']['undistorted_again'] for r in recs]):7.2f}")
    p99m = float(np.median(allm))
    res["summary"] = {
        "dpix_p99_median_px": float(np.median(allp)),
        "dmetric_p99_median_m": p99m,
        "immaterial_vs_one_voxel": bool(p99m <= ONE_VOXEL_M),
        "gradient_score_pinhole": float(np.mean(g_pin)),
        "gradient_score_undistorted_again": float(np.mean(g_und)),
        "gradient_favours": ("pinhole (images already rectified)"
                             if np.mean(g_pin) >= np.mean(g_und)
                             else "re-undistorted (images still distorted)")}
    sm = res["summary"]
    print(f"\n  M1 disagreement: {sm['dpix_p99_median_px']:.2f} px p99 -> "
          f"{sm['dmetric_p99_median_m']:.3f} m at depth; one voxel is {ONE_VOXEL_M} m")
    print(f"  M1 verdict: {'IMMATERIAL' if sm['immaterial_vs_one_voxel'] else 'MATERIAL'}")
    print(f"  M2 gradient alignment favours: {sm['gradient_favours']} "
          f"({sm['gradient_score_pinhole']:.2f} vs {sm['gradient_score_undistorted_again']:.2f})")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
