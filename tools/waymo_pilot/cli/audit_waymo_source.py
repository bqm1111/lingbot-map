#!/usr/bin/env python
"""Audit the extracted Waymo cache: what this source actually is, before any supervision.

Everything here is a property of the source and its calibration, measured from the frames on
disk. It deliberately stops short of supervision yields, which need the frozen causal
pipeline (LingBot, MoGe-2, the hard mapper, Trident-H) to have been run on Waymo images.

    /media/SSD1/MINH_DATASETS/envs/waymo/bin/python tools/waymo_pilot/cli/audit_waymo_source.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.waymo_pilot import config as C, sample as SA, waymo as W                  # noqa: E402


def _frustum_share(K: np.ndarray, w: int, h: int, origin_z: float, cam_z: float) -> float:
    """Share of the anchor grid a single view of this camera can see at all."""
    X, Y, Z = C.GRID_DIMS
    ix, iy, iz = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing="ij")
    c = (np.stack([ix, iy, iz], -1).reshape(-1, 3) + 0.5) * C.VOXEL_SIZE_M
    c = c + np.array([C.GRID_ORIGIN_M[0], C.GRID_ORIGIN_M[1], origin_z])
    px, py, pz = -c[:, 1], -(c[:, 2] - cam_z), c[:, 0]      # grid -> pinhole axes
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * px / pz + K[0, 2]
        v = K[1, 1] * py / pz + K[1, 2]
    ok = (pz >= 1) & (pz < 60) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return float(ok.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(C.Paths().logs, "waymo_source_audit.json"))
    a = ap.parse_args()
    C.ensure_dirs()
    p = C.Paths()
    segs = sorted(d for d in glob.glob(os.path.join(p.waymo_frames, "*")) if os.path.isdir(d))
    if not segs:
        print(f"no extracted frames under {p.waymo_frames}; run waymo_extract.py first",
              file=sys.stderr)
        return 2

    per_seg: List[Dict] = []
    road_z, sync, n_pts, no_cam, cal0 = [], {}, [], [], None
    total_bytes = 0
    for sd in segs:
        files = sorted(glob.glob(os.path.join(sd, "*.npz")))
        rows = []
        for f in files:
            total_bytes += os.path.getsize(f)
            with np.load(f, allow_pickle=False) as z:
                pts = z["points_vehicle"]
                cps = z["cp_points"]
                cal = json.loads(str(z["calibration_json"]))
                syn = json.loads(str(z["sync_error_json"]))
                if cal0 is None:
                    cal0 = cal
                near = pts[(np.abs(pts[:, 1]) < 3) & (pts[:, 0] > 3) & (pts[:, 0] < 15)]
                if len(near) > 100:
                    road_z.append(float(np.percentile(near[:, 2], 2)))
                n_pts.append(int(len(pts)))
                no_cam.append(float((cps[:, 0] == 0).mean()) if len(cps) else np.nan)
                for k, v in syn.items():
                    sync.setdefault(k, []).append(float(v))
                rows.append(int(z["frame_index"]))
        per_seg.append({"segment": os.path.basename(sd), "n_frames": len(files),
                        "frame_indices": [min(rows), max(rows)] if rows else [],
                        "anchors_supported": C.anchors_supported(len(files)),
                        "independent_anchor_ceiling":
                            SA.independence_ceiling(
                                list(range(C.anchors_supported(len(files)))),
                                C.MIN_ANCHOR_SEPARATION_FRAMES * C.WAYMO_STREAM_STRIDE)})

    road = float(np.median(road_z)) if road_z else float("nan")
    origin = C.waymo_grid_origin(road)
    cams = {}
    for name, c in (cal0 or {}).items():
        K = np.asarray(c["K"], float)
        cam_z = float(np.asarray(c["cam_to_vehicle"], float)[2, 3])
        cams[name] = {
            "width": c["width"], "height": c["height"],
            "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
            "hfov_deg": float(2 * np.degrees(np.arctan(K[0, 2] / K[0, 0]))),
            "vfov_deg": float(2 * np.degrees(np.arctan(K[1, 2] / K[1, 1]))),
            "height_above_vehicle_origin_m": cam_z,
            "rolling_shutter_direction": c["rolling_shutter_direction"],
            "grid_frustum_share_single_view":
                _frustum_share(K, c["width"], c["height"], origin[2], cam_z),
        }
    K_k360 = np.array([[552.0, 0, 704.0], [0, 552.0, 188.0]])
    k360_share = _frustum_share(K_k360, 1408, 376, C.GRID_ORIGIN_M[2], 0.0)

    rec = {
        "version": C.VERSION,
        "source": {"expected_version": C.WAYMO_EXPECTED_VERSION,
                   "n_segments": len(per_seg),
                   "n_frames": int(sum(s["n_frames"] for s in per_seg)),
                   "cache_gib": total_bytes / 2 ** 30,
                   "gib_per_frame": total_bytes / 2 ** 30 / max(sum(s["n_frames"] for s in per_seg), 1),
                   "segments": per_seg},
        "temporal_protocol": {
            "native_hz": C.WAYMO_NOMINAL_HZ, "stream_stride": C.WAYMO_STREAM_STRIDE,
            "stream_dt_s": C.WAYMO_STREAM_STRIDE / C.WAYMO_NOMINAL_HZ,
            "kitti360_stream_dt_s": 0.522,
            "causal_clip_span_s": (C.CLIP_LENGTH - 1) * C.WAYMO_STREAM_STRIDE / C.WAYMO_NOMINAL_HZ,
            **C.anchor_frame_budget()},
        "grid": {"road_z_in_vehicle_frame_m": road,
                 "origin": list(origin), "dims": list(C.GRID_DIMS),
                 "voxel_size_m": C.VOXEL_SIZE_M,
                 "kitti360_origin": list(C.GRID_ORIGIN_M),
                 "kitti360_road_z_in_grid_frame_m": C.K360_ROAD_Z_IN_GRID_FRAME_M,
                 "floor_below_road_m": C.GRID_FLOOR_BELOW_ROAD_M,
                 "note": ("z offset chosen so the grid floor sits the same distance below the "
                          "MEASURED road on both datasets; dims, voxel size and x/y extent "
                          "are the frozen ones")},
        "cameras": cams,
        "selected_camera": C.WAYMO_SELECTED_CAMERA,
        "field_of_view_gap": {
            "waymo_selected_hfov_deg": cams.get(C.WAYMO_SELECTED_CAMERA, {}).get("hfov_deg"),
            "kitti360_image00_hfov_deg": C.K360_IMAGE00_HFOV_DEG,
            "waymo_single_view_grid_share":
                cams.get(C.WAYMO_SELECTED_CAMERA, {}).get("grid_frustum_share_single_view"),
            "kitti360_single_view_grid_share": k360_share,
            "note": ("the deployment contract is monocular, so this gap cannot be closed by "
                     "protocol. It is the largest difference between the arms and any A/B "
                     "result has to be read with it in front.")},
        "synchronisation": {
            "per_camera_ms": {k: {"median": float(np.median(v) * 1000),
                                  "p90": float(np.percentile(v, 90) * 1000),
                                  "max": float(np.max(v) * 1000)} for k, v in sync.items()},
            "bound_s": C.MAX_SYNC_ERROR_S,
            "kitti360_reference_ms": 5.0,
            "note": ("CameraImage.pose is the vehicle pose at that image's own timestamp, so "
                     "ego motion is compensated per camera; the residual applies to moving "
                     "content, which the conflict and static-support rules already exclude")},
        "lidar": {"returns_per_frame": {"median": float(np.median(n_pts)),
                                        "min": int(np.min(n_pts)), "max": int(np.max(n_pts))},
                  "fraction_projecting_into_no_camera": float(np.nanmean(no_cam)),
                  "note": ("an upper bound on privileged semantic coverage before any "
                           "occlusion test: a return no camera sees can carry no teacher "
                           "feature")},
    }
    with open(a.out, "w") as fh:
        json.dump(rec, fh, indent=2, default=float)

    s = rec["source"]
    print(f"  segments / frames        {s['n_segments']} / {s['n_frames']}")
    print(f"  cache                    {s['cache_gib']:.2f} GiB "
          f"({s['gib_per_frame'] * 1024:.1f} MiB per frame)")
    print(f"  anchors supported        {sum(x['anchors_supported'] for x in per_seg)}")
    print(f"  road z (vehicle frame)   {road:+.3f} m -> grid origin z {origin[2]:+.3f} m")
    print(f"  selected camera hfov     {rec['field_of_view_gap']['waymo_selected_hfov_deg']:.1f} deg "
          f"vs KITTI-360 {C.K360_IMAGE00_HFOV_DEG:.1f} deg")
    print(f"  single-view grid share   "
          f"{100 * rec['field_of_view_gap']['waymo_single_view_grid_share']:.1f} % vs "
          f"{100 * k360_share:.1f} %")
    print(f"  sync error (FRONT)       "
          f"{rec['synchronisation']['per_camera_ms'][C.WAYMO_SELECTED_CAMERA]['median']:.1f} ms "
          f"vs KITTI-360 < 5 ms")
    print(f"  returns with no camera   "
          f"{100 * rec['lidar']['fraction_projecting_into_no_camera']:.1f} %")
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
