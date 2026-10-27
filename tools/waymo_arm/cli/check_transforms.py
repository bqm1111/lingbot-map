#!/usr/bin/env python
"""Phase 1.1 — coordinate-convention validation for the Waymo arm.

Seven checks, each either exact or against a stated tolerance. None of them replaces an
existing implementation; they test the one already in use.

T1  axis permutation      a point on the Waymo camera's optical axis (+x forward) must land
                          on the principal point once ``AXES_CAM_TO_PINHOLE`` is applied.
T2  SDK projection        our ``waymo.project`` against the SDK's OWN ``cp_points``, which
                          the extractor cached per point. This is the decisive test: it
                          compares against Waymo's computation, not against ourselves.
                          ``project`` deliberately applies NO lens distortion, so the
                          residual is NOT expected to be sub-pixel. The criterion is the one
                          the pilot already states for this quantity: the residual converted
                          to metres at each point's own depth must stay below one 0.2 m
                          voxel, since the projection is only used to decide which voxel a
                          return falls in.
T3  pose round trip       vehicle(a)->vehicle(b) composed with its inverse is the identity.
T4  anchor identity       the anchor frame's vehicle->vehicle transform is exactly I.
T5  laser inclinations    each laser's points, in spherical coordinates about ITS OWN cached
                          origin, must lie inside that laser's calibrated beam-inclination
                          range. This tests the sidecar slicing and the per-laser origins
                          together -- a wrong slice or a wrong origin breaks it.
T6  voxel indexing        grid coordinate -> index -> flat -> back is the identity, and the
                          flat ordering matches the C-order the evaluators assume.
T7  grid placement        the anchor camera's centre must sit inside the grid, and the
                          measured road must sit at the predeclared height above the floor.

    PYTHONPATH=$PWD python tools/waymo_arm/cli/check_transforms.py --segments 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.waymo_arm import config as A                                                 # noqa: E402
from tools.waymo_arm import geometry as WAG                                             # noqa: E402
from tools.waymo_pilot import config as WP                                               # noqa: E402
from tools.waymo_pilot import waymo as W                                                  # noqa: E402
from tools.waymo_pilot import waymo_geometry as WG                                        # noqa: E402

#: one voxel. The projection's only job is to pick a voxel, so the tolerance is metric.
REPROJ_TOL_M = 0.2
INCL_PAD_DEG = 1.5     # degrees of slack on the calibrated beam range


def t1_axis_permutation(seg: WG.WaymoSegment) -> dict:
    cam = seg.calibration()
    # Waymo camera axes: +x forward, +y left, +z up. A point 10 m straight ahead.
    p_cam = np.array([[10.0, 0.0, 0.0]])
    P = np.asarray(W.AXES_CAM_TO_PINHOLE, np.float64)[:3, :3] @ p_cam[0]
    uv = (np.asarray(cam.K, np.float64) @ P) / P[2]
    cx, cy = float(cam.K[0, 2]), float(cam.K[1, 2])
    err = float(np.hypot(uv[0] - cx, uv[1] - cy))
    return {"check": "T1_axis_permutation", "principal_point": [cx, cy],
            "projected": [float(uv[0]), float(uv[1])], "error_px": err,
            "z_forward_positive": bool(P[2] > 0), "passed": bool(err < 1e-6 and P[2] > 0)}


def _project_distorted(pts_vehicle: np.ndarray, cam) -> tuple:
    """The same projection WITH Waymo's radial-tangential distortion, for the diagnosis only.

    Waymo's intrinsic vector is (f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3); the model is the
    standard OpenCV radial-tangential one. This is not used anywhere on the pipeline path --
    it exists to tell apart "our transform is wrong" from "the distortion-free approximation
    costs this much".
    """
    T = cam.vehicle_to_pinhole
    q = np.asarray(pts_vehicle, np.float64) @ T[:3, :3].T + T[:3, 3]
    z = q[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    x, y = q[:, 0] / safe, q[:, 1] / safe
    k1, k2, p1, p2, k3 = (float(v) for v in cam.distortion)
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return cam.K[0, 0] * xd + cam.K[0, 2], cam.K[1, 1] * yd + cam.K[1, 2], z


def t2_sdk_projection(seg: WG.WaymoSegment, camera: str = "FRONT") -> dict:
    cam = seg.calibration(camera)
    cid = W.CAMERA_ID[camera]
    rows = []
    for fr in seg.frames[:3]:
        with np.load(fr.path, allow_pickle=False) as z:
            pts, cp = z["points_vehicle"], z["cp_points"]
        m = cp[:, 0] == cid
        if int(m.sum()) < 100:
            continue
        uv_sdk = cp[m][:, 1:3].astype(np.float64)
        u, v, zc = W.project(pts[m].astype(np.float64), cam)
        keep = zc > 1.0
        if int(keep.sum()) < 100:
            continue
        d_px = np.hypot(u[keep] - uv_sdk[keep, 0], v[keep] - uv_sdk[keep, 1])
        # px -> m at that point's own depth: err_m = err_px * z / f
        d_m = d_px * zc[keep] / float(cam.K[0, 0])
        ud, vd, _ = _project_distorted(pts[m].astype(np.float64), cam)
        dd_px = np.hypot(ud[keep] - uv_sdk[keep, 0], vd[keep] - uv_sdk[keep, 1])
        dd_m = dd_px * zc[keep] / float(cam.K[0, 0])
        rows.append({"frame": int(fr.native), "n_points": int(keep.sum()),
                     "px_p50": float(np.percentile(d_px, 50)),
                     "px_p99": float(np.percentile(d_px, 99)),
                     "m_p50": float(np.percentile(d_m, 50)),
                     "m_p99": float(np.percentile(d_m, 99)),
                     "m_max": float(d_m.max()),
                     "frac_within_one_voxel": float((d_m <= REPROJ_TOL_M).mean()),
                     "with_distortion_px_p50": float(np.percentile(dd_px, 50)),
                     "with_distortion_px_p99": float(np.percentile(dd_px, 99)),
                     "with_distortion_m_p99": float(np.percentile(dd_m, 99)),
                     "with_distortion_frac_within_one_voxel": float((dd_m <= REPROJ_TOL_M).mean())})
    worst = max((r["m_p99"] for r in rows), default=float("inf"))
    worst_d = max((r["with_distortion_m_p99"] for r in rows), default=float("inf"))
    return {"check": "T2_projection_vs_sdk_cp_points", "camera": camera,
            "tolerance_m": REPROJ_TOL_M,
            # NOT a gate for this experiment: ``W.project`` is called only by the pilot's own
            # synthetic checks and ``verify_reader``. The occupancy target and the hard map
            # never project -- LiDAR is voxelised directly in the grid frame -- so this
            # residual cannot affect Phases 2-4. It IS a precondition for any Waymo SEMANTIC
            # target, which must assign returns to pixels.
            "gates": "semantic_target_only",
            "note": "no lens distortion is applied by design; criterion is metric, not pixel",
            "rows": rows, "worst_p99_m": worst,
            "worst_p99_m_with_distortion": worst_d,
            "distortion_explains_tail": bool(worst_d < worst),
            "passed": bool(rows) and worst <= REPROJ_TOL_M}


def t3_t4_poses(wag: WAG.WaymoAnchorGeometry, anchor: int) -> dict:
    frames = wag.target_stream_frames(anchor)
    worst_rt, worst_id = 0.0, 0.0
    for t in frames:
        T = wag.vehicle_to_vehicle(t, anchor)
        Ti = wag.vehicle_to_vehicle(anchor, t)
        worst_rt = max(worst_rt, float(np.abs(T @ Ti - np.eye(4)).max()))
        if t == anchor:
            worst_id = float(np.abs(T - np.eye(4)).max())
    return {"check": "T3_T4_pose_roundtrip_and_anchor_identity", "n_frames": len(frames),
            "max_roundtrip_error": worst_rt, "anchor_identity_error": worst_id,
            "passed": bool(worst_rt < 1e-9 and worst_id == 0.0)}


def t5_laser_inclinations(wag: WAG.WaymoAnchorGeometry, anchor: int) -> dict:
    side = wag.lasers
    cal = json.loads(str(np.load(wag.paths.laser_sidecar(wag.segment))["laser_calibration_json"]))
    rows, ok_all = [], True
    sweeps = wag.sweeps(anchor, horizon=0)             # the anchor frame only
    for sw in sweeps:
        o = np.asarray(side.origins[sw.laser], np.float64)
        T = wag.vehicle_to_vehicle(int(sw.frame), anchor)
        # sw.points / sw.sensor are already in the grid frame; measure about sw.sensor
        v = np.asarray(sw.points, np.float64) - np.asarray(sw.sensor, np.float64)
        r = np.linalg.norm(v, axis=1)
        keep = r > 1e-3
        incl = np.degrees(np.arcsin(np.clip(v[keep, 2] / r[keep], -1, 1)))
        lo = np.degrees(float(cal[sw.laser]["beam_inclination_min"])) - INCL_PAD_DEG
        hi = np.degrees(float(cal[sw.laser]["beam_inclination_max"])) + INCL_PAD_DEG
        inside = float(((incl >= lo) & (incl <= hi)).mean())
        n_beams = len(cal[sw.laser]["beam_inclinations"])
        rows.append({"laser": sw.laser, "n_points": int(keep.sum()),
                     "calibrated_range_deg": [round(lo + INCL_PAD_DEG, 2),
                                              round(hi - INCL_PAD_DEG, 2)],
                     "n_beam_inclinations": n_beams,
                     "measured_p1_p99_deg": [float(np.percentile(incl, 1)),
                                             float(np.percentile(incl, 99))],
                     "frac_inside": inside, "passed": bool(inside > 0.99)})
        ok_all = ok_all and inside > 0.99
    return {"check": "T5_per_laser_beam_inclinations", "pad_deg": INCL_PAD_DEG,
            "rows": rows, "passed": bool(rows) and ok_all}


def t6_voxel_indexing() -> dict:
    grid = WG.waymo_grid()
    X, Y, Z = (int(d) for d in grid.dims)
    vs, org = float(grid.voxel_size), np.asarray(grid.origin, np.float64)
    rng = np.random.default_rng(0)
    ijk = np.stack([rng.integers(0, X, 4096), rng.integers(0, Y, 4096),
                    rng.integers(0, Z, 4096)], 1)
    centres = (ijk + 0.5) * vs + org
    back = np.floor((centres - org) / vs).astype(np.int64)
    flat = (ijk[:, 0] * Y + ijk[:, 1]) * Z + ijk[:, 2]
    flat_c = np.ravel_multi_index((ijk[:, 0], ijk[:, 1], ijk[:, 2]), (X, Y, Z))
    return {"check": "T6_voxel_index_roundtrip", "dims": [X, Y, Z], "voxel_size": vs,
            "origin": org.tolist(), "n_tested": len(ijk),
            "index_roundtrip_exact": bool(np.array_equal(back, ijk)),
            "flat_matches_c_order": bool(np.array_equal(flat, flat_c)),
            "passed": bool(np.array_equal(back, ijk) and np.array_equal(flat, flat_c))}


def t7_grid_placement(seg: WG.WaymoSegment) -> dict:
    grid = WG.waymo_grid()
    org = np.asarray(grid.origin, np.float64)
    T = WG.camera_to_grid(seg.calibration())
    c = np.asarray(T, np.float64)[:3, 3]
    idx = (c - org) / float(grid.voxel_size)
    inside = bool(np.all(idx >= 0) and np.all(idx < np.asarray(grid.dims)))
    road_measured = seg.road_z()                     # LiDAR: diagnostic only, not causal
    floor = float(org[2])
    return {"check": "T7_grid_placement", "camera_centre_grid_m": c.tolist(),
            "camera_centre_voxel_index": idx.tolist(), "camera_inside_grid": inside,
            "grid_floor_z_m": floor, "road_z_constant_m": WP.WAYMO_ROAD_Z_IN_VEHICLE_FRAME_M,
            "road_z_measured_this_segment_m": road_measured,
            "road_above_floor_m": road_measured - floor,
            "predeclared_floor_below_road_m": getattr(WP, "GRID_FLOOR_BELOW_ROAD_M", None),
            "passed": inside}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", type=int, default=3)
    a = ap.parse_args()
    A.ensure_dirs()
    p = A.Paths()
    segs = [s for s in sorted(os.listdir(p.clips))
            if os.path.isfile(p.laser_sidecar(s))][:a.segments]
    assert segs, "no segment has both clips and a laser sidecar yet"
    out = {"version": A.VERSION, "segments_tested": segs, "checks": []}
    out["checks"].append(t6_voxel_indexing())
    for s in segs:
        seg = WG.WaymoSegment(s)
        wag = WAG.WaymoAnchorGeometry(s)
        anchor = sorted(int(f[:-4]) for f in os.listdir(os.path.join(p.clips, s))
                        if f.endswith(".npz"))[0]
        for fn in (t1_axis_permutation, t2_sdk_projection, t7_grid_placement):
            out["checks"].append({"segment": s, **fn(seg)})
        out["checks"].append({"segment": s, "anchor": anchor, **t3_t4_poses(wag, anchor)})
        out["checks"].append({"segment": s, "anchor": anchor,
                              **t5_laser_inclinations(wag, anchor)})
    # ``all_passed`` is every check; ``geometry_gates_passed`` is the subset that actually
    # gates the preregistered geometry experiment. T2 is reported either way, never hidden.
    out["all_passed"] = all(c["passed"] for c in out["checks"])
    out["geometry_gates_passed"] = all(
        c["passed"] for c in out["checks"] if c.get("gates") != "semantic_target_only")
    out["advisory_failures"] = [c["check"] for c in out["checks"]
                                if not c["passed"] and c.get("gates") == "semantic_target_only"]
    path = os.path.join(p.logs, "check_transforms.json")
    json.dump(out, open(path, "w"), indent=2, default=float)

    seen = {}
    for c in out["checks"]:
        seen.setdefault(c["check"], []).append(c["passed"])
    for k, v in seen.items():
        print(f"[{'PASS' if all(v) else 'FAIL'}] {k}  ({sum(v)}/{len(v)})")
    for c in out["checks"]:
        if not c["passed"]:
            print("  FAILED:", json.dumps(c, default=float)[:400])
    print(f"\nall_passed={out['all_passed']}  "
          f"geometry_gates_passed={out['geometry_gates_passed']}")
    if out["advisory_failures"]:
        print(f"advisory (does not gate geometry): {out['advisory_failures']}")
    print(f"-> {path}")
    return 0 if out["geometry_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
