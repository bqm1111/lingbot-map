#!/usr/bin/env python
"""Geometry-only physical correspondences for one drive (Steps 1-4).

Landmarks are raw Velodyne returns of the sweep captured at source stream frame A,
transformed to the KITTI-360 world with the ground-truth pose chain
(``velo -> unrectified cam0 -> world``) and projected into rectified ``image_00`` of frame
A and of every partner stream frame B with the calibrated ``K`` and ``cam0_to_world``.
A projection is accepted only if the point is 3-50 m in front of the camera, inside the
image with a 12-px border, and passes a z-buffer visibility test against that frame's OWN
sweep: the minimum sweep depth inside a 7x7-pixel window around the projection must lie
within ``a + b*z`` of the projected depth (strict a=0.3 m, b=1 %; loose 0.6 m, 2 %). No
reference return in the window -> rejected as unverifiable. The semantic mapper, LingBot
depth and poses are not read.

    python tools/direct_correspondence_audit/build_pairs.py --drive 2013_05_28_drive_0006_sync
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.direct_correspondence_audit import common as DC                                # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402


def project(K, p_cam):
    z = p_cam[:, 2]
    u = K[0, 0] * p_cam[:, 0] / np.where(z > 0, z, 1) + K[0, 2]
    v = K[1, 1] * p_cam[:, 1] / np.where(z > 0, z, 1) + K[1, 2]
    return u, v, z


def min_depth_image(u, v, z, hw, window):
    """Per-pixel minimum sweep depth, then a min filter over the window (inf = no return)."""
    H, W = hw
    ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    img = np.full((H, W), np.inf)
    ui, vi = u[ok].astype(np.int64), v[ok].astype(np.int64)
    np.minimum.at(img, (vi, ui), z[ok])
    return ndimage.minimum_filter(img, size=window, mode="nearest")


def native_of(frame):
    return int(frame.key.split("_")[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    DC.require_data_root()
    src = SRC.source_of(a.drive); seg = [s for s in S.segments(src, REPO_ROOT) if s.name == a.drive][0]
    geo = TF.DriveGeometry(a.drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    K = np.asarray(geo.calib.K, np.float64); H, W = DC.NATIVE_HW
    assert tuple(geo.calib.native_hw) == (H, W)
    velo_to_cam = np.asarray(geo.calib.velo_to_rect_cam, np.float64)
    natives = [native_of(f) for f in seg.frames]
    stride = DC.A_STRIDE.get(a.drive, DC.A_STRIDE["default"])
    a_idx = list(range(0, len(natives), stride))
    if a.limit:
        a_idx = a_idx[:a.limit]
    out_dir = os.path.join(DC.PAIR_DIR, a.drive); os.makedirs(out_dir, exist_ok=True)
    tot = {"n_A": 0, "n_landmarks": 0, "n_candidates": 0, "accepted_loose": 0, "accepted_strict": 0,
           "rej_behind_or_range": 0, "rej_outside": 0, "rej_no_reference": 0, "rej_occluded": 0, "rej_in_front": 0}
    resid_hist = np.zeros(60, np.int64); resid_edges = np.linspace(-3, 3, 61)
    zcache = {}
    def zbuf(native):
        if native not in zcache:
            pts = geo.read_velodyne(native); pc = pts @ velo_to_cam[:3, :3].T + velo_to_cam[:3, 3]
            u, v, z = project(K, pc); zcache[native] = min_depth_image(u, v, z, (H, W), DC.ZBUF_WINDOW)
            if len(zcache) > 40:
                zcache.pop(next(iter(zcache)))
        return zcache[native]
    t0 = time.time()
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for ai in a_idx:
            fa = natives[ai]; op = DC.pair_path(a.drive, fa)
            if os.path.exists(op):
                continue
            pts = geo.read_velodyne(fa)
            Tvw = geo.velo_to_world(fa); Xw_all = pts @ Tvw[:3, :3].T + Tvw[:3, 3]
            pcA = pts @ velo_to_cam[:3, :3].T + velo_to_cam[:3, 3]
            uA, vA, zA = project(K, pcA)
            ok = (zA > DC.DEPTH_MIN) & (zA < DC.DEPTH_MAX) & (uA >= DC.BORDER_PX) & (uA < W - DC.BORDER_PX) & (vA >= DC.BORDER_PX) & (vA < H - DC.BORDER_PX)
            zb = zbuf(fa); ref = zb[np.clip(vA.astype(int), 0, H - 1), np.clip(uA.astype(int), 0, W - 1)]
            tol = DC.TOLS["strict"][0] + DC.TOLS["strict"][1] * zA
            ok &= np.isfinite(ref) & (np.abs(zA - ref) < tol)            # camera-LiDAR parallax occlusion in A itself
            idx = np.flatnonzero(ok)
            rng = np.random.default_rng(fa); sel = np.sort(rng.choice(idx, min(DC.N_LANDMARKS, len(idx)), replace=False))
            Xw = Xw_all[sel]; tot["n_A"] += 1; tot["n_landmarks"] += len(sel)
            cam_A = geo.rect_cam_to_world(fa)[:3, 3]
            rows = []
            for off in DC.B_OFFSETS:
                bi = ai + off
                if bi < 0 or bi >= len(natives):
                    continue
                fb = natives[bi]
                if fb not in geo.cam0_to_world:
                    continue
                Twc = TF.inv(geo.rect_cam_to_world(fb)); pcB = Xw @ Twc[:3, :3].T + Twc[:3, 3]
                uB, vB, zB = project(K, pcB); n = len(sel); tot["n_candidates"] += n
                r1 = ~((zB > DC.DEPTH_MIN) & (zB < DC.DEPTH_MAX)); tot["rej_behind_or_range"] += int(r1.sum())
                r2 = ~r1 & ~((uB >= DC.BORDER_PX) & (uB < W - DC.BORDER_PX) & (vB >= DC.BORDER_PX) & (vB < H - DC.BORDER_PX)); tot["rej_outside"] += int(r2.sum())
                cand = ~r1 & ~r2
                zbB = zbuf(fb); refB = np.full(n, np.inf); refB[cand] = zbB[vB[cand].astype(int), uB[cand].astype(int)]
                r3 = cand & ~np.isfinite(refB); tot["rej_no_reference"] += int(r3.sum())
                cand &= np.isfinite(refB); resid = zB - refB
                tl = DC.TOLS["loose"][0] + DC.TOLS["loose"][1] * zB; ts = DC.TOLS["strict"][0] + DC.TOLS["strict"][1] * zB
                r4 = cand & (resid >= tl); tot["rej_occluded"] += int(r4.sum())
                r5 = cand & (resid <= -tl); tot["rej_in_front"] += int(r5.sum())
                acc = cand & (np.abs(resid) < tl); strict = acc & (np.abs(resid) < ts)
                tot["accepted_loose"] += int(acc.sum()); tot["accepted_strict"] += int(strict.sum())
                resid_hist += np.histogram(resid[cand], resid_edges)[0]
                if not acc.any():
                    continue
                cam_B = geo.rect_cam_to_world(fb)[:3, 3]
                dA = Xw[acc] - cam_A; dB = Xw[acc] - cam_B
                ang = np.degrees(np.arccos(np.clip((dA * dB).sum(1) / (np.linalg.norm(dA, axis=1) * np.linalg.norm(dB, axis=1)), -1, 1)))
                rows.append(np.stack([np.flatnonzero(acc), np.full(acc.sum(), fb), uB[acc], vB[acc], zB[acc], resid[acc], strict[acc].astype(float),
                                      np.full(acc.sum(), float(np.linalg.norm(cam_A - cam_B))), ang, np.full(acc.sum(), off)], 1))
            rows = np.concatenate(rows) if rows else np.zeros((0, 10))
            np.savez_compressed(op, native_A=np.int64(fa), X_world=Xw.astype(np.float64), u_A=uA[sel].astype(np.float32), v_A=vA[sel].astype(np.float32),
                                z_A=zA[sel].astype(np.float32), resid_A=(zA[sel] - ref[sel]).astype(np.float32), cam_A=cam_A,
                                rows=rows.astype(np.float64), row_fields=np.asarray(["landmark", "native_B", "u_B", "v_B", "z_B", "resid_B", "strict", "baseline_m", "angle_deg", "offset"]))
            if tot["n_A"] % 25 == 0:
                print(f"  [{a.drive[17:21]}] {tot['n_A']} source frames, {tot['accepted_strict']:,} strict pairs, {(time.time()-t0)/tot['n_A']:.1f}s each", flush=True)
    DC.write_json(os.path.join(DC.ART, "pairs", f"{a.drive}.json"),
                  {"drive": a.drive, "counts": tot, "resid_hist": {"edges_m": resid_edges.tolist(), "counts": resid_hist.tolist()},
                   "rule": {"landmarks": "raw Velodyne returns of the sweep at A, velo->unrectified cam0->world (GT poses), rect_cam_to_velo = cam0_to_velo @ inv(R_rect_00)",
                            "projection": "K = P_rect_00[:3,:3] on the native 1408x376 image; cam0_to_world.txt poses; depth in (3, 50) m; 12-px border",
                            "visibility": f"min sweep depth of the same frame in a {DC.ZBUF_WINDOW}x{DC.ZBUF_WINDOW} px window; |z - ref| < a + b z, strict {DC.TOLS['strict']}, loose {DC.TOLS['loose']}; no return -> rejected",
                            "A_stride_stream_frames": stride, "B_offsets_stream_frames": DC.B_OFFSETS, "landmarks_per_A": DC.N_LANDMARKS, "rng": "default_rng(native frame index)"},
                   "calibration": geo.calib.to_dict(), "seconds": time.time() - t0, "firewall": audit.summary(), "env": DC.environment()})
    print(f"[{a.drive}] {tot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
