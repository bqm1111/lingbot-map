#!/usr/bin/env python
"""Steps 1, 2, 4, 5, 10, 12: relative poses, physical-point reprojection, depth S vs L.

Per clip (anchor t, frames i = t-4 .. t-1) and condition:
  S: T_i->t = inv(P_t) P_i from the stream cache, translation x frozen stream scale
  L: same from the clip-local LingBot run, translation x clip scale
  G: KITTI-360 GT, T_i->t = inv(rect_cam_to_world(t)) rect_cam_to_world(i)
Landmarks: raw Velodyne returns of the sweep at frame i, visible in camera i and camera t
(the direct-correspondence audit's z-buffer rule, 7x7 px, |dz| < 0.3 m + 1 %), max 1 000 per
(clip, i), identical for every condition. Pose-only reprojection: X_i (GT camera-i
coordinates) -> T_i->t -> K -> pixel, vs the GT projection. Endpoint error: |T X_i - T_G X_i|.
Depth+pose endpoint: LingBot depth at X's pixel in frame i (x scale) back-projected, mapped
by T, vs X in camera t (S depth for S and G, L depth for L; plus Lp = L pose with S depth).
Depth: LingBot depth x scale vs LiDAR depth at the projected pixels of frame t.

    python tools/local_registration_audit/pose_audit.py --drive 2013_05_28_drive_0006_sync
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.local_registration_audit import common as LR                                   # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402

H, W = 376, 1408; LH, LW = 140, 518; TOL = (0.3, 0.01); MAXPTS = 1000


def project(K, p):
    z = p[:, 2]; zz = np.where(z > 0, z, 1.0)
    return K[0, 0] * p[:, 0] / zz + K[0, 2], K[1, 1] * p[:, 1] / zz + K[1, 2], z


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--drive", required=True); ap.add_argument("--limit", type=int, default=None); a = ap.parse_args()
    d = a.drive; src = SRC.source_of(d); seg = [s for s in S.segments(src, REPO_ROOT) if s.name == d][0]
    nat = [int(f.order) for f in seg.frames]; geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    K = np.asarray(geo.calib.K, np.float64); v2c = np.asarray(geo.calib.velo_to_rect_cam, np.float64)
    with np.load(S.stream_path(src, d)) as z:
        s_dep = z["pred_depth"]; s_pose = z["pred_pose_c2w"].astype(np.float64); s_keys = [str(k) for k in z["keys"]]
    assert s_keys == [f.key for f in seg.frames]
    with np.load(S.scale_path(src, d)) as z:
        s_scale = float(np.exp(np.median(z["log_s"][:5])))
    zc = {}
    def zbuf(n):
        if n not in zc:
            pts = geo.read_velodyne(n); pc = pts @ v2c[:3, :3].T + v2c[:3, 3]; u, v, z_ = project(K, pc)
            ok = (z_ > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H); img = np.full((H, W), np.inf)
            np.minimum.at(img, (v[ok].astype(int), u[ok].astype(int)), z_[ok]); zc[n] = (ndimage.minimum_filter(img, 7, mode="nearest"), pc)
            if len(zc) > 12: zc.pop(next(iter(zc)))
        return zc[n]
    def lat(dep, u, v):
        ul = np.clip(((u + 0.5) * LW / W - 0.5).round().astype(int), 0, LW - 1); vl = np.clip(((v + 0.5) * LH / H - 0.5).round().astype(int), 0, LH - 1)
        return dep[vl, ul].astype(np.float64), ul, vl
    files = sorted(glob.glob(os.path.dirname(LR.clip_path(d, 0)) + "/*.npz"))[:a.limit]
    rows = []; drows = []; rng = np.random.default_rng(0); t0 = time.time(); rt_err = []
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for f in files:
            with np.load(f) as z:
                Lz = {k: z[k] for k in z.files}
            t = int(Lz["t"]); idx = Lz["stream_indices"].tolist(); assert idx == [t + o for o in LR.OFFSETS]
            l_scale = float(Lz["scale"]); l_pose = Lz["pred_pose_c2w"].astype(np.float64); l_dep = Lz["pred_depth"].astype(np.float32); Kl = Lz["pred_K"]
            nt = nat[t]; Gt = geo.rect_cam_to_world(nt)
            zb_t, pc_t = zbuf(nt)
            # depth vs LiDAR and S vs L at frame t
            ut, vt, ztt = project(K, pc_t); ok = (ztt > 3) & (ztt < 50) & (ut >= 0) & (ut < W) & (vt >= 0) & (vt < H)
            ds, _, _ = lat(s_dep[t], ut[ok], vt[ok]); dl, _, _ = lat(l_dep[4], ut[ok], vt[ok]); zl = ztt[ok]
            ds_m = ds * s_scale; dl_m = dl * l_scale
            drows.append((t, np.median(np.abs(ds_m - zl) / zl), np.median(np.abs(dl_m - zl) / zl), np.median(np.abs(ds - dl) / np.maximum(ds, 1e-6)),
                          np.corrcoef(ds, dl)[0, 1], np.median(dl / np.maximum(ds, 1e-6)), l_scale / s_scale, np.median(ds_m / zl), np.median(dl_m / zl)))
            for q, i in enumerate(idx[:4]):
                off = LR.OFFSETS[q]; ni = nat[i]; Gi = geo.rect_cam_to_world(ni)
                TG = TF.inv(Gt) @ Gi
                TS = np.linalg.inv(s_pose[t]) @ s_pose[i]; TS[:3, 3] *= s_scale
                TL = np.linalg.inv(l_pose[4]) @ l_pose[q]; TL[:3, 3] *= l_scale
                # round trip: GT world point through camera i and back
                X = pc_t[:5]; Xw = X @ Gt[:3, :3].T + Gt[:3, 3]; Xi = Xw @ TF.inv(Gi)[:3, :3].T + TF.inv(Gi)[:3, 3]
                rt_err.append(float(np.abs((Xi @ TG[:3, :3].T + TG[:3, 3]) - X).max()))
                # landmarks: sweep i, visible in i and t
                zb_i, pc_i = zbuf(ni); ui, vi, zi = project(K, pc_i)
                okv = (zi > 3) & (zi < 50) & (ui >= 2) & (ui < W - 2) & (vi >= 2) & (vi < H - 2)
                okv[okv] &= np.abs(zi[okv] - zb_i[vi[okv].astype(int), ui[okv].astype(int)]) < TOL[0] + TOL[1] * zi[okv]
                Xg = pc_i @ TG[:3, :3].T + TG[:3, 3]; ug, vg, zg = project(K, Xg)
                okt = okv & (zg > 3) & (zg < 50) & (ug >= 2) & (ug < W - 2) & (vg >= 2) & (vg < H - 2)
                okt[okt] &= np.abs(zg[okt] - zb_t[vg[okt].astype(int), ug[okt].astype(int)]) < TOL[0] + TOL[1] * zg[okt]
                sel = np.flatnonzero(okt)
                if len(sel) > MAXPTS: sel = np.sort(rng.choice(sel, MAXPTS, replace=False))
                Xi_, Xt_ = pc_i[sel], Xg[sel]; ugt, vgt = ug[sel], vg[sel]
                rec = {"t": t, "offset": off, "n_landmarks": len(sel), "gt_trans_m": float(np.linalg.norm(TG[:3, 3])), "gt_rot_deg": rot_deg(TG[:3, :3])}
                for c, T in (("S", TS), ("L", TL), ("G", TG)):
                    dt = T[:3, 3] - TG[:3, 3]; nt_ = np.linalg.norm(TG[:3, 3]); ne = np.linalg.norm(T[:3, 3])
                    rec[f"{c}_trans_err_m"] = float(np.linalg.norm(dt)); rec[f"{c}_trans_rel"] = float(np.linalg.norm(dt) / max(nt_, 1e-6))
                    rec[f"{c}_trans_dir_deg"] = float(np.degrees(np.arccos(np.clip(T[:3, 3] @ TG[:3, 3] / max(ne * nt_, 1e-9), -1, 1))))
                    rec[f"{c}_rot_err_deg"] = rot_deg(TG[:3, :3].T @ T[:3, :3]); rec[f"{c}_trans_scale_ratio"] = float(ne / max(nt_, 1e-6))
                    if len(sel):
                        Xp = Xi_ @ T[:3, :3].T + T[:3, 3]; up, vp, zp = project(K, Xp); e = np.sqrt((up - ugt) ** 2 + (vp - vgt) ** 2); e[zp <= 0] = np.inf
                        e3 = np.linalg.norm(Xp - Xt_, axis=1)
                        rec[f"{c}_reproj_px"] = e.astype(np.float32); rec[f"{c}_endpoint_pose_m"] = e3.astype(np.float32)
                if len(sel):
                    uI, vI, _ = project(K, Xi_)
                    for c, T, dep, sc in (("S", TS, s_dep[i], s_scale), ("L", TL, l_dep[q], l_scale), ("Lp", TL, s_dep[i], s_scale), ("G", TG, s_dep[i], s_scale), ("GLd", TG, l_dep[q], l_scale)):
                        dd, _, _ = lat(dep, uI, vI); dm = dd * sc
                        ray = np.stack([(uI - K[0, 2]) / K[0, 0], (vI - K[1, 2]) / K[1, 1], np.ones_like(uI)], 1)
                        P = ray * dm[:, None]; Pt = P @ T[:3, :3].T + T[:3, 3]
                        rec[f"{c}_endpoint_depthpose_m"] = np.linalg.norm(Pt - Xt_, axis=1).astype(np.float32)
                rows.append(rec)
            if len(drows) % 100 == 0:
                print(f"  [{d[17:21]}] {len(drows)}/{len(files)} clips, {(time.time()-t0)/len(drows):.2f}s each", flush=True)
    out_dir = os.path.join(LR.DATA_ROOT, "pose_audit"); os.makedirs(out_dir, exist_ok=True)
    arrays = {}
    scal_keys = [k for k in rows[0] if not isinstance(rows[0][k], np.ndarray)]
    for k in scal_keys:
        arrays[f"clip_{k}"] = np.asarray([r.get(k, np.nan) for r in rows], np.float64)
    vec_keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, np.ndarray)})
    for k in vec_keys:
        arrays[f"pt_{k}"] = np.concatenate([r[k] if k in r else np.zeros(0, np.float32) for r in rows])
    arrays["pt_row"] = np.concatenate([np.full(r["n_landmarks"], i, np.int32) for i, r in enumerate(rows)])
    arrays["depth"] = np.asarray(drows, np.float64)
    np.savez_compressed(os.path.join(out_dir, f"{d}.npz"), **arrays,
                        depth_fields=np.asarray(["t", "absrel_S", "absrel_L", "absrel_S_vs_L", "corr_S_L_canonical", "median_ratio_L_over_S_canonical", "scale_L_over_S", "median_ratio_S_metric_over_lidar", "median_ratio_L_metric_over_lidar"]))
    LR_json = {"drive": d, "n_clips": len(files), "n_pairs": len(rows), "n_landmarks": int(arrays["pt_row"].size), "stream_scale": s_scale,
               "gt_round_trip_max_err_m": float(max(rt_err)), "firewall": audit.summary(), "seconds": time.time() - t0}
    from tools.kitti360_data_scaling import common as C0
    C0.write_json(os.path.join(LR.ART, "pose_audit", f"{d}.json"), LR_json)
    print(json.dumps(LR_json)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
