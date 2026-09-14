#!/usr/bin/env python
"""Gate 9 per-clip audit: oracle ladder, mapper matrix, pixel-level depth error.

    python tools/gate9_depth_occupancy/audit.py --drive 2013_05_28_drive_0006_sync --device cuda:0 [--shard i --shards n]

Per five-frame causal clip (t-4 .. t, the local-registration audit's clips):
* poses: KITTI-360 GT, T_j->t = inv(rect_cam_to_world(t)) rect_cam_to_world(j); anchor camera = identity;
* camera: the calibrated rectified K scaled to the 140x518 LingBot lattice (one camera model for
  every condition); grid: the anchor's SSC velodyne grid, query T = inv(rect_cam_to_velo);
* V: raw Velodyne returns of the five sweeps, GT poses, voxelised directly (all returns, and
  camera-visible returns only);
* LD: each sweep projected into its own camera on the lattice, nearest-depth z-buffer per pixel,
  depth = LiDAR z at supported pixels, through the mapper; PD: frozen local-reset LingBot depth
  (x its clip MoGe scale) at exactly LD's pixels; GF: dense LingBot depth with its native
  confidence gate; Gs: GF with a per-clip LiDAR median-ratio scale (diagnostic oracle);
* reference: the same five sweeps through ``rawtarget.build`` measurement (occupied = return and
  never traversed; free = traversed and never a return; conflicts excluded); evaluation mask =
  reference-measured voxels whose centre lies inside at least one of the five camera frusta
  (1-60 m, in image). Identical mask for every condition.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import common as G9                                                                        # noqa: E402
from variant_mapper import VariantMapper                                                  # noqa: E402
from tools.local_registration_audit import common as LR                                   # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402
from core.mapping.mapper import FrameInput                                                # noqa: E402
from core.mapping import depth as D7                                                      # noqa: E402
from core.evaluation.voxels import dilate                                                 # noqa: E402
from core.supervision import rawtarget as RT                                              # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402

H, W = G9.NATIVE_HW; LH, LW = G9.LATTICE_HW; DIMS = G9.DIMS; ORIGIN = np.asarray(G9.ORIGIN); VS = G9.VS; NV = int(np.prod(DIMS))


def lattice_K(K):
    Kl = np.eye(3); sx, sy = LW / W, LH / H
    Kl[0, 0] = K[0, 0] * sx; Kl[1, 1] = K[1, 1] * sy; Kl[0, 2] = (K[0, 2] + 0.5) * sx - 0.5; Kl[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return Kl


def zbuffer(pc, Kl):
    """nearest-depth z-buffer on the lattice: depth [LH,LW] (0 = none), index of the winning point [LH,LW] (-1)."""
    z = pc[:, 2]; ok = z > 0.1
    u = np.floor(Kl[0, 0] * pc[ok, 0] / z[ok] + Kl[0, 2] + 0.5).astype(np.int64); v = np.floor(Kl[1, 1] * pc[ok, 1] / z[ok] + Kl[1, 2] + 0.5).astype(np.int64)
    ins = (u >= 0) & (u < LW) & (v >= 0) & (v < LH); ids = np.flatnonzero(ok)[ins]; pix = v[ins] * LW + u[ins]; zz = z[ids]
    order = np.lexsort((zz, pix)); pix_s, first = np.unique(pix[order], return_index=True)
    win = ids[order][first]
    depth = np.zeros(LH * LW, np.float64); idx = np.full(LH * LW, -1, np.int64); depth[pix_s] = z[win]; idx[pix_s] = win
    return depth.reshape(LH, LW), idx.reshape(LH, LW)


def to_grid_index(p_anchor_cam, T_c2g):
    g = p_anchor_cam @ T_c2g[:3, :3].T + T_c2g[:3, 3]
    return np.floor((g - ORIGIN) / VS).astype(np.int64)


def flat_of(ijk):
    ok = (ijk >= 0).all(1) & (ijk < np.asarray(DIMS)).all(1)
    fl = np.full(len(ijk), -1, np.int64); fl[ok] = np.ravel_multi_index(ijk[ok].T, DIMS)
    return fl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); d = a.drive; src = SRC.source_of(d)
    seg = [s for s in S.segments(src, REPO_ROOT) if s.name == d][0]; nat = [int(f.order) for f in seg.frames]
    geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    K = np.asarray(geo.calib.K, np.float64); Kl = lattice_K(K); v2c = np.asarray(geo.calib.velo_to_rect_cam, np.float64); T_c2g = np.asarray(geo.rect_cam_to_velo, np.float64)
    Tgw = np.linalg.inv(T_c2g)
    files = sorted(glob.glob(os.path.dirname(LR.clip_path(d, 0)) + "/*.npz"))[a.shard::a.shards][:a.limit]
    out_dir = a.out or os.path.join(G9.DATA_ROOT, "clips", d); os.makedirs(out_dir, exist_ok=True)
    # voxel centres in the anchor camera (fixed per drive)
    gi = np.stack(np.unravel_index(np.arange(NV), DIMS), 1).astype(np.float64)
    ctr_g = (gi + 0.5) * VS + ORIGIN; ctr_c = torch.from_numpy(ctr_g @ Tgw[:3, :3].T + Tgw[:3, 3]).to(dev)
    t0 = time.time(); nw = 0
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for f in files:
            with np.load(f) as z:
                Lz = {k: z[k] for k in z.files}
            t = int(Lz["t"]); op = os.path.join(out_dir, f"{t:05d}.npz")
            if os.path.exists(op):
                continue
            idx = Lz["stream_indices"].tolist(); l_scale = float(Lz["scale"]); Gt = geo.rect_cam_to_world(nat[t])
            TG = [TF.inv(Gt) @ geo.rect_cam_to_world(nat[j]) for j in idx]
            rec = {"t": np.int64(t), "stream_indices": np.asarray(idx), "natives": np.asarray([nat[j] for j in idx]), "l_scale": l_scale,
                   "T_j_to_t": np.stack(TG), "K_lattice": Kl}
            # ---- reference and evaluation mask
            ref = RT.build(geo, nat[t], [nat[j] for j in idx], dev)
            n_obs = np.asarray(ref.n_obs) > 0; n_free = np.asarray(ref.n_free) > 0
            r_occ = n_obs & ~n_free; r_free = n_free & ~n_obs
            fr_ = torch.zeros(NV, dtype=torch.bool, device=dev)
            for T in TG:
                Ti = torch.as_tensor(np.linalg.inv(T), device=dev); pc = ctr_c @ Ti[:3, :3].T + Ti[:3, 3]; zc = pc[:, 2]
                uu = Kl[0, 0] * pc[:, 0] / zc.clamp_min(1e-6) + Kl[0, 2]; vv = Kl[1, 1] * pc[:, 1] / zc.clamp_min(1e-6) + Kl[1, 2]
                fr_ |= (zc > D7.MIN_DEPTH_M) & (zc < D7.MAX_DEPTH_M) & (uu >= -0.5) & (uu < LW - 0.5) & (vv >= -0.5) & (vv < LH - 0.5)
            frustum = fr_.cpu().numpy(); E = (r_occ | r_free) & frustum
            rec["eval_mask_digest"] = hashlib.sha256(np.packbits(E).tobytes()).hexdigest()[:16]
            rec["n_eval"] = int(E.sum()); rec["n_ref_occ"] = int((r_occ & E).sum()); rec["n_ref_free"] = int((r_free & E).sum())
            # ---- per-frame LiDAR and depth images
            sup, zl, zp, cf, ptidx, pcs = [], [], [], [], [], []
            V_all = np.zeros(NV, bool); V_cam = np.zeros(NV, bool)
            for q, j in enumerate(idx):
                pts = geo.read_velodyne(nat[j]); pc = pts @ v2c[:3, :3].T + v2c[:3, 3]; pcs.append(pc)
                pa = pc @ TG[q][:3, :3].T + TG[q][:3, 3]; fl = flat_of(to_grid_index(pa, T_c2g)); V_all[fl[fl >= 0]] = True
                zb, zi = zbuffer(pc, Kl)
                zz = pc[:, 2]; incam = (zz > 0.1)
                uu = Kl[0, 0] * pc[:, 0] / np.maximum(zz, 1e-6) + Kl[0, 2]; vv = Kl[1, 1] * pc[:, 1] / np.maximum(zz, 1e-6) + Kl[1, 2]
                incam &= (uu >= -0.5) & (uu < LW - 0.5) & (vv >= -0.5) & (vv < LH - 0.5) & (zz > D7.MIN_DEPTH_M) & (zz < D7.MAX_DEPTH_M)
                V_cam[fl[(fl >= 0) & incam]] = True
                m = (zb > D7.MIN_DEPTH_M) & (zb < D7.MAX_DEPTH_M)
                sup.append(m); zl.append(zb); ptidx.append(zi)
                zp.append(Lz["pred_depth"][q].astype(np.float64) * l_scale); cf.append(Lz["pred_depth_conf"][q].astype(np.float64))
            # oracle scale for Gs: median LiDAR / LingBot at the anchor's supported pixels
            oracle = float(np.median(zl[4][sup[4]] / np.maximum(zp[4][sup[4]], 1e-6))); rec["oracle_ratio"] = oracle
            rec["support_digest"] = hashlib.sha256(np.stack(sup).tobytes()).hexdigest()[:16]
            # ---- mapper runs
            def run(name):
                depth_src, support, band, carve, margin, gate = G9.RUNS[name]
                mp = VariantMapper(dev, band, margin, carve); mp.scale_state.force(1.0)
                for q in range(5):
                    if depth_src == "lidar":
                        dm = zl[q]
                    elif depth_src == "lingbot":
                        dm = zp[q]
                    else:
                        dm = zp[q] * oracle
                    if support == "dense":
                        conf = cf[q]
                    elif support == "lidar":
                        conf = np.where(sup[q], G9.SUPPORT_CONF, 0.0)
                    else:
                        conf = np.where(sup[q] & (cf[q] >= D7.CONF_THRESHOLD), G9.SUPPORT_CONF, 0.0)
                    fi = FrameInput(index=q, depth_canonical=torch.from_numpy(np.where(np.isfinite(dm), dm, 0.0).astype(np.float32)).to(dev),
                                    conf=torch.from_numpy(conf.astype(np.float32)).to(dev), K=Kl, pose_c2w_canonical=TG[q], log_scale_candidate=0.0)
                    mp.step(fi)
                qq = mp.query(type("G", (), {"dims": DIMS, "voxel_size": VS, "origin": tuple(ORIGIN)})(), Tgw)
                return qq["occupied"].cpu().numpy(), qq["free"].cpu().numpy()
            maps = {}
            for name in G9.RUNS:
                occ, fre = run(name); maps[name] = (occ, fre)
                tp = int((occ & r_occ & E).sum()); fp = int((occ & ~r_occ & E).sum()); fn = int((~occ & r_occ & E).sum())
                rec[f"m_{name}"] = np.asarray([tp, fp, fn, int((occ & E).sum()), int((occ & r_free & E).sum()), int((fre & r_occ & E).sum()), int(occ.sum()), int((occ & ~E).sum())], np.int64)
            for name, occ in (("V_all", V_all), ("V_cam", V_cam)):
                tp = int((occ & r_occ & E).sum()); fp = int((occ & ~r_occ & E).sum()); fn = int((~occ & r_occ & E).sum())
                rec[f"m_{name}"] = np.asarray([tp, fp, fn, int((occ & E).sum()), int((occ & r_free & E).sum()), 0, int(occ.sum()), int((occ & ~E).sum())], np.int64)
            rec["metric_fields"] = np.asarray(["tp", "fp", "fn", "pred_occ_in_mask", "pred_occ_on_ref_free", "ref_occ_marked_free", "pred_occ_total", "pred_occ_outside_mask"])
            for name in ("LD", "PD", "GF", "LD_nocarve", "PD_nocarve", "GF_nocarve"):
                rec[f"occ_{name}"] = np.flatnonzero(maps[name][0]).astype(np.int32)
            rec["occ_V_cam"] = np.flatnonzero(V_cam).astype(np.int32); rec["ref_occ"] = np.flatnonzero(r_occ & E).astype(np.int32); rec["ref_free_digest_n"] = int(r_free.sum())
            # ---- pixel records (LD vs PD matched pixels)
            r_occ_dil = dilate(torch.from_numpy(r_occ.reshape(DIMS)).to(dev), 1).reshape(-1).cpu().numpy()
            pd_free = maps["PD"][1]; gf_free = maps["GF"][1]
            ep_frames = {k: np.zeros(NV, np.uint8) for k in ("LD", "PD")}
            prow = []
            rng = np.random.default_rng(t)
            vv_, uu_ = np.meshgrid(np.arange(LH), np.arange(LW), indexing="ij")
            rays = np.stack([(uu_ - Kl[0, 2]) / Kl[0, 0], (vv_ - Kl[1, 2]) / Kl[1, 1], np.ones_like(uu_, dtype=np.float64)], -1)
            for q in range(5):
                m = sup[q]; yy, xx = np.nonzero(m); zlq = zl[q][yy, xx]; zpq = zp[q][yy, xx]; cfq = cf[q][yy, xx]; ray = rays[yy, xx]; rl = np.linalg.norm(ray, axis=1)
                # edge: max relative jump to supported neighbours (5x5)
                zt = torch.from_numpy(np.where(m, zl[q], np.nan)).to(dev)[None, None]
                pad = G9.EDGE_WIN // 2
                un = F.unfold(torch.nan_to_num(zt, nan=0.0), G9.EDGE_WIN, padding=pad)[0]; msk = F.unfold((~torch.isnan(zt)).float(), G9.EDGE_WIN, padding=pad)[0]
                cz = torch.nan_to_num(zt, nan=0.0).reshape(-1)
                jump = ((un - cz.unsqueeze(0)).abs() / cz.clamp_min(1e-6).unsqueeze(0) * msk).amax(0); nnb = msk.sum(0) - 1
                jump = jump.reshape(LH, LW).cpu().numpy()[yy, xx]; nnb = nnb.reshape(LH, LW).cpu().numpy()[yy, xx]
                edge = np.where(nnb < 1, 3, np.where(jump < G9.EDGE_EDGES[0], 0, np.where(jump < G9.EDGE_EDGES[1], 1, 2)))
                # planarity from the LiDAR camera points of supported pixels in a 5x5 window
                P = torch.from_numpy(np.where(m[..., None], rays * zl[q][..., None], 0.0)).to(dev).permute(2, 0, 1)[None]
                pu = F.unfold(P, G9.PLANAR_WIN, padding=G9.PLANAR_WIN // 2)[0].reshape(3, G9.PLANAR_WIN ** 2, -1)
                mu = F.unfold(torch.from_numpy(m.astype(np.float32)).to(dev)[None, None], G9.PLANAR_WIN, padding=G9.PLANAR_WIN // 2)[0]
                sel_lin = torch.from_numpy(yy * LW + xx).to(dev)
                pu = pu[:, :, sel_lin].permute(2, 1, 0).double(); mu = mu[:, sel_lin].T.double(); cnt = mu.sum(1)
                mean = (pu * mu.unsqueeze(-1)).sum(1) / cnt.clamp_min(1).unsqueeze(-1); dvv = (pu - mean.unsqueeze(1)) * mu.unsqueeze(-1)
                cov = dvv.transpose(1, 2) @ dvv / cnt.clamp_min(1).view(-1, 1, 1); ev = torch.linalg.eigvalsh(cov)
                plan = (ev[:, 0] / ev.sum(1).clamp_min(1e-12)).cpu().numpy(); cnt = cnt.cpu().numpy()
                surf = np.where(cnt < G9.PLANAR_MIN_PTS, 2, np.where(plan < G9.PLANAR_MAX, 0, 1))
                # endpoints in the anchor grid
                Pl = ray * zlq[:, None]; Pp = ray * zpq[:, None]; T = TG[q]
                gl = to_grid_index(Pl @ T[:3, :3].T + T[:3, 3], T_c2g); gp = to_grid_index(Pp @ T[:3, :3].T + T[:3, 3], T_c2g)
                fl_l = flat_of(gl); fl_p = flat_of(gp)
                for name, fl_ in (("LD", fl_l), ("PD", fl_p)):
                    u_ = np.unique(fl_[fl_ >= 0]); ep_frames[name][u_] += 1
                true_pt = pcs[q][ptidx[q][yy, xx]]; true_err = np.linalg.norm(Pl - true_pt, axis=1)                      # pixel-centre quantisation of the LiDAR endpoint
                inside_p = fl_p >= 0; inside_l = fl_l >= 0
                on_free = np.zeros(len(yy), bool); on_free[inside_p] = r_free[fl_p[inside_p]]
                near_occ = np.zeros(len(yy), bool); near_occ[inside_p] = r_occ_dil[fl_p[inside_p]]
                true_vox_free_pd = np.zeros(len(yy), bool); true_vox_free_pd[inside_l] = pd_free[fl_l[inside_l]]
                true_vox_free_gf = np.zeros(len(yy), bool); true_vox_free_gf[inside_l] = gf_free[fl_l[inside_l]]
                carve_through = (zpq - 0.2 / rl) > zlq + 0.5 * VS / rl                                                 # the ray's carve would pass the true endpoint (any pixel)
                keep = rng.random(len(yy)) < G9.PIXEL_KEEP
                cols = np.stack([np.full(len(yy), q), yy, xx, zlq, zpq, cfq, rl, true_err, gp[:, 0] - gl[:, 0], gp[:, 1] - gl[:, 1], gp[:, 2] - gl[:, 2],
                                 (fl_p == fl_l) & inside_l, on_free, near_occ, carve_through, true_vox_free_pd, true_vox_free_gf, edge, surf, jump, inside_l & inside_p], 1)[keep]
                prow.append(cols.astype(np.float32))
            rec["pixels"] = np.concatenate(prow); rec["pixel_fields"] = np.asarray(["frame", "v", "u", "z_lidar", "z_pred", "conf", "ray_norm", "lidar_pixel_quant_err_m", "dvx", "dvy", "dvz", "same_voxel", "pred_ep_on_ref_free",
                                                                                   "pred_ep_near_ref_occ", "carve_through_true_endpoint", "true_voxel_free_in_PD_map", "true_voxel_free_in_GF_map", "edge_class", "surface_class", "edge_jump", "both_in_grid"])
            rec["n_support_pixels"] = np.asarray([int(s.sum()) for s in sup])
            ro = r_occ & E
            for name in ("LD", "PD"):
                rec[f"ep_support_{name}"] = np.asarray([int((ro & (ep_frames[name] >= 1)).sum()), int((ro & (ep_frames[name] >= 2)).sum()), int(ro.sum())])
            tmp = op + ".tmp.npz"; np.savez_compressed(tmp, **rec); os.replace(tmp, op); nw += 1
            if nw % 25 == 0:
                print(f"  [{d[17:21]} s{a.shard}] {nw}/{len(files)} clips, {(time.time()-t0)/nw:.1f}s each", flush=True)
    print(f"[{d} s{a.shard}] {nw} clips, {(time.time()-t0)/60:.1f} min, firewall violations {len(audit.violations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
