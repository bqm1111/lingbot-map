#!/usr/bin/env python
"""Gate 14 Stage 1 per-clip audit: 2 x 2 depth (LBM / NEW = MUSt3R-512) x pose (L / G) observed maps.

    python tools/gate14_depth_pose/audit14.py --drive 2013_05_28_drive_0006_sync --device cuda:0 [--shard i --shards n] [--limit k] [--out dir]

Per clip t (Gate 9 / Gate 13 five-frame clip, frames t-4..t):
  LBM depth   cached local-reset LingBot depth (canonical) x s_clip (causal MoGe-2 clip scale; applied once by the mapper)
  NEW depth   MUSt3R-512 z-depth (official code, cache from must3r_infer.py) resampled to the 140x518 LingBot lattice by
              nearest neighbour through the official crop/resize matrix (lattice pixel centre -> native -> rescaled);
              lattice pixels outside the MUSt3R crop are invalid; metric via s_new = exp(median over the five frames of
              gates.gate7b.scale.frame_candidate(MoGe-2 depth, MoGe mask, NEW depth, conf = 1e3)) -- the LB-M clip-scale
              estimator with no confidence gate (MUSt3R confidence is not on LingBot's scale); no LiDAR
  L pose      LingBot local poses inv(P_4) P_q (canonical), LingBot pred_K per frame, mapper scale forced to s_clip
              (translation x s_clip; depth entered as canonical-equivalent value)
  G pose      KITTI-360 GT relative poses (metric), calibrated lattice camera (Gate 9 lattice_K), mapper scale 1
  supports    shared = LingBot conf >= 1.5 AND NEW valid (inside crop, finite, > 0);  native: LBM conf >= 1.5, NEW valid
Mapper: frozen VariantMapper(0.2, 0.2, carve) with conf = SUPPORT_CONF on the support, 0 elsewhere. Reference, evaluation
mask E, voxel range (anchor-camera z) and pixel subsample are Gate 9's. LD-G: the Gate 13 oracle input (== Gate 9 LD).
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as TF_

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g14                                                                                   # noqa: E402
sys.path.insert(0, g14.G9_DIR); sys.path.insert(0, g14.REPO_ROOT)
sys.path.insert(0, os.path.join(g14.REPO_ROOT, "tools", "depth_source_comparison"))
import common as G9                                                                          # noqa: E402  (Gate 9)
from audit import lattice_K, zbuffer, to_grid_index, flat_of                                 # noqa: E402  (Gate 9)
from variant_mapper import VariantMapper                                                     # noqa: E402  (Gate 9)
import dsc as DSC                                                                            # noqa: E402  (MoGe-2 cache loader)
from gates.gate7b import scale as SC                                                         # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S   # noqa: E402
from core.datasets.audit import FileAudit                                                    # noqa: E402
from core.mapping.mapper import FrameInput                                                   # noqa: E402
from core.mapping import depth as D7                                                         # noqa: E402
from core.supervision import rawtarget as RT                                                 # noqa: E402
from core.run._common import REPO_ROOT                                                       # noqa: E402

LH, LW = G9.LATTICE_HW; NH, NW = G9.NATIVE_HW; DIMS = G9.DIMS; ORIGIN = np.asarray(G9.ORIGIN); VS = G9.VS; NV = int(np.prod(DIMS))
COND_RUNS = [(c, s) for s in g14.SUPPORTS for c in g14.CONDITIONS]
MFIELDS = ["tp", "fp", "fn", "pred_occ_in_mask", "pred_occ_on_ref_free", "ref_occ_marked_free", "pred_occ_total", "ref_occ"]


def edge_classes(zl_q, m, dev):
    """Gate 9 audit.py lines 171-177, unchanged."""
    yy, xx = np.nonzero(m)
    zt = torch.from_numpy(np.where(m, zl_q, np.nan)).to(dev)[None, None]
    pad = G9.EDGE_WIN // 2
    un = TF_.unfold(torch.nan_to_num(zt, nan=0.0), G9.EDGE_WIN, padding=pad)[0]; msk = TF_.unfold((~torch.isnan(zt)).float(), G9.EDGE_WIN, padding=pad)[0]
    cz = torch.nan_to_num(zt, nan=0.0).reshape(-1)
    jump = ((un - cz.unsqueeze(0)).abs() / cz.clamp_min(1e-6).unsqueeze(0) * msk).amax(0); nnb = msk.sum(0) - 1
    jump = jump.reshape(LH, LW).cpu().numpy()[yy, xx]; nnb = nnb.reshape(LH, LW).cpu().numpy()[yy, xx]
    return np.where(nnb < 1, 3, np.where(jump < G9.EDGE_EDGES[0], 0, np.where(jump < G9.EDGE_EDGES[1], 1, 2)))


def lattice_to_must3r(to_rescaled):
    """Nearest MUSt3R pixel for every lattice pixel centre (pixel-corner continuous coordinates), -1 if outside."""
    vv, uu = np.meshgrid(np.arange(LH), np.arange(LW), indexing="ij")
    xn = (uu + 0.5) * NW / LW; yn = (vv + 0.5) * NH / LH                                   # native continuous coords of lattice centres
    p = np.stack([xn, yn, np.ones_like(xn)], -1) @ np.asarray(to_rescaled).T
    xm, ym = np.floor(p[..., 0]).astype(np.int64), np.floor(p[..., 1]).astype(np.int64)
    ok = (xm >= 0) & (xm < 512) & (ym >= 0) & (ym < 160)
    return np.where(ok, ym, -1), np.where(ok, xm, -1), ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1); ap.add_argument("--out", default=None)
    a = ap.parse_args(); d = a.drive
    dev = torch.device(a.device); torch.cuda.set_device(dev); src = SRC.source_of(d)
    seg = [s for s in S.segments(src, REPO_ROOT) if s.name == d][0]; nat = [int(f.order) for f in seg.frames]
    geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    Kl = lattice_K(np.asarray(geo.calib.K, np.float64)); v2c = np.asarray(geo.calib.velo_to_rect_cam, np.float64); T_c2g = np.asarray(geo.rect_cam_to_velo, np.float64)
    Tgw = np.linalg.inv(T_c2g); grid = type("G", (), {"dims": DIMS, "voxel_size": VS, "origin": tuple(ORIGIN)})()
    files = sorted(glob.glob(os.path.join(g14.LR_ROOT, d, "*.npz")))[a.shard::a.shards][:a.limit]
    out_dir = a.out or os.path.join(g14.CLIP_ROOT, d); os.makedirs(out_dir, exist_ok=True)
    gi = np.stack(np.unravel_index(np.arange(NV), DIMS), 1).astype(np.float64)
    ctr_np = ((gi + 0.5) * VS + ORIGIN) @ Tgw[:3, :3].T + Tgw[:3, 3]; ctr_c = torch.from_numpy(ctr_np).to(dev)
    rbin = torch.from_numpy(np.clip(np.digitize(ctr_np[:, 2], np.asarray(g14.RANGE_BINS[1:-1])), 0, 4)).to(dev)
    vv_, uu_ = np.meshgrid(np.arange(LH), np.arange(LW), indexing="ij")
    rays_G = np.stack([(uu_ - Kl[0, 2]) / Kl[0, 0], (vv_ - Kl[1, 2]) / Kl[1, 1], np.ones_like(uu_, dtype=np.float64)], -1)
    reps = representative(d)
    load_moge = DSC.moge_loader(src); t0 = time.time(); nw = 0
    with FileAudit(patterns=tuple(SRC.FORBIDDEN_PATH_TOKENS) + ("semantickitti", "occ3d", "nuscenes"), strict=True) as audit:
        for f in files:
            with np.load(f) as z:
                Lz = {k: z[k] for k in z.files}
            t = int(Lz["t"]); op = os.path.join(out_dir, f"{t:05d}.npz")
            if os.path.exists(op):
                continue
            idx = [int(x) for x in Lz["stream_indices"]]; s_clip = float(Lz["scale"])
            with np.load(g14.new_cache_path(d, t)) as zn:
                N = {k: zn[k] for k in zn.files}
            assert [int(x) for x in N["stream_indices"]] == idx and [str(k) for k in N["keys"]] == [str(k) for k in Lz["keys"]]
            Gt = geo.rect_cam_to_world(nat[t]); TG = [TF.inv(Gt) @ geo.rect_cam_to_world(nat[j]) for j in idx]
            P = Lz["pred_pose_c2w"].astype(np.float64); TLc = [np.linalg.inv(P[4]) @ P[q] for q in range(5)]
            canon = Lz["pred_depth"].astype(np.float32); cf = Lz["pred_depth_conf"].astype(np.float64); pK = Lz["pred_K"].astype(np.float64)
            # ---- NEW depth on the lattice + causal MoGe-2 clip scale
            ym, xm, in_crop = lattice_to_must3r(N["to_rescaled"])
            z_new = np.full((5, LH, LW), np.nan); c_new = np.full((5, LH, LW), np.nan)
            z_new[:, in_crop] = N["depth_z"][:, ym[in_crop], xm[in_crop]]; c_new[:, in_crop] = N["conf"][:, ym[in_crop], xm[in_crop]]
            new_valid = np.isfinite(z_new) & (z_new > 0)
            logs = []
            for q in range(5):
                md, mm, _ = load_moge(str(Lz["keys"][q]))
                logs.append(SC.frame_candidate(md.astype(np.float32), mm, np.where(new_valid[q], z_new[q], 0.0).astype(np.float32), np.full((LH, LW), 1e3, np.float32))["log_s"])
            s_new = float(np.exp(np.nanmedian(logs)))
            lb_valid = (cf >= D7.CONF_THRESHOLD) & np.isfinite(canon) & (canon > 0)
            sup_ = {"shared": {"LBM": lb_valid & new_valid, "NEW": lb_valid & new_valid}, "native": {"LBM": lb_valid, "NEW": new_valid}}
            zmet = {"LBM": canon.astype(np.float64) * s_clip, "NEW": np.where(new_valid, z_new, 0.0) * s_new}
            rec = {"t": np.int64(t), "stream_indices": np.asarray(idx), "keys": Lz["keys"], "s_clip": s_clip, "s_new": s_new, "log_s_new_frames": np.asarray(logs),
                   "n_pixels": np.asarray([[int(sup_[s][c].sum()) for c in ("LBM", "NEW")] for s in g14.SUPPORTS]), "n_in_crop": int(in_crop.sum()),
                   "T_G": np.stack(TG), "T_L_canonical": np.stack(TLc), "K_lattice": Kl, "metric_fields": np.asarray(MFIELDS), "conditions": np.asarray([f"{c}|{s}" for c, s in COND_RUNS])}
            # ---- reference and evaluation mask (Gate 9)
            ref = RT.build(geo, nat[t], [nat[j] for j in idx], dev)
            n_obs = np.asarray(ref.n_obs) > 0; n_free = np.asarray(ref.n_free) > 0; r_occ = n_obs & ~n_free; r_free = n_free & ~n_obs
            fr_ = torch.zeros(NV, dtype=torch.bool, device=dev)
            for T in TG:
                Ti = torch.as_tensor(np.linalg.inv(T), device=dev); pc = ctr_c @ Ti[:3, :3].T + Ti[:3, 3]; zc = pc[:, 2]
                uu = Kl[0, 0] * pc[:, 0] / zc.clamp_min(1e-6) + Kl[0, 2]; vv = Kl[1, 1] * pc[:, 1] / zc.clamp_min(1e-6) + Kl[1, 2]
                fr_ |= (zc > D7.MIN_DEPTH_M) & (zc < D7.MAX_DEPTH_M) & (uu >= -0.5) & (uu < LW - 0.5) & (vv >= -0.5) & (vv < LH - 0.5)
            E = (r_occ | r_free) & fr_.cpu().numpy()
            rec["eval_mask_digest"] = hashlib.sha256(np.packbits(E).tobytes()).hexdigest()[:16]
            Rt, Frt, Et = (torch.from_numpy(x).to(dev) for x in (r_occ & E, r_free & E, E))

            def score(occ_t, fre_t):
                out = np.zeros((6, len(MFIELDS)), np.int64)
                sels = (occ_t & Rt, occ_t & ~Rt & Et, ~occ_t & Rt, occ_t & Et, occ_t & Frt, fre_t & Rt, occ_t, Rt)
                for i, x in enumerate(sels):
                    out[:5, i] = torch.bincount(rbin[x], minlength=5).cpu().numpy(); out[5, i] = int(x.sum())
                return out

            maps = {}
            for cond, supn in COND_RUNS:
                dsrc, pose = cond.split("-"); support = sup_[supn][dsrc]
                mp = VariantMapper(dev, 0.2, 0.2, True)
                if pose == "L":
                    mp.scale_state.force(s_clip)
                    dcan = canon if dsrc == "LBM" else (zmet["NEW"] / s_clip).astype(np.float32)
                    frames = [FrameInput(index=q, depth_canonical=torch.from_numpy(np.where(np.isfinite(dcan[q]), dcan[q], 0.0).astype(np.float32)).to(dev),
                                         conf=torch.from_numpy(np.where(support[q], G9.SUPPORT_CONF, 0.0).astype(np.float32)).to(dev), K=pK[q], pose_c2w_canonical=TLc[q],
                                         log_scale_candidate=float(np.log(s_clip))) for q in range(5)]
                else:
                    mp.scale_state.force(1.0)
                    frames = [FrameInput(index=q, depth_canonical=torch.from_numpy(zmet[dsrc][q].astype(np.float32)).to(dev),
                                         conf=torch.from_numpy(np.where(support[q], G9.SUPPORT_CONF, 0.0).astype(np.float32)).to(dev), K=Kl, pose_c2w_canonical=TG[q],
                                         log_scale_candidate=0.0) for q in range(5)]
                for fr in frames:
                    mp.step(fr)
                qq = mp.query(grid, Tgw); occ_t, fre_t = qq["occupied"], qq["free"]
                rec[f"m_{cond}|{supn}"] = score(occ_t, fre_t)
                if t in reps.values() and supn == "shared":
                    maps[cond] = np.flatnonzero(occ_t.cpu().numpy()).astype(np.int32)
                if cond == "LBM-L" and supn == "native":
                    rec["LBM-L_native_occ_sha"] = hashlib.sha256(np.packbits(occ_t.cpu().numpy()).tobytes()).hexdigest()
                    rec["LBM-L_native_free_sha"] = hashlib.sha256(np.packbits(fre_t.cpu().numpy()).tobytes()).hexdigest()
                del mp, qq
            with np.load(os.path.join(g14.O_ROOT, d, f"{t:05d}.npz")) as zo:
                SO = torch.from_numpy(np.unpackbits(zo["S_O"])[:NV].astype(bool)).to(dev); FO = torch.from_numpy(np.unpackbits(zo["F_O"])[:NV].astype(bool)).to(dev)
            rec["m_LD-G|lidar"] = score(SO, FO)
            if maps:
                for k, v in maps.items():
                    rec[f"occ_{k}"] = v
                rec["ref_occ"] = np.flatnonzero(r_occ & E).astype(np.int32); rec["occ_LD-G"] = np.flatnonzero(SO.cpu().numpy()).astype(np.int32)
            # ---- registration: L vs G relative pose (metric via s_clip)
            reg = []
            for q in range(4):
                tL = TLc[q][:3, 3] * s_clip; tG = TG[q][:3, 3]; Rr = TLc[q][:3, :3].T @ TG[q][:3, :3]
                reg.append([np.linalg.norm(tL - tG), np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1))), np.linalg.norm(tG), np.linalg.norm(tL)])
            rec["registration"] = np.asarray(reg)                                                   # frames t-4..t-1: trans err m, rot err deg, |t_G|, |t_L|
            # ---- pixel diagnostics on Gate 9's deterministic 20 % subsample of the LiDAR support
            rng = np.random.default_rng(t); prow = []
            for q, j in enumerate(idx):
                pc = geo.read_velodyne(nat[j]) @ v2c[:3, :3].T + v2c[:3, 3]; zb, _ = zbuffer(pc, Kl); sup = (zb > D7.MIN_DEPTH_M) & (zb < D7.MAX_DEPTH_M)
                yy, xx = np.nonzero(sup); zl = zb[yy, xx]; rayG = rays_G[yy, xx]; rl = np.linalg.norm(rayG, axis=1)
                edge = edge_classes(zb, sup, dev)
                T = TG[q]; fl_l = flat_of(to_grid_index((rayG * zl[:, None]) @ T[:3, :3].T + T[:3, 3], T_c2g)); inl = fl_l >= 0
                cols = [np.full(len(yy), q), yy, xx, zl, rl, edge, cf[q][yy, xx], lb_valid[q][yy, xx], new_valid[q][yy, xx], c_new[q][yy, xx], inl]
                for ds in ("LBM", "NEW"):
                    zx = zmet[ds][q][yy, xx]; ok = np.isfinite(zx) & (zx > 0); fl = np.full(len(yy), -1, np.int64)
                    fl[ok] = flat_of(to_grid_index((rayG[ok] * zx[ok, None]) @ T[:3, :3].T + T[:3, 3], T_c2g))
                    onf = np.zeros(len(yy), bool); onf[fl >= 0] = r_free[fl[fl >= 0]]
                    cols += [np.where(ok, zx, np.nan), (fl == fl_l) & inl, onf]
                # pose-induced displacement of the LiDAR endpoint: pred_K ray + L pose vs calibrated ray + G pose; reprojection into the anchor
                rayL = np.stack([(xx - pK[q][0, 2]) / pK[q][0, 0], (yy - pK[q][1, 2]) / pK[q][1, 1], np.ones(len(yy))], 1)
                TLm = TLc[q].copy(); TLm[:3, 3] *= s_clip
                pL = (rayL * zl[:, None]) @ TLm[:3, :3].T + TLm[:3, 3]; pG = (rayG * zl[:, None]) @ T[:3, :3].T + T[:3, 3]
                # reprojection into the anchor camera only where both endpoints are in front of it (z > 1 m); nan otherwise.
                # (Amended before the code freeze: the first 0006 run divided by a clamped depth for endpoints behind the camera.)
                prj = lambda p: np.stack([Kl[0, 0] * p[:, 0] / np.maximum(p[:, 2], 1e-6) + Kl[0, 2], Kl[1, 1] * p[:, 1] / np.maximum(p[:, 2], 1e-6) + Kl[1, 2]], 1)
                front = (pL[:, 2] > 1.0) & (pG[:, 2] > 1.0)
                cols += [np.linalg.norm(pL - pG, axis=1), np.where(front, np.linalg.norm(prj(pL) - prj(pG), axis=1), np.nan)]
                keep = rng.random(len(yy)) < G9.PIXEL_KEEP
                prow.append(np.stack(cols, 1)[keep].astype(np.float32))
            rec["pixels"] = np.concatenate(prow)
            rec["pixel_fields"] = np.asarray(["frame", "v", "u", "z_lidar", "ray_norm", "edge_class", "conf_lb", "lb_valid", "new_valid", "conf_new", "lidar_in_grid",
                                              "z_LBM", "same_voxel_LBM", "ep_on_ref_free_LBM", "z_NEW", "same_voxel_NEW", "ep_on_ref_free_NEW", "pose_disp_m", "reproj_px"])
            tmp = op + ".tmp.npz"; np.savez_compressed(tmp, **rec); os.replace(tmp, op); nw += 1
            if nw % 25 == 0:
                print(f"  [{d[17:21]} s{a.shard}] {nw}/{len(files)} {(time.time()-t0)/nw:.1f}s each", flush=True)
    print(f"[{d} s{a.shard}] {nw} clips, {(time.time()-t0)/60:.1f} min, firewall violations {len(audit.violations)}")
    return 0


def representative(d):
    """Gate 9 rule: Gate 9 PD (LB-M on LiDAR support) clip IoU at the 10th / 50th / 90th percentile."""
    fs = sorted(glob.glob(os.path.join(g14.G9_CLIPS, d, "*.npz"))); v = []
    for f in fs:
        m = np.load(f)["m_PD"]; v.append(m[0] / max(m[:3].sum(), 1))
    order = np.argsort(np.asarray(v), kind="stable")
    return {lab: int(os.path.basename(fs[order[int(q * (len(order) - 1))]])[:-4]) for lab, q in (("bad", 0.1), ("median", 0.5), ("good", 0.9))}


if __name__ == "__main__":
    raise SystemExit(main())
