#!/usr/bin/env python
"""Per-clip controlled depth-source comparison: LD / LB-M / MG / LB-O on identical pixels.

    python tools/depth_source_comparison/compare_sources.py --drive 2013_05_28_drive_0006_sync --device cuda:0 [--shard i --shards n]

Everything but the depth values is Gate 9's (reused by import or copied line for line): clips,
GT relative poses, lattice camera, anchor grid, reference and evaluation mask, LiDAR z-buffer
support, the frozen mapper at band 0.2 / margin 0.2 / carving, conf = SUPPORT_CONF on the mask
and 0 elsewhere, and the deterministic 20 % pixel subsample. See dsc.py for conditions/masks.
"""
from __future__ import annotations
import argparse, glob, hashlib, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dsc as D                                                                            # noqa: E402
sys.path.insert(0, D.G9_DIR); sys.path.insert(0, D.REPO_ROOT)
import common as G9                                                                        # noqa: E402
from audit import lattice_K, zbuffer, to_grid_index, flat_of                               # noqa: E402  (Gate 9)
from variant_mapper import VariantMapper                                                   # noqa: E402  (Gate 9)
from tools.local_registration_audit import common as LR                                    # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S  # noqa: E402
from core.datasets.audit import FileAudit                                                  # noqa: E402
from core.mapping.mapper import FrameInput                                                 # noqa: E402
from core.mapping import depth as D7                                                       # noqa: E402
from core.supervision import rawtarget as RT                                               # noqa: E402
from core.run._common import REPO_ROOT                                                     # noqa: E402

LH, LW = G9.LATTICE_HW; DIMS = G9.DIMS; ORIGIN = np.asarray(G9.ORIGIN); VS = G9.VS; NV = int(np.prod(DIMS))
PIX_FIELDS = ["frame", "v", "u", "z_lidar", "conf", "ray_norm", "in_S1", "in_S0", "lidar_in_grid", "z_LB_cached"] + \
             [f"{k}_{X}" for X in D.PIX_SOURCES for k in ("z", "same_voxel", "ep_on_ref_free")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); d = a.drive; src = SRC.source_of(d)
    seg = [s for s in S.segments(src, REPO_ROOT) if s.name == d][0]; nat = [int(f.order) for f in seg.frames]
    geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    K = np.asarray(geo.calib.K, np.float64); Kl = lattice_K(K); v2c = np.asarray(geo.calib.velo_to_rect_cam, np.float64); T_c2g = np.asarray(geo.rect_cam_to_velo, np.float64)
    Tgw = np.linalg.inv(T_c2g); grid = type("G", (), {"dims": DIMS, "voxel_size": VS, "origin": tuple(ORIGIN)})()
    files = sorted(glob.glob(os.path.dirname(LR.clip_path(d, 0)) + "/*.npz"))[a.shard::a.shards][:a.limit]
    out_dir = a.out or os.path.join(D.DATA_ROOT, "clips", d); os.makedirs(out_dir, exist_ok=True)
    gi = np.stack(np.unravel_index(np.arange(NV), DIMS), 1).astype(np.float64)
    ctr_g = (gi + 0.5) * VS + ORIGIN; ctr_c = torch.from_numpy(ctr_g @ Tgw[:3, :3].T + Tgw[:3, 3]).to(dev)
    vv_, uu_ = np.meshgrid(np.arange(LH), np.arange(LW), indexing="ij")
    rays = np.stack([(uu_ - Kl[0, 2]) / Kl[0, 0], (vv_ - Kl[1, 2]) / Kl[1, 1], np.ones_like(uu_, dtype=np.float64)], -1)
    t0 = time.time(); nw = 0; n_moge_missing = 0
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        load_moge = D.moge_loader(src)
        for f in files:
            with np.load(f) as z:
                Lz = {k: z[k] for k in z.files}
            t = int(Lz["t"]); op = os.path.join(out_dir, f"{t:05d}.npz")
            if os.path.exists(op):
                continue
            idx = Lz["stream_indices"].tolist(); l_scale = float(Lz["scale"]); Gt = geo.rect_cam_to_world(nat[t])
            TG = [TF.inv(Gt) @ geo.rect_cam_to_world(nat[j]) for j in idx]
            rec = {"t": np.int64(t), "stream_indices": np.asarray(idx), "keys": Lz["keys"], "l_scale": l_scale, "T_j_to_t": np.stack(TG), "K_lattice": Kl}
            # ---- reference and evaluation mask (Gate 9 audit.py, unchanged)
            ref = RT.build(geo, nat[t], [nat[j] for j in idx], dev)
            n_obs = np.asarray(ref.n_obs) > 0; n_free = np.asarray(ref.n_free) > 0
            r_occ = n_obs & ~n_free; r_free = n_free & ~n_obs
            fr_ = torch.zeros(NV, dtype=torch.bool, device=dev)
            for T in TG:
                Ti = torch.as_tensor(np.linalg.inv(T), device=dev); pc = ctr_c @ Ti[:3, :3].T + Ti[:3, 3]; zc = pc[:, 2]
                uu = Kl[0, 0] * pc[:, 0] / zc.clamp_min(1e-6) + Kl[0, 2]; vv = Kl[1, 1] * pc[:, 1] / zc.clamp_min(1e-6) + Kl[1, 2]
                fr_ |= (zc > D7.MIN_DEPTH_M) & (zc < D7.MAX_DEPTH_M) & (uu >= -0.5) & (uu < LW - 0.5) & (vv >= -0.5) & (vv < LH - 0.5)
            E = (r_occ | r_free) & fr_.cpu().numpy()
            rec["eval_mask_digest"] = hashlib.sha256(np.packbits(E).tobytes()).hexdigest()[:16]; rec["n_eval"] = int(E.sum())
            # ---- per-frame depth images on the one 140x518 lattice
            sup, zl, canon, cf, md, mvalid, fovs = [], [], [], [], [], [], []
            for q, j in enumerate(idx):
                pc = geo.read_velodyne(nat[j]) @ v2c[:3, :3].T + v2c[:3, 3]; zb, _ = zbuffer(pc, Kl)
                sup.append((zb > D7.MIN_DEPTH_M) & (zb < D7.MAX_DEPTH_M)); zl.append(zb)
                canon.append(Lz["pred_depth"][q].astype(np.float64)); cf.append(Lz["pred_depth_conf"][q].astype(np.float64))
                mo = load_moge(str(Lz["keys"][q]))
                if mo is None:
                    n_moge_missing += 1; md.append(np.full((LH, LW), np.nan)); mvalid.append(np.zeros((LH, LW), bool)); fovs.append(np.nan)
                else:
                    assert mo[0].shape == (LH, LW), f"MoGe lattice {mo[0].shape}"
                    md.append(mo[0]); mvalid.append(mo[1] & np.isfinite(mo[0]) & (mo[0] > 0)); fovs.append(mo[2])
            lbv = [np.isfinite(canon[q]) & (canon[q] > 0) for q in range(5)]
            gate = [cf[q] >= D7.CONF_THRESHOLD for q in range(5)]
            masks = {"g9": sup, "g9gate": [sup[q] & gate[q] for q in range(5)],
                     "S0": [sup[q] & mvalid[q] & lbv[q] for q in range(5)], "S1": [sup[q] & mvalid[q] & lbv[q] & gate[q] for q in range(5)]}
            for k, mk in masks.items():
                rec[f"n_pixels_{k}"] = np.asarray([int(m.sum()) for m in mk])
            rec["support_digest"] = hashlib.sha256(np.stack(sup).tobytes()).hexdigest()[:16]; rec["S1_digest"] = hashlib.sha256(np.stack(masks["S1"]).tobytes()).hexdigest()[:16]
            rec["moge_fov_x_deg"] = np.asarray(fovs)
            # ---- LB-O diagnostic scalars: median LiDAR / cached LingBot over the mask (all five frames)
            def fit(mk):
                return float(np.median(np.concatenate([zl[q][mk[q]] for q in range(5)]) / np.concatenate([canon[q][mk[q]] for q in range(5)])))
            s_LBO = fit(masks["S1"]); s_LBO_u = fit(masks["S0"]); rec["lbo_scale_S1"] = s_LBO; rec["lbo_scale_S0"] = s_LBO_u
            depth = {"lidar": zl, "LBM": [canon[q] * l_scale for q in range(5)], "MG": md,
                     "LBO": [canon[q] * s_LBO for q in range(5)], "LBO_u": [canon[q] * s_LBO_u for q in range(5)]}
            rec["range_rejected_S1"] = np.asarray([sum(int((masks["S1"][q] & ~((depth[X][q] > D7.MIN_DEPTH_M) & (depth[X][q] < D7.MAX_DEPTH_M))).sum()) for q in range(5))
                                                   for X in ("lidar", "LBM", "MG", "LBO")])
            # ---- mapper runs: only the depth array and the (source-independent) mask differ
            maps = {}
            for name, (dk, mkey) in D.RUNS.items():
                support = masks[mkey]
                mp = VariantMapper(dev, 0.2, 0.2, True); mp.scale_state.force(1.0)
                for q in range(5):
                    dm = depth[dk][q]; conf = np.where(support[q], G9.SUPPORT_CONF, 0.0)
                    fi = FrameInput(index=q, depth_canonical=torch.from_numpy(np.where(np.isfinite(dm), dm, 0.0).astype(np.float32)).to(dev),
                                    conf=torch.from_numpy(conf.astype(np.float32)).to(dev), K=Kl, pose_c2w_canonical=TG[q], log_scale_candidate=0.0)
                    mp.step(fi)
                qq = mp.query(grid, Tgw)
                occ, fre = qq["occupied"].cpu().numpy(), qq["free"].cpu().numpy(); maps[name] = occ
                tp = int((occ & r_occ & E).sum()); fp = int((occ & ~r_occ & E).sum()); fn = int((~occ & r_occ & E).sum())
                rec[f"m_{name}"] = np.asarray([tp, fp, fn, int((occ & E).sum()), int((occ & r_free & E).sum()), int((fre & r_occ & E).sum()), int(occ.sum()), int((occ & ~E).sum())], np.int64)
                del mp, qq
            for name in D.PRIMARY_COND:
                rec[f"occ_{name}"] = np.flatnonzero(maps[name]).astype(np.int32)
            rec["ref_occ"] = np.flatnonzero(r_occ & E).astype(np.int32)
            # ---- pixel rows: Gate 9's deterministic 20 % subsample of the LiDAR support (same draws)
            rng = np.random.default_rng(t); prow = []
            for q in range(5):
                yy, xx = np.nonzero(sup[q]); zlq = zl[q][yy, xx]; ray = rays[yy, xx]; rl = np.linalg.norm(ray, axis=1); T = TG[q]
                fl_l = flat_of(to_grid_index((ray * zlq[:, None]) @ T[:3, :3].T + T[:3, 3], T_c2g)); inside_l = fl_l >= 0
                cols = [np.full(len(yy), q), yy, xx, zlq, cf[q][yy, xx], rl, masks["S1"][q][yy, xx], masks["S0"][q][yy, xx], inside_l, canon[q][yy, xx]]
                for X in D.PIX_SOURCES:
                    zx = depth[X][q][yy, xx]; fin = np.isfinite(zx); fl = np.full(len(yy), -1, np.int64)
                    fl[fin] = flat_of(to_grid_index((ray[fin] * zx[fin, None]) @ T[:3, :3].T + T[:3, 3], T_c2g))
                    ins = fl >= 0; onf = np.zeros(len(yy), bool); onf[ins] = r_free[fl[ins]]
                    cols += [zx, (fl == fl_l) & inside_l, onf]
                keep = rng.random(len(yy)) < G9.PIXEL_KEEP
                prow.append(np.stack(cols, 1)[keep].astype(np.float32))
            rec["pixels"] = np.concatenate(prow); rec["pixel_fields"] = np.asarray(PIX_FIELDS)
            tmp = op + ".tmp.npz"; np.savez_compressed(tmp, **rec); os.replace(tmp, op); nw += 1
            if nw % 25 == 0:
                print(f"  [{d[17:21]} s{a.shard}] {nw}/{len(files)} clips, {(time.time()-t0)/nw:.1f}s each", flush=True)
    print(f"[{d} s{a.shard}] {nw} clips, {(time.time()-t0)/60:.1f} min, frames without MoGe {n_moge_missing}, firewall violations {len(audit.violations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
