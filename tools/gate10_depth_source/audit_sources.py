#!/usr/bin/env python
"""Gate 10 per-clip audit: one pass per clip, every depth source on identical pixels.

    python tools/gate10_depth_source/audit_sources.py --drive 2013_05_28_drive_0006_sync --device cuda:0 [--shard i --shards n]

Setup lines (poses, lattice camera, reference, evaluation mask, LiDAR z-buffer, edge classes,
pixel subsample) are Gate 9's audit.py, reused by import or copied line for line; only the
depth source fed to the frozen mapper changes. Matched pixels = Gate 9 LiDAR support AND
MoGe-2 valid (MoGe mask, finite, > 0); every condition below LD_g9 / LBM_g9 uses exactly
this set, with conf = SUPPORT_CONF there and 0 elsewhere (Gate 9's PD convention).
"""
from __future__ import annotations
import argparse, glob, hashlib, os, sys, time
import numpy as np, torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g10 as G10                                                                          # noqa: E402
sys.path.insert(0, G10.G9_DIR); sys.path.insert(0, G10.REPO_ROOT)
import common as G9                                                                        # noqa: E402
from audit import lattice_K, zbuffer, to_grid_index, flat_of                              # noqa: E402  (Gate 9)
from variant_mapper import VariantMapper                                                  # noqa: E402  (Gate 9)
from tools.local_registration_audit import common as LR                                   # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402
from core.mapping.mapper import FrameInput                                                # noqa: E402
from core.mapping import depth as D7                                                      # noqa: E402
from core.supervision import rawtarget as RT                                              # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402

LH, LW = G9.LATTICE_HW; DIMS = G9.DIMS; ORIGIN = np.asarray(G9.ORIGIN); VS = G9.VS; NV = int(np.prod(DIMS))
PIX_FIELDS = ["frame", "v", "u", "z_lidar", "conf", "ray_norm", "edge_class", "edge_jump", "matched", "lidar_in_grid"] + \
             [f"{k}_{X}" for X in G10.PRED for k in ("z", "same_voxel", "ep_on_ref_free", "in_grid")]
METRIC_FIELDS = ["tp", "fp", "fn", "pred_occ_in_mask", "pred_occ_on_ref_free", "ref_occ_marked_free", "pred_occ_total", "pred_occ_outside_mask"]


def edge_classes(zl_q, m, dev):
    """Gate 9 audit.py lines 171-177: max relative LiDAR depth jump to supported neighbours (5x5)."""
    yy, xx = np.nonzero(m)
    zt = torch.from_numpy(np.where(m, zl_q, np.nan)).to(dev)[None, None]
    pad = G9.EDGE_WIN // 2
    un = F.unfold(torch.nan_to_num(zt, nan=0.0), G9.EDGE_WIN, padding=pad)[0]; msk = F.unfold((~torch.isnan(zt)).float(), G9.EDGE_WIN, padding=pad)[0]
    cz = torch.nan_to_num(zt, nan=0.0).reshape(-1)
    jump = ((un - cz.unsqueeze(0)).abs() / cz.clamp_min(1e-6).unsqueeze(0) * msk).amax(0); nnb = msk.sum(0) - 1
    jump = jump.reshape(LH, LW).cpu().numpy()[yy, xx]; nnb = nnb.reshape(LH, LW).cpu().numpy()[yy, xx]
    edge = np.where(nnb < 1, 3, np.where(jump < G9.EDGE_EDGES[0], 0, np.where(jump < G9.EDGE_EDGES[1], 1, 2)))
    return edge, jump


def moge_loader(src):
    """Frozen MoGe-2 B-variant caches, exactly the ones the clip scale was computed from
    (run_local_lingbot.py): clip-keyed Gate 5.2 cache for 0006, per-frame Gate 8B cache otherwise."""
    if src == "kitti360":
        index = D7.moge_frame_index("kitti360", REPO_ROOT)
        def load(key):
            r = D7.load_moge_frame("kitti360", key, index)
            return None if r is None else (r[0].astype(np.float64), r[1], r[2])
    else:
        def load(key):
            p = S.moge_path(src, key)
            if not os.path.exists(p):
                return None
            with np.load(p) as z:
                return z["moge_depth"].astype(np.float32).astype(np.float64), D7.unpack_mask(z["moge_mask"], z["mask_shape"]), float(z["fov_x_deg"])
    return load


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
    out_dir = a.out or os.path.join(G10.DATA_ROOT, "clips", d); os.makedirs(out_dir, exist_ok=True)
    gi = np.stack(np.unravel_index(np.arange(NV), DIMS), 1).astype(np.float64)
    ctr_g = (gi + 0.5) * VS + ORIGIN; ctr_c = torch.from_numpy(ctr_g @ Tgw[:3, :3].T + Tgw[:3, 3]).to(dev)
    vv_, uu_ = np.meshgrid(np.arange(LH), np.arange(LW), indexing="ij")
    rays = np.stack([(uu_ - Kl[0, 2]) / Kl[0, 0], (vv_ - Kl[1, 2]) / Kl[1, 1], np.ones_like(uu_, dtype=np.float64)], -1)
    t0 = time.time(); nw = 0; n_moge_missing = 0
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        load_moge = moge_loader(src)
        for f in files:
            with np.load(f) as z:
                Lz = {k: z[k] for k in z.files}
            t = int(Lz["t"]); op = os.path.join(out_dir, f"{t:05d}.npz")
            if os.path.exists(op):
                continue
            idx = Lz["stream_indices"].tolist(); l_scale = float(Lz["scale"]); Gt = geo.rect_cam_to_world(nat[t])
            TG = [TF.inv(Gt) @ geo.rect_cam_to_world(nat[j]) for j in idx]
            rec = {"t": np.int64(t), "stream_indices": np.asarray(idx), "natives": np.asarray([nat[j] for j in idx]), "keys": Lz["keys"], "l_scale": l_scale,
                   "T_j_to_t": np.stack(TG), "K_lattice": Kl}
            # ---- reference and evaluation mask (Gate 9 audit.py lines 98-108)
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
            # ---- per-frame LiDAR support and depth images
            sup, zl, canon, cf, md, mvalid, fovs = [], [], [], [], [], [], []
            for q, j in enumerate(idx):
                pts = geo.read_velodyne(nat[j]); pc = pts @ v2c[:3, :3].T + v2c[:3, 3]
                zb, _ = zbuffer(pc, Kl)
                sup.append((zb > D7.MIN_DEPTH_M) & (zb < D7.MAX_DEPTH_M)); zl.append(zb)
                canon.append(Lz["pred_depth"][q].astype(np.float64)); cf.append(Lz["pred_depth_conf"][q].astype(np.float64))
                mo = load_moge(str(Lz["keys"][q]))
                if mo is None:
                    n_moge_missing += 1; md.append(np.full((LH, LW), np.nan)); mvalid.append(np.zeros((LH, LW), bool)); fovs.append(np.nan)
                else:
                    assert mo[0].shape == (LH, LW), f"MoGe lattice {mo[0].shape}"
                    md.append(mo[0]); mvalid.append(mo[1] & np.isfinite(mo[0]) & (mo[0] > 0)); fovs.append(mo[2])
            matched = [sup[q] & mvalid[q] & np.isfinite(canon[q]) & (canon[q] > 0) for q in range(5)]
            rec["moge_fov_x_deg"] = np.asarray(fovs); rec["n_support_pixels"] = np.asarray([int(s.sum()) for s in sup]); rec["n_matched_pixels"] = np.asarray([int(m.sum()) for m in matched])
            rec["support_digest"] = hashlib.sha256(np.stack(sup).tobytes()).hexdigest()[:16]; rec["matched_digest"] = hashlib.sha256(np.stack(matched).tobytes()).hexdigest()[:16]
            # ---- ORACLE scalars (LiDAR). Used only by LBL / ML below.
            zl_m = np.concatenate([zl[q][matched[q]] for q in range(5)])
            s_LBL = float(np.median(zl_m / np.concatenate([canon[q][matched[q]] for q in range(5)])))
            s_ML = float(np.median(zl_m / np.concatenate([md[q][matched[q]] for q in range(5)])))
            rec["oracle_scale_LBL"] = s_LBL; rec["oracle_scale_ML"] = s_ML
            depth = {"lidar": zl, "LB": canon, "LBM": [canon[q] * l_scale for q in range(5)], "M": md,
                     "LBL": [canon[q] * s_LBL for q in range(5)], "ML": [md[q] * s_ML for q in range(5)]}
            # ---- mapper runs (frozen mapper, identical FrameInput apart from depth and support)
            maps = {}
            for name, (dk, sk) in G10.RUNS.items():
                support = sup if sk == "gate9" else matched
                mp = VariantMapper(dev, 0.2, 0.2, True); mp.scale_state.force(1.0)
                for q in range(5):
                    dm = depth[dk][q]; conf = np.where(support[q], G9.SUPPORT_CONF, 0.0)
                    fi = FrameInput(index=q, depth_canonical=torch.from_numpy(np.where(np.isfinite(dm), dm, 0.0).astype(np.float32)).to(dev),
                                    conf=torch.from_numpy(conf.astype(np.float32)).to(dev), K=Kl, pose_c2w_canonical=TG[q], log_scale_candidate=0.0)
                    mp.step(fi)
                qq = mp.query(type("G", (), {"dims": DIMS, "voxel_size": VS, "origin": tuple(ORIGIN)})(), Tgw)
                occ, fre = qq["occupied"].cpu().numpy(), qq["free"].cpu().numpy(); maps[name] = occ
                tp = int((occ & r_occ & E).sum()); fp = int((occ & ~r_occ & E).sum()); fn = int((~occ & r_occ & E).sum())
                rec[f"m_{name}"] = np.asarray([tp, fp, fn, int((occ & E).sum()), int((occ & r_free & E).sum()), int((fre & r_occ & E).sum()), int(occ.sum()), int((occ & ~E).sum())], np.int64)
                del mp, qq
            rec["metric_fields"] = np.asarray(METRIC_FIELDS)
            for name in G10.MAIN:
                rec[f"occ_{name}"] = np.flatnonzero(maps[name]).astype(np.int32)
            rec["ref_occ"] = np.flatnonzero(r_occ & E).astype(np.int32)
            # ---- pixel records: Gate 9's deterministic 20 % subsample of the LiDAR support (same draws)
            rng = np.random.default_rng(t); prow = []
            for q in range(5):
                m = sup[q]; yy, xx = np.nonzero(m); zlq = zl[q][yy, xx]; ray = rays[yy, xx]; rl = np.linalg.norm(ray, axis=1)
                edge, jump = edge_classes(zl[q], m, dev)
                T = TG[q]
                fl_l = flat_of(to_grid_index((ray * zlq[:, None]) @ T[:3, :3].T + T[:3, 3], T_c2g)); inside_l = fl_l >= 0
                cols = [np.full(len(yy), q), yy, xx, zlq, cf[q][yy, xx], rl, edge, jump, matched[q][yy, xx], inside_l]
                for X in G10.PRED:
                    zx = depth[X][q][yy, xx]; fin = np.isfinite(zx)
                    fl = np.full(len(yy), -1, np.int64)
                    fl[fin] = flat_of(to_grid_index((ray[fin] * zx[fin, None]) @ T[:3, :3].T + T[:3, 3], T_c2g))
                    ins = fl >= 0; onf = np.zeros(len(yy), bool); onf[ins] = r_free[fl[ins]]
                    cols += [zx, (fl == fl_l) & inside_l, onf, ins]
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
