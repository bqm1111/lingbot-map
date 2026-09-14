#!/usr/bin/env python
"""Gate 11 per-clip audit: H and the uncertainty representations on identical rays.

    python tools/gate11_uncertainty/audit_u.py --drive 2013_05_28_drive_0006_sync --device cuda:0 [--shard i --shards n]

Setup (clips, GT relative poses, lattice camera, anchor grid, reference, evaluation mask, LiDAR z-buffer
support, pixel subsample, edge classes) is Gate 9's, reused by import or copied line for line. Rays =
Gate 9 support AND LingBot conf >= 1.5, conf = SUPPORT_CONF on them (Gate 9 PD_gate / LD_gate). Width
rules are read from uncertainty_rules.json (derived from depth residuals before this script ran).
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as TF_

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g11 as G                                                                            # noqa: E402
sys.path.insert(0, G.G9_DIR); sys.path.insert(0, G.REPO_ROOT)
import common as G9                                                                        # noqa: E402
from audit import lattice_K, zbuffer, to_grid_index, flat_of                               # noqa: E402  (Gate 9)
from channel_mapper import ChannelMapper                                                   # noqa: E402
from tools.local_registration_audit import common as LR                                    # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S  # noqa: E402
from core.datasets.audit import FileAudit                                                  # noqa: E402
from core.mapping.mapper import FrameInput                                                 # noqa: E402
from core.mapping import depth as D7                                                       # noqa: E402
from core.supervision import rawtarget as RT                                               # noqa: E402
from core.run._common import REPO_ROOT                                                     # noqa: E402

LH, LW = G9.LATTICE_HW; DIMS = G9.DIMS; ORIGIN = np.asarray(G9.ORIGIN); VS = G9.VS; NV = int(np.prod(DIMS))
COUNT_FIELDS = ["R", "Fr", "E", "S_R", "U_R", "F_R", "S_Fr", "U_Fr", "F_Fr", "S_E", "U_E", "F_E", "S_all", "U_all", "F_all"]
MAP_CONDS = ("H", "U1_0.4", "U2", "U3", "U4", "U3h", "U4h", "H_LD", "U3_LD")
G9_CLIPS = "/media/SSD1/MINH_DATASETS/lingbot_gate9_depth_occupancy/clips"


def edge_classes(zl_q, m, dev):
    """Gate 9 audit.py lines 171-177, unchanged."""
    yy, xx = np.nonzero(m)
    zt = torch.from_numpy(np.where(m, zl_q, np.nan)).to(dev)[None, None]
    pad = G9.EDGE_WIN // 2
    un = TF_.unfold(torch.nan_to_num(zt, nan=0.0), G9.EDGE_WIN, padding=pad)[0]; msk = TF_.unfold((~torch.isnan(zt)).float(), G9.EDGE_WIN, padding=pad)[0]
    cz = torch.nan_to_num(zt, nan=0.0).reshape(-1)
    jump = ((un - cz.unsqueeze(0)).abs() / cz.clamp_min(1e-6).unsqueeze(0) * msk).amax(0); nnb = msk.sum(0) - 1
    jump = jump.reshape(LH, LW).cpu().numpy()[yy, xx]; nnb = nnb.reshape(LH, LW).cpu().numpy()[yy, xx]
    edge = np.where(nnb < 1, 3, np.where(jump < G9.EDGE_EDGES[0], 0, np.where(jump < G9.EDGE_EDGES[1], 1, 2)))
    return edge, jump


def representative_clips(drive):
    """Predeclared: Gate 9 PD_gate (== H) clip IoU at the 10th / 50th / 90th percentile."""
    fs = sorted(glob.glob(os.path.join(G9_CLIPS, drive, "*.npz"))); v = []
    for f in fs:
        m = np.load(f)["m_PD_gate"]; v.append(m[0] / max(m[:3].sum(), 1))
    order = np.argsort(np.asarray(v), kind="stable")
    return {lab: int(os.path.basename(fs[order[int(q * (len(order) - 1))]])[:-4]) for lab, q in (("bad", 0.1), ("median", 0.5), ("good", 0.9))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); d = a.drive; src = SRC.source_of(d)
    rules = json.load(open(G.RULES)); rules_sha = hashlib.sha256(open(G.RULES, "rb").read()).hexdigest()
    conds = G.conditions(rules, with_ld=(d == G.PRIMARY)); reps = representative_clips(d)
    seg = [s for s in S.segments(src, REPO_ROOT) if s.name == d][0]; nat = [int(f.order) for f in seg.frames]
    geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
    K = np.asarray(geo.calib.K, np.float64); Kl = lattice_K(K); v2c = np.asarray(geo.calib.velo_to_rect_cam, np.float64); T_c2g = np.asarray(geo.rect_cam_to_velo, np.float64)
    Tgw = np.linalg.inv(T_c2g); grid = type("G", (), {"dims": DIMS, "voxel_size": VS, "origin": tuple(ORIGIN)})()
    files = sorted(glob.glob(os.path.dirname(LR.clip_path(d, 0)) + "/*.npz"))[a.shard::a.shards][:a.limit]
    out_dir = a.out or os.path.join(G.DATA_ROOT, "clips", d); os.makedirs(out_dir, exist_ok=True)
    gi = np.stack(np.unravel_index(np.arange(NV), DIMS), 1).astype(np.float64)
    ctr_g = (gi + 0.5) * VS + ORIGIN; ctr_np = ctr_g @ Tgw[:3, :3].T + Tgw[:3, 3]; ctr_c = torch.from_numpy(ctr_np).to(dev)
    rbin = torch.from_numpy(np.clip(np.digitize(ctr_np[:, 2], np.asarray(G.RANGE_BINS[1:-1])), 0, 4)).to(dev)   # voxel range = anchor-camera z
    vv_, uu_ = np.meshgrid(np.arange(LH), np.arange(LW), indexing="ij")
    rays = np.stack([(uu_ - Kl[0, 2]) / Kl[0, 0], (vv_ - Kl[1, 2]) / Kl[1, 1], np.ones_like(uu_, dtype=np.float64)], -1)
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
            rec = {"t": np.int64(t), "stream_indices": np.asarray(idx), "keys": Lz["keys"], "l_scale": l_scale, "T_j_to_t": np.stack(TG), "K_lattice": Kl,
                   "rules_sha256": rules_sha, "condition_names": np.asarray(list(conds)), "count_fields": np.asarray(COUNT_FIELDS),
                   "representative": np.asarray([reps["bad"], reps["median"], reps["good"]])}
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
            Rt = torch.from_numpy(r_occ & E).to(dev); Frt = torch.from_numpy(r_free & E).to(dev); Et = torch.from_numpy(E).to(dev)
            # ---- rays
            sup, zl, canon, cf = [], [], [], []
            for q, j in enumerate(idx):
                pc = geo.read_velodyne(nat[j]) @ v2c[:3, :3].T + v2c[:3, 3]; zb, _ = zbuffer(pc, Kl)
                sup.append((zb > D7.MIN_DEPTH_M) & (zb < D7.MAX_DEPTH_M)); zl.append(zb)
                canon.append(Lz["pred_depth"][q].astype(np.float64)); cf.append(Lz["pred_depth_conf"][q].astype(np.float64))
            gate = [sup[q] & (cf[q] >= D7.CONF_THRESHOLD) & np.isfinite(canon[q]) & (canon[q] > 0) for q in range(5)]
            rec["support_digest"] = hashlib.sha256(np.stack(sup).tobytes()).hexdigest()[:16]; rec["n_gate_pixels"] = np.asarray([int(m.sum()) for m in gate])
            depth = {"LBM": [canon[q] * l_scale for q in range(5)], "lidar": zl}
            for sname, dd in depth.items():
                rec[f"n_rays_{sname}"] = int(sum(int((gate[q] & (dd[q] > D7.MIN_DEPTH_M) & (dd[q] < D7.MAX_DEPTH_M)).sum()) for q in range(5)))
            # ---- representations
            states, W = {}, {}
            for name, (sname, rule, carve) in conds.items():
                uf_l, ub_l = zip(*[G.widths(rule, depth[sname][q], cf[q], rules) for q in range(5)]); W[name] = (uf_l, ub_l)
                frames = [FrameInput(index=q, depth_canonical=torch.from_numpy(np.where(np.isfinite(depth[sname][q]), depth[sname][q], 0.0).astype(np.float32)).to(dev),
                                     conf=torch.from_numpy(np.where(gate[q], G9.SUPPORT_CONF, 0.0).astype(np.float32)).to(dev), K=Kl, pose_c2w_canonical=TG[q], log_scale_candidate=0.0) for q in range(5)]
                mS = ChannelMapper(dev, "surface"); mS.scale_state.force(1.0)
                for q in range(5):
                    mS.set_pixels(margin=(uf_l[q] if carve == "conservative" else np.full((LH, LW), 0.2))); mS.step(frames[q])
                qS = mS.query(grid, Tgw); Sm, Fm = qS["occupied"], qS["free"]; del mS, qS
                if rule is None:
                    Um = torch.zeros_like(Sm)
                else:
                    mU = ChannelMapper(dev, "interval"); mU.scale_state.force(1.0)
                    for q in range(5):
                        mU.set_pixels(u_front=uf_l[q], u_back=ub_l[q]); mU.step(frames[q])
                    Um = ~Sm & ~Fm & (mU.query(grid, Tgw)["w_occ"] > 0); del mU
                cnt = torch.zeros((6, len(COUNT_FIELDS)), dtype=torch.int64, device=dev)
                for i, x in enumerate((Rt, Frt, Et, Sm & Rt, Um & Rt, Fm & Rt, Sm & Frt, Um & Frt, Fm & Frt, Sm & Et, Um & Et, Fm & Et, Sm, Um, Fm)):
                    cnt[:5, i] = torch.bincount(rbin[x], minlength=5); cnt[5, i] = x.sum()
                rec[f"c_{name}"] = cnt.cpu().numpy()
                occ = Sm.cpu().numpy(); fre = Fm.cpu().numpy()
                tp = int((occ & r_occ & E).sum()); fp = int((occ & ~r_occ & E).sum()); fn = int((~occ & r_occ & E).sum())
                rec[f"m_{name}"] = np.asarray([tp, fp, fn, int((occ & E).sum()), int((occ & r_free & E).sum()), int((fre & r_occ & E).sum()), int(occ.sum()), int((occ & ~E).sum())], np.int64)
                st = torch.zeros(NV, dtype=torch.uint8, device=dev); st[Sm] = 1; st[Um] = 2; st[Fm] = 3; states[name] = st.cpu().numpy()
                if t in reps.values() and name in MAP_CONDS:
                    rec[f"map_{name}"] = states[name]
            if t in reps.values():
                rec["map_ref"] = (r_occ.astype(np.uint8) + 2 * r_free.astype(np.uint8)) * E
            # ---- pixel rows (Gate 9 deterministic 20 % subsample of the LiDAR support, same draws)
            rng = np.random.default_rng(t); prow = []; fields = None
            r_free_flat = r_free
            for q in range(5):
                m = sup[q]; yy, xx = np.nonzero(m); zlq = zl[q][yy, xx]; ray = rays[yy, xx]; rl = np.linalg.norm(ray, axis=1); T = TG[q]
                edge, jump = edge_classes(zl[q], m, dev)
                fl_l = flat_of(to_grid_index((ray * zlq[:, None]) @ T[:3, :3].T + T[:3, 3], T_c2g)); inl = fl_l >= 0
                zb_ = depth["LBM"][q][yy, xx]; fl_p = flat_of(to_grid_index((ray * zb_[:, None]) @ T[:3, :3].T + T[:3, 3], T_c2g))
                onf = np.zeros(len(yy), bool); onf[fl_p >= 0] = r_free_flat[fl_p[fl_p >= 0]]
                cols = [np.full(len(yy), q), yy, xx, zlq, zb_, cf[q][yy, xx], rl, edge, jump, gate[q][yy, xx], inl, onf]
                fields = ["frame", "v", "u", "z_lidar", "z_lbm", "conf", "ray_norm", "edge_class", "edge_jump", "gate", "lidar_in_grid", "lbm_ep_on_ref_free"]
                for name, (sname, rule, carve) in conds.items():
                    stt = np.full(len(yy), 4, np.uint8); stt[inl] = states[name][fl_l[inl]]
                    r = (zlq - depth[sname][q][yy, xx]) * rl
                    if rule is None:
                        cont = np.abs(r) <= 0.2
                    else:
                        cont = (r >= -W[name][0][q][yy, xx]) & (r <= W[name][1][q][yy, xx])
                    cols += [stt, cont]; fields += [f"state_{name}", f"contains_{name}"]
                for name in ("U2", "U3", "U4"):
                    if name in conds:
                        cols += [W[name][0][q][yy, xx], W[name][1][q][yy, xx]]; fields += [f"uf_{name}", f"ub_{name}"]
                keep = rng.random(len(yy)) < G9.PIXEL_KEEP
                prow.append(np.stack(cols, 1)[keep].astype(np.float32))
            rec["pixels"] = np.concatenate(prow); rec["pixel_fields"] = np.asarray(fields)
            tmp = op + ".tmp.npz"; np.savez_compressed(tmp, **rec); os.replace(tmp, op); nw += 1
            if nw % 10 == 0:
                print(f"  [{d[17:21]} s{a.shard}] {nw}/{len(files)} clips, {(time.time()-t0)/nw:.1f}s each", flush=True)
    print(f"[{d} s{a.shard}] {nw} clips, {(time.time()-t0)/60:.1f} min, firewall violations {len(audit.violations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
