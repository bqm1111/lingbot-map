#!/usr/bin/env python
"""Steps 6-9 and 11: five-frame maps under each registration condition.

For a clip (t-4 .. t) and condition c the frozen IncrementalMapper integrates the five frames
with FrameInputs built from that condition's depth/confidence and anchor-relative poses
(anchor camera = identity, translations in canonical units = metric / scale, scale forced):

  S   stream depth, stream poses, stream scale
  L   clip depth, clip poses, clip scale
  Lp  stream depth, clip relative poses (metric via clip scale, expressed in the stream gauge)
  G   stream depth, GT relative poses (metric, expressed in the stream gauge)
  GLd clip depth, GT relative poses (clip gauge)
  Gs  DIAGNOSTIC ORACLE: stream depth rescaled by the clip's median LiDAR/LingBot depth ratio at the
      anchor frame, GT relative poses -- isolates depth *scale* from depth shape; uses LiDAR, not deployable

The map is queried on the anchor's SSC grid (velodyne of t). References, all raw LiDAR + GT:
  past5   voxels measured by the same five sweeps (occupied = return-only, free = traversal-only)
  target  the current raw-LiDAR training target (anchor + 20 future sweeps, conflicts excluded)
Reported: occupied precision / recall / IoU (raw, and after the adopted r=2 dilation), free
contradiction (map-occupied on LiDAR-free), multi-view consistency (each frame mapped alone:
nearest-neighbour distance of frame i endpoints to the anchor frame's endpoints, conflicting
occupied/free voxels of the joint map), the frozen completion head (sky seed 0, adopted
threshold) on each map, and (0006 only) the frozen Trident same-voxel semantic check.
No threshold, dilation or checkpoint differs across conditions.

    python tools/local_registration_audit/geometry_audit.py --drive 2013_05_28_drive_0006_sync --device cuda:0
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np, torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.local_registration_audit import common as LR                                   # noqa: E402
from core.datasets import kitti360_transforms as TF, kitti360_splits as SRC, sources as S, union_vocab as V8, vocab, grids as G6G  # noqa: E402
from core.datasets.audit import FileAudit                                                 # noqa: E402
from core.mapping.feed import CachedFeed                                                  # noqa: E402
from core.mapping.mapper import IncrementalMapper, FrameInput                             # noqa: E402
from core.mapping import depth as D7                                                      # noqa: E402
from core.evaluation.voxels import dilate                                                 # noqa: E402
from core.supervision import rawtarget as RT                                              # noqa: E402
from core.run.build_targets import load as load_target                                    # noqa: E402
from core.model.net import load_checkpoint                                                # noqa: E402
from core.run._common import REPO_ROOT                                                    # noqa: E402
from tools.kitti360_data_scaling import common as C0                                      # noqa: E402

DIMS = (256, 256, 32); ORIGIN = np.array([0.0, -25.6, -2.0]); VS = 0.2
HEAD = "artifacts/gate8c1/checkpoints/sky_seed0_last.pt"; HEAD_TAU = -0.125


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev); d = a.drive; src = SRC.source_of(d)
    seg = [s for s in S.segments(src, REPO_ROOT) if s.name == d][0]; nat = [int(f.order) for f in seg.frames]
    geo = TF.DriveGeometry(d, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT); T_c2g = np.asarray(geo.rect_cam_to_velo, np.float64)
    feed = CachedFeed(seg, dev)                                                           # stream depth/conf/pose + Trident probs per frame
    with np.load(S.scale_path(src, d)) as z:
        s_scale = float(np.exp(np.median(z["log_s"][:5])))
    MAP = G6G.PREDICTION_GRID["kitti360"]; into = V8.into_matrix("kitti360"); C = len(vocab.load("kitti360"))
    head = load_checkpoint(os.path.join(REPO_ROOT, HEAD), dev)
    files = sorted(glob.glob(os.path.dirname(LR.clip_path(d, 0)) + "/*.npz"))[a.shard::a.shards][:a.limit]
    out_dir = os.path.join(LR.DATA_ROOT, "geometry", d); os.makedirs(out_dir, exist_ok=True)
    n = int(np.prod(DIMS)); t0 = time.time(); nw = 0
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for f in files:
            with np.load(f) as z:
                Lz = {k: z[k] for k in z.files}
            t = int(Lz["t"]); op = os.path.join(out_dir, f"{t:05d}.npz")
            if os.path.exists(op):
                continue
            idx = Lz["stream_indices"].tolist(); l_scale = float(Lz["scale"])
            Gt = geo.rect_cam_to_world(nat[t])
            frames_s = [feed.frame(j) for j in idx]
            TG = [TF.inv(Gt) @ geo.rect_cam_to_world(nat[j]) for j in idx]
            TS = [np.linalg.inv(feed.pose[t]) @ feed.pose[j] for j in idx]                  # canonical stream units
            TLc = [np.linalg.inv(Lz["pred_pose_c2w"][4].astype(np.float64)) @ Lz["pred_pose_c2w"][q].astype(np.float64) for q in range(5)]  # canonical clip units
            def canon(Ts_metric, sc):
                out = []
                for T in Ts_metric:
                    T = T.copy(); T[:3, 3] /= sc; out.append(T)
                return out
            TLm = [T.copy() for T in TLc]
            for T in TLm: T[:3, 3] *= l_scale
            # per-clip LiDAR/LingBot median depth ratio at the anchor frame (scale oracle, diagnostic G* only)
            pts_t = geo.read_velodyne(nat[t]); v2c = np.asarray(geo.calib.velo_to_rect_cam); pc = pts_t @ v2c[:3, :3].T + v2c[:3, 3]
            Kn = np.asarray(geo.calib.K); zz = pc[:, 2]; uu_ = Kn[0, 0] * pc[:, 0] / np.maximum(zz, 1e-6) + Kn[0, 2]; vv_ = Kn[1, 1] * pc[:, 1] / np.maximum(zz, 1e-6) + Kn[1, 2]
            okp = (zz > 3) & (zz < 50) & (uu_ >= 0) & (uu_ < 1408) & (vv_ >= 0) & (vv_ < 376)
            ul = np.clip(((uu_[okp] + 0.5) * 518 / 1408 - 0.5).round().astype(int), 0, 517); vl = np.clip(((vv_[okp] + 0.5) * 140 / 376 - 0.5).round().astype(int), 0, 139)
            dS = frames_s[4].depth_canonical.cpu().numpy()[vl, ul].astype(np.float64) * s_scale
            oracle_ratio = float(np.median(zz[okp] / np.maximum(dS, 1e-6)))
            gs_scale = s_scale * oracle_ratio
            conds = {"S": ([fr.depth_canonical for fr in frames_s], [fr.conf for fr in frames_s], [fr.K for fr in frames_s], TS, s_scale),
                     "L": ([torch.from_numpy(Lz["pred_depth"][q].astype(np.float32)).to(dev) for q in range(5)], [torch.from_numpy(Lz["pred_depth_conf"][q].astype(np.float32)).to(dev) for q in range(5)],
                           [Lz["pred_K"][q].astype(np.float64) for q in range(5)], TLc, l_scale),
                     "Lp": ([fr.depth_canonical for fr in frames_s], [fr.conf for fr in frames_s], [fr.K for fr in frames_s], canon(TLm, s_scale), s_scale),
                     "G": ([fr.depth_canonical for fr in frames_s], [fr.conf for fr in frames_s], [fr.K for fr in frames_s], canon(TG, s_scale), s_scale),
                     "GLd": ([torch.from_numpy(Lz["pred_depth"][q].astype(np.float32)).to(dev) for q in range(5)], [torch.from_numpy(Lz["pred_depth_conf"][q].astype(np.float32)).to(dev) for q in range(5)],
                             [Lz["pred_K"][q].astype(np.float64) for q in range(5)], canon(TG, l_scale), l_scale),
                     "Gs": ([fr.depth_canonical for fr in frames_s], [fr.conf for fr in frames_s], [fr.K for fr in frames_s], canon(TG, gs_scale), gs_scale)}
            # S reproduction: the adopted past5 map in the stream's own world frame (eval_target.window_map convention)
            ms = IncrementalMapper(dev, sem_into=into, n_teacher=C); ms.scale_state.force(s_scale)
            for j in idx:
                ms.step(feed.frame(j))
            Pw = feed.pose[t].copy(); Pw[:3, 3] *= s_scale
            occ_stream_frame = ms.query(MAP, Pw @ np.linalg.inv(T_c2g))["occupied"].cpu().numpy(); del ms
            # references
            past = RT.build(geo, nat[t], [nat[j] for j in idx], dev)
            p_occ = (np.asarray(past.n_obs) > 0) & ~(np.asarray(past.n_free) > 0); p_free = (np.asarray(past.n_free) > 0) & ~(np.asarray(past.n_obs) > 0)
            tgt = load_target(SRC.target_path(d, t)); t_occ, t_val = tgt["occupied"], tgt["valid"]
            rec = {}
            for c, (deps, confs, Ks, Ts, sc) in conds.items():
                m = IncrementalMapper(dev, sem_into=into, n_teacher=C); m.scale_state.force(sc)
                ep_sets = []
                for q in range(5):
                    fr = FrameInput(index=q, depth_canonical=deps[q], conf=confs[q], K=Ks[q], pose_c2w_canonical=Ts[q],
                                    log_scale_candidate=float(np.log(sc)), sem_probs=frames_s[q].sem_probs, moge_depth=None, moge_mask=None)
                    m.step(fr)
                    # single-frame endpoints in the anchor camera (for multi-view consistency)
                    dd = (deps[q].double() * sc); ok = (confs[q].double() >= D7.CONF_THRESHOLD) & torch.isfinite(dd) & (dd > D7.MIN_DEPTH_M) & (dd < D7.MAX_DEPTH_M)
                    Kq = torch.as_tensor(Ks[q], device=dev); H_, W_ = dd.shape
                    vv, uu = torch.meshgrid(torch.arange(H_, device=dev, dtype=torch.float64), torch.arange(W_, device=dev, dtype=torch.float64), indexing="ij")
                    P = torch.stack([(uu - Kq[0, 2]) / Kq[0, 0], (vv - Kq[1, 2]) / Kq[1, 1], torch.ones_like(uu)], -1)[ok] * dd[ok].unsqueeze(1)
                    Tq = torch.as_tensor(Ts[q], device=dev).clone(); Tq[:3, 3] *= sc
                    ep_sets.append((P @ Tq[:3, :3].T + Tq[:3, 3]).float().cpu().numpy()[::4])
                Tgw = np.linalg.inv(T_c2g)                                  # anchor camera = identity; the mapper's world is metric
                q_ = m.query(MAP, Tgw)
                occ = q_["occupied"].cpu().numpy(); wo = q_["w_occ"].cpu().numpy(); wf = q_["w_free"].cpu().numpy()
                dil = dilate(q_["occupied"].view(DIMS), 2).reshape(-1).cpu().numpy()
                def prf(pred, gocc, gval):
                    tp = int((pred & gocc & gval).sum()); fp = int((pred & ~gocc & gval).sum()); fn = int((~pred & gocc & gval).sum()); return tp, fp, fn
                meas = p_occ | p_free
                rec[f"{c}_past5_raw"] = prf(occ, p_occ, meas); rec[f"{c}_past5_dil"] = prf(dil, p_occ, meas)
                rec[f"{c}_target_raw"] = prf(occ, t_occ, t_val); rec[f"{c}_target_dil"] = prf(dil, t_occ, t_val)
                rec[f"{c}_occ_on_lidar_free"] = int((occ & p_free).sum()); rec[f"{c}_n_occ"] = int(occ.sum())
                rec[f"{c}_conflict_voxels"] = int(((wo > 0) & (wf > 0)).sum())
                A_ = ep_sets[4]; tree = cKDTree(A_) if len(A_) else None; nn = []
                for q in range(4):
                    if tree is not None and len(ep_sets[q]):
                        dq, _ = tree.query(ep_sets[q], k=1, distance_upper_bound=5.0); dq = np.where(np.isfinite(dq), dq, 5.0)
                        zq = ep_sets[q][:, 2]; sel = (zq > 3) & (zq < 30)
                        nn.append(float(np.median(dq[sel])) if sel.any() else np.nan)
                rec[f"{c}_mv_nn_median_m"] = np.asarray(nn, np.float32)
                # frozen completion head on this map (diagnostic)
                q_["age"] = torch.where(q_["last_time"] >= 0, (4 - q_["last_time"]).float(), torch.full_like(q_["last_time"], -1).float())
                fin, _ = head.raw(q_, MAP)
                pred = (fin >= HEAD_TAU).cpu().numpy(); rec[f"{c}_head_target"] = prf(pred, t_occ, t_val)
                rec.setdefault("occ_maps", {})[c] = np.flatnonzero(occ).astype(np.int32)
                if c == "S":
                    inter = int((occ & occ_stream_frame).sum()); uni = int((occ | occ_stream_frame).sum())
                    rec["S_vs_streamframe_jaccard"] = inter / max(uni, 1); rec["S_streamframe_past5_raw"] = prf(occ_stream_frame, p_occ, meas)
                del m, q_
            flat = {}
            for k, v in rec.items():
                if k == "occ_maps":
                    for c, arr in v.items(): flat[f"occmap_{c}"] = arr
                else:
                    flat[k] = np.asarray(v)
            flat["past5_occ"] = np.flatnonzero(p_occ).astype(np.int32); flat["t"] = np.int64(t); flat["scale_S"] = s_scale; flat["scale_L"] = l_scale; flat["oracle_ratio"] = oracle_ratio
            tmp = op + ".tmp.npz"; np.savez_compressed(tmp, **flat); os.replace(tmp, op); nw += 1
            if nw % 25 == 0:
                print(f"  [{d[17:21]} s{a.shard}] {nw}/{len(files)} clips, {(time.time()-t0)/nw:.1f}s each", flush=True)
    print(f"[{d} s{a.shard}] {nw} clips, {(time.time()-t0)/60:.1f} min, firewall violations {len(audit.violations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
