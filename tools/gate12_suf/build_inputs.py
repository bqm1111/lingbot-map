#!/usr/bin/env python
"""Build the paired H and S/U/F input volumes from dense local-reset L clips. No LiDAR, no GT pose.

    python tools/gate12_suf/build_inputs.py --drive 2013_05_28_drive_0006_sync --device cuda:0 [--shard i --shards n] [--limit k] [--out dir]

Per anchor t (clip frames t-4..t), identical for both conditions: LingBot depth (canonical, cached
fp16), LingBot confidence, LingBot local poses T_q = inv(P_4) P_q (canonical; anchor camera =
identity), LingBot pred_K per frame, mapper scale forced to the causal MoGe-2 clip scale s_clip
(applied once, by the mapper, to depth and translation). The grid transform is the drive's camera
calibration rect_cam_to_velo (read from the calibration files only). Runs inside a strict
file-access audit that forbids LiDAR, GT poses, targets, evaluation masks and the target datasets.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g12                                                                                  # noqa: E402
sys.path.insert(0, g12.G11_DIR); sys.path.insert(0, g12.G9_DIR); sys.path.insert(0, g12.REPO_ROOT)
import g11 as G11                                                                           # noqa: E402  (frozen width function)
from channel_mapper import ChannelMapper                                                    # noqa: E402  (Gate 11)
from core.datasets import kitti360 as K3, kitti360_transforms as TF, kitti360_splits as SRC  # noqa: E402
from core.datasets.audit import FileAudit                                                   # noqa: E402
from core.mapping.mapper import FrameInput                                                  # noqa: E402

AUDIT_PATTERNS = tuple(SRC.FORBIDDEN_PATH_TOKENS) + ("data_poses", "velodyne", "data_3d_raw", "cam0_to_world", "lingbot_k360_7drives/targets",
                                                     "eval_masks", "lingbot_gate9_depth_occupancy", "lingbot_gate11_uncertainty", "semantickitti", "occ3d")
DIMS = TF.GRID.dims; NV = int(np.prod(DIMS))


def frames_for(z, conf_arrays):
    P = z["pred_pose_c2w"].astype(np.float64); Pinv = np.linalg.inv(P[4]); s = float(z["scale"])
    return [FrameInput(index=q, depth_canonical=torch.from_numpy(z["pred_depth"][q].astype(np.float32)), conf=torch.from_numpy(conf_arrays[q].astype(np.float32)),
                       K=z["pred_K"][q].astype(np.float64), pose_c2w_canonical=Pinv @ P[q], log_scale_candidate=float(np.log(s))) for q in range(5)]


def integrate(dev, z, frames, margins, intervals, grid, Tgw):
    """(S, F, U or None) on the grid. margins: per-frame carve stop margin (m); intervals: per-frame (u_front, u_back) or None."""
    s = float(z["scale"])
    mS = ChannelMapper(dev, "surface"); mS.scale_state.force(s)
    for q, fr in enumerate(frames):
        fr.depth_canonical = fr.depth_canonical.to(dev); fr.conf = fr.conf.to(dev)
        mS.set_pixels(margin=margins[q]); mS.step(fr)
    qS = mS.query(grid, Tgw); Sv, Fv = qS["occupied"], qS["free"]; del mS, qS
    if intervals is None:
        return Sv, Fv, None
    mU = ChannelMapper(dev, "interval"); mU.scale_state.force(s)
    for q, fr in enumerate(frames):
        mU.set_pixels(u_front=intervals[q][0], u_back=intervals[q][1]); mU.step(fr)
    Uv = ~Sv & ~Fv & (mU.query(grid, Tgw)["w_occ"] > 0)
    return Sv, Fv, Uv


def build_one(dev, z, rules, grid, Tgw):
    s = float(z["scale"]); conf = z["pred_depth_conf"].astype(np.float64)
    zm = s * z["pred_depth"].astype(np.float32).astype(np.float64)                    # the mapper's own metric depth
    H_ = (zm.shape[1], zm.shape[2])
    # ---- H: unchanged mapper
    S_H, F_H, _ = integrate(dev, z, frames_for(z, conf), [np.full(H_, g12.BAND_MARGIN_M)] * 5, None, grid, Tgw)
    # ---- SUF: rays < 20 m only, frozen U4 intervals, conservative carving
    conf_suf = np.where(zm < g12.RANGE_CUTOFF_M, conf, 0.0)
    W = [G11.widths("U4", zm[q], conf[q], rules) for q in range(5)]
    S, F, U = integrate(dev, z, frames_for(z, conf_suf), [W[q][0] for q in range(5)], W, grid, Tgw)
    acc = (conf >= 1.5) & (zm > 1.0) & (zm < 60.0)
    stats = {"n_rays_H": int(acc.sum()), "n_rays_SUF": int(((conf_suf >= 1.5) & (zm > 1.0) & (zm < 60.0)).sum()),
             "n_rays_cut_ge20": int((acc & (zm >= g12.RANGE_CUTOFF_M)).sum()), "n_suf_rays_ge20": int(((conf_suf >= 1.5) & (zm >= g12.RANGE_CUTOFF_M)).sum()),
             "u_front_mean_suf_rays": float(np.mean(np.concatenate([W[q][0][(conf_suf[q] >= 1.5) & (zm[q] > 1)] for q in range(5)]))),
             "u_back_mean_suf_rays": float(np.mean(np.concatenate([W[q][1][(conf_suf[q] >= 1.5) & (zm[q] > 1)] for q in range(5)])))}
    vols = {"S_H": S_H, "F_H": F_H, "S": S, "U": U, "F": F}
    vols = {k: v.cpu().numpy() for k, v in vols.items()}
    stats.update({f"n_{k}": int(v.sum()) for k, v in vols.items()})
    stats.update({"overlap_S_U": int((vols["S"] & vols["U"]).sum()), "overlap_S_F": int((vols["S"] & vols["F"]).sum()), "overlap_U_F": int((vols["U"] & vols["F"]).sum()),
                  "overlap_SH_FH": int((vols["S_H"] & vols["F_H"]).sum())})
    return vols, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1); ap.add_argument("--out", default=None)
    a = ap.parse_args(); d = a.drive
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    rules = g12.load_rules(); rules_sha = g12.sha256_file(g12.G11_RULES)
    calib = K3.parse_calibration(os.path.join(SRC.SSCBENCH_ROOT, "calibration"))
    Tgw = np.linalg.inv(np.asarray(calib.rect_cam_to_velo, np.float64)); grid = TF.GRID
    clips = sorted(glob.glob(os.path.join(os.path.dirname(g12.clip_path(d, 0)), "*.npz")))[a.shard::a.shards][:a.limit]
    out_dir = a.out or os.path.dirname(g12.input_path(d, 0)); os.makedirs(out_dir, exist_ok=True)
    t0 = time.time(); nw = 0
    with FileAudit(patterns=AUDIT_PATTERNS, strict=True) as audit:
        for cp in clips:
            t = int(os.path.basename(cp)[:-4]); op = os.path.join(out_dir, f"{t:05d}.npz")
            if os.path.exists(op):
                continue
            with np.load(cp) as zz:
                z = {k: zz[k] for k in zz.files}
            assert int(z["t"]) == t and list(z["stream_indices"]) == [t - 4, t - 3, t - 2, t - 1, t]
            vols, stats = build_one(dev, z, rules, grid, Tgw)
            rec = {"t": np.int64(t), "stream_indices": z["stream_indices"], "keys": z["keys"], "scale": np.float64(z["scale"]), "dims": np.asarray(DIMS, np.int32),
                   "rules_sha256": rules_sha, "clip_sha256": hashlib.sha256(open(cp, "rb").read()).hexdigest(), "stats": json.dumps(stats),
                   **{k: np.packbits(v) for k, v in vols.items()}}
            tmp = op + ".tmp.npz"; np.savez_compressed(tmp, **rec); os.replace(tmp, op); nw += 1
            if nw % 50 == 0:
                print(f"  [{d[17:21]} s{a.shard}] {nw}/{len(clips)} {(time.time()-t0)/nw:.2f}s each", flush=True)
    print(f"[{d} s{a.shard}] {nw} inputs, {(time.time()-t0)/60:.1f} min, files opened {len(audit.opened)}, firewall violations {len(audit.violations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
