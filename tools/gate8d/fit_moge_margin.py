#!/usr/bin/env python
"""Gate 8D Phase 2.2: fit the MoGe free-space safety margin on KITTI-360 TRAIN drives.

A MoGe ray may carve free space only up to ``depth - margin(depth)``. The margin must be an
empirical property of the depth model on this sensor, not a number chosen by eye, so it is
the ``MOGE_MARGIN_QUANTILE`` quantile of ``|d_moge - d_lidar|`` inside each depth bin,
measured by projecting the synchronous LiDAR sweep into the rectified camera.

Fitted on drives 0003 / 0007 / 0010, then *validated* on 0006: the report states the
achieved coverage on the held-out drive, which is what tells you whether the margin
generalises. No target dataset is opened.

    python tools/gate8d/fit_moge_margin.py --frames 300 --device cuda:1
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT, default_device                               # noqa: E402
from gates.gate8c0 import transforms as TF                                             # noqa: E402
from gates.gate8d import protocol as P, sources as SRC                                 # noqa: E402
from sscbench_kitti360.audit import FileAudit                                    # noqa: E402


def moge_model(dev):
    from moge_gauge.calibrated import CalibratedMoGe
    from gates.scale_gate.config import load_config
    g5 = load_config("configs/gate5/moge_metric_gauge.yaml")
    return CalibratedMoGe(dev, moge_src=g5.moge.src_dir, hf_repo=g5.moge.hf_repo)


def lidar_depth_map(geo, nf, hw):
    """Sparse metric depth from the synchronous sweep, in the rectified camera."""
    pts = geo.read_velodyne(nf)[:, :3]
    T = np.linalg.inv(geo.rect_cam_to_velo)
    P_ = pts @ T[:3, :3].T + T[:3, 3]   # velodyne -> rectified camera
    z = P_[:, 2]
    keep = z > 1.0
    P_, z = P_[keep], z[keep]
    K = np.asarray(geo.calib.K, np.float64)          # P_rect_00[:3,:3], native lattice
    uv = P_ @ K.T
    u = (uv[:, 0] / uv[:, 2]).round().astype(int)
    v = (uv[:, 1] / uv[:, 2]).round().astype(int)
    H, W = hw
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    d = np.full((H, W), np.nan, np.float32)
    order = np.argsort(-z[ok])                       # nearest wins
    d[v[ok][order], u[ok][order]] = z[ok][order]
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=300, help="frames per drive")
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    from PIL import Image
    moge = moge_model(dev)
    bins = np.asarray(P.MOGE_MARGIN_DEPTH_BINS_M, float)
    err = {i: [] for i in range(len(bins) - 1)}
    val_err = {i: [] for i in range(len(bins) - 1)}
    stats = {"drives": {}, "quantile": P.MOGE_MARGIN_QUANTILE}
    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        for drive in SRC.ALL_DRIVES:
            is_val = drive == SRC.VAL_DRIVE
            geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
            # the calibrated horizontal FOV, from P_rect_00 -- never MoGe's own guess
            K0 = np.asarray(geo.calib.K, np.float64)
            W0 = int(geo.calib.native_hw[1])
            fov_x = float(np.degrees(2 * np.arctan(W0 / (2 * K0[0, 0]))))
            frames = sorted(geo.cam0_to_world)
            frames = frames[::max(1, len(frames) // a.frames)][:a.frames]
            n_ok = 0
            for nf in frames:
                ip = SRC.image_path(drive, nf)
                if not os.path.exists(ip):
                    continue
                try:
                    rgb = np.asarray(Image.open(ip).convert("RGB"), np.float32) / 255.0
                    dl = lidar_depth_map(geo, nf, rgb.shape[:2])
                except Exception:                                       # noqa: BLE001
                    continue
                o = moge.infer_calibrated(rgb.transpose(2, 0, 1), fov_x_deg=fov_x)
                dm = np.asarray(o.depth_z, np.float32)
                ok_m = np.asarray(o.mask, bool)
                if dm.shape != dl.shape:
                    continue
                m = (ok_m & np.isfinite(dl) & np.isfinite(dm)
                     & (dl > 1.0) & (dl < P.MOGE_MAX_DEPTH_M))
                if not m.any():
                    continue
                e = np.abs(dm[m] - dl[m]); d = dl[m]
                tgt = val_err if is_val else err
                for i in range(len(bins) - 1):
                    s = (d >= bins[i]) & (d < bins[i + 1])
                    if s.any():
                        tgt[i].append(e[s].astype(np.float32))
                n_ok += 1
            stats["drives"][drive] = {"frames_used": n_ok, "fov_x_deg": fov_x,
                                      "role": "val" if is_val else "train"}
            print(f"  {drive[17:21]}: {n_ok} frames", flush=True)
    margin = []
    for i in range(len(bins) - 1):
        if err[i]:
            v = np.concatenate(err[i])
            margin.append(float(max(P.MOGE_MIN_MARGIN_M,
                                    np.quantile(v, P.MOGE_MARGIN_QUANTILE))))
        else:
            margin.append(float(P.MOGE_MIN_MARGIN_M))
    cov = []
    for i in range(len(bins) - 1):
        v = np.concatenate(val_err[i]) if val_err[i] else np.zeros(1)
        cov.append(float((v <= margin[i]).mean()))
    out = {"depth_bins_m": bins.tolist(), "margin_m": margin,
           "val_drive_coverage": cov, "val_drive": SRC.VAL_DRIVE,
           "fitted_on": list(SRC.TRAIN_DRIVES), "stats": stats,
           "firewall": audit.summary(),
           "rule": ("margin(d) is the %.2f quantile of |MoGe - LiDAR| in the depth bin "
                    "containing d, floored at %.2f m; a ray carves free only to "
                    "d - margin(d) and leaves [d-margin, d+margin] unknown"
                    % (P.MOGE_MARGIN_QUANTILE, P.MOGE_MIN_MARGIN_M))}
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "moge_margin.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{'depth bin (m)':>18} {'margin (m)':>11} {'val coverage':>13}")
    for i in range(len(bins) - 1):
        hi = "inf" if bins[i + 1] > 1e8 else f"{bins[i+1]:.0f}"
        print(f"{f'{bins[i]:.0f}-{hi}':>18} {margin[i]:11.2f} {100*cov[i]:12.1f}%")
    print(f"\nfirewall violations: {len(out['firewall']['violations'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
