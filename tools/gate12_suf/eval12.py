#!/usr/bin/env python
"""Evaluate one Gate 12 checkpoint on KITTI-360 0006 (590 anchors): geometry completion occupancy.

    python tools/gate12_suf/eval12.py --checkpoint <path> --device cuda:0 [--stride 1]

Full-grid inference (256x256x32, bf16 autocast, replicate padding as trained); prediction
occupied := final log-odds > 0 with final = apply_residual(0, residual) (the frozen threshold).
Per anchor, tp / fp / fn over the unchanged target (valid, occupied) in regions x voxel-range bins:
  full               target valid (PRIMARY)
  past5 / unobs5_later         LiDAR-measured by the five causal sweeps / not, but resolved by the future sweeps
  observed20 / unobserved_later20   frozen data-scaling masks (20-sweep past window)
  H_evidence / no_H_evidence   S_H or F_H present in the causal input (condition-independent reference)
Range bins: anchor-camera z of the voxel centre, 0-5 / 5-10 / 10-20 / 20-30 / >30 m.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g12                                                                                  # noqa: E402
sys.path.insert(0, g12.REPO_ROOT)
from core.datasets import kitti360 as K3, kitti360_transforms as TF, kitti360_splits as SRC  # noqa: E402
from core.datasets.audit import FileAudit                                                   # noqa: E402
from core.model.net import load_checkpoint, apply_residual                                  # noqa: E402

DIMS = (256, 256, 32); N = int(np.prod(DIMS))
REGIONS = ("full", "past5", "unobs5_later", "observed20", "unobserved_later20", "H_evidence", "no_H_evidence")


def range_bins():
    calib = K3.parse_calibration(os.path.join(SRC.SSCBENCH_ROOT, "calibration")); Tgw = np.linalg.inv(np.asarray(calib.rect_cam_to_velo, np.float64))
    gi = np.stack(np.unravel_index(np.arange(N), DIMS), 1).astype(np.float64)
    ctr = (gi + 0.5) * TF.GRID.voxel_size + np.asarray(TF.GRID.origin); zc = (ctr @ Tgw[:3, :3].T + Tgw[:3, 3])[:, 2]
    return np.clip(np.digitize(zc, np.asarray(g12.RANGE_BINS[1:-1])), 0, 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--stride", type=int, default=1); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False); cond = ck["meta"]["cond"]
    net = load_checkpoint(a.checkpoint, dev).net.eval(); rb = torch.from_numpy(range_bins()).to(dev)
    ids = sorted(int(os.path.basename(f)[:-4]) for f in glob.glob(os.path.join(g12.DATA_ROOT, "regions", g12.VAL_DRIVE, "*.npz")))[::a.stride]
    counts = np.zeros((len(ids), len(REGIONS), 6, 3), np.int64); t0 = time.time()
    u = lambda z, k: torch.from_numpy(np.unpackbits(z[k])[:N].astype(bool)).to(dev)
    with FileAudit(patterns=tuple(SRC.FORBIDDEN_PATH_TOKENS) + ("semantickitti", "occ3d", "nuscenes"), strict=True) as audit:
        for k, t in enumerate(ids):
            zi = np.load(g12.input_path(g12.VAL_DRIVE, t)); zr = np.load(os.path.join(g12.DATA_ROOT, "regions", g12.VAL_DRIVE, f"{t:05d}.npz"))
            SH, FH = u(zi, "S_H"), u(zi, "F_H")
            x = torch.stack([SH, torch.zeros_like(SH), FH]) if cond == "H" else torch.stack([u(zi, "S"), u(zi, "U"), u(zi, "F")])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                res, _ = net(x.float().reshape((1, 3) + DIMS))
            pred = apply_residual(torch.zeros(DIMS, device=dev), res[0, 0].float()).reshape(-1) > 0
            valid, occ = u(zr, "valid"), u(zr, "occupied"); gt = occ & valid
            M = {"full": valid, "past5": u(zr, "past5"), "unobs5_later": u(zr, "unobs5_later"), "observed20": u(zr, "observed20"),
                 "unobserved_later20": u(zr, "unobserved_later20"), "H_evidence": valid & (SH | FH), "no_H_evidence": valid & ~(SH | FH)}
            for ri, r in enumerate(REGIONS):
                m = M[r]
                for c, sel in enumerate((pred & gt & m, pred & ~gt & m, ~pred & gt & m)):
                    counts[k, ri, :5, c] = torch.bincount(rb[sel], minlength=5).cpu().numpy(); counts[k, ri, 5, c] = int(sel.sum())
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.3f}s each", flush=True)
    s = counts.sum(0)
    summ = {r: {"iou": float(s[i, 5, 0] / max(s[i, 5].sum(), 1)), "precision": float(s[i, 5, 0] / max(s[i, 5, 0] + s[i, 5, 1], 1)), "recall": float(s[i, 5, 0] / max(s[i, 5, 0] + s[i, 5, 2], 1)),
                "by_range_iou": {g12.RANGE_NAMES[b]: float(s[i, b, 0] / max(s[i, b].sum(), 1)) for b in range(5)}} for i, r in enumerate(REGIONS)}
    out = a.out or os.path.join(g12.ART, "eval", os.path.basename(a.checkpoint)[:-3] + (f"_stride{a.stride}" if a.stride > 1 else ""))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out + "_counts.npz", ids=np.asarray(ids), counts=counts, regions=np.asarray(REGIONS), ranges=np.asarray(list(g12.RANGE_NAMES) + ["all"]))
    json.dump({"checkpoint": os.path.relpath(a.checkpoint, g12.REPO_ROOT), "checkpoint_sha256": g12.sha256_file(a.checkpoint), "cond": cond, "stride": a.stride, "n_anchors": len(ids),
               "threshold": 0.0, "summary": summ, "firewall": {"opened": len(audit.opened), "violations": audit.violations}, "seconds": time.time() - t0}, open(out + ".json", "w"), indent=1)
    print(f"[{os.path.basename(a.checkpoint)}] full IoU {summ['full']['iou']:.4f} P {summ['full']['precision']:.4f} R {summ['full']['recall']:.4f}; unobs5_later {summ['unobs5_later']['iou']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
