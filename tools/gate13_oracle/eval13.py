#!/usr/bin/env python
"""Evaluate one Gate 13 checkpoint on KITTI-360 0006 (Gate 12's 590 anchors, regions and protocol).

    python tools/gate13_oracle/eval13.py --checkpoint <path> --device cuda:0 [--stride 1]

Identical to tools/gate12_suf/eval12.py (full-grid bf16 inference, occupied := apply_residual(0, res) > 0,
tp / fp / fn per region x anchor-camera range bin) except that the input comes from the M or O cache
and two regions are added: O_evidence / no_O_evidence (S_O or F_O present in the oracle input).
Regions: full (PRIMARY), past5 / unobs5_later (LiDAR five-sweep measured / not but later resolved),
observed20 / unobserved_later20, H_evidence / no_H_evidence (monocular input), O_evidence / no_O_evidence.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g13                                                                                   # noqa: E402
from g13 import g12                                                                          # noqa: E402
sys.path.insert(0, g13.REPO_ROOT)
import eval12 as E12                                                                         # noqa: E402  (Gate 12, frozen)
from core.datasets import kitti360_splits as SRC                                             # noqa: E402
from core.datasets.audit import FileAudit                                                    # noqa: E402
from core.model.net import load_checkpoint, apply_residual                                   # noqa: E402

DIMS, N = E12.DIMS, E12.N
REGIONS = E12.REGIONS + ("O_evidence", "no_O_evidence")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--stride", type=int, default=1); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False); cond = ck["meta"]["cond"]; assert cond in g13.CONDITIONS
    net = load_checkpoint(a.checkpoint, dev).net.eval(); rb = torch.from_numpy(E12.range_bins()).to(dev)
    reg_dir = os.path.join(g12.DATA_ROOT, "regions", g13.VAL_DRIVE)
    ids = sorted(int(f[:-4]) for f in os.listdir(reg_dir) if f.endswith(".npz"))[::a.stride]
    counts = np.zeros((len(ids), len(REGIONS), 6, 3), np.int64); t0 = time.time()
    u = lambda z, k: torch.from_numpy(np.unpackbits(z[k])[:N].astype(bool)).to(dev)
    with FileAudit(patterns=tuple(SRC.FORBIDDEN_PATH_TOKENS) + ("semantickitti", "occ3d", "nuscenes"), strict=True) as audit:
        for k, t in enumerate(ids):
            zm = np.load(g13.m_path(g13.VAL_DRIVE, t)); zo = np.load(g13.o_path(g13.VAL_DRIVE, t)); zr = np.load(os.path.join(reg_dir, f"{t:05d}.npz"))
            SH, FH, SO, FO = u(zm, "S_H"), u(zm, "F_H"), u(zo, "S_O"), u(zo, "F_O")
            S, F = (SH, FH) if cond == "M" else (SO, FO)
            x = torch.stack([S, torch.zeros_like(S), F]).float()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                res, _ = net(x.reshape((1, 3) + DIMS))
            pred = apply_residual(torch.zeros(DIMS, device=dev), res[0, 0].float()).reshape(-1) > 0
            valid, occ = u(zr, "valid"), u(zr, "occupied"); gt = occ & valid
            M = {"full": valid, "past5": u(zr, "past5"), "unobs5_later": u(zr, "unobs5_later"), "observed20": u(zr, "observed20"), "unobserved_later20": u(zr, "unobserved_later20"),
                 "H_evidence": valid & (SH | FH), "no_H_evidence": valid & ~(SH | FH), "O_evidence": valid & (SO | FO), "no_O_evidence": valid & ~(SO | FO)}
            for ri, r in enumerate(REGIONS):
                m = M[r]
                for c, sel in enumerate((pred & gt & m, pred & ~gt & m, ~pred & gt & m)):
                    counts[k, ri, :5, c] = torch.bincount(rb[sel], minlength=5).cpu().numpy(); counts[k, ri, 5, c] = int(sel.sum())
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.3f}s each", flush=True)
    s = counts.sum(0)
    summ = {r: {"iou": float(s[i, 5, 0] / max(s[i, 5].sum(), 1)), "precision": float(s[i, 5, 0] / max(s[i, 5, 0] + s[i, 5, 1], 1)), "recall": float(s[i, 5, 0] / max(s[i, 5, 0] + s[i, 5, 2], 1))}
            for i, r in enumerate(REGIONS)}
    out = a.out or os.path.join(g13.ART, "eval", os.path.basename(a.checkpoint)[:-3] + (f"_stride{a.stride}" if a.stride > 1 else ""))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out + "_counts.npz", ids=np.asarray(ids), counts=counts, regions=np.asarray(REGIONS), ranges=np.asarray(list(g12.RANGE_NAMES) + ["all"]))
    json.dump({"checkpoint": os.path.relpath(a.checkpoint, g13.REPO_ROOT), "checkpoint_sha256": g12.sha256_file(a.checkpoint), "cond": cond, "stride": a.stride, "n_anchors": len(ids), "threshold": 0.0,
               "summary": summ, "firewall": {"opened": len(audit.opened), "violations": audit.violations}, "seconds": time.time() - t0}, open(out + ".json", "w"), indent=1)
    print(f"[{os.path.basename(a.checkpoint)}] full IoU {summ['full']['iou']:.4f} P {summ['full']['precision']:.4f} R {summ['full']['recall']:.4f}; unobs5_later {summ['unobs5_later']['iou']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
