#!/usr/bin/env python
"""Fit and freeze the 64-d projection of the frozen SAM-H embedding.

The projection is fitted on **KITTI-360 training drives only** (0003, 0007, 0010) from a
deterministic sample -- a fixed frame stride and every valid feature cell of each sampled
frame -- and is written once. Drive 0006 (source validation) is never read here, and
neither target dataset is reachable: the fit runs inside the gate's file-access audit.

Refusing to refit an existing basis is deliberate. The same mean, basis and normalisation
must reach every condition and every seed, so the only supported way to change it is to
delete it, which leaves a trace.

    python tools/feature_study/fit_pca.py --device cuda:0
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# study first: it pins G8C1_DATA_ROOT, which core.datasets reads at import time
from tools.feature_study import study as ST                                       # noqa: E402
from core.datasets.config import REPO_ROOT                                        # noqa: E402
from core.datasets import sources as S, kitti360_splits as SRC                    # noqa: E402
from core.datasets.audit import FileAudit                                         # noqa: E402
from tools.feature_study.sam_teacher import PATCH, preprocess_shape               # noqa: E402

#: every ``FRAME_STRIDE``-th frame of each training drive contributes cells
FRAME_STRIDE = 4


def valid_rows(native_hw, grid: int = ST.SAM_GRID) -> int:
    """Feature rows the image actually reaches; the rest of the 64 is SAM's pad."""
    h1, _ = preprocess_shape(tuple(int(x) for x in native_hw))
    return min(grid, int(np.ceil(h1 / PATCH)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stride", type=int, default=FRAME_STRIDE)
    a = ap.parse_args()
    ST.require_data_root()
    out = ST.pca_path()
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; the projection is frozen. Delete it to refit.")
    dev = torch.device(a.device)
    t0 = time.time()

    with FileAudit(patterns=SRC.FORBIDDEN_PATH_TOKENS, strict=True) as audit:
        keys, src_of = [], {}
        for drive in SRC.TRAIN_DRIVES:                    # 0003 / 0007 / 0010 ONLY
            src = SRC.source_of(drive)
            seg = [s for s in S.segments(src, REPO_ROOT) if s.name == drive][0]
            for f in list(seg.frames)[::a.stride]:
                keys.append(f.key); src_of[f.key] = src
        assert SRC.VAL_DRIVE not in {k[:4] for k in keys}, "drive 0006 must not be sampled"

        D = ST.SAM_NATIVE_DIM
        n = 0
        s1 = torch.zeros(D, dtype=torch.float64, device=dev)
        s2 = torch.zeros(D, D, dtype=torch.float64, device=dev)
        for k in keys:
            with np.load(ST.sam_feature_path(src_of[k], k)) as z:
                f = torch.from_numpy(z["feat"].astype(np.float32)).to(dev)
                r = valid_rows(z["native_hw"])
            x = f[:, :r, :].reshape(D, -1).T.double()                # [cells, 256]
            n += x.shape[0]
            s1 += x.sum(0)
            s2 += x.T @ x
        mean = s1 / n
        cov = s2 / n - torch.outer(mean, mean)
        cov = 0.5 * (cov + cov.T)
        evals, evecs = torch.linalg.eigh(cov)                        # ascending
        order = torch.argsort(evals, descending=True)
        evals, evecs = evals[order], evecs[:, order]
        basis = evecs[:, : ST.FEATURE_DIM]                           # [256, 64]
        total = float(evals.clamp_min(0).sum())
        kept = float(evals[: ST.FEATURE_DIM].clamp_min(0).sum())

    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp.npz"
    np.savez(tmp, mean=mean.cpu().numpy().astype(np.float32),
             basis=basis.cpu().numpy().astype(np.float32),
             eigenvalues=evals.cpu().numpy().astype(np.float64),
             fit_keys=np.asarray(keys), frame_stride=np.int32(a.stride),
             n_cells=np.int64(n))
    os.replace(tmp, out)
    rec = {"output": out, "sha256": ST.sha256_file(out),
           "native_dim": D, "feature_dim": ST.FEATURE_DIM,
           "fit_drives": list(SRC.TRAIN_DRIVES), "excluded": [SRC.VAL_DRIVE],
           "frame_stride": a.stride, "n_frames": len(keys), "n_cells": int(n),
           "explained_variance_fraction": kept / max(total, 1e-12),
           "top_eigenvalues": [float(x) for x in evals[:8].cpu().numpy()],
           "sam_checkpoint_sha256": ST.sha256_file(ST.SAM_CKPT),
           "fit_keys": keys, "seconds": time.time() - t0,
           "firewall": dict(audit.summary(), note="PCA fit ran inside a file-access audit"),
           "env": ST.environment()}
    ST.write_json(os.path.join(ST.ART, "pca_fit.json"), rec)
    print(f"PCA {D}->{ST.FEATURE_DIM} on {len(keys)} frames / {n} cells: "
          f"explained {100*kept/max(total,1e-12):.1f}%  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
