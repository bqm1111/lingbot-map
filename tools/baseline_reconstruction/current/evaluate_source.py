"""Source-side evaluation of one checkpoint on KITTI-360 0006 (590 anchors, stride-3 anchor set).

Same regions and range bins as Gates 12/13 so the result is directly comparable with Gate 13 M:

    full                target valid (PRIMARY)
    past5               measured by the five causal LiDAR sweeps (observed region)
    unobs5_later        not measured by them but resolved by the future sweeps (completion region)
    observed20 / unobserved_later20   the frozen 20-sweep masks
    H_evidence / no_H_evidence        S or F present in the causal input
    map_only            secondary: the raw map S as the prediction (no network)

Decision rule: occupied := final log-odds > OCCUPANCY_THRESHOLD (0) with base log-odds 0.
Range bins: anchor-camera z of the voxel centre.
"""
from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from core.datasets import kitti360 as K3, kitti360_splits as SRC, kitti360_transforms as TF
from core.datasets.audit import FileAudit
from core.model.net import apply_residual, load_checkpoint
from tools.baseline_reconstruction import config as C
from tools.baseline_reconstruction.current.dataset import DIMS, N_VOXELS
from tools.baseline_reconstruction.current.geometry import network_input
from tools.baseline_reconstruction.current.train import sha256_file
from tools.baseline_reconstruction.metrics import occupancy as M

REGIONS = ("full", "past5", "unobs5_later", "observed20", "unobserved_later20", "H_evidence", "no_H_evidence")
EVAL_FORBIDDEN = tuple(SRC.FORBIDDEN_PATH_TOKENS)


def kitti360_range_bins(device) -> torch.Tensor:
    """Range bin of every grid voxel by z in the anchor (rectified) camera, from the calibration only."""
    calib = K3.parse_calibration(os.path.join(SRC.SSCBENCH_ROOT, "calibration"))
    T_grid_to_cam = np.linalg.inv(np.asarray(calib.rect_cam_to_velo, np.float64))
    return torch.from_numpy(M.range_bins_for_grid(DIMS, TF.GRID.voxel_size, TF.GRID.origin, T_grid_to_cam)).to(device)


def _unpack(z, key, device) -> torch.Tensor:
    return torch.from_numpy(np.unpackbits(z[key])[:N_VOXELS].astype(bool)).to(device)


@torch.no_grad()
def predict(completer, x: torch.Tensor) -> torch.Tensor:
    """Flat boolean prediction on the grid from a ``[1, 3, X, Y, Z]`` input."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        res, _ = completer.net(x)
    residual = res[0, 0].float()
    final = apply_residual(torch.zeros_like(residual), residual)        # base log-odds 0 on whatever grid x lives on
    return final.reshape(-1) > C.OCCUPANCY_THRESHOLD


@dataclass
class SourceEvalRecord:
    checkpoint: str
    checkpoint_sha256: str
    seed: int
    step: int
    stride: int
    n_anchors: int
    threshold: float
    summary: Dict            # region -> counts + by_range
    map_only: Dict           # region "full" for the raw map S
    seconds: float
    firewall: Dict


def evaluate_checkpoint(checkpoint: str, device: str, paths: C.CurrentPaths = C.CurrentPaths(), stride: int = 1,
                        out_path: Optional[str] = None) -> SourceEvalRecord:
    dev = torch.device(device); torch.cuda.set_device(dev)
    meta = torch.load(checkpoint, map_location="cpu", weights_only=False)["meta"]
    completer = load_checkpoint(checkpoint, dev)
    rb = kitti360_range_bins(dev); n_bins = len(C.RANGE_NAMES)
    ids = sorted(int(os.path.basename(f)[:-4]) for f in glob.glob(os.path.join(paths.regions_root, C.VAL_DRIVE, "*.npz")))[::stride]
    acc = M.RegionAccumulator(REGIONS, n_bins); acc_map = M.RegionAccumulator(("full",), n_bins)
    t0 = time.time()
    with FileAudit(patterns=EVAL_FORBIDDEN, strict=True) as audit:
        for k, t in enumerate(ids):
            zi, zr = np.load(paths.input_path(C.VAL_DRIVE, t)), np.load(paths.region_path(t))
            S, F = _unpack(zi, "S_H", dev), _unpack(zi, "F_H", dev)
            pred = predict(completer, network_input(S, F, DIMS))
            valid, occ = _unpack(zr, "valid", dev), _unpack(zr, "occupied", dev); gt = occ & valid
            masks = {"full": valid, "past5": _unpack(zr, "past5", dev), "unobs5_later": _unpack(zr, "unobs5_later", dev),
                     "observed20": _unpack(zr, "observed20", dev), "unobserved_later20": _unpack(zr, "unobserved_later20", dev),
                     "H_evidence": valid & (S | F), "no_H_evidence": valid & ~(S | F)}
            for r in REGIONS:
                acc.add(r, M.counts_by_range(pred, gt, masks[r], rb, n_bins))
            acc_map.add("full", M.counts_by_range(S, gt, valid, rb, n_bins))
            acc.n_clips += 1
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(ids)} {(time.time()-t0)/(k+1):.3f}s each", flush=True)
    rec = SourceEvalRecord(checkpoint=os.path.relpath(checkpoint, C.REPO_ROOT), checkpoint_sha256=sha256_file(checkpoint),
                           seed=int(meta["seed"]), step=int(meta["step"]), stride=stride, n_anchors=len(ids),
                           threshold=C.OCCUPANCY_THRESHOLD, summary=acc.summary(), map_only=acc_map.summary(),
                           seconds=time.time() - t0, firewall={"opened": len(audit.opened), "violations": audit.violations})
    out_path = out_path or os.path.join(paths.source_eval, os.path.basename(checkpoint)[:-3] + (f"_stride{stride}" if stride > 1 else "") + ".json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(asdict(rec), open(out_path, "w"), indent=1)
    f = rec.summary["full"]
    print(f"[{os.path.basename(checkpoint)}] full IoU {f['iou']:.4f} P {f['precision']:.4f} R {f['recall']:.4f} | "
          f"unobs5_later {rec.summary['unobs5_later']['iou']:.4f} | map-only {rec.map_only['full']['iou']:.4f} ({len(ids)} anchors)", flush=True)
    return rec


__all__ = ["REGIONS", "kitti360_range_bins", "predict", "SourceEvalRecord", "evaluate_checkpoint"]
