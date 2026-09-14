"""Build and cache the sparse evidence of every anchor of a drive (inside a strict file-access audit)."""
from __future__ import annotations

import hashlib
import os
import time
from typing import Dict

import numpy as np
import torch

from core.datasets import kitti360 as K3, kitti360_splits as SRC
from core.datasets.audit import FileAudit
from tools.baseline_reconstruction.current.geometry import LocalClip
from tools.semantic_completion import config as C
from tools.semantic_completion.evidence import lift as LF, trident as TR


def anchor_ids(drive: str, paths: C.Paths = C.Paths()) -> list:
    d = os.path.dirname(paths.input_path(drive, 0))
    return sorted(int(f[:-4]) for f in os.listdir(d) if f.endswith(".npz"))


def camera_to_grid() -> np.ndarray:
    """Anchor rectified camera -> anchor velodyne grid frame (metres), from the calibration files only."""
    calib = K3.parse_calibration(os.path.join(SRC.SSCBENCH_ROOT, "calibration"))
    return np.asarray(calib.rect_cam_to_velo, np.float64)


def build_drive(drive: str, device: str, shard: int = 0, shards: int = 1, paths: C.Paths = C.Paths()) -> Dict:
    dev = torch.device(device); torch.cuda.set_device(dev)
    T_cam_to_grid = camera_to_grid()
    ids = anchor_ids(drive, paths)[shard::shards]
    n_written = n_present = 0; sizes = []; t0 = time.time()
    with FileAudit(patterns=C.TRAIN_FORBIDDEN_TOKENS, strict=True) as audit:
        for t in ids:
            out = paths.evidence_path(drive, t)
            if os.path.exists(out):
                n_present += 1; continue
            clip = LocalClip.load(paths.clip_path(drive, t))
            with np.load(paths.input_path(drive, t)) as z:
                S = torch.from_numpy(np.unpackbits(z["S_H"])[:C.N_VOXELS].astype(bool)).to(dev)
                assert str(z["clip_sha256"]) == hashlib.sha256(open(paths.clip_path(drive, t), "rb").read()).hexdigest(), "input cache not built from this clip"
            tokens = [TR.load_token_map(paths.token_path(drive, str(k)), dev) for k in clip.keys]
            ev = LF.lift_clip_evidence(clip, tokens, S, T_cam_to_grid, dev)
            LF.save_evidence(ev, out, {"t": np.int64(t), "keys": np.asarray(clip.keys), "stream_indices": clip.stream_indices,
                                       "n_S": np.int64(int(S.sum())), "scale": np.float64(clip.scale_m_per_unit)})
            n_written += 1; sizes.append(len(ev.flat_index))
            if n_written % 100 == 0:
                print(f"  [{drive[17:21]} s{shard}] {n_written}/{len(ids)} {(time.time()-t0)/n_written:.2f}s each", flush=True)
    rec = {"drive": drive, "shard": shard, "n_anchors": len(ids), "n_written": n_written, "n_present": n_present,
           "evidence_voxels_mean": float(np.mean(sizes)) if sizes else None, "seconds": time.time() - t0,
           "firewall": {"opened": len(audit.opened), "violations": audit.violations, "forbidden": list(C.TRAIN_FORBIDDEN_TOKENS)}}
    print(f"[{drive} s{shard}] {n_written} written (+{n_present}), mean evidence voxels {rec['evidence_voxels_mean']}, {rec['seconds']/60:.1f} min", flush=True)
    return rec


__all__ = ["anchor_ids", "camera_to_grid", "build_drive"]
