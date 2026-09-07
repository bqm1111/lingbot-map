"""Per-frame dataset for the depth-refinement head.

Everything is read from the Gate-0 caches, which fixes the pixel grid: RGB, LingBot
depth, LingBot confidence and projected metric LiDAR all live on the same 154x518
processed lattice produced by the validated ``mode="crop"`` transform.

Leakage discipline, enforced here rather than by convention:

* splits are by **complete sequence**, never by frame or clip;
* the model never receives sequence id, dataset id, filename, LiDAR counts or any
  ground-truth statistic -- only the tensors listed in :class:`Frame`;
* normalisation statistics are computed on training sequences and passed in, so
  sequence 08 cannot influence them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from gates.scale_gate.config import REPO_ROOT
from gates.scale_gate.kitti import read_manifest


@dataclass
class FrameRef:
    clip_id: str
    sequence: str
    frame_index: int          # position within the clip
    frame_id: int             # native KITTI frame id


class DepthRefineDataset(Dataset):
    """One item per frame. Metadata is returned for bookkeeping, never as model input."""

    def __init__(self, scale_cfg, depth_cfg, sequences: Sequence[str], split: str,
                 metric_scale: float, stats: Optional[Dict[str, float]] = None):
        self.cache = os.path.join(REPO_ROOT, scale_cfg.cache.root)
        self.rgb_dir = os.path.join(REPO_ROOT, depth_cfg.data.rgb_cache)
        self.metric_scale = float(metric_scale)
        self.stats = stats
        seqs = set(sequences)
        recs = read_manifest(os.path.join(REPO_ROOT, scale_cfg.experiment.output_dir,
                                          "manifests", f"{split}.jsonl"))
        self.frames: List[FrameRef] = []
        for r in recs:
            if r["sequence"] not in seqs:
                continue
            if not os.path.exists(os.path.join(self.cache, "lingbot", f"{r['clip_id']}.npz")):
                continue
            for i, fid in enumerate(r["frame_ids"]):
                self.frames.append(FrameRef(r["clip_id"], r["sequence"], i, int(fid)))
        if not self.frames:
            raise ValueError(f"no frames for sequences {sorted(seqs)} in split {split!r}")
        self._cache: Dict[str, tuple] = {}

    @property
    def sequences(self) -> List[str]:
        return sorted({f.sequence for f in self.frames})

    def _clip(self, clip_id: str):
        if clip_id not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            L = np.load(os.path.join(self.cache, "lingbot", f"{clip_id}.npz"), allow_pickle=False)
            D = np.load(os.path.join(self.cache, "lidar_depth", f"{clip_id}.npz"), allow_pickle=False)
            R = np.load(os.path.join(self.rgb_dir, f"{clip_id}.npz"), allow_pickle=False)
            self._cache[clip_id] = (L["pred_depth"], L["pred_depth_conf"],
                                    D["depth"], D["valid"], R["rgb"])
        return self._cache[clip_id]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        f = self.frames[i]
        dep, conf, gtd, val, rgb = self._clip(f.clip_id)
        t = f.frame_index
        d = torch.from_numpy(dep[t].astype(np.float32))[None]
        c = torch.from_numpy(conf[t].astype(np.float32))[None]
        g = torch.from_numpy(gtd[t].astype(np.float32))[None]
        v = torch.from_numpy(val[t].astype(bool))[None]
        im = torch.from_numpy(rgb[t].astype(np.float32) / 255.0)

        base = d * self.metric_scale                      # metric, before refinement
        valid_ling = torch.isfinite(base) & (base > 0)
        v = v & valid_ling & (g > 0)

        assert im.shape[-2:] == d.shape[-2:] == g.shape[-2:], "resolution mismatch"
        assert torch.isfinite(d).all() and torch.isfinite(c).all(), "non-finite cache entry"
        return {
            "rgb": im, "lingbot_depth": d, "lingbot_confidence": c,
            "valid_lingbot_mask": valid_ling, "metric_scale": torch.tensor(self.metric_scale),
            "base_depth": base,
            "projected_lidar_depth": g, "projected_lidar_valid_mask": v,
            "clip_id": f.clip_id, "frame_id": f.frame_id,
            "sequence": f.sequence, "frame_index": f.frame_index,
        }


def compute_stats(ds: DepthRefineDataset, n: int = 400, seed: int = 0) -> Dict[str, float]:
    """Normalisation statistics from *training* frames only."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    ld, cf = [], []
    for i in idx:
        s = ds[int(i)]
        b = s["base_depth"][s["valid_lingbot_mask"]]
        ld.append(torch.log(b.clamp_min(1e-6)))
        cf.append(s["lingbot_confidence"].reshape(-1))
    ld, cf = torch.cat(ld), torch.cat(cf)
    return {"log_depth_mean": float(ld.mean()), "log_depth_std": float(ld.std() + 1e-6),
            "conf_mean": float(cf.mean()), "conf_std": float(cf.std() + 1e-6),
            "n_frames": int(len(idx))}


def collate(batch):
    """Stack tensors; keep metadata as plain lists so it can never reach the model."""
    out = {}
    for k in batch[0]:
        v = batch[0][k]
        out[k] = torch.stack([b[k] for b in batch]) if torch.is_tensor(v) else [b[k] for b in batch]
    return out
