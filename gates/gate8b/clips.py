"""The matched five-frame setting: one fresh map per official clip, nothing carried over.

This is the closest thing to OccAny's input budget that our pipeline can produce: exactly
the five real frames of one official clip from one camera, LingBot run on those five
frames alone (the Gate 5.2/6 per-clip caches, never the causal stream), the metric scale
fixed from those same five frames (the pinned G51-B scalar of every prior gate), each frame
integrated exactly once into a fresh :class:`IncrementalMapper`, the map queried and the
completion applied after the fifth frame, and both ``ScaleState`` and map state discarded
before the next clip. The primary all-past streaming setting shares none of this except
the frozen models and the completion weights.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from gates.gate6 import frames as G6F, pipelines as G6P
from gates.gate8 import sources as S
from gates.gate8.mapper import FrameInput, IncrementalMapper


@dataclass
class Clip:
    dataset: str
    clip_id: str
    group: str
    frames: List[FrameInput]
    T_cam_to_grid: np.ndarray        # anchor (last) camera -> the benchmark grid frame
    pose_anchor: np.ndarray          # canonical camera-to-world of the anchor frame
    gt_ref: dict
    scale: float


class ClipFeed:
    """Official clips of one benchmark, each as five ``FrameInput`` from the per-clip cache."""

    def __init__(self, dataset: str, repo_root: str, device, with_semantics: bool = True):
        self.dataset, self.repo_root, self.device = dataset, repo_root, device
        self.recs = list(G6F.read_manifest(dataset, repo_root))
        self.scales = G6P.load_scales(dataset, repo_root)
        self.groups = G6P.load_groups(dataset, repo_root)
        self.with_semantics = with_semantics
        self._sk_T = None
        if dataset == "semantickitti":
            self._sk_T = {}

    def __len__(self) -> int:
        return len(self.recs)

    def _cam_to_grid(self, rec, z) -> np.ndarray:
        if self.dataset == "semantickitti":
            seq = str(rec.raw["sequence"])
            if seq not in self._sk_T:
                self._sk_T[seq] = S._sk_calib(self.repo_root, seq)[0]
            return np.asarray(self._sk_T[seq], np.float64)
        if self.dataset == "kitti360":
            return np.asarray(z["rect_cam_to_velo"], np.float64)
        return np.asarray(z["T_camera_to_ego"][-1], np.float64)

    def clip(self, i: int) -> Optional[Clip]:
        rec = self.recs[i]
        s = self.scales.get(rec.clip_id)
        if s is None:
            return None
        with np.load(G6F.lingbot_cache_path(self.dataset, rec.clip_id, self.repo_root)) as z:
            dep = z["pred_depth"].astype(np.float32); conf = z["pred_depth_conf"].astype(np.float32)
            K = z["pred_K"].astype(np.float64); pose = z["pred_pose_c2w"].astype(np.float64)
            T = self._cam_to_grid(rec, z)
        frames = []
        for j, key in enumerate(rec.keys):
            sem = None
            if self.with_semantics:
                p = G6F.semantic_cache_path(self.dataset, key)
                if os.path.exists(p):
                    with np.load(p) as zz:
                        sem = torch.from_numpy(zz["probs"].astype(np.float32)).permute(1, 2, 0)
            frames.append(FrameInput(
                index=j, depth_canonical=torch.from_numpy(dep[j]).to(self.device),
                conf=torch.from_numpy(conf[j]).to(self.device), K=K[j],
                pose_c2w_canonical=pose[j], log_scale_candidate=float(np.log(s)),
                sem_probs=sem))
        return Clip(self.dataset, rec.clip_id, str(self.groups.get(rec.clip_id, "")),
                    frames, T, pose[-1], rec.raw, float(s))


def build_map(clip: Clip, device, sem_into, n_teacher: int) -> IncrementalMapper:
    """Fresh mapper, scale fixed from the clip's own five frames, each integrated once."""
    m = IncrementalMapper(device, sem_into=sem_into, n_teacher=n_teacher)
    m.scale_state.force(clip.scale)
    for f in clip.frames:
        m.step(f)
    assert m.n_integrated == len(clip.frames)
    return m


def grid_to_world(clip: Clip, scale: float) -> np.ndarray:
    P = clip.pose_anchor.copy()
    P[:3, 3] *= float(scale)
    return P @ np.linalg.inv(clip.T_cam_to_grid)


__all__ = ["Clip", "ClipFeed", "build_map", "grid_to_world"]
