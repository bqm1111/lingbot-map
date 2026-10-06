"""Causal evidence accumulation for a chosen history length K, and the K ladder's evaluation.

K <= 5 uses only the anchor's own clip, so it is fully deployable and K=5 must reproduce the frozen evidence cache.
K > 5 needs evidence from earlier clips, and each clip is locally reset with its own origin and scale, so the two are
related by **GT poses**. Those rows are a registration UPPER BOUND and are labelled as such.

Accumulated evidence is kept on the anchor's own frozen hard map S_t. That is deliberate: it isolates *refinement of
the semantics of mapped voxels* from *extension of the map*, which is a geometry question and belongs to Gate 19.
So a longer history can improve the feature at an evidence voxel, but never invent a new one -- which is exactly the
question T3 asks, "if the evidence entering B0 gets better, does the completed region improve?".
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from core.datasets import kitti360_splits as SRC, kitti360_transforms as TF
from tools.baseline_reconstruction.current.geometry import LocalClip
from tools.delayed_semantic import config as C
from tools.semantic_completion.evidence import build as G15B, lift as LF, trident as TR
from tools.temporal_semantics import config as G18

DIMS = tuple(int(d) for d in C.GRID_DIMS)
VOXEL = float(C.VOXEL_SIZE_M)
ORIGIN = np.asarray(C.GRID_ORIGIN_M, np.float64)


def centres(flat: torch.Tensor) -> torch.Tensor:
    X, Y, Z = DIMS
    k = flat.long(); i = k // (Y * Z); j = (k // Z) % Y; l = k % Z
    o = torch.as_tensor(ORIGIN, dtype=torch.float64, device=flat.device)
    return (torch.stack([i, j, l], 1).double() + 0.5) * VOXEL + o


def voxel_of(pts: torch.Tensor) -> torch.Tensor:
    o = torch.as_tensor(ORIGIN, dtype=pts.dtype, device=pts.device)
    idx = torch.floor((pts - o) / VOXEL).long()
    dims = torch.as_tensor(DIMS, device=pts.device)
    ok = ((idx >= 0) & (idx < dims)).all(1)
    flat = (idx[:, 0] * DIMS[1] + idx[:, 1]) * DIMS[2] + idx[:, 2]
    return torch.where(ok, flat, torch.full_like(flat, -1))


def clip_frame_evidence(clip: LocalClip, token_maps: Sequence[torch.Tensor], slots: Sequence[int], device):
    """``(voxel flat [n], feature [n,1024] unit, frame_id [n])`` from the chosen clip slots only.

    Uses the frozen ``ray_endpoints`` so the accepted rays are exactly the mapper's."""
    lattice_hw = clip.depth_canonical.shape[1:]
    ends = LF.ray_endpoints(clip, device)
    keys, feats, fid = [], [], []
    for q in slots:
        k, pix = ends[q]
        keys.append(k)
        feats.append(LF.sample_at_lattice(token_maps[q], lattice_hw).reshape(-1, C.TEACHER_DIM)[pix])
        fid.append(torch.full((len(k),), int(clip.stream_indices[q]), dtype=torch.long, device=device))
    return torch.cat(keys), torch.cat(feats), torch.cat(fid)


def fuse(world_keys: torch.Tensor, feats: torch.Tensor, frame_ids: torch.Tensor, grid_keys: torch.Tensor,
         S_occupied: torch.Tensor, rule: str, text: torch.Tensor, device) -> Dict[str, torch.Tensor]:
    """Fuse observations onto the anchor's occupied voxels with one deterministic rule."""
    uq, inv = torch.unique(world_keys, return_inverse=True)
    n = len(uq)
    cnt = torch.zeros(n, dtype=torch.float32, device=device).index_add_(0, inv, torch.ones_like(inv, dtype=torch.float32))
    if rule in ("mean_feature", "count_weighted"):
        # every accepted ray carries one unit feature, so summing rays IS the count weighting; the two rules coincide
        # on this evidence and are reported as one. Kept separate in the config so the equality is explicit, not hidden.
        fsum = torch.zeros(n, C.TEACHER_DIM, device=device).index_add_(0, inv, feats)
        val = torch.nn.functional.normalize(fsum, dim=-1)
    elif rule == "latest":
        order = torch.argsort(frame_ids)                       # last write wins => most recent frame
        val = torch.zeros(n, C.TEACHER_DIM, device=device)
        val[inv[order]] = feats[order]
        val = torch.nn.functional.normalize(val, dim=-1)
    elif rule == "mean_probability":
        p = torch.softmax(C.LOGIT_SCALE * (feats @ text.T), dim=-1)
        psum = torch.zeros(n, p.shape[1], device=device).index_add_(0, inv, p)
        val = psum / cnt[:, None]                               # [n, 18] distribution, not a feature
    else:
        raise ValueError(rule)
    # distinct frames per voxel
    fu = torch.unique(torch.stack([inv, frame_ids]), dim=1)
    nfr = torch.zeros(n, dtype=torch.long, device=device).index_add_(0, fu[0], torch.ones_like(fu[0]))
    pos = torch.searchsorted(uq, grid_keys).clamp_(max=n - 1)
    hit = (uq[pos] == grid_keys) & S_occupied
    flat = hit.nonzero(as_tuple=True)[0]
    row = pos[flat]
    return {"flat_index": flat, "value": val[row], "count": cnt[row], "n_frames": nfr[row], "is_prob": rule == "mean_probability"}


def label_of(ev: Dict[str, torch.Tensor], text: torch.Tensor) -> torch.Tensor:
    """Frozen readout: argmax over the text embeddings, or over the averaged distribution."""
    v = ev["value"]
    return (v.argmax(-1) + 1) if ev["is_prob"] else TR.classify(v, text)


class HistoryBuilder:
    """Accumulated evidence at an anchor for any K, reusing the frozen caches."""

    def __init__(self, drive: str, device, paths: C.Paths = C.Paths()):
        self.drive, self.dev, self.paths = drive, device, paths
        self.geo = TF.DriveGeometry(drive, SRC.SSCBENCH_ROOT, SRC.KITTI360_ROOT)
        self.anchors = G15B.anchor_ids(drive, paths.g15)
        self.T_cam_to_grid = G15B.camera_to_grid()
        self.grid_keys = LF.grid_world_keys(np.linalg.inv(np.asarray(self.T_cam_to_grid, np.float64)), device)
        self._tok: Dict[str, torch.Tensor] = {}

    def tokens(self, key: str) -> torch.Tensor:
        if key not in self._tok:
            if len(self._tok) > 64:
                self._tok.clear()
            self._tok[key] = TR.load_token_map(self.paths.g15.token_path(self.drive, key), self.dev)
        return self._tok[key]

    def earlier_anchors(self, t: int, k: int) -> List[int]:
        """Anchors strictly before t whose clip covers a frame inside t-k+1 .. t-5 (only needed when k > 5)."""
        lo = t - k + 1
        return [a for a in self.anchors if a < t and a >= lo - 4]

    def build(self, t: int, k: int, rule: str, S: torch.Tensor, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        clip = LocalClip.load(self.paths.g15.clip_path(self.drive, t))
        toks = [self.tokens(str(key)) for key in clip.keys]
        n_slots = len(clip.stream_indices)
        slots = list(range(max(0, n_slots - k), n_slots))                    # the K most recent frames of this clip
        keys, feats, fid = clip_frame_evidence(clip, toks, slots, self.dev)
        if k > n_slots:
            for a in self.earlier_anchors(t, k):
                ep = self.paths.g15.evidence_path(self.drive, a)
                if not os.path.exists(ep):
                    continue
                with np.load(ep) as z:
                    fa = torch.from_numpy(z["flat_index"].astype(np.int64)).to(self.dev)
                    fe = torch.from_numpy(z["feature"].astype(np.float32)).to(self.dev)
                    ca = torch.from_numpy(z["count"].astype(np.int64)).to(self.dev)
                # anchor a's grid -> anchor t's grid, GT poses (ORACLE registration)
                T = TF.inv(self.geo.velo_to_world(self.stream_native(t))) @ self.geo.velo_to_world(self.stream_native(a))
                Tt = torch.as_tensor(T, device=self.dev, dtype=torch.float64)
                pts = centres(fa) @ Tt[:3, :3].T + Tt[:3, 3]
                fl = voxel_of(pts)
                ok = fl >= 0
                # one observation per contributing voxel, weighted by the rays behind it
                rep = ca[ok].clamp(max=8)
                keys = torch.cat([keys, LF.grid_world_keys(np.linalg.inv(np.asarray(self.T_cam_to_grid, np.float64)), self.dev)[fl[ok]].repeat_interleave(rep)])
                feats = torch.cat([feats, torch.nn.functional.normalize(fe[ok], dim=-1).repeat_interleave(rep, dim=0)])
                fid = torch.cat([fid, torch.full((int(rep.sum()),), a, dtype=torch.long, device=self.dev)])
        return fuse(keys, feats, fid, self.grid_keys, S, rule, text, self.dev)

    def stream_native(self, t: int) -> int:
        from tools.delayed_semantic.stream import DriveStream
        if not hasattr(self, "_stream"):
            self._stream = DriveStream(self.drive)
        return self._stream.frames[t].native


__all__ = ["HistoryBuilder", "clip_frame_evidence", "fuse", "label_of", "centres", "voxel_of"]
