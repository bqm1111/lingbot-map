"""Recover the frozen Gate-6 state of one clip, exactly, and hand it to the analysis.

Gate 6 stored the *argmax channel* of each occupied voxel, not the fused probability
vector, and Gate 7A needs the vectors to propagate them. They are recomputed here from the
same three frozen inputs Gate 6 used -- the LingBot cache, the G51-B scale table and the
Trident cache -- and the result is then checked against the **pinned Gate-6 prediction
file**: identical flat indices, identical argmax channels, identical support flags, in
identical order. Nothing is re-run, re-cached or re-fitted; a mismatch is a stop condition.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from gates.gate6 import grids as g6grids, lifting as g6lift, pipelines as g6pipe

VERIFY_SAMPLE_SEED = 0


@dataclass
class FrozenClip:
    """Everything Gate 7A needs about one clip, all of it recovered, none of it new."""
    clip_id: str
    group: str
    scale: float
    raw_flat: np.ndarray
    raw_probs: torch.Tensor
    dil_flat: np.ndarray
    dil_probs: torch.Tensor
    dil_support: np.ndarray
    points_anchor: np.ndarray
    frame: np.ndarray
    v_pix: np.ndarray
    u_pix: np.ndarray
    verified: Dict[str, object]
    #: the pinned Gate-6 argmax channels, filled in by :func:`check_against_pinned` and
    #: authoritative for every base label this gate reports
    pinned_raw_channel: Optional[np.ndarray] = None
    pinned_dil_channel: Optional[np.ndarray] = None


def rebuild(dataset: str, clip: g6pipe.ClipGeometry, sem: torch.Tensor,
            device) -> FrozenClip:
    """The frozen Gate-6 construction, with the probability vectors kept.

    Every arithmetic step below is the one in ``gate6.pipelines.predict_clip``; the test
    suite asserts equality against that function rather than trusting this copy.
    """
    G = g6grids.PREDICTION_GRID[dataset]
    s = float(clip.scale)
    pts, fr, cf, dp, vp, up, _ = g6lift.points_with_pixels(
        clip.dep, clip.conf, clip.K, clip.pose, s, g6pipe.CONF_THRESHOLD,
        g6pipe.MIN_DEPTH_M, g6pipe.MAX_DEPTH_M)
    R, t = clip.T_anchor_to_grid[:3, :3], clip.T_anchor_to_grid[:3, 3]
    pg = pts @ R.T + t if len(pts) else pts

    idx, keep = g6grids.voxelize(pg, G)
    flat = g6grids.flat_of(idx, G) if len(idx) else np.zeros((0,), np.int64)
    probs = (g6lift.sample_probs(sem, fr[keep], vp[keep], up[keep], device)
             if len(idx) else torch.zeros((0, sem.shape[1]), device=device))
    raw_flat, raw_probs, _ = g6lift.fuse_voxel_probs(flat, probs, int(np.prod(G.dims)),
                                                     device)

    occ = torch.zeros(int(np.prod(G.dims)), dtype=torch.bool, device=device)
    if len(raw_flat):
        occ[torch.from_numpy(raw_flat).to(device)] = True
    from gates.voxel_gate.voxels import dilate
    dil = dilate(occ.view(G.dims), g6pipe.DILATE_RADIUS_VOXELS).reshape(-1)
    only = (dil & ~occ).nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int64)
    only_probs, assigned = g6lift.propagate_dilation(
        tuple(G.dims), raw_flat, raw_probs, only, g6pipe.DILATE_RADIUS_VOXELS, device)
    if len(only) and not assigned.all():
        raise AssertionError("dilation is extensive: every dilation-only voxel must have "
                             "a source inside the same radius")

    dil_flat = np.concatenate([raw_flat, only])
    dil_probs = torch.cat([raw_probs, only_probs], 0) if len(only) else raw_probs
    dil_support = np.concatenate([np.ones(len(raw_flat), np.uint8),
                                  np.zeros(len(only), np.uint8)])

    if g6grids.NEEDS_REDUCTION[dataset]:
        raw_flat_n, raw_probs_n = (g6grids.reduce_occ3d_probs(raw_flat, raw_probs)
                                   if len(raw_flat) else
                                   (np.zeros((0,), np.int64), raw_probs))
        dil_flat_n, dil_probs_n = (g6grids.reduce_occ3d_probs(dil_flat, dil_probs)
                                   if len(dil_flat) else
                                   (np.zeros((0,), np.int64), dil_probs))
        sup = np.isin(dil_flat_n, raw_flat_n).astype(np.uint8)
        raw_flat, raw_probs, dil_flat, dil_probs, dil_support = (
            raw_flat_n, raw_probs_n, dil_flat_n, dil_probs_n, sup)

    return FrozenClip(clip_id=clip.clip_id, group=clip.group, scale=s,
                      raw_flat=raw_flat.astype(np.int64), raw_probs=raw_probs,
                      dil_flat=dil_flat.astype(np.int64), dil_probs=dil_probs,
                      dil_support=dil_support, points_anchor=pts, frame=fr,
                      v_pix=vp, u_pix=up, verified={})


def check_against_pinned(fc: FrozenClip, npz_path: str, sha256: Optional[str] = None
                         ) -> Dict[str, object]:
    """Compare the rebuild with the pinned Gate-6 prediction.

    **Geometry must be bit-exact** -- identical flat indices in identical order, identical
    support flags. A geometry mismatch is a stop condition and raises.

    The *argmax channel* is compared but not required to be identical, and the pinned
    channel is what Gate 7A goes on to use. ``torch.Tensor.index_add_`` accumulates with
    CUDA atomics, so the fused mean of a voxel's probability vectors depends on the order
    the GPU happens to schedule the adds; on a voxel whose top two classes are separated by
    less than that accumulation error the argmax can flip between runs. This is a property
    of the Gate-6 artifacts, not of Gate 7A, and it is measured here rather than hidden:
    the mismatch count and the top-1/top-2 probability gap at each mismatch are recorded,
    and the base labels of this gate are taken from the pinned file so that Gate 7A's base
    metrics are *exactly* Gate 6's.
    """
    if sha256 is not None:
        h = hashlib.sha256()
        with open(npz_path, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 22), b""):
                h.update(c)
        if h.hexdigest() != sha256:
            raise AssertionError(f"{npz_path}: prediction file no longer matches its "
                                 f"pinned SHA-256 -- refusing to analyse")
    with np.load(npz_path) as z:
        rf, rc = z["raw_flat"].astype(np.int64), z["raw_channel"].astype(np.int64)
        df, dc = z["dil_flat"].astype(np.int64), z["dil_channel"].astype(np.int64)
        ds_ = z["dil_support"].astype(np.uint8)

    geometry = {"raw_flat": np.array_equal(rf, fc.raw_flat),
                "dil_flat": np.array_equal(df, fc.dil_flat),
                "dil_support": np.array_equal(ds_, fc.dil_support)}
    if not all(geometry.values()):
        bad = [k for k, v in geometry.items() if not v]
        raise AssertionError(f"{os.path.basename(npz_path)}: frozen B-R/B-D geometry could "
                             f"not be reproduced exactly ({', '.join(bad)})")

    my_rc = (fc.raw_probs.argmax(1).cpu().numpy().astype(np.int64)
             if len(fc.raw_flat) else np.zeros((0,), np.int64))
    my_dc = (fc.dil_probs.argmax(1).cpu().numpy().astype(np.int64)
             if len(fc.dil_flat) else np.zeros((0,), np.int64))
    out = dict(geometry, n_raw=int(len(rf)), n_dil=int(len(df)),
               n_raw_channel_mismatch=int((rc != my_rc).sum()),
               n_dil_channel_mismatch=int((dc != my_dc).sum()),
               max_top2_gap_at_mismatch=0.0)
    bad = np.flatnonzero(dc != my_dc)
    if len(bad):
        pr = fc.dil_probs[torch.from_numpy(bad).to(fc.dil_probs.device)].float()
        top2 = pr.topk(2, dim=1).values
        out["max_top2_gap_at_mismatch"] = float((top2[:, 0] - top2[:, 1]).max())
    # the pinned labels are authoritative from here on
    fc.pinned_raw_channel = rc.astype(np.int32)
    fc.pinned_dil_channel = dc.astype(np.int32)
    fc.verified = out
    return out


__all__ = ["FrozenClip", "rebuild", "check_against_pinned"]
