"""The head's input: every channel the causal system has at time t, and nothing else.

The causal state itself is Gate 20's ``build_targets.causal_state``, imported unchanged. It runs inside a
``FileAudit`` with ``C.PREDICT_FORBIDDEN``, so LiDAR, GT poses, delayed targets, human labels and target
datasets cannot be opened while it is computed. This module only arranges that state into named channels.

Why each channel is here (section 9), and why the grouping matters:

    trunk          the frozen CompletionUNet decoder feature d1 -- the only representation in the stack
                   that has already reasoned about completed geometry at three resolutions
    hard_S/hard_F  the hard mapper's observed-surface and carved-free state: "observed / free / unknown"
    occupancy      the frozen completed occupancy O -- which voxels the model is being asked about
    height, range  the voxel's own place in the grid; both are pure functions of the index, so they add no
                   information about this scene, only about where in the sensor's frame the voxel sits
    evidence_mask  where causal semantic evidence exists
    evidence_dist  how far the nearest causal semantic evidence is, from B0's OWN EDT (not a new metric)
    evidence_proj  what that evidence says, as a learned projection of the unit teacher feature

``evidence_dist`` and ``evidence_mask`` are semantic-evidence channels, not geometric ones: both are
functions of where Trident-H evidence landed. A "geometry-only" control that used them would not be
geometry-only, which is why ``config.GEOMETRY_CHANNELS`` excludes them.

Nothing derived from a future frame appears anywhere in this file. ``channel_names`` is the complete list,
``tests/future_semantic_prior`` asserts it against a whitelist, and ``build_input`` takes no reveal mask.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as C
from tools.future_revealability.build_targets import causal_state
from tools.future_semantic_prior import config as G
from tools.future_semantic_prior.model import channel_spec

#: every channel name this gate can produce -- the audit list for the leakage test
ALL_CHANNEL_NAMES: Tuple[str, ...] = tuple(n for n, _ in G.GEOMETRY_CHANNELS + G.SEMANTIC_CHANNELS)


@lru_cache(maxsize=8)
def _static_channels(device_str: str) -> torch.Tensor:
    """``[2, X, Y, Z]``: voxel-centre height and horizontal range, normalised. Index-only, cached."""
    dev = torch.device(device_str)
    X, Y, Z = (int(d) for d in G.GRID_DIMS)
    ix, iy, iz = torch.meshgrid(torch.arange(X, device=dev), torch.arange(Y, device=dev),
                                torch.arange(Z, device=dev), indexing="ij")
    ox, oy, oz = C.GRID_ORIGIN_M
    cx = (ix.float() + 0.5) * C.VOXEL_SIZE_M + ox
    cy = (iy.float() + 0.5) * C.VOXEL_SIZE_M + oy
    cz = (iz.float() + 0.5) * C.VOXEL_SIZE_M + oz
    return torch.stack([cz / 2.0, torch.sqrt(cx * cx + cy * cy) / 25.6])


def causal_channels(st: Dict, head, device) -> Dict[str, torch.Tensor]:
    """Named dense channels ``[w, X, Y, Z]`` from one causal state. Keys are a superset of any control's."""
    dims = tuple(int(d) for d in G.GRID_DIMS)
    stat = _static_channels(str(device))
    dist = torch.from_numpy(np.minimum(st["dist_m"], 25.6)).to(device).reshape(dims)
    ch = {"trunk": st["trunk"],
          "hard_S": st["S"].float().reshape(dims)[None],
          "hard_F": st["F"].float().reshape(dims)[None],
          "occupancy": st["O"].float().reshape(dims)[None],
          "height": stat[0:1],
          "range": stat[1:2],
          "evidence_mask": st["E"].float().reshape(dims)[None],
          "evidence_dist": (torch.log1p(dist) / float(np.log1p(25.6)))[None]}
    if head.project_evidence is not None:
        ch["evidence_proj"] = head.scatter_evidence(st["ev"].flat_index, st["ev"].feature)
    return ch


def build_input(st: Dict, head, device) -> torch.Tensor:
    """``[1, cin, X, Y, Z]`` in the control's own channel order. ``C1_prior`` gets an empty placeholder."""
    spec = channel_spec(head.control)
    if not spec:
        return torch.zeros(1, 0, *[int(d) for d in G.GRID_DIMS], device=device)
    ch = causal_channels(st, head, device)
    parts = []
    for name, width in spec:
        x = ch[name]
        assert x.shape == (width, *[int(d) for d in G.GRID_DIMS]), f"channel {name} has shape {tuple(x.shape)}"
        parts.append(x)
    return torch.cat(parts, 0)[None]


@torch.no_grad()
def frozen_causal_state(drive: str, t: int, net, text, device, paths: C.Paths = C.Paths(),
                        audit: bool = True) -> Dict:
    """Gate 20's causal state plus the frozen trunk, under the same no-future file audit.

    ``audit=False`` exists only for the inner loop of training, where the audit is already open around the
    whole epoch; the file set opened is identical either way."""
    def _go() -> Dict:
        st = causal_state(drive, t, net, text, device, paths, want_trunk=True)
        assert st["trunk"].shape == (24, *[int(d) for d in G.GRID_DIMS]), tuple(st["trunk"].shape)
        return st
    if not audit:
        return _go()
    with FileAudit(patterns=G.INPUT_FORBIDDEN, strict=True):
        return _go()


#: stride of the per-step fingerprint's sampled sum. Coprime with every grid dimension and with 24 and 63,
#: so the sample is not confined to one channel or one slice of the grid.
_STRIDE = 7919


def fingerprint(x: torch.Tensor) -> Tuple[float, ...]:
    """A cheap EXACT fingerprint of a tensor, computed on the device in float64.

    Four reductions -- element count, sum, sum of squares and a strided partial sum -- are compared for
    exact equality. This is a mutation detector, not a hash: it has to catch an in-place write between the
    moment a target file is opened and the moment the backward pass finishes, and any such write changes at
    least one of these unless it is bit-for-bit a no-op. It costs a few milliseconds; a sha256 of the same
    tensor costs about a second, and the training loop runs this three thousand times."""
    d = x.detach()
    f = d.reshape(-1).double()
    return (float(d.numel()), float(f.sum()), float((f * f).sum()), float(f[::_STRIDE].sum()))


def state_fingerprint(st: Dict) -> Tuple[Tuple[float, ...], ...]:
    """``fingerprint`` of every causal quantity in one state, in a fixed order."""
    return tuple(fingerprint(st[k] if torch.is_tensor(st[k]) else torch.from_numpy(np.asarray(st[k])))
                 for k in ("O", "E", "S", "F", "dist_m", "b0", "trunk"))


def state_digest(st: Dict) -> str:
    """sha256 of every causal quantity the head can see. Used once per run, for the record; the per-step
    check uses ``state_fingerprint``, which does not move a gigabyte to the host three thousand times."""
    h = hashlib.sha256()
    for k in ("O", "E", "S", "F"):
        h.update(np.ascontiguousarray(st[k].cpu().numpy()).tobytes())
    h.update(np.ascontiguousarray(st["dist_m"]).tobytes())
    h.update(np.ascontiguousarray(st["b0"].cpu().numpy().astype(np.int32)).tobytes())
    h.update(np.ascontiguousarray(st["trunk"].float().cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


__all__ = ["ALL_CHANNEL_NAMES", "causal_channels", "build_input", "frozen_causal_state", "state_digest",
           "fingerprint", "state_fingerprint"]
