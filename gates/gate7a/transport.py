"""Semantic transport: give a newly added voxel the semantics of its nearest support.

The rule is Gate 6's, extended to larger distances: a voxel copies the **complete fused
probability vector** of its nearest occupied base voxel, and where several sources are
exactly equidistant their vectors are averaged. Ground truth never enters a class; the
oracle uses the target only to decide *which* voxels exist.

The implementation is exact rather than approximate. The distance field already says each
destination voxel's exact squared lattice distance, so the tied source set is precisely the
offset shell of that squared distance -- destinations are grouped by shell and each group
is gathered once.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from . import distance as _dist

#: Largest number of (destination, offset) pairs materialised at once.
CHUNK_PAIRS = 40_000_000


def source_lut(dims: Tuple[int, int, int], src_flat: np.ndarray, device) -> torch.Tensor:
    """``flat voxel -> row of src_probs``, ``-1`` where the base is not occupied."""
    n = int(dims[0]) * int(dims[1]) * int(dims[2])
    lut = torch.full((n,), -1, dtype=torch.int32, device=device)
    if len(src_flat):
        lut[torch.from_numpy(np.asarray(src_flat, np.int64)).to(device)] = \
            torch.arange(len(src_flat), dtype=torch.int32, device=device)
    return lut


def propagate_stats(dims: Tuple[int, int, int], src_flat: np.ndarray,
                    src_probs: torch.Tensor, dst_flat: np.ndarray, dst_d2: np.ndarray,
                    device) -> dict:
    """Propagate to ``dst_flat`` and return per-voxel statistics, not the vectors.

    Returns ``channel`` (argmax teacher channel), ``max_prob``, ``entropy`` (natural log,
    over the fused probability vector) and ``n_tied`` (how many exactly-nearest sources
    were averaged). Keeping only these keeps a 2-million-voxel destination set in a few
    megabytes instead of a few hundred.
    """
    D = int(len(dst_flat))
    C = int(src_probs.shape[1])
    out = {"channel": np.zeros(D, np.uint8), "max_prob": np.zeros(D, np.float32),
           "entropy": np.zeros(D, np.float32), "n_tied": np.zeros(D, np.int32),
           "assigned": np.zeros(D, bool)}
    if D == 0 or len(src_flat) == 0:
        return out

    X, Y, Z = (int(v) for v in dims)
    lut = source_lut((X, Y, Z), src_flat, device)
    sp = src_probs.to(device).float()

    d2 = np.asarray(dst_d2, np.int64)
    order = np.argsort(d2, kind="stable")
    d2s, dfs = d2[order], np.asarray(dst_flat, np.int64)[order]
    shells = _dist.shells(int(d2s.max()))

    starts = np.searchsorted(d2s, np.unique(d2s))
    uq = np.unique(d2s)
    for gi, val in enumerate(uq):
        lo = int(starts[gi])
        hi = int(starts[gi + 1]) if gi + 1 < len(starts) else len(d2s)
        offs = shells.get(int(val))
        if offs is None or hi <= lo:
            continue
        S = len(offs)
        ot = torch.from_numpy(offs.astype(np.int64)).to(device)          # [S, 3]
        step = max(1, CHUNK_PAIRS // max(S, 1))
        for a in range(lo, hi, step):
            b = min(a + step, hi)
            f = torch.from_numpy(dfs[a:b]).to(device)
            z = f % Z
            y = (f // Z) % Y
            x = f // (Z * Y)
            nx = x[:, None] + ot[None, :, 0]
            ny = y[:, None] + ot[None, :, 1]
            nz = z[:, None] + ot[None, :, 2]
            ok = ((nx >= 0) & (nx < X) & (ny >= 0) & (ny < Y) & (nz >= 0) & (nz < Z))
            nf = (nx.clamp(0, X - 1) * Y + ny.clamp(0, Y - 1)) * Z + nz.clamp(0, Z - 1)
            j = lut[nf.reshape(-1)].reshape(nf.shape)
            hit = ok & (j >= 0)
            cnt = hit.sum(dim=1)
            rows, cols = hit.nonzero(as_tuple=True)
            acc = torch.zeros(b - a, C, dtype=torch.float32, device=device)
            if rows.numel():
                acc.index_add_(0, rows, sp[j[rows, cols].long()])
            good = cnt > 0
            probs = torch.zeros_like(acc)
            probs[good] = acc[good] / cnt[good].unsqueeze(1).float()
            mx, ch = probs.max(dim=1)
            ent = -(probs.clamp_min(1e-12).log() * probs).sum(dim=1)
            sl = order[a:b]
            out["channel"][sl] = ch.to(torch.uint8).cpu().numpy()
            out["max_prob"][sl] = mx.cpu().numpy()
            out["entropy"][sl] = ent.cpu().numpy()
            out["n_tied"][sl] = cnt.to(torch.int32).cpu().numpy()
            out["assigned"][sl] = good.cpu().numpy()
    return out


def propagate_probs(dims, src_flat, src_probs, dst_flat, dst_d2, device) -> torch.Tensor:
    """The full ``[D, C]`` propagated vectors. Used by the tests and small cases only."""
    D, C = int(len(dst_flat)), int(src_probs.shape[1])
    outp = torch.zeros(D, C, dtype=torch.float32, device=device)
    if D == 0 or len(src_flat) == 0:
        return outp
    X, Y, Z = (int(v) for v in dims)
    lut = source_lut((X, Y, Z), src_flat, device)
    sp = src_probs.to(device).float()
    d2 = np.asarray(dst_d2, np.int64)
    shells = _dist.shells(int(d2.max()))
    for val in np.unique(d2):
        sel = np.flatnonzero(d2 == val)
        offs = shells.get(int(val))
        if offs is None:
            continue
        f = torch.from_numpy(np.asarray(dst_flat, np.int64)[sel]).to(device)
        ot = torch.from_numpy(offs.astype(np.int64)).to(device)
        z, y, x = f % Z, (f // Z) % Y, f // (Z * Y)
        nx, ny, nz = x[:, None] + ot[None, :, 0], y[:, None] + ot[None, :, 1], z[:, None] + ot[None, :, 2]
        ok = (nx >= 0) & (nx < X) & (ny >= 0) & (ny < Y) & (nz >= 0) & (nz < Z)
        nf = (nx.clamp(0, X - 1) * Y + ny.clamp(0, Y - 1)) * Z + nz.clamp(0, Z - 1)
        jj = lut[nf.reshape(-1)].reshape(nf.shape)
        hit = ok & (jj >= 0)
        cnt = hit.sum(dim=1)
        rows, cols = hit.nonzero(as_tuple=True)
        acc = torch.zeros(len(sel), C, dtype=torch.float32, device=device)
        if rows.numel():
            acc.index_add_(0, rows, sp[jj[rows, cols].long()])
        good = cnt > 0
        res = torch.zeros_like(acc)
        res[good] = acc[good] / cnt[good].unsqueeze(1).float()
        outp[torch.from_numpy(sel).to(device)] = res
    return outp


__all__ = ["propagate_stats", "propagate_probs", "source_lut", "CHUNK_PAIRS"]
