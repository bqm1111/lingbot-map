"""Exact metric distance to the frozen occupancy, and nothing else.

Two traps this module exists to avoid:

* **voxel indices are not metres.** Every distance is produced on the voxel lattice by an
  exact Euclidean transform and then multiplied by the *benchmark's own* voxel size
  (0.2 m for the KITTI family, 0.4 m for the official Occ3D grid);
* **the radius comparison is not naive.** ``1.2 / 0.2`` is ``5.999999999999999`` in binary
  floating point, so a voxel at exactly six voxels would fall outside a 1.2 m radius if the
  test were written the obvious way. Comparisons are therefore done on the *integer*
  squared lattice distance against a relatively-toleranced threshold.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

#: Relative slack on the squared-radius test. Chosen to absorb double-precision
#: representation error in ``r / voxel_size`` and nothing more: the smallest gap between
#: two distinct integer squared distances is 1, twelve orders of magnitude larger.
RADIUS_EPS = 1e-9


def edt(occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(d_vox, d2_int)``: exact Euclidean distance to the nearest ``True`` of ``occ``.

    ``d_vox`` is in **voxel units** and is ``+inf`` everywhere when ``occ`` is empty.
    ``d2_int`` is the exact integer squared lattice distance (``-1`` for the empty case),
    which is what every radius test uses.
    """
    occ = np.ascontiguousarray(occ, dtype=bool)
    if not occ.any():
        return (np.full(occ.shape, np.inf, np.float64),
                np.full(occ.shape, -1, np.int64))
    d = ndimage.distance_transform_edt(~occ)
    d2 = np.rint(d * d).astype(np.int64)
    return d, d2


def edt_with_source(occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """As :func:`edt`, plus the flat index of *one* exactly-nearest occupied voxel.

    The representative is only used for diagnostics and for the fast path of
    :func:`gate7a.transport.propagate`; the reported semantics always average **all**
    exactly-tied sources.
    """
    occ = np.ascontiguousarray(occ, dtype=bool)
    if not occ.any():
        return (np.full(occ.shape, np.inf, np.float64),
                np.full(occ.shape, -1, np.int64), None)
    d, ind = ndimage.distance_transform_edt(~occ, return_indices=True)
    d2 = np.rint(d * d).astype(np.int64)
    X, Y, Z = occ.shape
    src = (ind[0].astype(np.int64) * Y + ind[1].astype(np.int64)) * Z + ind[2].astype(np.int64)
    return d, d2, src


def metres(d_vox: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel-unit distance -> metres. The one place the conversion happens."""
    return d_vox * float(voxel_size)


def d2_threshold(radius_m: float, voxel_size: float) -> float:
    """The squared-lattice-distance threshold of a metric radius, with the fp slack."""
    rv = float(radius_m) / float(voxel_size)
    t = rv * rv
    return t * (1.0 + RADIUS_EPS) + RADIUS_EPS


def within_radius(d2_int: np.ndarray, radius_m: float, voxel_size: float) -> np.ndarray:
    """``True`` where the exact metric distance is ``<= radius_m``.

    ``d2_int < 0`` (the empty-base case) is never within any radius.
    """
    return (d2_int >= 0) & (d2_int <= d2_threshold(radius_m, voxel_size))


# --------------------------------------------------------------------------- #
# Offset shells, shared by the transport propagation
# --------------------------------------------------------------------------- #
_SHELL_CACHE: Dict[int, Dict[int, np.ndarray]] = {}


def shells(max_d2: int) -> Dict[int, np.ndarray]:
    """``{squared distance: (S, 3) integer offsets}`` for every shell up to ``max_d2``."""
    max_d2 = int(max_d2)
    hit = _SHELL_CACHE.get(max_d2)
    if hit is not None:
        return hit
    r = int(np.floor(np.sqrt(max_d2)))
    ax = np.arange(-r, r + 1)
    ox, oy, oz = np.meshgrid(ax, ax, ax, indexing="ij")
    off = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1)
    d2 = (off * off).sum(axis=1)
    keep = (d2 > 0) & (d2 <= max_d2)
    off, d2 = off[keep], d2[keep]
    order = np.argsort(d2, kind="stable")
    off, d2 = off[order], d2[order]
    bounds = np.searchsorted(d2, np.unique(d2))
    out = {}
    uq = np.unique(d2)
    for i, v in enumerate(uq):
        lo = bounds[i]
        hi = bounds[i + 1] if i + 1 < len(bounds) else len(d2)
        out[int(v)] = np.ascontiguousarray(off[lo:hi])
    _SHELL_CACHE[max_d2] = out
    return out


def histogram(d_m: np.ndarray, bin_m: float, max_m: float) -> np.ndarray:
    """Fixed-width metric histogram with a final overflow bin. Deterministic."""
    n = int(round(max_m / bin_m))
    if d_m.size == 0:
        return np.zeros(n + 1, np.int64)
    b = np.floor(np.asarray(d_m, np.float64) / bin_m).astype(np.int64)
    b = np.clip(b, 0, n)
    return np.bincount(b, minlength=n + 1).astype(np.int64)


def quantiles_from_histogram(hist: np.ndarray, qs, bin_m: float) -> Dict[str, float]:
    """Quantiles read off :func:`histogram`, reported at the bin's upper edge.

    Resolution-limited by ``bin_m`` and honest about it: a quantile is the smallest bin
    upper edge whose cumulative count reaches the requested fraction.
    """
    tot = int(hist.sum())
    if tot == 0:
        return {f"p{int(q*100)}": None for q in qs}
    cum = np.cumsum(hist)
    out = {}
    for q in qs:
        k = int(np.searchsorted(cum, q * tot, side="left"))
        out[f"p{int(q*100)}"] = float((k + 1) * bin_m)
    return out


__all__ = ["edt", "edt_with_source", "metres", "d2_threshold", "within_radius", "shells",
           "histogram", "quantiles_from_histogram", "RADIUS_EPS"]
