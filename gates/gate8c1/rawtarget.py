"""Occupancy supervision rebuilt from raw KITTI-360 Velodyne sweeps.

Gate 8C-0 established that the published SSCBench-KITTI-360 completion label is not
geometrically consistent with its own sensor, so this module never reads it. The target is
built from the measurement itself, under the rules the Gate 8C-1 brief fixes:

* ray **endpoints** are occupied;
* ray **interiors** are free, carved from the near plane up to ``BAND_HALF_M`` before the
  endpoint -- the frozen Gate 7B rule, so the target's surface convention matches the
  map's -- and a voxel in which *this same sweep* recorded a return is never carved. That
  is not a conflict resolution: within one sweep a measured return at a voxel is a direct
  observation that the voxel is occupied, and a neighbouring ray grazing past it does not
  observe its interior as empty. Without this, grazing ground returns make roughly three
  quarters of all surface voxels self-conflicting and the target collapses;
* **nothing beyond an endpoint**: the space behind a surface stays unknown;
* a voxel with occupied evidence from one sweep and free evidence from **another** is
  **unknown**, never resolved by a heuristic or a vote -- that disagreement is real
  (dynamic objects, thin structure, occlusion boundaries) and is not ours to adjudicate;
* unobserved voxels are unknown and are excluded from the loss.

Because occupancy is written where a return actually landed, the target agrees with the
sweep by construction -- the property Gate 8C-0 found missing. ``n_obs`` and
``n_frames_occ`` are carried so single-return and repeatedly-observed targets can be
reported apart, and so a caller can require temporal support without changing the rule.

Future sweeps and ground-truth poses appear **only** here. Nothing in this module is
reachable from the inference path; ``tests/gate8c1`` asserts that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from gates.gate7b import rays as RY
from gates.gate8c0 import transforms as TF

#: Free-space carve starts here, in metres from the sensor.
CARVE_NEAR_M = RY.CARVE_NEAR_M
#: Carving stops this far before the endpoint, so a surface is not carved away.
BAND_HALF_M = RY.BAND_HALF_M
#: Keep one ray in this many for the free carve. Free space is spatially redundant; this
#: is the LiDAR analogue of Gate 7B's 4x4 pixel decimation and is never tuned.
CARVE_DECIMATION = 4
#: Chunk size for the ray march, in rays. Bounds peak memory, changes no result.
CHUNK = 20000

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
#: voxels this many cells above a column's topmost return are left UNKNOWN, so thin
#: structure just over a surface is never carved by the sky rule
SKY_MARGIN_VOX = 1
#: half-width, in voxels, of the neighbourhood a column may borrow a roof height from.
#: A column with no return of its own -- 43 % of the grid -- has no top to measure from,
#: so the plain column-wise rule cannot reach it and its sky stays unsupervised. Taking
#: the MAXIMUM column top over a window is the conservative choice: it frees only what is
#: above everything nearby, so a column beside a building is not carved through the
#: building. A window containing no return at all frees nothing.
SKY_NEIGHBOURHOOD_VOX = 7


@dataclass
class RawTarget:
    """One anchor's rebuilt supervision on the benchmark grid, flattened."""
    state: np.ndarray            # uint8, UNKNOWN / FREE / OCCUPIED
    n_obs: np.ndarray            # uint16, occupied returns that landed in the voxel
    n_frames_occ: np.ndarray     # uint8, distinct sweeps that marked it occupied
    n_free: np.ndarray           # uint16, free traversals
    frames_used: Sequence[int]
    stats: Dict[str, float]

    @property
    def occupied(self) -> np.ndarray:
        return self.state == OCCUPIED

    @property
    def free(self) -> np.ndarray:
        return self.state == FREE

    @property
    def valid(self) -> np.ndarray:
        """Supervised voxels: everything the sweeps actually resolved."""
        return self.state != UNKNOWN


def _flat(idx: torch.Tensor, dims) -> torch.Tensor:
    return (idx[:, 0] * int(dims[1]) + idx[:, 1]) * int(dims[2]) + idx[:, 2]


def _voxelize(pts: torch.Tensor, origin: torch.Tensor, vs: float, dims):
    idx = torch.floor((pts - origin) / vs).to(torch.int64)
    ok = torch.ones(len(idx), dtype=torch.bool, device=idx.device)
    for a in range(3):
        ok &= (idx[:, a] >= 0) & (idx[:, a] < int(dims[a]))
    return idx[ok], ok


def build(geo: TF.DriveGeometry, anchor_native: int, future_natives: Sequence[int],
          device, grid=TF.GRID, decimation: int = CARVE_DECIMATION) -> RawTarget:
    """Rebuild the occupancy target for ``anchor_native`` from the listed raw sweeps.

    ``future_natives`` must include the anchor itself and the future frames; ground-truth
    poses transform each sweep into the anchor's velodyne frame, which *is* the grid frame.
    """
    dims = tuple(int(d) for d in grid.dims)
    n = int(np.prod(dims))
    vs = float(grid.voxel_size)
    origin = torch.as_tensor(np.asarray(grid.origin, np.float64), device=device)
    n_occ = torch.zeros(n, dtype=torch.int32, device=device)
    n_fr = torch.zeros(n, dtype=torch.int32, device=device)
    n_free = torch.zeros(n, dtype=torch.int32, device=device)
    used, n_pts_total, n_pts_in = [], 0, 0
    for nf in future_natives:
        nf = int(nf)
        if nf not in geo.cam0_to_world:
            continue
        try:
            pts_np = geo.read_velodyne(nf)
        except (FileNotFoundError, OSError):
            continue
        T = np.eye(4) if nf == anchor_native else geo.velo_to_velo(nf, anchor_native)
        pts = torch.as_tensor(TF.apply(T, pts_np), device=device)
        sensor = torch.as_tensor(T[:3, 3], device=device)          # sensor origin in grid frame
        rng = (pts - sensor).norm(dim=1)
        keep = rng > CARVE_NEAR_M
        pts, rng = pts[keep], rng[keep]
        n_pts_total += int(len(pts))
        # ---- occupied: every endpoint that lands in the grid --------------------------
        idx, ok = _voxelize(pts, origin, vs, dims)
        n_pts_in += int(ok.sum())
        hit = torch.zeros(n, dtype=torch.bool, device=device)       # THIS sweep's returns
        if len(idx):
            f = _flat(idx, dims)
            n_occ.index_add_(0, f, torch.ones(len(f), dtype=torch.int32, device=device))
            hit[f] = True
            n_fr += hit.to(torch.int32)
        # ---- free: march the ray, stopping BAND_HALF_M before the endpoint ------------
        sub = pts[::decimation]
        srng = rng[::decimation]
        d = (sub - sensor) / srng.unsqueeze(1)                      # unit directions
        stop = (srng - BAND_HALF_M).clamp(min=0.0)
        sweep_free = torch.zeros(n, dtype=torch.int32, device=device)
        for c0 in range(0, len(sub), CHUNK):
            dd = d[c0:c0 + CHUNK]; ss = stop[c0:c0 + CHUNK]
            n_steps = int(torch.ceil(ss.max() / vs).item()) if len(ss) else 0
            if n_steps <= 0:
                continue
            step = torch.arange(n_steps, device=device, dtype=torch.float64) * vs
            t = CARVE_NEAR_M + step.unsqueeze(0)                    # [chunk, n_steps]
            m = t < ss.unsqueeze(1)
            p = sensor + dd.unsqueeze(1) * t.unsqueeze(2)
            fi, fok = _voxelize(p[m], origin, vs, dims)
            if len(fi):
                ff = _flat(fi, dims)
                sweep_free.index_add_(0, ff, torch.ones(len(ff), dtype=torch.int32,
                                                        device=device))
        # a voxel this sweep measured a return in is not carved by this sweep
        sweep_free[hit] = 0
        n_free += sweep_free
        used.append(nf)
    occ = n_occ > 0
    fre = n_free > 0
    state = torch.full((n,), UNKNOWN, dtype=torch.uint8, device=device)
    state[fre & ~occ] = FREE
    state[occ & ~fre] = OCCUPIED                                    # conflict stays UNKNOWN
    conflict = int((occ & fre).sum())

    # ---- open sky is FREE, not unknown -------------------------------------------------
    # A Velodyne ray never climbs above its topmost ring, so a voxel high over the road is
    # unmeasured. It is not *occluded*, though: nothing stands between it and the sensor,
    # and the column beneath it was measured all the way down to a surface. Leaving it
    # UNKNOWN drops it from the loss, and an unsupervised region is one the completion can
    # fill for free -- which is exactly the ceiling slab. This rule is derived from
    # KITTI-360 LiDAR geometry alone; no target dataset is consulted.
    X, Y, Z = (int(d) for d in dims)
    sv = state.view(X, Y, Z)
    # Every voxel a sweep put a return in -- including one the conflict rule left UNKNOWN.
    # Resolved OCCUPIED is not enough here. A conflicting voxel is still a real surface
    # detection, so it must raise the roof (or the sky is carved straight through it) and
    # it must never itself be carved (or a measured return is labelled FREE, which is the
    # contradiction Gate 8C-0 rejected SSCBench's own label for).
    measured = (n_occ > 0).view(X, Y, Z)
    zi = torch.arange(Z, device=device).view(1, 1, Z)
    neg = torch.full((X, Y, Z), -1, device=device, dtype=zi.dtype)
    top = torch.where(measured, zi.expand(X, Y, Z), neg).amax(dim=2)      # [X, Y]
    # a column with no return of its own borrows the highest roof in its neighbourhood
    k = 2 * SKY_NEIGHBOURHOOD_VOX + 1
    top_local = torch.nn.functional.max_pool2d(
        top.to(torch.float32)[None, None], kernel_size=k, stride=1,
        padding=SKY_NEIGHBOURHOOD_VOX)[0, 0].to(top.dtype)
    reachable = top_local >= 0                       # some return within the window
    # ``~measured`` is implied by the roof test once ``top`` counts every return, and is
    # kept explicit because it is the invariant the rule must not break.
    sky = ((zi > (top_local + SKY_MARGIN_VOX).unsqueeze(2))
           & reachable.unsqueeze(2) & (sv == UNKNOWN) & ~measured)
    n_sky = int(sky.sum())
    sv[sky] = FREE
    state = sv.reshape(-1)
    st = {"n_frames_used": len(used), "n_points_total": n_pts_total,
          "n_sky_free": n_sky, "sky_margin_vox": SKY_MARGIN_VOX,
          "sky_neighbourhood_vox": SKY_NEIGHBOURHOOD_VOX,
          "n_points_in_grid": n_pts_in,
          "n_occupied": int((state == OCCUPIED).sum()), "n_free": int((state == FREE).sum()),
          "n_unknown": int((state == UNKNOWN).sum()), "n_conflict_to_unknown": conflict,
          "conflict_fraction_of_touched": conflict / max(int((occ | fre).sum()), 1),
          "valid_fraction": float((state != UNKNOWN).sum() / n),
          "prevalence_in_valid": float((state == OCCUPIED).sum()
                                       / max(int((state != UNKNOWN).sum()), 1))}
    return RawTarget(state=state.cpu().numpy(), n_obs=n_occ.clamp(max=65535).to(torch.int32).cpu().numpy().astype(np.uint16),
                     n_frames_occ=n_fr.clamp(max=255).cpu().numpy().astype(np.uint8),
                     n_free=n_free.clamp(max=65535).to(torch.int32).cpu().numpy().astype(np.uint16),
                     frames_used=used, stats=st)


def load(path: str) -> Dict[str, np.ndarray]:
    """Read a cached target back: ``occupied``/``valid`` boolean volumes plus support counts.

    The on-disk schema stores the two masks bit-packed and the observation counts only for
    occupied voxels, which is why a target is ~160 kB rather than 12 MB.
    """
    with np.load(path, allow_pickle=False) as z:
        dims = tuple(int(x) for x in z["dims"]); n = int(np.prod(dims))
        occ = np.unpackbits(z["occupied"])[:n].astype(bool)
        val = np.unpackbits(z["valid"])[:n].astype(bool)
        rows = z["occ_rows"].astype(np.int64)
        n_obs = np.zeros(n, np.uint16); n_obs[rows] = z["n_obs"]
        n_fr = np.zeros(n, np.uint8); n_fr[rows] = z["n_frames_occ"]
        return {"occupied": occ, "valid": val, "n_obs": n_obs, "n_frames_occ": n_fr,
                "dims": dims, "stream_index": int(z["stream_index"]),
                "native_frame": int(z["native_frame"]),
                "future_stream_indices": z["future_stream_indices"].astype(np.int64),
                "future_natives": z["future_natives"].astype(np.int64),
                "frames_used": z["frames_used"].astype(np.int64)}


__all__ = ["RawTarget", "build", "load", "UNKNOWN", "FREE", "OCCUPIED", "CARVE_NEAR_M",
           "BAND_HALF_M", "CARVE_DECIMATION"]
