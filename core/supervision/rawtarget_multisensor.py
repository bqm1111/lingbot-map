"""The Gate 8C-1 occupancy supervision rule, applied to a platform with SEVERAL lasers.

``core/supervision/rawtarget.py`` carves free space from ONE sensor origin: the sweep's
points arrive in the sensor's own frame, so ``T[:3, 3]`` is that origin. KITTI-360 has a
single Velodyne and that is exact. Waymo has five lasers -- MEASURED origins in the vehicle
frame: TOP (1.430, 0.000, 2.184), FRONT (4.070, 0.000, 0.689), SIDE_LEFT
(3.245, 1.025, 0.979), SIDE_RIGHT (3.245, -1.025, 0.979), REAR (-1.155, 0.000, 0.464) --
and its cached point cloud is the concatenation of all five in the *vehicle* frame. Carving
those rays from one origin misdirects them by up to ~15 deg at the measured median planar
range of 8.4 m, which carves free space through real surfaces.

This module keeps the rule and changes only the unit of accumulation: a **sweep** is a
(frame, laser) pair with its own origin. Two properties of the original are preserved
deliberately:

* the "a voxel this sweep measured a return in is not carved by this sweep" rule is applied
  per **frame**, over the union of that frame's lasers. All five fire at one timestamp, so
  a return from any of them is a simultaneous direct observation; using the union is also
  the conservative choice, since it carves strictly less than a per-laser rule would.
* a voxel with occupied evidence from one sweep and free evidence from another stays
  UNKNOWN. Across lasers this is the same disagreement the original refuses to adjudicate.

:func:`resolve` below is a transcription of ``rawtarget.build``'s resolution and sky rule,
lines ``occ = n_occ > 0`` to the ``RawTarget`` return, with no logic changed.
``tools/waymo_arm/cli/check_equivalence.py`` asserts that this module reproduces
``rawtarget.build`` bit for bit on KITTI-360 anchors when every sweep is given the single
Velodyne origin -- the gate that keeps the two arms comparable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from core.supervision import rawtarget as RT
from core.supervision.rawtarget import (BAND_HALF_M, CARVE_DECIMATION, CARVE_NEAR_M, CHUNK,
                                        FREE, OCCUPIED, SKY_MARGIN_VOX,
                                        SKY_NEIGHBOURHOOD_VOX, UNKNOWN, RawTarget)


@dataclass
class Sweep:
    """One laser's returns at one timestamp, already in the GRID frame."""
    frame: int                 # the timestamp this sweep belongs to (the per-frame hit rule)
    sensor: np.ndarray         # [3] laser origin in the grid frame, metres
    points: np.ndarray         # [N, 3] returns in the grid frame, metres
    laser: str = ""            # for the record only


def accumulate(sweeps: Iterable[Sweep], dims, origin, voxel_size: float, device,
               decimation: int = CARVE_DECIMATION):
    """``(n_occ, n_frames_occ, n_free, frames_used, stats)`` over the grid, flattened.

    Sweeps are grouped by ``frame``; the carve of every laser in a frame is held back until
    the frame's returns are known, so the frame's own measurements are never carved.
    """
    n = int(np.prod([int(d) for d in dims]))
    vs = float(voxel_size)
    org = torch.as_tensor(np.asarray(origin, np.float64), device=device)
    n_occ = torch.zeros(n, dtype=torch.int32, device=device)
    n_fr = torch.zeros(n, dtype=torch.int32, device=device)
    n_free = torch.zeros(n, dtype=torch.int32, device=device)

    by_frame: Dict[int, List[Sweep]] = {}
    for s in sweeps:
        by_frame.setdefault(int(s.frame), []).append(s)

    used, n_pts_total, n_pts_in, n_sweeps = [], 0, 0, 0
    for fr in sorted(by_frame):
        hit = torch.zeros(n, dtype=torch.bool, device=device)        # THIS frame's returns
        frame_free = torch.zeros(n, dtype=torch.int32, device=device)
        for sw in by_frame[fr]:
            if len(sw.points) == 0:
                continue
            pts = torch.as_tensor(np.ascontiguousarray(sw.points), dtype=torch.float64, device=device)
            sensor = torch.as_tensor(np.asarray(sw.sensor, np.float64), device=device)
            rng = (pts - sensor).norm(dim=1)
            keep = rng > CARVE_NEAR_M
            pts, rng = pts[keep], rng[keep]
            if not len(pts):
                continue
            n_sweeps += 1
            n_pts_total += int(len(pts))
            # ---- occupied: every endpoint that lands in the grid ----------------------
            idx, ok = RT._voxelize(pts, org, vs, dims)
            n_pts_in += int(ok.sum())
            if len(idx):
                f = RT._flat(idx, dims)
                n_occ.index_add_(0, f, torch.ones(len(f), dtype=torch.int32, device=device))
                hit[f] = True
            # ---- free: march the ray, stopping BAND_HALF_M before the endpoint --------
            sub, srng = pts[::decimation], rng[::decimation]
            d = (sub - sensor) / srng.unsqueeze(1)
            stop = (srng - BAND_HALF_M).clamp(min=0.0)
            for c0 in range(0, len(sub), CHUNK):
                dd, ss = d[c0:c0 + CHUNK], stop[c0:c0 + CHUNK]
                n_steps = int(torch.ceil(ss.max() / vs).item()) if len(ss) else 0
                if n_steps <= 0:
                    continue
                step = torch.arange(n_steps, device=device, dtype=torch.float64) * vs
                t = CARVE_NEAR_M + step.unsqueeze(0)
                m = t < ss.unsqueeze(1)
                p = sensor + dd.unsqueeze(1) * t.unsqueeze(2)
                fi, _fok = RT._voxelize(p[m], org, vs, dims)
                if len(fi):
                    ff = RT._flat(fi, dims)
                    frame_free.index_add_(0, ff, torch.ones(len(ff), dtype=torch.int32,
                                                            device=device))
        # Added unconditionally: an empty frame contributes an all-false ``hit`` and an
        # all-zero carve, so this is a no-op for the counts and keeps ``frames_used``
        # identical to ``rawtarget.build``'s, which appends every frame it was given.
        n_fr += hit.to(torch.int32)
        frame_free[hit] = 0          # the frame's own returns are never carved by it
        n_free += frame_free
        used.append(fr)
    return n_occ, n_fr, n_free, used, {"n_points_total": n_pts_total,
                                       "n_points_in_grid": n_pts_in, "n_sweeps": n_sweeps}


def resolve(n_occ, n_fr, n_free, used: Sequence[int], dims, device, extra: Dict = None) -> RawTarget:
    """Transcription of ``rawtarget.build``'s resolution + open-sky rule. No logic changed."""
    n = int(np.prod([int(d) for d in dims]))
    occ, fre = n_occ > 0, n_free > 0
    state = torch.full((n,), UNKNOWN, dtype=torch.uint8, device=device)
    state[fre & ~occ] = FREE
    state[occ & ~fre] = OCCUPIED                                    # conflict stays UNKNOWN
    conflict = int((occ & fre).sum())

    X, Y, Z = (int(d) for d in dims)
    sv = state.view(X, Y, Z)
    measured = (n_occ > 0).view(X, Y, Z)
    zi = torch.arange(Z, device=device).view(1, 1, Z)
    neg = torch.full((X, Y, Z), -1, device=device, dtype=zi.dtype)
    top = torch.where(measured, zi.expand(X, Y, Z), neg).amax(dim=2)
    k = 2 * SKY_NEIGHBOURHOOD_VOX + 1
    top_local = torch.nn.functional.max_pool2d(
        top.to(torch.float32)[None, None], kernel_size=k, stride=1,
        padding=SKY_NEIGHBOURHOOD_VOX)[0, 0].to(top.dtype)
    reachable = top_local >= 0
    sky = ((zi > (top_local + SKY_MARGIN_VOX).unsqueeze(2))
           & reachable.unsqueeze(2) & (sv == UNKNOWN) & ~measured)
    n_sky = int(sky.sum())
    sv[sky] = FREE
    state = sv.reshape(-1)
    st = {"n_frames_used": len(used), "n_sky_free": n_sky, "sky_margin_vox": SKY_MARGIN_VOX,
          "sky_neighbourhood_vox": SKY_NEIGHBOURHOOD_VOX,
          "n_occupied": int((state == OCCUPIED).sum()), "n_free": int((state == FREE).sum()),
          "n_unknown": int((state == UNKNOWN).sum()), "n_conflict_to_unknown": conflict,
          "conflict_fraction_of_touched": conflict / max(int((occ | fre).sum()), 1),
          "valid_fraction": float((state != UNKNOWN).sum() / n),
          "prevalence_in_valid": float((state == OCCUPIED).sum()
                                       / max(int((state != UNKNOWN).sum()), 1))}
    st.update(extra or {})
    return RawTarget(state=state.cpu().numpy(),
                     n_obs=n_occ.clamp(max=65535).to(torch.int32).cpu().numpy().astype(np.uint16),
                     n_frames_occ=n_fr.clamp(max=255).cpu().numpy().astype(np.uint8),
                     n_free=n_free.clamp(max=65535).to(torch.int32).cpu().numpy().astype(np.uint16),
                     frames_used=used, stats=st)


def build(sweeps: Iterable[Sweep], dims, origin, voxel_size: float, device,
          decimation: int = CARVE_DECIMATION) -> RawTarget:
    n_occ, n_fr, n_free, used, extra = accumulate(sweeps, dims, origin, voxel_size, device,
                                                  decimation)
    return resolve(n_occ, n_fr, n_free, used, dims, device, extra)


__all__ = ["Sweep", "accumulate", "resolve", "build"]
