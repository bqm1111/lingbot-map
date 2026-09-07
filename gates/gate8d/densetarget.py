"""Gate 8D: privileged supervision fused from dense future KITTI-360 evidence.

Gate 8C-1 built its target from 21 stride-5 LiDAR sweeps and nothing else, which left most
of the volume UNKNOWN and therefore outside the loss. Gate 8D keeps the horizon and the
evidence *rules* identical and makes the evidence dense:

* **LiDAR** -- every native sweep inside the same horizon, not one in five;
* **MoGe-2** -- metric depth on future images, carving free space only to
  ``depth - margin(depth)`` where the margin is a quantile of the model's own error fitted
  on the training drives;
* **Trident-H** -- rays through pixels the frozen teacher calls sky, carved to the grid
  boundary.

Precedence is strict and is the whole safety argument: a LiDAR endpoint can never be
overwritten by a pseudo-teacher, a genuine cross-source conflict becomes UNKNOWN, and no
rule ever converts unknown space to free because of where it sits. There is deliberately no
column-height or neighbourhood-height propagation anywhere in this module.

Future sweeps, future images and ground-truth poses appear only here. Nothing in this module
is reachable from the inference path; ``tests/gate8d`` asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from gates.gate7b import rays as RY
from gates.gate8c0 import transforms as TF
from gates.gate8d import protocol as P

UNKNOWN, FREE, OCCUPIED = 0, 1, 2

CARVE_NEAR_M = P.CARVE_NEAR_M
BAND_HALF_M = P.BAND_HALF_M
CARVE_DECIMATION = P.CARVE_DECIMATION
CHUNK = 20000


@dataclass
class DenseTarget:
    state: np.ndarray            # uint8, UNKNOWN / FREE / OCCUPIED
    tudf_m: np.ndarray           # float16 metres, truncated unsigned distance to a surface
    tudf_valid: np.ndarray       # bool, where the distance is trustworthy
    n_obs: np.ndarray            # uint16, LiDAR endpoint hits
    n_frames_occ: np.ndarray     # uint8, distinct sweeps that saw an endpoint
    w_pseudo_free: np.ndarray    # float16, accumulated MoGe + sky free weight
    frames_used: Sequence[int]
    stats: Dict

    @property
    def occupied(self) -> np.ndarray:
        return self.state == OCCUPIED

    @property
    def free(self) -> np.ndarray:
        return self.state == FREE

    @property
    def valid(self) -> np.ndarray:
        return self.state != UNKNOWN


def _flat(idx: torch.Tensor, dims) -> torch.Tensor:
    X, Y, Z = (int(d) for d in dims)
    return (idx[:, 0] * Y + idx[:, 1]) * Z + idx[:, 2]


def _voxelize(pts: torch.Tensor, origin: torch.Tensor, vs: float, dims):
    idx = torch.floor((pts - origin) / vs).to(torch.int64)
    X, Y, Z = (int(d) for d in dims)
    ok = ((idx[:, 0] >= 0) & (idx[:, 0] < X) & (idx[:, 1] >= 0) & (idx[:, 1] < Y)
          & (idx[:, 2] >= 0) & (idx[:, 2] < Z))
    return idx[ok], ok


def _march(sensor, dirs, t_start, t_stop, origin, vs, dims, device, out, weight=1.0):
    """Accumulate ``weight`` into every voxel a ray crosses between two ranges."""
    n_steps = int(np.ceil(float(t_stop.max().item() - t_start) / vs)) + 1
    if n_steps <= 0:
        return
    step = torch.arange(n_steps, device=device, dtype=torch.float64) * vs
    for s in range(0, dirs.shape[0], CHUNK):
        dd = dirs[s:s + CHUNK]
        ss = t_stop[s:s + CHUNK]
        t = t_start + step.unsqueeze(0)
        m = t < ss.unsqueeze(1)
        if not m.any():
            continue
        p = sensor.view(1, 1, 3) + dd.unsqueeze(1) * t.unsqueeze(2)
        fi, _ = _voxelize(p[m], origin, vs, dims)
        if len(fi):
            ff = _flat(fi, dims)
            out.index_add_(0, ff, torch.full((len(ff),), float(weight),
                                             dtype=out.dtype, device=device))


def _margin_lookup(depth: torch.Tensor, bins, margins) -> torch.Tensor:
    b = torch.as_tensor(np.asarray(bins[1:], np.float64), device=depth.device)
    m = torch.as_tensor(np.asarray(margins, np.float64), device=depth.device)
    return m[torch.searchsorted(b, depth.double(), right=False).clamp(max=len(m) - 1)]


def build(geo: TF.DriveGeometry, anchor_native: int, dense_natives: Sequence[int],
          device, grid=TF.GRID, image_natives: Optional[Sequence[int]] = None,
          moge: Optional[Dict[int, Dict]] = None,
          sky: Optional[Dict[int, Dict]] = None,
          margin: Optional[Dict] = None,
          decimation: int = CARVE_DECIMATION,
          min_occ_frames: int = None) -> DenseTarget:
    """Fuse LiDAR, MoGe and sky evidence for one anchor into one supervision volume."""
    dims = tuple(int(d) for d in grid.dims)
    n = int(np.prod(dims))
    vs = float(grid.voxel_size)
    origin = torch.as_tensor(np.asarray(grid.origin, np.float64), device=device)
    n_occ = torch.zeros(n, dtype=torch.int32, device=device)
    n_fr = torch.zeros(n, dtype=torch.int32, device=device)
    n_free = torch.zeros(n, dtype=torch.int32, device=device)
    w_pf = torch.zeros(n, dtype=torch.float32, device=device)
    n_moge_views = torch.zeros(n, dtype=torch.int16, device=device)
    n_sky_views = torch.zeros(n, dtype=torch.int16, device=device)
    used, n_pts_total, n_pts_in, n_missing = [], 0, 0, 0

    # ---- 1. LiDAR: every native sweep in the horizon --------------------------------
    for nf in dense_natives:
        nf = int(nf)
        if nf not in geo.cam0_to_world:
            n_missing += 1
            continue
        try:
            pts_np = geo.read_velodyne(nf)
        except (FileNotFoundError, OSError):
            n_missing += 1
            continue
        T = torch.as_tensor(geo.velo_to_velo(nf, int(anchor_native)), device=device)
        pts = torch.as_tensor(pts_np[:, :3].astype(np.float64), device=device)
        pts = pts @ T[:3, :3].T + T[:3, 3]
        n_pts_total += len(pts)
        idx, ok = _voxelize(pts, origin, vs, dims)
        n_pts_in += len(idx)
        sensor = T[:3, 3]
        sweep_free = torch.zeros(n, dtype=torch.int32, device=device)
        hit = None
        if len(idx):
            f = _flat(idx, dims)
            hit = torch.zeros(n, dtype=torch.bool, device=device)
            hit[f] = True
            n_occ.index_add_(0, f, torch.ones(len(f), dtype=torch.int32, device=device))
            n_fr += hit.to(torch.int32)
        d = pts[ok] - sensor
        rng = torch.linalg.norm(d, dim=1)
        keep = rng > (CARVE_NEAR_M + BAND_HALF_M)
        d, rng = d[keep], rng[keep]
        if len(d):
            d = d[::decimation] / rng[::decimation].unsqueeze(1)
            stop = rng[::decimation] - BAND_HALF_M
            _march(sensor, d, CARVE_NEAR_M, stop, origin, vs, dims, device, sweep_free)
        if hit is not None:
            sweep_free[hit] = 0          # a sweep never carves its own endpoint
        n_free += sweep_free
        used.append(nf)

    # ---- 2 & 3. image-based pseudo-free evidence ------------------------------------
    K = np.asarray(geo.calib.K, np.float64)
    Kinv = torch.as_tensor(np.linalg.inv(K), device=device)
    for nf in (image_natives or []):
        nf = int(nf)
        if nf not in geo.cam0_to_world:
            continue
        T = torch.as_tensor(geo.velo_to_velo(nf, int(anchor_native))
                            @ geo.rect_cam_to_velo, device=device)
        cam = T[:3, 3]
        md = (moge or {}).get(nf)
        sk = (sky or {}).get(nf)
        for src, rec in (("moge", md), ("sky", sk)):
            if rec is None:
                continue
            st = int(rec["stride"])
            H, W = rec["shape"]
            vv, uu = np.meshgrid(np.arange(H) * st, np.arange(W) * st, indexing="ij")
            pix = torch.as_tensor(np.stack([uu.ravel(), vv.ravel(),
                                            np.ones(uu.size)], 1).astype(np.float64),
                                  device=device)
            ray = pix @ Kinv.T
            ray = ray / torch.linalg.norm(ray, dim=1, keepdim=True)
            ray = ray @ T[:3, :3].T                       # camera -> anchor velodyne
            if src == "moge":
                dep = torch.as_tensor(np.asarray(rec["depth_z"], np.float64).ravel(),
                                      device=device)
                ok = (torch.as_tensor(rec["mask"].ravel(), device=device)
                      & torch.isfinite(dep) & (dep > CARVE_NEAR_M + BAND_HALF_M)
                      & (dep < P.MOGE_MAX_DEPTH_M))
                if not ok.any():
                    continue
                mg = _margin_lookup(dep[ok], margin["depth_bins_m"], margin["margin_m"])
                stop = dep[ok] - mg                      # never inside the uncertainty band
                good = stop > CARVE_NEAR_M
                if not good.any():
                    continue
                acc = torch.zeros(n, dtype=torch.float32, device=device)
                _march(cam, ray[ok][good], CARVE_NEAR_M, stop[good], origin, vs, dims,
                       device, acc)
                n_moge_views += (acc > 0).to(torch.int16)
            else:
                sky_p = torch.as_tensor(np.asarray(rec["sky"], np.float64).ravel(),
                                        device=device)
                ok = sky_p >= P.SKY_MIN_PROB
                if not ok.any():
                    continue
                far = torch.full((int(ok.sum()),), float(np.linalg.norm(
                    np.asarray(dims, float) * vs)), dtype=torch.float64, device=device)
                acc = torch.zeros(n, dtype=torch.float32, device=device)
                _march(cam, ray[ok], CARVE_NEAR_M, far, origin, vs, dims, device, acc)
                n_sky_views += (acc > 0).to(torch.int16)

    # ---- 4. precedence ---------------------------------------------------------------
    if min_occ_frames is None:
        min_occ_frames = P.MIN_OCC_FRAMES
    occ = n_occ > 0
    fre = n_free > 0
    #: an endpoint corroborated by this many distinct sweeps is a surface, not a grazing
    #: artefact, so precedence rule 1 outranks rule 2 for it
    corroborated = n_fr >= int(min_occ_frames)
    #: a pseudo-free vote only counts once enough independent future views agree
    moge_free = n_moge_views >= P.MOGE_MIN_VIEWS
    sky_free = n_sky_views >= P.SKY_MIN_VIEWS
    w_pf = (n_moge_views.float() + n_sky_views.float())
    pseudo = (moge_free | sky_free) & (w_pf >= P.PSEUDO_FREE_MIN_WEIGHT)

    state = torch.full((n,), UNKNOWN, dtype=torch.uint8, device=device)
    state[fre & ~occ] = FREE                      # 2. consistent LiDAR interior
    state[pseudo & ~occ & ~fre] = FREE            # 3. confidence-weighted pseudo-free
    state[occ & ~fre] = OCCUPIED                  # 1. uncontested endpoint
    state[occ & fre & corroborated] = OCCUPIED    # 1 outranks 2 once corroborated
    # 4. an endpoint seen by too few sweeps and contradicted by free evidence is a genuine
    #    conflict and stays UNKNOWN; pseudo evidence never breaks that tie, and never
    #    overrides an endpoint.
    conflict = int((occ & fre & ~corroborated).sum())
    conflict_resolved = int((occ & fre & corroborated).sum())
    pseudo_vs_lidar = int((pseudo & occ).sum())

    # ---- 5. truncated unsigned distance to the nearest reliable surface --------------
    from scipy.ndimage import distance_transform_edt as edt
    occ3 = (state == OCCUPIED).view(*dims).cpu().numpy()
    if occ3.any():
        dist_vox = edt(~occ3, sampling=(vs, vs, vs))
    else:
        dist_vox = np.full(dims, P.TUDF_TRUNCATION_M, np.float32)
    tudf = np.minimum(dist_vox, P.TUDF_TRUNCATION_M).astype(np.float32)
    # the distance is only meaningful where the volume is supervised at all
    tudf_valid = (state != UNKNOWN).view(*dims).cpu().numpy()

    st = {"n_frames_used": len(used), "n_frames_missing": n_missing,
          "n_points_total": int(n_pts_total), "n_points_in_grid": int(n_pts_in),
          "n_occupied": int((state == OCCUPIED).sum()),
          "n_free": int((state == FREE).sum()),
          "n_unknown": int((state == UNKNOWN).sum()),
          "n_conflict_to_unknown": conflict,
          "n_conflict_resolved_by_corroboration": conflict_resolved,
          "min_occ_frames": int(min_occ_frames),
          "n_free_from_lidar": int(((state == FREE) & fre).sum()),
          "n_free_from_pseudo": int(((state == FREE) & ~fre).sum()),
          "n_moge_voxels": int((n_moge_views >= P.MOGE_MIN_VIEWS).sum()),
          "n_sky_voxels": int((n_sky_views >= P.SKY_MIN_VIEWS).sum()),
          "n_pseudo_refused_by_lidar_endpoint": pseudo_vs_lidar,
          "valid_fraction": float((state != UNKNOWN).sum() / n),
          "prevalence_in_valid": float((state == OCCUPIED).sum()
                                       / max(int((state != UNKNOWN).sum()), 1)),
          "tudf_truncation_m": P.TUDF_TRUNCATION_M}
    return DenseTarget(
        state=state.cpu().numpy(),
        tudf_m=tudf.reshape(-1).astype(np.float16),
        tudf_valid=tudf_valid.reshape(-1),
        n_obs=n_occ.clamp(max=65535).cpu().numpy().astype(np.uint16),
        n_frames_occ=n_fr.clamp(max=255).cpu().numpy().astype(np.uint8),
        w_pseudo_free=w_pf.cpu().numpy().astype(np.float16),
        frames_used=used, stats=st)


def save(path: str, t: DenseTarget, dims, anchor_native: int, stream_index: int,
         future_stream_indices, future_natives) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = np.nonzero(t.state == OCCUPIED)[0].astype(np.int32)
    np.savez_compressed(
        path,
        valid_packed=np.packbits(t.valid), occ_packed=np.packbits(t.occupied),
        free_packed=np.packbits(t.free),
        occ_rows=rows, occ_n_obs=t.n_obs[rows], occ_n_frames=t.n_frames_occ[rows],
        tudf_m=t.tudf_m, tudf_valid_packed=np.packbits(t.tudf_valid),
        w_pseudo_free=t.w_pseudo_free,
        dims=np.asarray(dims, np.int32), stream_index=np.int32(stream_index),
        native_frame=np.int32(anchor_native),
        future_stream_indices=np.asarray(future_stream_indices, np.int32),
        future_natives=np.asarray(future_natives, np.int32),
        frames_used=np.asarray(t.frames_used, np.int32),
        stats=np.asarray(str(t.stats)))


def load(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        dims = tuple(int(x) for x in z["dims"])
        n = int(np.prod(dims))
        rows = z["occ_rows"].astype(np.int64)
        n_obs = np.zeros(n, np.uint16); n_obs[rows] = z["occ_n_obs"]
        n_fr = np.zeros(n, np.uint8); n_fr[rows] = z["occ_n_frames"]
        return {"valid": np.unpackbits(z["valid_packed"])[:n].astype(bool),
                "occupied": np.unpackbits(z["occ_packed"])[:n].astype(bool),
                "free": np.unpackbits(z["free_packed"])[:n].astype(bool),
                "tudf_m": z["tudf_m"].astype(np.float32),
                "tudf_valid": np.unpackbits(z["tudf_valid_packed"])[:n].astype(bool),
                "w_pseudo_free": z["w_pseudo_free"].astype(np.float32),
                "n_obs": n_obs, "n_frames_occ": n_fr, "dims": dims,
                "occ_rows": rows, "occ_n_obs": z["occ_n_obs"],
                "occ_n_frames": z["occ_n_frames"],
                "stream_index": int(z["stream_index"]),
                "native_frame": int(z["native_frame"]),
                "future_stream_indices": z["future_stream_indices"].astype(np.int64),
                "future_natives": z["future_natives"].astype(np.int64),
                "frames_used": z["frames_used"].astype(np.int64)}


__all__ = ["DenseTarget", "build", "save", "load", "UNKNOWN", "FREE", "OCCUPIED"]
