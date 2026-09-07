"""Privileged training targets for one causal map state.

For an anchor timestamp ``t`` the **input** is the causal map exported at ``t`` (frames
``<= t`` only). The **targets** use what only training may see:

* binary occupancy from the source benchmark's own voxel ground truth (LiDAR-derived);
  no semantic class label is ever read from it -- only ``occupied / empty / invalid``;
* frozen Trident-H evidence from the following ``FUTURE_FRAMES`` frames, fused through
  the frozen geometry of those frames into a *separate* volume with the *same* frozen
  scale, so the semantic target at a voxel is the teacher's opinion from views the causal
  map has not yet had.

Every mask the loss needs is explicit, and every sample records which frames built its
input and which built its target, so leakage can be audited from the file alone.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from gate6 import grids as G6G, targets as G6T
from gate8 import sources as S, vocab as V8
from gate8.feed import CachedFeed
from gate8.mapper import IncrementalMapper

FUTURE_FRAMES = 20
MIN_FUTURE = 5


def binary_occupancy(dataset: str, gt_ref: dict, repo_root: str, map_grid):
    """``(occupied, valid)`` on the MAP grid. Semantic ids are collapsed to occupancy here
    and are never returned: the class label does not leave this function."""
    target, keep = G6T.semantic_target(dataset, gt_ref, repo_root)
    if target is None:
        return None, None
    eval_grid = G6G.EVAL_GRID[dataset]
    occ = (target != eval_grid.empty_class) & keep
    if G6G.NEEDS_REDUCTION[dataset]:
        r = G6G.RATIO
        occ = np.repeat(np.repeat(np.repeat(occ, r, 0), r, 1), r, 2)
        keep = np.repeat(np.repeat(np.repeat(keep, r, 0), r, 1), r, 2)
        assert occ.shape == tuple(map_grid.dims)
    return occ.reshape(-1), keep.reshape(-1)


def future_volume(feed: CachedFeed, t: int, scale: float, into: np.ndarray, device,
                  n_future: int = FUTURE_FRAMES) -> Tuple[IncrementalMapper, list]:
    """A throwaway mapper holding frames ``t+1 .. t+n_future`` only."""
    m = IncrementalMapper(device, sem_into=into)
    m.scale_state.force(scale)
    used = []
    for j in range(t + 1, min(t + 1 + n_future, len(feed))):
        m.step(feed.frame(j))
        used.append(j)
    return m, used


def build_sample(feed: CachedFeed, mapper: IncrementalMapper, t: int, map_grid,
                 T_grid_to_world: np.ndarray, dataset: str, gt_ref: dict, repo_root: str,
                 into: np.ndarray, device) -> Optional[Dict[str, np.ndarray]]:
    """Export the causal input at ``t`` and construct its privileged targets."""
    if len(feed) - 1 - t < MIN_FUTURE:
        return None
    q = mapper.query(map_grid, T_grid_to_world)
    gt_occ, gt_valid = binary_occupancy(dataset, gt_ref, repo_root, map_grid)
    if gt_occ is None:
        return None
    fut, used = future_volume(feed, t, mapper.scale_state.scale, into, device)
    qf = fut.query(map_grid, T_grid_to_world)

    obs = q["observed"].cpu().numpy()
    fut_obs = qf["observed"].cpu().numpy()
    fut_semw = qf["sem_w"].cpu().numpy()
    rows = np.flatnonzero(obs | fut_obs | (gt_occ & gt_valid))
    sem_rows = np.flatnonzero(obs & (q["sem_w"].cpu().numpy() > 0))
    fut_rows = np.flatnonzero(fut_semw > 0)

    def u8(p):
        return np.clip(np.rint(p * 255.0), 0, 255).astype(np.uint8)

    sem = q["sem"].cpu().numpy(); semw = q["sem_w"].cpu().numpy()
    sem_p = sem[sem_rows] / np.maximum(semw[sem_rows, None], 1e-9)
    fsem = qf["sem"].cpu().numpy()
    fut_p = fsem[fut_rows] / np.maximum(fut_semw[fut_rows, None], 1e-9)
    age = np.where(q["last_time"].cpu().numpy() >= 0,
                   t - q["last_time"].cpu().numpy(), -1)
    out = {
        "t": np.int64(t), "input_frames": np.arange(0, t + 1, dtype=np.int32),
        "target_frames": np.asarray(used, np.int32),
        "scale": np.float64(mapper.scale_state.scale),
        "dims": np.asarray(map_grid.dims, np.int32),
        # ---- input state (sparse over rows)
        "rows": rows.astype(np.int32),
        "logodds": q["logodds"].cpu().numpy()[rows].astype(np.float16),
        "w_occ": q["w_occ"].cpu().numpy()[rows].astype(np.float16),
        "w_free": q["w_free"].cpu().numpy()[rows].astype(np.float16),
        "n_obs": np.clip(q["n_obs"].cpu().numpy()[rows], 0, 255).astype(np.uint8),
        "n_lb": np.clip(q["n_lb"].cpu().numpy()[rows], 0, 255).astype(np.uint8),
        "age": np.clip(age[rows], -1, 254).astype(np.int16),
        "observed": obs[rows],
        "sem_rows": sem_rows.astype(np.int32), "sem_p": u8(sem_p),
        "sem_w": semw[sem_rows].astype(np.float16),
        # ---- targets
        "gt_occ": np.packbits(gt_occ), "gt_valid": np.packbits(gt_valid),
        "fut_observed": np.packbits(fut_obs),
        "fut_rows": fut_rows.astype(np.int32), "fut_p": u8(fut_p),
        "fut_w": fut_semw[fut_rows].astype(np.float16),
        "fut_occ": qf["occupied"].cpu().numpy()[fut_rows],
    }
    return out


def unpack_sample(z, device) -> Dict[str, torch.Tensor]:
    """Sparse file -> dense tensors on ``device``: inputs ``[Cin, X, Y, Z]``, targets."""
    dims = tuple(int(d) for d in z["dims"]); n = int(np.prod(dims))
    rows = torch.from_numpy(z["rows"].astype(np.int64)).to(device)
    def dense(vals, dt=torch.float32, fill=0.0):
        out = torch.full((n,), fill, dtype=dt, device=device)
        out[rows] = torch.from_numpy(np.asarray(vals)).to(device=device, dtype=dt)
        return out
    logodds = dense(z["logodds"].astype(np.float32))
    w_free = dense(z["w_free"].astype(np.float32))
    n_obs = dense(z["n_obs"].astype(np.float32))
    age = dense(z["age"].astype(np.float32), fill=-1.0)
    observed = dense(z["observed"].astype(np.float32))
    sem = torch.zeros(n, V8.U, device=device)
    semw = torch.zeros(n, device=device)
    sr = torch.from_numpy(z["sem_rows"].astype(np.int64)).to(device)
    if sr.numel():
        sem[sr] = torch.from_numpy(z["sem_p"].astype(np.float32) / 255.0).to(device)
        semw[sr] = torch.from_numpy(z["sem_w"].astype(np.float32)).to(device)
    gt_occ = torch.from_numpy(np.unpackbits(z["gt_occ"])[:n].astype(np.float32)).to(device)
    gt_valid = torch.from_numpy(np.unpackbits(z["gt_valid"])[:n].astype(np.bool_)).to(device)
    fut_obs = torch.from_numpy(np.unpackbits(z["fut_observed"])[:n].astype(np.bool_)).to(device)
    fut_p = torch.zeros(n, V8.U, device=device)
    fut_valid = torch.zeros(n, dtype=torch.bool, device=device)
    fr = torch.from_numpy(z["fut_rows"].astype(np.int64)).to(device)
    if fr.numel():
        fut_p[fr] = torch.from_numpy(z["fut_p"].astype(np.float32) / 255.0).to(device)
        fut_valid[fr] = True
    X, Y, Z = dims
    def g(x): return x.reshape(X, Y, Z)
    inp = torch.stack([g(logodds) / 4.0, g(w_free).clamp(max=20) / 20.0, g(observed),
                       1.0 - g(observed), g(n_obs).clamp(max=50) / 50.0,
                       (g(age).clamp(min=0) / 100.0) * (g(age) >= 0),
                       g(semw).clamp(max=20) / 20.0] +
                      [g(sem[:, u]) for u in range(V8.U)], dim=0)
    return {"input": inp, "gt_occ": g(gt_occ), "gt_valid": g(gt_valid),
            "observed": g(observed) > 0, "fut_observed": g(fut_obs),
            "fut_p": fut_p.reshape(X, Y, Z, V8.U), "fut_valid": g(fut_valid),
            "base_logodds": g(logodds), "t": int(z["t"])}


N_INPUT_CHANNELS = 7 + V8.U

__all__ = ["build_sample", "unpack_sample", "binary_occupancy", "future_volume",
           "FUTURE_FRAMES", "MIN_FUTURE", "N_INPUT_CHANNELS"]
