# extracted from gate8/mapper.py -- imports rewritten, logic unchanged.
# see core/README.md; the original is still in place and is the audited copy.
"""The incremental causal mapper: one frame in, a persistent world-frame map updated.

Two states, kept apart on purpose:

* :class:`ScaleState` -- buffers the first five frames, estimates one metric scale from
  them (the frozen G-A rule of Gate 7B), and freezes it. It is never stored inside a
  voxel, and the map cannot be re-gauged: the scale is decided *before* the first voxel
  is written and the buffered frames are integrated exactly once, after it is known;
* :class:`VoxelTable` -- a sorted hashed voxel table in the **scaled world frame** of
  the sequence. Every frame merges its contributions into the table by key; nothing is
  ever re-integrated and no history is replayed. Dense grids are an *export* --
  :meth:`IncrementalMapper.query` looks each grid cell up in the table -- never a
  reconstruction of the map.

The per-frame evidence rule is Gate 7B's, unchanged: free-space before the surface,
an occupied band around it, nothing behind it; fixed OctoMap log-odds; semantic evidence
kept in its own accumulator and weighted by geometry reliability, never by the
teacher's own confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from core.mapping import depth as D7, rays as RY, scale as SC, voxmap as VM

VOXEL_SIZE = 0.2
_OFF = 1 << 20          # key packing offset; world extent +-200 km at 0.2 m
_BITS = 21


def pack(ijk: torch.Tensor) -> torch.Tensor:
    """``[N, 3]`` int64 lattice indices -> ``[N]`` int64 keys."""
    i = ijk[:, 0] + _OFF
    j = ijk[:, 1] + _OFF
    k = ijk[:, 2] + _OFF
    return (i << (2 * _BITS)) | (j << _BITS) | k


def unpack(keys: torch.Tensor) -> torch.Tensor:
    m = (1 << _BITS) - 1
    k = (keys & m) - _OFF
    j = ((keys >> _BITS) & m) - _OFF
    i = (keys >> (2 * _BITS)) - _OFF
    return torch.stack([i, j, k], dim=1)


# --------------------------------------------------------------------------- #
class ScaleState:
    """Five buffered frames -> one frozen scalar. Separate from the map by construction."""

    def __init__(self, n_anchor: int = SC.N_ANCHOR_FRAMES):
        self.n_anchor = int(n_anchor)
        self._logs: List[float] = []
        self.scale: Optional[float] = None
        self.n_seen = 0

    def observe(self, log_candidate: float) -> None:
        if self.scale is not None:
            return                                  # frozen: later frames are ignored
        self.n_seen += 1
        if np.isfinite(log_candidate):
            self._logs.append(float(log_candidate))
        if self.n_seen >= self.n_anchor:
            self.scale = float(np.exp(np.median(self._logs))) if self._logs else None

    @property
    def frozen(self) -> bool:
        return self.scale is not None

    def force(self, scale: float) -> None:
        """Adopt an already-decided scale (the target builder's future volume must use the
        *same* frozen scalar as the causal map it supervises)."""
        self.scale = float(scale)


# --------------------------------------------------------------------------- #
class VoxelTable:
    """Sorted hashed voxel table on the GPU, merged by key.

    Storage is **capacity-doubling and double-buffered**: every attribute lives in a
    preallocated buffer of ``cap`` rows, of which the first ``n`` are live and sorted by
    key. A merge scatters old and new rows into the spare buffer and swaps, so a frame
    allocates nothing once the capacity is reached. (The first version re-concatenated
    every attribute per frame; the allocator fragmented across ever-growing sizes and a
    single 815-frame sequence exhausted a 95 GB GPU. This is the fix.)
    """

    ATTRS = ("logodds", "w_occ", "w_free", "n_obs", "n_lb", "n_moge", "first_time",
             "last_time", "sem_w")
    DTYPE = {"logodds": torch.float32, "w_occ": torch.float32, "w_free": torch.float32,
             "n_obs": torch.int32, "n_lb": torch.int32, "n_moge": torch.int32,
             "first_time": torch.int32, "last_time": torch.int32, "sem_w": torch.float32}

    def __init__(self, n_sem: int, device, capacity: int = 1 << 18):
        self.device = device
        self.n_sem = int(n_sem)
        self.n = 0
        self._alloc(int(capacity))

    def _alloc(self, cap: int):
        self.cap = cap
        self._buf = {}
        for side in ("a", "b"):
            self._buf[side] = {"keys": torch.empty(cap, dtype=torch.int64, device=self.device),
                               "sem": torch.empty(cap, self.n_sem, dtype=torch.float32,
                                                  device=self.device)}
            for a in self.ATTRS:
                self._buf[side][a] = torch.empty(cap, dtype=self.DTYPE[a], device=self.device)
        self._live = "a"

    def _grow(self, need: int):
        cap = self.cap
        while cap < need:
            cap *= 2
        old = self._buf[self._live]; n = self.n
        self._alloc(cap)
        cur = self._buf[self._live]
        cur["keys"][:n] = old["keys"][:n]; cur["sem"][:n] = old["sem"][:n]
        for a in self.ATTRS:
            cur[a][:n] = old[a][:n]
        del old

    def __getattr__(self, name):
        if name in ("keys", "sem") or name in VoxelTable.ATTRS:
            b = self.__dict__.get("_buf")
            if b is None:
                raise AttributeError(name)
            return b[self.__dict__["_live"]][name][: self.__dict__["n"]]
        raise AttributeError(name)

    def __len__(self) -> int:
        return int(self.n)

    def memory_bytes(self) -> int:
        per_row = 8 + self.n_sem * 4 + sum(torch.empty(0, dtype=self.DTYPE[a]).element_size()
                                          for a in self.ATTRS)
        return int(2 * self.cap * per_row)

    def lookup(self, keys: torch.Tensor) -> torch.Tensor:
        if self.n == 0:
            return torch.full_like(keys, -1)
        k = self.keys
        pos = torch.searchsorted(k, keys).clamp(max=self.n - 1)
        return torch.where(k[pos] == keys, pos, torch.full_like(pos, -1))

    def merge(self, keys, d_logodds, d_wocc, d_wfree, d_nobs, d_nlb, d_nmoge, d_sem, d_semw,
              time_index: int) -> int:
        """Add one frame's *already reduced, sorted-unique* contributions. Rows added."""
        if keys.numel() == 0:
            return 0
        row = self.lookup(keys)
        hit = row >= 0
        t_i = int(time_index)
        if hit.any():
            r = row[hit]
            self.logodds.index_add_(0, r, d_logodds[hit])
            self.w_occ.index_add_(0, r, d_wocc[hit])
            self.w_free.index_add_(0, r, d_wfree[hit])
            self.n_obs.index_add_(0, r, d_nobs[hit])
            self.n_lb.index_add_(0, r, d_nlb[hit])
            self.n_moge.index_add_(0, r, d_nmoge[hit])
            if d_sem is not None:
                self.sem.index_add_(0, r, d_sem[hit])
                self.sem_w.index_add_(0, r, d_semw[hit])
            oh = d_nobs[hit] > 0
            if oh.any():
                rr = r[oh]
                self.last_time[rr] = torch.maximum(self.last_time[rr],
                                                   torch.full_like(rr, t_i, dtype=torch.int32))
                ft = self.first_time[rr]
                self.first_time[rr] = torch.where(ft < 0, torch.full_like(ft, t_i), ft)
        miss = ~hit
        m = int(miss.sum())
        if m == 0:
            self.logodds.clamp_(-VM.L_CLAMP, VM.L_CLAMP)
            return 0
        n = self.n
        if n + m > self.cap:
            self._grow(n + m)
        nk = keys[miss]
        old = self._buf[self._live]; new = self._buf["b" if self._live == "a" else "a"]
        ok = old["keys"][:n]
        pos_old = torch.arange(n, device=self.device) + torch.searchsorted(nk, ok)
        pos_new = torch.arange(m, device=self.device) + torch.searchsorted(ok, nk)
        occ_new = d_nobs[miss] > 0
        t_new = torch.where(occ_new, torch.full_like(nk, t_i, dtype=torch.int32),
                            torch.full_like(nk, -1, dtype=torch.int32))
        contrib = {"logodds": d_logodds[miss], "w_occ": d_wocc[miss], "w_free": d_wfree[miss],
                   "n_obs": d_nobs[miss], "n_lb": d_nlb[miss], "n_moge": d_nmoge[miss],
                   "first_time": t_new, "last_time": t_new,
                   "sem_w": (d_semw[miss] if d_sem is not None
                             else torch.zeros(m, device=self.device))}
        new["keys"][pos_old] = ok; new["keys"][pos_new] = nk
        for a in self.ATTRS:
            new[a][pos_old] = old[a][:n]; new[a][pos_new] = contrib[a].to(self.DTYPE[a])
        new["sem"][pos_old] = old["sem"][:n]
        new["sem"][pos_new] = d_sem[miss] if d_sem is not None else 0.0
        self._live = "b" if self._live == "a" else "a"
        self.n = n + m
        self.logodds.clamp_(-VM.L_CLAMP, VM.L_CLAMP)
        return m

    def occupied(self) -> torch.Tensor:
        occ = self.logodds > VM.L_OCCUPIED_AT
        moge_only = (self.n_lb == 0) & (self.n_moge > 0)
        return occ & ~(moge_only & (self.n_moge < VM.MOGE_CONFIRMATIONS))

    def free(self) -> torch.Tensor:
        return (self.logodds <= VM.L_OCCUPIED_AT) & (self.w_free > 0)

# --------------------------------------------------------------------------- #
@dataclass
class FrameInput:
    """One frame of frozen-model output, in canonical units, plus its teacher map."""
    index: int
    depth_canonical: torch.Tensor          # [H, W] float32
    conf: torch.Tensor                     # [H, W] float32
    K: np.ndarray                          # [3, 3] processed-lattice intrinsics
    pose_c2w_canonical: np.ndarray         # [4, 4] camera-to-world, canonical units
    log_scale_candidate: float             # from gate7b.scale.frame_candidate, or nan
    sem_probs: Optional[torch.Tensor] = None   # [H, W, C_teacher] float32
    moge_depth: Optional[torch.Tensor] = None
    moge_mask: Optional[torch.Tensor] = None


class IncrementalMapper:
    """``initialize(anchor_frames)`` -> ``step(frame)`` -> ``query(grid, T)``."""

    def __init__(self, device, sem_into: Optional[np.ndarray] = None, n_teacher: int = 0,
                 voxel_size: float = VOXEL_SIZE, moge_rescue: bool = False,
                 conf_threshold: float = D7.CONF_THRESHOLD, carve: bool = True):
        self.device = device
        self.vs = float(voxel_size)
        self.scale_state = ScaleState()
        self.into = (torch.as_tensor(sem_into, device=device, dtype=torch.float32)
                     if sem_into is not None else None)
        n_sem = int(self.into.shape[1]) if self.into is not None else int(n_teacher)
        self.table = VoxelTable(n_sem, device)
        self.moge_rescue = bool(moge_rescue)
        self.conf_threshold = float(conf_threshold)
        self.carve = bool(carve)
        self._buffer: List[FrameInput] = []
        self.n_integrated = 0
        self.last_stats: Dict[str, float] = {}
        self._dirs_cache: Dict[Tuple, torch.Tensor] = {}

    # ---- the public API ----------------------------------------------------
    def initialize(self, anchor_frames: Sequence[FrameInput]) -> None:
        """Buffer the anchor frames, fix the scale, then integrate them exactly once."""
        assert self.n_integrated == 0 and not self.scale_state.frozen
        for f in anchor_frames:
            self.scale_state.observe(f.log_scale_candidate)
            self._buffer.append(f)
        if not self.scale_state.frozen:
            raise RuntimeError(f"scale not fixed after {len(anchor_frames)} anchor frames")
        for f in self._buffer:
            self._integrate(f)
        self._buffer = []

    def step(self, frame: FrameInput) -> Dict[str, float]:
        """Integrate one new frame into the existing state. Never touches older ones."""
        if not self.scale_state.frozen:
            self._buffer.append(frame)
            self.scale_state.observe(frame.log_scale_candidate)
            if self.scale_state.frozen:
                for f in self._buffer:
                    self._integrate(f)
                self._buffer = []
            return {"buffered": 1}
        self.scale_state.observe(frame.log_scale_candidate)   # no-op once frozen
        return self._integrate(frame)

    def query(self, grid, T_grid_to_world: np.ndarray, allow_provisional: bool = False
              ) -> Dict[str, torch.Tensor]:
        """Dense export onto a benchmark grid: a lookup per cell, never a rebuild."""
        X, Y, Z = (int(d) for d in grid.dims)
        ix, iy, iz = torch.meshgrid(torch.arange(X, device=self.device),
                                    torch.arange(Y, device=self.device),
                                    torch.arange(Z, device=self.device), indexing="ij")
        c = (torch.stack([ix, iy, iz], -1).reshape(-1, 3).double() + 0.5) * float(grid.voxel_size)
        c = c + torch.as_tensor(np.asarray(grid.origin, np.float64), device=self.device)
        T = torch.as_tensor(np.asarray(T_grid_to_world, np.float64), device=self.device)
        w = c @ T[:3, :3].T + T[:3, 3]
        keys = pack(torch.floor(w / self.vs).to(torch.int64))
        row = self.table.lookup(keys)
        hit = row >= 0
        rr = row.clamp(min=0)
        def take(x, fill=0):
            out = torch.full((keys.numel(),) + tuple(x.shape[1:]), fill, dtype=x.dtype,
                             device=self.device)
            out[hit] = x[rr[hit]]
            return out
        occ_rows = self.table.occupied() if not allow_provisional else \
            (self.table.logodds > VM.L_OCCUPIED_AT)
        out = {
            "hit": hit,
            "occupied": take(occ_rows.to(torch.bool)),
            "free": take(self.table.free().to(torch.bool)),
            "logodds": take(self.table.logodds),
            "w_occ": take(self.table.w_occ), "w_free": take(self.table.w_free),
            "n_obs": take(self.table.n_obs), "n_lb": take(self.table.n_lb),
            "n_moge": take(self.table.n_moge),
            "first_time": take(self.table.first_time, -1),
            "last_time": take(self.table.last_time, -1),
            "sem": take(self.table.sem), "sem_w": take(self.table.sem_w),
        }
        out["observed"] = hit
        return out

    # ---- internals ----------------------------------------------------------
    def _dirs(self, K: np.ndarray, hw) -> torch.Tensor:
        key = (tuple(np.round(np.asarray(K, np.float64), 6).ravel()), tuple(int(x) for x in hw))
        if key not in self._dirs_cache:
            self._dirs_cache[key] = RY.pixel_rays(K, hw, self.device)
        return self._dirs_cache[key]

    def _integrate(self, f: FrameInput) -> Dict[str, float]:
        s = float(self.scale_state.scale)
        d_m = s * f.depth_canonical.to(torch.float64)
        acc = (f.conf.to(torch.float64) >= self.conf_threshold) & torch.isfinite(d_m) \
            & (d_m > D7.MIN_DEPTH_M) & (d_m < D7.MAX_DEPTH_M)
        T = np.asarray(f.pose_c2w_canonical, np.float64).copy()
        T[:3, 3] *= s                                      # identical scalar on translation
        Tw = torch.as_tensor(T, device=self.device)
        dirs = self._dirs(f.K, d_m.shape)
        H, W = d_m.shape
        stats = {"n_rays": int(acc.sum()), "n_new_rows": 0, "n_occ": 0, "n_free": 0}

        probs_u = None
        if f.sem_probs is not None:
            p = f.sem_probs.to(self.device).float()
            probs_u = (p @ self.into) if self.into is not None else p
            sem_w_pix = (VM.W_LINGBOT * (f.conf.to(self.device) / self.conf_threshold)
                         .clamp(0.0, 2.0).clamp(min=0.25)).float()

        k_occ, k_sem, w_sem, k_free = [], [], [], []
        sel = acc.reshape(-1).nonzero(as_tuple=True)[0]
        if sel.numel():
            d = d_m.reshape(-1)[sel]
            dv = dirs.reshape(-1, 3)[sel]
            rl = dv.norm(dim=1)
            n_off = max(int(np.floor(RY.BAND_HALF_M / self.vs)), 0)
            for o in range(-n_off, n_off + 1):
                dz = d + o * self.vs / rl
                p = dv * dz.unsqueeze(1)
                pw = p @ Tw[:3, :3].T + Tw[:3, 3]
                ok = dz > 0
                k_occ.append(pack(torch.floor(pw[ok] / self.vs).to(torch.int64)))
                if probs_u is not None:
                    k_sem.append(k_occ[-1])
                    w_sem.append(sem_w_pix.reshape(-1)[sel][ok])
            if self.carve:
                m2 = torch.zeros_like(acc)
                m2[::RY.CARVE_STRIDE, ::RY.CARVE_STRIDE] = True
                cs = (acc & m2).reshape(-1).nonzero(as_tuple=True)[0]
                if cs.numel():
                    cd = d_m.reshape(-1)[cs]
                    cdir = dirs.reshape(-1, 3)[cs]
                    cl = cdir.norm(dim=1)
                    stop = (cd - RY.BAND_HALF_M / cl).clamp(min=RY.CARVE_NEAR_M)
                    n_steps = int(np.ceil(D7.MAX_DEPTH_M / self.vs))
                    step = torch.arange(n_steps, device=self.device, dtype=torch.float64)
                    zz = RY.CARVE_NEAR_M + step.unsqueeze(0) * self.vs / cl.unsqueeze(1)
                    valid = zz < stop.unsqueeze(1)
                    pts = (cdir.unsqueeze(1) * zz.unsqueeze(2))[valid]
                    pw = pts @ Tw[:3, :3].T + Tw[:3, 3]
                    k_free.append(pack(torch.floor(pw / self.vs).to(torch.int64)))

        ko = torch.cat(k_occ) if k_occ else torch.zeros(0, dtype=torch.int64, device=self.device)
        kf = torch.cat(k_free) if k_free else torch.zeros(0, dtype=torch.int64, device=self.device)
        keys = torch.cat([ko, kf])
        if keys.numel() == 0:
            self.n_integrated += 1
            self.last_stats = stats
            return stats
        uq, inv = torch.unique(keys, return_inverse=True)
        n = uq.numel()
        io, if_ = inv[:ko.numel()], inv[ko.numel():]
        one = torch.ones(1, device=self.device)
        d_wocc = torch.zeros(n, device=self.device).index_add_(0, io, one.expand(io.numel()))
        d_wfree = torch.zeros(n, device=self.device).index_add_(0, if_, one.expand(if_.numel()))
        d_log = d_wocc * VM.L_OCC + d_wfree * VM.L_FREE
        d_nobs = torch.zeros(n, dtype=torch.int32, device=self.device)
        d_nobs.index_add_(0, io, torch.ones(io.numel(), dtype=torch.int32, device=self.device))
        d_nlb = d_nobs.clone()
        d_nmoge = torch.zeros_like(d_nobs)
        d_sem = d_semw = None
        if probs_u is not None and k_sem:
            ks = torch.cat(k_sem)
            ws = torch.cat(w_sem).float()
            pv = probs_u.reshape(-1, probs_u.shape[-1])[sel]
            pv = pv.repeat(2 * n_off + 1, 1)
            # keep only the (in-front-of-camera) rows that k_occ kept
            keep = torch.cat([(d + o * self.vs / rl) > 0 for o in range(-n_off, n_off + 1)])
            pv = pv[keep]
            i_s = self.table_index_for(uq, ks)
            d_sem = torch.zeros(n, pv.shape[1], device=self.device).index_add_(0, i_s, pv * ws.unsqueeze(1))
            d_semw = torch.zeros(n, device=self.device).index_add_(0, i_s, ws)
        stats["n_new_rows"] = self.table.merge(uq, d_log, d_wocc, d_wfree, d_nobs, d_nlb,
                                               d_nmoge, d_sem, d_semw, f.index)
        stats["n_occ"] = int((d_wocc > 0).sum())
        stats["n_free"] = int((d_wfree > 0).sum())
        stats["n_rows"] = len(self.table)
        self.n_integrated += 1
        self.last_stats = stats
        return stats

    @staticmethod
    def table_index_for(uq: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        pos = torch.searchsorted(uq, keys)
        return pos


__all__ = ["IncrementalMapper", "FrameInput", "ScaleState", "VoxelTable", "pack", "unpack",
           "VOXEL_SIZE"]
