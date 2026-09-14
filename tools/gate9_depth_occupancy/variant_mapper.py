"""``IncrementalMapper`` with the occupied band and the carve stop margin exposed separately.

``core.mapping.mapper.IncrementalMapper._integrate`` uses one constant,
``rays.BAND_HALF_M = 0.2``, for two things: the half-width of the occupied band
(``n_off = floor(BAND_HALF_M / 0.2)`` samples each side along the ray, i.e. d-0.2, d, d+0.2 m)
and the distance before the surface at which free-space carving stops. This subclass copies
the method line for line, replacing those two uses by ``band_half_m`` and
``carve_margin_m``; semantics are not integrated (no teacher in this audit). With (0.2, 0.2)
it is asserted bit-identical to the original (``verify.py``). ``core/`` is not edited.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import torch

from core.mapping import mapper as MP, depth as D7, rays as RY, voxmap as VM
from core.mapping.mapper import IncrementalMapper, FrameInput, pack


class VariantMapper(IncrementalMapper):
    def __init__(self, device, band_half_m: float = RY.BAND_HALF_M, carve_margin_m: float = RY.BAND_HALF_M, carve: bool = True):
        super().__init__(device, sem_into=None, n_teacher=0, carve=carve)
        self.band_half_m = float(band_half_m); self.carve_margin_m = float(carve_margin_m)

    def _integrate(self, f: FrameInput) -> Dict[str, float]:
        s = float(self.scale_state.scale)
        d_m = s * f.depth_canonical.to(torch.float64)
        acc = (f.conf.to(torch.float64) >= self.conf_threshold) & torch.isfinite(d_m) \
            & (d_m > D7.MIN_DEPTH_M) & (d_m < D7.MAX_DEPTH_M)
        T = np.asarray(f.pose_c2w_canonical, np.float64).copy()
        T[:3, 3] *= s
        Tw = torch.as_tensor(T, device=self.device)
        dirs = self._dirs(f.K, d_m.shape)
        stats = {"n_rays": int(acc.sum()), "n_new_rows": 0, "n_occ": 0, "n_free": 0}
        k_occ, k_free = [], []
        sel = acc.reshape(-1).nonzero(as_tuple=True)[0]
        if sel.numel():
            d = d_m.reshape(-1)[sel]
            dv = dirs.reshape(-1, 3)[sel]
            rl = dv.norm(dim=1)
            n_off = max(int(np.floor(self.band_half_m / self.vs)), 0)
            for o in range(-n_off, n_off + 1):
                dz = d + o * self.vs / rl
                p = dv * dz.unsqueeze(1)
                pw = p @ Tw[:3, :3].T + Tw[:3, 3]
                ok = dz > 0
                k_occ.append(pack(torch.floor(pw[ok] / self.vs).to(torch.int64)))
            if self.carve:
                m2 = torch.zeros_like(acc)
                m2[::RY.CARVE_STRIDE, ::RY.CARVE_STRIDE] = True
                cs = (acc & m2).reshape(-1).nonzero(as_tuple=True)[0]
                if cs.numel():
                    cd = d_m.reshape(-1)[cs]
                    cdir = dirs.reshape(-1, 3)[cs]
                    cl = cdir.norm(dim=1)
                    stop = (cd - self.carve_margin_m / cl).clamp(min=RY.CARVE_NEAR_M)
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
            self.n_integrated += 1; self.last_stats = stats
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
        d_nlb = d_nobs.clone(); d_nmoge = torch.zeros_like(d_nobs)
        stats["n_new_rows"] = self.table.merge(uq, d_log, d_wocc, d_wfree, d_nobs, d_nlb, d_nmoge, None, None, f.index)
        stats["n_occ"] = int((d_wocc > 0).sum()); stats["n_free"] = int((d_wfree > 0).sum()); stats["n_rows"] = len(self.table)
        self.n_integrated += 1; self.last_stats = stats
        return stats


def table_state(m: IncrementalMapper):
    t = m.table; n = t.n
    return {k: getattr(t, k)[:n].detach().cpu().numpy() for k in ("keys", "logodds", "w_occ", "w_free", "n_obs")}
