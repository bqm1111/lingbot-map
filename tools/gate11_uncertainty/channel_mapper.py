"""Gate 9 VariantMapper with per-pixel geometry, for the uncertainty representation.

mode "surface":  VariantMapper._integrate line for line (+-0.2 m band, stride-4 carving), except that
                 the carve stop margin is a per-pixel array (metres along the ray). With a constant
                 0.2 m margin it is bit-identical to VariantMapper(0.2, 0.2, carve) (checked in verify).
mode "interval": occupied-evidence samples every 0.2 m along the ray over [d - u_front, d + u_back]
                 (per pixel, metres along the ray), no carving. Only w_occ > 0 is read from it.
No log-odds constant, gate, range or band is changed; core/ is not edited.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gate9_depth_occupancy"))
from variant_mapper import VariantMapper                                                   # noqa: E402  (Gate 9)
from core.mapping import depth as D7, rays as RY, voxmap as VM                             # noqa: E402
from core.mapping.mapper import FrameInput, pack                                           # noqa: E402


class ChannelMapper(VariantMapper):
    def __init__(self, device, mode: str):
        assert mode in ("surface", "interval")
        super().__init__(device, RY.BAND_HALF_M, RY.BAND_HALF_M, carve=(mode == "surface"))
        self.mode = mode; self.px = None

    def set_pixels(self, **arrays):
        self.px = {k: torch.as_tensor(np.asarray(v, np.float64), device=self.device) for k, v in arrays.items()}

    def _integrate(self, f: FrameInput):
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
            if self.mode == "surface":
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
                        stop = (cd - self.px["margin"].reshape(-1)[cs] / cl).clamp(min=RY.CARVE_NEAR_M)
                        n_steps = int(np.ceil(D7.MAX_DEPTH_M / self.vs))
                        step = torch.arange(n_steps, device=self.device, dtype=torch.float64)
                        zz = RY.CARVE_NEAR_M + step.unsqueeze(0) * self.vs / cl.unsqueeze(1)
                        valid = zz < stop.unsqueeze(1)
                        pts = (cdir.unsqueeze(1) * zz.unsqueeze(2))[valid]
                        pw = pts @ Tw[:3, :3].T + Tw[:3, 3]
                        k_free.append(pack(torch.floor(pw / self.vs).to(torch.int64)))
            else:
                uf = self.px["u_front"].reshape(-1)[sel]; ub = self.px["u_back"].reshape(-1)[sel]
                omax = int(np.ceil(float(torch.maximum(uf.max(), ub.max())) / self.vs - 1e-9))
                for o in range(-omax, omax + 1):
                    dist = o * self.vs
                    keep = (dist >= -uf - 1e-9) & (dist <= ub + 1e-9)
                    if not bool(keep.any()):
                        continue
                    dz = d[keep] + dist / rl[keep]
                    p = dv[keep] * dz.unsqueeze(1)
                    pw = p @ Tw[:3, :3].T + Tw[:3, 3]
                    ok = dz > 0
                    k_occ.append(pack(torch.floor(pw[ok] / self.vs).to(torch.int64)))
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
