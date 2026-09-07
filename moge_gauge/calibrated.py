"""Gate 5.1 — MoGe-2 wrapper that may additionally receive a scalar calibrated FOV.

Subclasses the frozen Gate-5 adapter so that :class:`moge_gauge.adapter.FrozenMoGe` and its
Gate-5 tests remain untouched. Only ONE extra scalar may be passed to MoGe: the horizontal
field of view in degrees. Full intrinsics, extrinsics, poses, camera height, LiDAR and
occupancy information remain forbidden.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .adapter import FrozenMoGe, MoGeOutput


class CalibratedMoGe(FrozenMoGe):
    @torch.no_grad()
    def infer_calibrated(self, rgb01: np.ndarray,
                         fov_x_deg: Optional[float] = None) -> MoGeOutput:
        """``rgb01``: ``(3, H, W)`` in [0, 1]. ``fov_x_deg``: scalar degrees, or None.

        ``fov_x_deg=None`` reproduces Gate 5 exactly -- MoGe infers its own FOV from RGB.
        """
        assert rgb01.ndim == 3 and rgb01.shape[0] == 3, f"expected (3,H,W), got {rgb01.shape}"
        assert np.isfinite(rgb01).all(), "non-finite RGB"
        lo, hi = float(rgb01.min()), float(rgb01.max())
        assert -1e-6 <= lo and hi <= 1.0 + 1e-6, f"RGB must be in [0,1], got [{lo},{hi}]"
        if fov_x_deg is not None:
            fov_x_deg = float(fov_x_deg)
            assert np.isfinite(fov_x_deg) and 1.0 < fov_x_deg < 179.0, \
                f"implausible calibrated fov_x {fov_x_deg} deg"
        t = torch.from_numpy(np.ascontiguousarray(rgb01)).float().to(self.device)
        out = self.model.infer(t, apply_mask=False, fov_x=fov_x_deg)
        depth, points = out["depth"], out["points"]
        z = points[..., 2]
        fin = torch.isfinite(depth) & torch.isfinite(z)
        if fin.any():
            dmax = float((depth[fin] - z[fin]).abs().max())
            assert dmax < 1e-4, f"depth != points[...,2] (max |diff| {dmax})"
        K = out["intrinsics"].float().cpu().numpy()
        fov_out = float(np.degrees(2.0 * np.arctan(0.5 / max(K[0, 0], 1e-9))))
        mask = out["mask"].bool().cpu().numpy() if "mask" in out else \
            np.isfinite(depth.cpu().numpy())
        return MoGeOutput(depth_z=depth.float().cpu().numpy(), mask=mask,
                          intrinsics=K, fov_x_deg=fov_out)
