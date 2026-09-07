"""Gate 8D: the Gate 8C-1 completion network plus one surface-aware auxiliary output.

The occupancy path is untouched -- same residual, same lock, same 32 input channels, same
widths, same ``replicate`` padding. The only change is a second 1x1x1 head predicting the
**truncated unsigned distance to the nearest future surface, in metres**.

Why metres and not voxels: the stress protocol varies the voxel size between 0.15 m and
0.45 m, so a distance expressed in voxel indices would mean a different physical thing in
every variant and the head could not transfer. Metres are invariant.

Why a distance head at all: a thick surface and a thin one are equally good under a pure
occupancy loss as long as the occupied set overlaps, so nothing in Gate 8C-1 discouraged
smearing the road across several voxels. A distance target penalises thickness directly,
without a hardcoded height filter and without touching the decision rule -- occupancy
remains the only inference output.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from gates.gate8 import vocab as V8
from gates.gate8.net import LOCK_LOGODDS, CompletionUNet, apply_residual
from gates.gate8d import protocol as P


class SurfaceCompletionUNet(CompletionUNet):
    """``CompletionUNet`` with an added metric surface-distance head."""

    def __init__(self, cin: int = None, width: int = 24, n_sem: int = V8.U,
                 padding_mode: str = "replicate"):
        kw = {} if cin is None else {"cin": cin}
        super().__init__(width=width, n_sem=n_sem, padding_mode=padding_mode, **kw)
        self.head_dist = nn.Conv3d(width, 1, 1)
        nn.init.zeros_(self.head_dist.weight)
        nn.init.constant_(self.head_dist.bias, float(P.TUDF_TRUNCATION_M))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        dist = self.head_dist(d1).clamp(0.0, float(P.TUDF_TRUNCATION_M))
        return self.head_occ(d1), self.head_sem(d1), dist


def surface_distance_loss(pred_m: torch.Tensor, target_m: torch.Tensor,
                          valid: torch.Tensor) -> torch.Tensor:
    """L1 in metres, only where KITTI-360 evidence makes the distance meaningful.

    Genuinely unknown space carries no distance target and contributes no gradient, exactly
    as the occupancy loss ignores it.
    """
    w = valid.float()
    if w.sum() < 1:
        return pred_m.sum() * 0.0
    return ((pred_m - target_m).abs() * w).sum() / w.sum().clamp_min(1.0)


def save_checkpoint(net: SurfaceCompletionUNet, path: str, meta: Dict) -> None:
    torch.save({"state_dict": net.state_dict(), "meta": meta,
                "cin": net.enc1[0].in_channels, "width": net.enc1[0].out_channels,
                "padding_mode": getattr(net, "padding_mode", "replicate"),
                "arch": "gate8d.net.SurfaceCompletionUNet",
                "tudf_truncation_m": float(P.TUDF_TRUNCATION_M)}, path)


class SurfaceCompleter:
    """Inference wrapper. Occupancy is the only output the evaluator ever reads."""

    def __init__(self, net: SurfaceCompletionUNet, device):
        self.net = net.to(device).eval()
        self.device = device

    @torch.no_grad()
    def raw(self, q: Dict[str, torch.Tensor], grid, pad_z: int = 0):
        import torch.nn.functional as F
        X, Y, Z = (int(d) for d in grid.dims)
        n = X * Y * Z
        sem, semw = q["sem"], q["sem_w"]
        obs = q["observed"].float()
        age = q.get("age", torch.full((n,), -1.0, device=self.device)).float()
        base = q["logodds"].reshape(X, Y, Z)
        inp = torch.stack([base / 4.0, q["w_free"].reshape(X, Y, Z).clamp(max=20) / 20.0,
                           obs.reshape(X, Y, Z), 1.0 - obs.reshape(X, Y, Z),
                           q["n_obs"].float().reshape(X, Y, Z).clamp(max=50) / 50.0,
                           (age.clamp(min=0) / 100.0 * (age >= 0)).reshape(X, Y, Z),
                           semw.reshape(X, Y, Z).clamp(max=20) / 20.0] +
                          [(sem[:, u] / semw.clamp_min(1e-9)).reshape(X, Y, Z)
                           for u in range(V8.U)], 0)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            occ_res, sem_log, _dist = self.net(inp)
        occ_res = occ_res[0, 0].float()
        sem_log = sem_log[0].float()
        final = apply_residual(base, occ_res).reshape(-1)
        map_probs = sem / semw.clamp_min(1e-9).unsqueeze(1)
        net_probs = F.softmax(sem_log.reshape(V8.U, -1).T, dim=1)
        #: observed semantics are never overwritten -- where the frozen teacher fused any
        #: evidence the map's own distribution is returned unchanged
        probs = torch.where((semw > 0).unsqueeze(1), map_probs, net_probs)
        return final, probs

    @torch.no_grad()
    def complete(self, q, grid):
        final, probs = self.raw(q, grid)
        return final > 0, probs


def load_checkpoint(path: str, device) -> SurfaceCompleter:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = SurfaceCompletionUNet(cin=ck["cin"], width=ck["width"],
                                padding_mode=ck.get("padding_mode", "replicate"))
    net.load_state_dict(ck["state_dict"])
    return SurfaceCompleter(net, device)


__all__ = ["SurfaceCompletionUNet", "SurfaceCompleter", "surface_distance_loss",
           "save_checkpoint", "load_checkpoint", "apply_residual", "LOCK_LOGODDS"]
