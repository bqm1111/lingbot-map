"""Cross-view correspondence between patch grids, from depth + camera poses.

Everything is expressed on the **patch lattice**: a patch ``(gy, gx)`` is represented
by its centre pixel, back-projected with the depth at that pixel, transported to the
other frame with the two world-to-camera extrinsics, and reprojected.  The resulting
pixel is snapped to the nearest patch.

A correspondence survives only if it passes, in order:

1. positive source depth and positive reprojected depth;
2. the reprojection lands inside the image;
3. **occlusion**: the reprojected depth agrees, to within ``occlusion_rel_tol``
   relative error, with the depth the target frame assigns to the patch the point
   lands in -- if the target frame sees something much nearer there, the source point
   is hidden.  The comparison is made against the *patch centre* depth rather than the
   sub-pixel depth so that the test is exactly symmetric between the two frames and
   independent of floating-point rounding in the pixel index;
4. **forward-backward consistency**: mapping the matched target patch back into the
   source frame returns to within ``fb_tol_patches`` of the original patch.

The same code runs on predicted geometry (LingBot depth + poses, arbitrary but
self-consistent scale) and on oracle geometry (KITTI ``poses.txt`` + metric depth),
which is what separates "bad semantic tokens" from "bad geometry".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class Correspondences:
    """Matched patch indices between two frames, flattened as ``gy * gw + gx``."""

    src_idx: torch.Tensor    # [N] int64
    dst_idx: torch.Tensor    # [N] int64
    src_depth: torch.Tensor  # [N] float32, depth in the source camera
    num_candidates: int      # patches considered before filtering

    def __len__(self) -> int:
        return int(self.src_idx.numel())

    @property
    def valid_fraction(self) -> float:
        return len(self) / max(self.num_candidates, 1)


def patch_center_pixels(grid_hw: Tuple[int, int], image_hw: Tuple[int, int],
                        device=None) -> torch.Tensor:
    """Centre pixel of every patch, ``[gh*gw, 2]`` as ``(u, v)`` in image coordinates."""
    gh, gw = grid_hw
    H, W = image_hw
    ph, pw = H / gh, W / gw
    v = (torch.arange(gh, device=device, dtype=torch.float32) + 0.5) * ph
    u = (torch.arange(gw, device=device, dtype=torch.float32) + 0.5) * pw
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)


def _sample_depth(depth: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour depth lookup at pixel coordinates ``[N, 2]`` -> ``[N]``."""
    H, W = depth.shape
    x = uv[:, 0].long().clamp(0, W - 1)
    y = uv[:, 1].long().clamp(0, H - 1)
    return depth[y, x]


def _transfer(uv: torch.Tensor, z: torch.Tensor, K: torch.Tensor,
              E_src: torch.Tensor, E_dst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move pixels from the source camera into the destination camera.

    Args:
        uv: ``[N, 2]`` source pixels; ``z``: ``[N]`` source depths.
        K: ``[3, 3]`` intrinsics shared by both frames.
        E_src, E_dst: ``[3, 4]`` **world-to-camera** extrinsics.

    Returns:
        ``(uv_dst [N, 2], z_dst [N])``.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cam = torch.stack([(uv[:, 0] - cx) * z / fx, (uv[:, 1] - cy) * z / fy, z], dim=-1)
    R_s, t_s = E_src[:3, :3], E_src[:3, 3]
    world = (cam - t_s) @ R_s                       # R_s^T (cam - t_s)
    R_d, t_d = E_dst[:3, :3], E_dst[:3, 3]
    cam_d = world @ R_d.T + t_d
    z_d = cam_d[:, 2]
    safe = torch.where(z_d.abs() < 1e-6, torch.full_like(z_d, 1e-6), z_d)
    u_d = cam_d[:, 0] * fx / safe + cx
    v_d = cam_d[:, 1] * fy / safe + cy
    return torch.stack([u_d, v_d], dim=-1), z_d


def compute_correspondences(
    depth_src: torch.Tensor, depth_dst: torch.Tensor,
    E_src: torch.Tensor, E_dst: torch.Tensor, K: torch.Tensor,
    grid_hw: Tuple[int, int],
    occlusion_rel_tol: float = 0.05,
    fb_tol_patches: float = 1.0,
    valid_src: Optional[torch.Tensor] = None,
    valid_dst: Optional[torch.Tensor] = None,
) -> Correspondences:
    """Patch-level correspondences from the source to the destination frame.

    Args:
        depth_src, depth_dst: ``[H, W]`` depth maps in a *shared* scale.
        E_src, E_dst: ``[3, 4]`` world-to-camera extrinsics.
        K: ``[3, 3]`` intrinsics on the same pixel grid as the depth maps.
        grid_hw: ``(gh, gw)`` patch lattice.
        valid_src, valid_dst: optional ``[gh*gw]`` boolean patch masks (e.g. depth
            confidence filtering).

    Returns:
        :class:`Correspondences` after occlusion and forward-backward filtering.
    """
    device = depth_src.device
    H, W = depth_src.shape
    gh, gw = grid_hw
    ph, pw = H / gh, W / gw

    uv = patch_center_pixels(grid_hw, (H, W), device=device)
    n = uv.shape[0]
    # Patch-level depth: one integer lookup per patch, reused everywhere below so no
    # later floating-point pixel index can flip a bin.
    z_src = _sample_depth(depth_src, uv)
    z_dst = _sample_depth(depth_dst, uv)

    keep = torch.isfinite(z_src) & (z_src > 0)
    if valid_src is not None:
        keep = keep & valid_src.to(device)

    uv_d, z_d = _transfer(uv, z_src, K, E_src, E_dst)
    keep = keep & torch.isfinite(z_d) & (z_d > 0) & torch.isfinite(uv_d).all(-1)
    keep = keep & (uv_d[:, 0] >= 0) & (uv_d[:, 0] < W) & (uv_d[:, 1] >= 0) & (uv_d[:, 1] < H)

    uv_d_safe = torch.where(keep.unsqueeze(-1), uv_d, torch.zeros_like(uv_d))
    gx = (uv_d_safe[:, 0] / pw).floor().clamp(0, gw - 1).long()
    gy = (uv_d_safe[:, 1] / ph).floor().clamp(0, gh - 1).long()
    dst_idx = gy * gw + gx

    # Occlusion: does the destination frame agree the surface sits at z_d?
    z_obs = z_dst[dst_idx]
    rel = (z_d - z_obs).abs() / z_obs.clamp_min(1e-6)
    keep = keep & torch.isfinite(z_obs) & (z_obs > 0) & (rel < occlusion_rel_tol)
    if valid_dst is not None:
        keep = keep & valid_dst.to(device)[dst_idx]

    # Forward-backward: send the matched patch centre back and compare to the source.
    uv_ret, z_ret = _transfer(uv[dst_idx], z_obs, K, E_dst, E_src)
    err_patches = (uv_ret - uv).norm(dim=-1) / max(ph, pw)
    keep = keep & torch.isfinite(err_patches) & (err_patches < fb_tol_patches) & (z_ret > 0)

    src_idx = torch.arange(n, device=device)[keep]
    return Correspondences(
        src_idx=src_idx, dst_idx=dst_idx[keep], src_depth=z_src[keep], num_candidates=n,
    )
