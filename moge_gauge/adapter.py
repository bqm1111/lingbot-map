"""Frozen MoGe-2 adapter — a metric *gauge*, never a geometry source.

MoGe-2 contributes exactly one scalar per five-frame clip. Its depth, point map,
predicted intrinsics and predicted FOV never enter the fused reconstruction; the fusion
keeps LingBot's depth shape, LingBot's intrinsics and LingBot's relative poses.

Conventions, asserted at load time and per call:

* MoGe returns points in OpenCV camera coordinates (+x right, +y down, +z forward), so
  ``depth == points[..., 2]`` is **optical-axis z-depth**. Euclidean ray distance
  ``norm(points, axis=-1)`` is a different quantity and is never used.
* the model receives RGB in ``[0, 1]`` and nothing else: ``fov_x=None`` forces it to infer
  its own field of view from the image. No ground-truth intrinsics, extrinsics, focal
  length, camera height, pose or LiDAR is passed.
* every parameter has ``requires_grad=False``; no optimiser is ever constructed.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# Pinned upstream. Recorded in the report and asserted by the tests.
MOGE_REPO = "https://github.com/microsoft/MoGe"
MOGE_COMMIT = "74fbce054ebed49800de42d0ad0e83495065719a"
MOGE_HF_REPO = "Ruicheng/moge-2-vitl"
MOGE_HF_REVISION = "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"
MOGE_LICENSE = "MIT"
MOGE_MODEL_CLASS = "moge.model.v2.MoGeModel"          # v2 = the metric model
FORBIDDEN_INFER_KWARGS = ("fov_x", "intrinsics", "focal", "camera_height", "extrinsics",
                          "pose", "lidar", "depth")


@dataclass
class MoGeOutput:
    depth_z: np.ndarray           # (H, W) optical-axis z-depth, metres
    mask: np.ndarray              # (H, W) bool, MoGe's own validity mask
    intrinsics: np.ndarray        # (3, 3) normalised, MoGe's own prediction
    fov_x_deg: float              # derived from the predicted intrinsics


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class FrozenMoGe:
    """Loads ``Ruicheng/moge-2-vitl`` at a pinned revision and keeps it frozen."""

    def __init__(self, device, moge_src: Optional[str] = None,
                 hf_repo: str = MOGE_HF_REPO, revision: str = MOGE_HF_REVISION):
        import sys
        if moge_src and moge_src not in sys.path:
            sys.path.insert(0, moge_src)
        from moge.model.v2 import MoGeModel                      # v2, never v1 or v3

        self.model = MoGeModel.from_pretrained(hf_repo, revision=revision)
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        if trainable:
            raise RuntimeError(f"{len(trainable)} MoGe parameters are trainable")
        if not hasattr(self.model, "scale_head"):
            raise RuntimeError("loaded checkpoint has no scale_head: not the metric "
                               "MoGe-2 model")
        self.device = device
        self.hf_repo, self.revision = hf_repo, revision
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        try:
            from huggingface_hub import hf_hub_download
            wp = hf_hub_download(repo_id=hf_repo, filename="model.pt", revision=revision)
            self.weight_path, self.weight_sha256 = wp, _sha256(wp)
        except Exception:                                         # noqa: BLE001
            self.weight_path, self.weight_sha256 = None, None

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def infer(self, rgb01: np.ndarray) -> MoGeOutput:
        """``rgb01``: ``(3, H, W)`` float in [0, 1]. Returns z-depth at the same H, W."""
        assert rgb01.ndim == 3 and rgb01.shape[0] == 3, f"expected (3,H,W), got {rgb01.shape}"
        assert np.isfinite(rgb01).all(), "non-finite RGB"
        lo, hi = float(rgb01.min()), float(rgb01.max())
        assert -1e-6 <= lo and hi <= 1.0 + 1e-6, f"RGB must be in [0,1], got [{lo},{hi}]"
        t = torch.from_numpy(np.ascontiguousarray(rgb01)).float().to(self.device)
        # fov_x is deliberately omitted: MoGe infers its own field of view from RGB.
        out = self.model.infer(t, apply_mask=False)
        depth, points = out["depth"], out["points"]
        z = points[..., 2]
        fin = torch.isfinite(depth) & torch.isfinite(z)
        if fin.any():
            dmax = float((depth[fin] - z[fin]).abs().max())
            assert dmax < 1e-4, f"depth != points[...,2] (max |diff| {dmax})"
        K = out["intrinsics"].float().cpu().numpy()
        fov_x = float(np.degrees(2.0 * np.arctan(0.5 / max(K[0, 0], 1e-9))))
        mask = out["mask"].bool().cpu().numpy() if "mask" in out else \
            np.isfinite(depth.cpu().numpy())
        return MoGeOutput(depth_z=depth.float().cpu().numpy(), mask=mask,
                          intrinsics=K, fov_x_deg=fov_x)

    def provenance(self) -> Dict[str, object]:
        return {"repo": MOGE_REPO, "commit": MOGE_COMMIT, "hf_repo": self.hf_repo,
                "hf_revision": self.revision, "model_class": MOGE_MODEL_CLASS,
                "license": MOGE_LICENSE, "n_params": self.n_params,
                "weight_sha256": self.weight_sha256,
                "depth_convention": "optical-axis z (points[...,2]), OpenCV camera frame",
                "fov_source": "predicted by MoGe from RGB (fov_x=None)"}
