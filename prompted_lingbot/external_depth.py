"""Frozen external metric-depth adapter.

Nothing here is trained. The external model is loaded frozen, run once per frame
into a cache, and never touched again -- exactly the treatment LingbotMap gets.

**Which model, and why not DA3.** OccAny's published baseline uses Depth Anything 3
(`from depth_anything_3.api import DepthAnything3`, DA3-LARGE). Neither the
`depth_anything_3` package nor any DA3 checkpoint exists on this machine, and the
brief forbids downloading one without confirmation. The substitute used here is
**Depth Anything V2 metric, VKITTI-finetuned, ViT-S**, which *is* present locally.
It is labelled as a substitute everywhere and is never presented as OccAny's
baseline. See `docs/frozen_depth_substitution_feasibility.md` §2.

**Depth convention (verified, not assumed).** DA-V2 metric outputs **Z-depth in
metres** at the input image resolution, with no pose, no intrinsics and no
confidence. Verified against projected LiDAR on SemanticKITTI seq 08 frame 100:
AbsRel 0.127, median(pred/lidar) 0.974, delta1 0.867 -- a scale-fitted model would
not land within 3 % of unity by accident.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

DA_V2_REPO = "/home/minh/workspace/Depth-Anything-V2/metric_depth"
DA_V2_METRIC_VKITTI_VITS = (
    "/home/minh/workspace/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_vkitti_vits.pth")

# Encoder presets, from Depth-Anything-V2/metric_depth/run.py
_ENCODER_CFG = {
    "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
    "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
    "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
}

MODELS: Dict[str, dict] = {
    # name -> everything needed to rebuild it, recorded into every cache manifest
    "da_v2_metric_vkitti_vits": dict(
        family="depth_anything_v2_metric",
        encoder="vits",
        checkpoint=DA_V2_METRIC_VKITTI_VITS,
        max_depth=80.0,
        input_size=518,
        depth_convention="z_depth_camera_frame",
        units="metres",
        provides_pose=False,
        provides_intrinsics=False,
        provides_confidence=False,
        trained_on="VKITTI (virtual KITTI), outdoor driving, metric",
    ),
}


@dataclass
class FrozenExternalDepth:
    """Frozen metric-depth model, loaded once, inference only."""

    name: str = "da_v2_metric_vkitti_vits"
    device: str = "cuda:0"
    _model: object = field(default=None, repr=False)

    @property
    def spec(self) -> dict:
        return MODELS[self.name]

    def load(self):
        import sys
        import torch

        if self._model is not None:
            return self
        spec = self.spec
        if spec["family"] != "depth_anything_v2_metric":
            raise NotImplementedError(f"no loader for {spec['family']!r}")
        if DA_V2_REPO not in sys.path:
            sys.path.insert(0, DA_V2_REPO)
        from depth_anything_v2.dpt import DepthAnythingV2

        model = DepthAnythingV2(**_ENCODER_CFG[spec["encoder"]], max_depth=spec["max_depth"])
        sd = torch.load(spec["checkpoint"], map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: {len(missing)} missing, "
                               f"{len(unexpected)} unexpected")
        model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        return self

    def infer_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        """``(H, W, 3)`` BGR uint8 -> ``(H, W)`` metric Z-depth in metres."""
        if self._model is None:
            self.load()
        import torch

        with torch.no_grad():
            return self._model.infer_image(image_bgr, input_size=self.spec["input_size"])

    def checkpoint_sha256(self) -> str:
        import hashlib

        h = hashlib.sha256()
        with open(self.spec["checkpoint"], "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()


# --------------------------------------------------------------------------- #
def resample_to_cached_lattice(depth: np.ndarray, source_hw: Tuple[int, int],
                               target_hw: Tuple[int, int], image_size: int = 518,
                               patch_size: int = 14, depth_stride: int = 2) -> np.ndarray:
    """Put external depth on the SAME lattice as the cached LingbotMap depth.

    Both models are then sampled identically, so an occupancy comparison isolates
    depth *quality* rather than point density. Nearest-neighbour, because averaging
    across a depth discontinuity invents surfaces that exist in neither map.
    """
    from PIL import Image

    from .preprocess import target_grid

    H, W = source_hw
    out_h, new_w, crop_top, new_h = target_grid((H, W), image_size, patch_size)
    d = np.array(Image.fromarray(depth.astype(np.float32)).resize((new_w, new_h),
                                                                  Image.NEAREST))
    if crop_top or new_h != out_h:
        d = d[crop_top:crop_top + out_h]
    d = d[::depth_stride, ::depth_stride]
    Ht, Wt = target_hw
    if d.shape != (Ht, Wt):
        d = np.array(Image.fromarray(d).resize((Wt, Ht), Image.NEAREST))
    return d.astype(np.float32)


# --------------------------------------------------------------------------- #
def causal_translation_scale(
    external_depth: np.ndarray, lingbot_depth: np.ndarray,
    lingbot_conf: Optional[np.ndarray] = None,
    conf_threshold: float = 1.5, min_depth: float = 1.0, max_depth: float = 70.0,
    min_samples: int = 64, mad_k: float = 3.0,
):
    """Robust ``median(D_external / D_lingbot)`` on overlapping valid pixels.

    Returns ``(log_scale, dispersion, n_inliers)`` or ``None``. Estimated in log
    space with MAD-based outlier rejection and confidence weighting, matching the
    depth-prompt estimator already validated in :mod:`prompted_lingbot.anchors`.

    This scale is applied to LingbotMap's **camera translations only**. External
    depth is already metric and is never rescaled.
    """
    from .anchors import robust_log_scale

    e = np.asarray(external_depth, np.float64)
    l = np.asarray(lingbot_depth, np.float64)
    if e.shape != l.shape:
        raise ValueError(f"lattice mismatch: external {e.shape} vs lingbot {l.shape}")
    ok = (np.isfinite(e) & np.isfinite(l) & (e > min_depth) & (e < max_depth) & (l > 1e-6))
    w = None
    if lingbot_conf is not None:
        c = np.asarray(lingbot_conf, np.float64)
        ok &= c >= conf_threshold
        w = c[ok]
    if ok.sum() < min_samples:
        return None
    return robust_log_scale(e[ok], l[ok], weights=w, mad_k=mad_k, min_samples=min_samples)
