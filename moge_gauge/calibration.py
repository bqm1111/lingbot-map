"""Gate 5.1 — calibrated FOV and the deterministic aspect-safe crop.

One generic policy, applied identically to every dataset. Nothing here is dataset-specific
and nothing is hard-coded for SemanticKITTI.

Camera intrinsics are ordinary calibrated inference metadata -- the voxel projection
already requires them. Occupancy labels, LiDAR depth and oracle scale remain forbidden.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gates.scale_gate.kitti import Preprocess                                      # noqa: E402

MAX_ASPECT_RATIO = 2.0        # MoGe's documented operating range is 2:1 .. 1:2
PATCH = 14                    # the LingBot/DINOv2 patch lattice the depth lives on


@dataclass(frozen=True)
class CropSpec:
    """An integer, axis-aligned horizontal crop. No resize, pad or letterbox, ever."""
    x0: int
    x1: int
    width: int
    height: int
    cx_crop: float
    identity: bool

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def apply(self, arr: np.ndarray) -> np.ndarray:
        """Pure integer slice of the last axis. Height untouched; no resize/pad/letterbox."""
        if arr.shape[-1] < self.x1:
            raise ValueError(f"array width {arr.shape[-1]} < crop x1 {self.x1}")
        if arr.shape[-2] != self.height:
            raise ValueError(f"array height {arr.shape[-2]} != crop height {self.height}")
        out = arr[..., self.x0:self.x1]
        assert out.shape[-1] == self.width
        return out

    def to_dict(self) -> dict:
        return {"x0": self.x0, "x1": self.x1, "width": self.width, "height": self.height,
                "cx_crop": self.cx_crop, "identity": self.identity,
                "aspect_ratio": self.aspect_ratio}


def processed_intrinsics(K_native: np.ndarray, orig_hw: Tuple[int, int],
                         image_size: int = 518, patch_size: int = PATCH):
    """Native calibrated K through the exact frozen LingBot preprocessing transform."""
    pre = Preprocess.build((int(orig_hw[0]), int(orig_hw[1])), image_size, patch_size)
    return pre.scale_intrinsics(np.asarray(K_native, dtype=np.float64)), pre.proc_hw


def aspect_safe_crop(width: int, height: int, cx: float, patch: int = PATCH,
                     max_ar: float = MAX_ASPECT_RATIO) -> CropSpec:
    """Deterministic crop keeping the aspect ratio inside MoGe's operating range.

    * ``W/H <= max_ar``  -> identity (the full lattice).
    * otherwise          -> the widest crop with ``W_crop <= max_ar * H`` that is a whole
      number of ``patch`` columns, full height, centred as closely as possible on the
      calibrated principal point and clipped to the image.

    Deterministic: the same inputs always give the same integer coordinates.
    """
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"bad lattice {width}x{height}")
    if width / height <= max_ar:
        return CropSpec(0, width, width, height, float(cx), True)
    w_crop = int((max_ar * height) // patch) * patch
    if w_crop < patch:
        raise ValueError(f"lattice {width}x{height} admits no {patch}-aligned crop")
    w_crop = min(w_crop, width)
    x0 = int(round(float(cx) - w_crop / 2.0))
    x0 = max(0, min(x0, width - w_crop))            # clip to the valid image boundary
    return CropSpec(x0, x0 + w_crop, w_crop, height, float(cx) - x0, False)


def crop_for(K_native: np.ndarray, orig_hw: Tuple[int, int], image_size: int = 518,
             patch_size: int = PATCH, max_ar: float = MAX_ASPECT_RATIO):
    """``(CropSpec, K_processed, proc_hw)`` for one camera."""
    Kp, proc_hw = processed_intrinsics(K_native, orig_hw, image_size, patch_size)
    H, W = int(proc_hw[0]), int(proc_hw[1])
    # A horizontal slice never changes the focal length, so fx is a property of
    # ``K_processed`` alone and is deliberately not duplicated inside CropSpec.
    c = aspect_safe_crop(W, H, float(Kp[0, 2]), patch_size, max_ar)
    return c, Kp, (H, W)


def calibrated_fov_x_deg(fx: float, width: int) -> float:
    """Horizontal field of view in **degrees** -- the unit MoGe-2 v2 `infer` expects.

        fov_x = 2 * atan(W / (2 * fx))

    Verified against the pinned implementation, which applies ``torch.deg2rad(fov_x / 2)``.
    """
    fx = float(fx)
    if not (fx > 0) or width <= 0:
        raise ValueError(f"bad camera: fx={fx}, width={width}")
    return math.degrees(2.0 * math.atan(float(width) / (2.0 * fx)))
