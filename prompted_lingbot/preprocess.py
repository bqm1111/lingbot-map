"""Replicate ``load_and_preprocess_images``' geometry so ground truth can be put
on exactly the same pixel grid as the predictions.

``lingbot_map.utils.load_fn.load_and_preprocess_images`` (mode="crop") does:

    new_w = image_size
    new_h = round(H_src * (new_w / W_src) / patch_size) * patch_size
    resize (bicubic) to (new_w, new_h)
    if new_h > image_size: centre-crop the height to image_size

Depth is a Z coordinate, so resampling the image grid does not change depth
values -- only the pixel lattice and the intrinsics change.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def target_grid(src_hw: Tuple[int, int], image_size: int = 518, patch_size: int = 14):
    """Return ``(new_h, new_w, crop_top)`` for the preprocessed frame."""
    H, W = src_hw
    new_w = image_size
    new_h = int(round(H * (new_w / W) / patch_size) * patch_size)
    crop_top = (new_h - image_size) // 2 if new_h > image_size else 0
    out_h = min(new_h, image_size)
    return out_h, new_w, crop_top, new_h


def resample_gt_intrinsics(K: np.ndarray, src_hw: Tuple[int, int], image_size: int = 518,
                           patch_size: int = 14) -> np.ndarray:
    """Map source-image intrinsics onto the preprocessed grid."""
    H, W = src_hw
    out_h, new_w, crop_top, new_h = target_grid(src_hw, image_size, patch_size)
    sx, sy = new_w / W, new_h / H
    K2 = K.astype(np.float64).copy()
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] = K2[1, 2] * sy - crop_top
    return K2


def resample_depth(depth: np.ndarray, valid: np.ndarray, image_size: int = 518,
                   patch_size: int = 14):
    """Nearest-neighbour resample a depth map + mask onto the preprocessed grid.

    Nearest neighbour (not bilinear) because averaging across a depth
    discontinuity invents surfaces that exist in neither frame.
    """
    H, W = depth.shape
    out_h, new_w, crop_top, new_h = target_grid((H, W), image_size, patch_size)
    d = np.array(Image.fromarray(depth.astype(np.float32)).resize((new_w, new_h), Image.NEAREST))
    m = np.array(Image.fromarray(valid.astype(np.uint8)).resize((new_w, new_h), Image.NEAREST)) > 0
    if crop_top or new_h != out_h:
        d = d[crop_top:crop_top + out_h]
        m = m[crop_top:crop_top + out_h]
    return d, m
