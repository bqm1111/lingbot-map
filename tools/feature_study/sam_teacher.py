"""The frozen SAM-H image encoder, read at its documented dense embedding.

Trident-H already builds ``segment_anything``'s ViT-H and drives it through
``SamPredictor``; the refinement stage consumes the encoder's dense embedding and then a
mask decoder turns it into region masks. This module stops one step earlier and keeps the
embedding itself -- ``sam.image_encoder(x) -> [1, 256, 64, 64]`` -- which is the only thing
it takes from the stack. Nothing is trained, nothing upstream is edited, and the
preprocessing is ``SamPredictor.set_image``'s, byte for byte.

Geometry of the embedding
-------------------------
``SamPredictor`` resizes the long edge to 1024, normalises, and pads the bottom and right
to a 1024x1024 square. The encoder's patch size is 16, so the output is a 64x64 lattice in
which cell ``k`` covers padded pixels ``[16k, 16k+16)``. For a 376x1408 KITTI-360 image the
picture occupies the top 273 rows, i.e. about 17 of the 64 feature rows: the embedding is
coarse in the vertical direction by construction, and that is a property of the teacher,
not of this study.

:func:`lattice_grid` composes the three maps -- LingBot processed lattice -> native pixel
-> padded pixel -> feature cell -- into one ``grid_sample`` field, so a feature can be read
at exactly the pixel a fused ray came from. The same composition (as a single bilinear
resize) is what ``gates/gate6/trident_adapter.resample_to_lattice`` does for the class
probabilities, so both teacher representations land on the same lattice the same way.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tools.feature_study.study import (SAM_CKPT, SAM_GRID, SAM_IMG_SIZE, SAM_MODEL_TYPE,
                                       SAM_NATIVE_DIM)

TRIDENT_ROOT = os.environ.get("TRIDENT_ROOT", "/home/minh/workspace/third_party/Trident")
#: encoder patch size; 1024 / 64
PATCH = SAM_IMG_SIZE // SAM_GRID


def _ensure_path() -> None:
    for p in (TRIDENT_ROOT, os.path.dirname(TRIDENT_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def preprocess_shape(hw: Tuple[int, int], long_side: int = SAM_IMG_SIZE) -> Tuple[int, int]:
    """``ResizeLongestSide.get_preprocess_shape`` -- the official rounding, reproduced."""
    h, w = float(hw[0]), float(hw[1])
    s = long_side * 1.0 / max(h, w)
    return int(h * s + 0.5), int(w * s + 0.5)


class SamFeatureTeacher:
    """``image path -> [256, 64, 64]`` dense embedding. Frozen; float32; eval mode."""

    def __init__(self, device: str = "cuda:0", ckpt: str = SAM_CKPT):
        _ensure_path()
        from segment_anything import sam_model_registry           # noqa: E402
        from segment_anything.utils.transforms import ResizeLongestSide  # noqa: E402
        self.ckpt = ckpt
        self.device = torch.device(device)
        # float32, as Trident-H itself builds ViT-H (segment-anything issue #540: ViT-H
        # overflows in fp16). Identical construction, identical weights.
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=ckpt).to(self.device).eval()
        for p in sam.parameters():
            p.requires_grad_(False)
        self.sam = sam
        self.transform = ResizeLongestSide(sam.image_encoder.img_size)
        assert sam.image_encoder.img_size == SAM_IMG_SIZE, sam.image_encoder.img_size

    @torch.no_grad()
    def encode(self, img_path: str) -> Dict[str, object]:
        """One image -> the encoder's dense embedding plus the geometry needed to read it."""
        import cv2
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        native_hw = (rgb.shape[0], rgb.shape[1])
        resized = self.transform.apply_image(rgb)                      # HWC uint8
        x = torch.as_tensor(resized, device=self.device).permute(2, 0, 1).contiguous()[None]
        input_hw = (int(x.shape[-2]), int(x.shape[-1]))
        assert input_hw == preprocess_shape(native_hw), (input_hw, native_hw)
        x = self.sam.preprocess(x)                                     # normalise + pad
        feat = self.sam.image_encoder(x)[0]                            # [256, 64, 64]
        assert tuple(feat.shape) == (SAM_NATIVE_DIM, SAM_GRID, SAM_GRID), feat.shape
        return {"feat": feat, "native_hw": native_hw, "input_hw": input_hw}


def lattice_grid(native_hw: Tuple[int, int], input_hw: Tuple[int, int],
                 lattice_hw: Tuple[int, int], device) -> torch.Tensor:
    """``grid_sample`` field mapping the LingBot lattice onto the 64x64 feature lattice.

    Composition, all in ``align_corners=False`` pixel-centre conventions:

    1. lattice cell ``(v, u)`` -> native pixel ``((v+.5) * H0/hp, (u+.5) * W0/wp)``
       (both lattices are full-extent resamplings of the same rectified image);
    2. native -> padded ``(y * h1/H0, x * w1/W0)`` (the long-edge resize, which is
       very slightly anisotropic because ``h1`` and ``w1`` are rounded independently);
    3. padded -> feature cell ``/ 16``;
    4. feature cell -> normalised ``2 * f / 64 - 1``.
    """
    H0, W0 = native_hw
    h1, w1 = input_hw
    hp, wp = lattice_hw
    v = (torch.arange(hp, device=device, dtype=torch.float64) + 0.5) * (H0 / hp)
    u = (torch.arange(wp, device=device, dtype=torch.float64) + 0.5) * (W0 / wp)
    fy = v * (h1 / H0) / PATCH
    fx = u * (w1 / W0) / PATCH
    gy = (2.0 * fy / SAM_GRID - 1.0).float()
    gx = (2.0 * fx / SAM_GRID - 1.0).float()
    gyy, gxx = torch.meshgrid(gy, gx, indexing="ij")
    return torch.stack([gxx, gyy], dim=-1)[None]                       # [1, hp, wp, 2]


def to_lattice(feat: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """``[C, 64, 64]`` -> ``[C, hp, wp]`` by bilinear ``grid_sample``."""
    out = F.grid_sample(feat[None].float(), grid.to(feat.device), mode="bilinear",
                        padding_mode="border", align_corners=False)
    return out[0]


__all__ = ["SamFeatureTeacher", "lattice_grid", "to_lattice", "preprocess_shape",
           "TRIDENT_ROOT", "PATCH"]
