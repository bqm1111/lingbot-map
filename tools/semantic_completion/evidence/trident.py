"""Frozen Trident-H dense features and text embeddings, read from the Gate feature-first caches.

Token map: ``[1024, 34, 128]`` float32, **unit L2 norm per token**, covering the full native frame.
The official coarse prediction is ``bilinear(tokens @ text^T)`` to native with ``align_corners=False``;
sampling the tokens with the same interpolation at any image point is therefore the teacher's own
feature at that point (up to the renormalisation applied afterwards).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tools.semantic_completion import config as C


def load_token_map(path: str, device) -> torch.Tensor:
    """``[1024, h, w]`` float32 unit-norm tokens."""
    with np.load(path) as z:
        h, w = (int(x) for x in z["grid_hw"])
        tok = torch.from_numpy(z["tokens"].astype(np.float32)).to(device)
    assert tok.shape == (h * w, C.TEACHER_DIM), tok.shape
    return tok.T.reshape(C.TEACHER_DIM, h, w)


def sample_at_lattice(token_map: torch.Tensor, lattice_hw: Tuple[int, int]) -> torch.Tensor:
    """Teacher feature at every LingBot lattice pixel centre: ``[LH, LW, 1024]``, unit norm.

    The lattice is a full-frame resize of the native image (LingBot ``mode="crop"`` never crops a
    1408x376 frame), so lattice pixel centre ``(v + 0.5) / LH`` and the token map share one normalised
    coordinate.
    """
    LH, LW = (int(x) for x in lattice_hw)
    ys = (torch.arange(LH, device=token_map.device, dtype=torch.float32) + 0.5) / LH * 2 - 1
    xs = (torch.arange(LW, device=token_map.device, dtype=torch.float32) + 0.5) / LW * 2 - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], -1)[None]
    out = F.grid_sample(token_map[None], grid, mode="bilinear", align_corners=False)[0]
    return F.normalize(out.permute(1, 2, 0), dim=-1)


def load_text_embeddings(path: str = C.Paths().text_embeddings) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """``(text [18, 1024] unit norm, phrases [18], labels [18] = 1..18)`` -- the frozen Trident vocabulary."""
    with np.load(path) as z:
        text = torch.from_numpy(z["text"].astype(np.float32))
        phrases, labels = z["phrases"].astype(str), z["labels"].astype(np.int64)
    assert text.shape == (C.N_CLASSES, C.TEACHER_DIM) and list(labels) == list(range(1, C.N_CLASSES + 1))
    return F.normalize(text, dim=-1), phrases, labels


def classify(features: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    """Open-vocabulary readout: label id ``1 + argmax_c cos(f, text_c)`` for ``features [..., 1024]``."""
    return 1 + (F.normalize(features, dim=-1) @ text.to(features.device, features.dtype).T).argmax(-1)


__all__ = ["load_token_map", "sample_at_lattice", "load_text_embeddings", "classify"]
