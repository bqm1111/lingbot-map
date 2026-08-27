"""Qualitative figures: cross-view correspondences and per-representation similarity.

Each figure takes one held-out frame pair and shows, for a handful of query patches,
where the geometry says the match is and how strongly each representation actually
agrees there.  Rendered with a non-interactive matplotlib backend so it runs headless.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from research.lingbot_semantic_memory.reprojection import Correspondences, patch_center_pixels


def _to_img(x: np.ndarray) -> np.ndarray:
    return np.clip(x.transpose(1, 2, 0), 0, 1)


def correspondence_figure(
    img_src: np.ndarray, img_dst: np.ndarray, corr: Correspondences,
    grid_hw: Tuple[int, int], image_hw: Tuple[int, int], path: str,
    n_show: int = 12, title: str = "", seed: int = 0,
) -> None:
    """Draw ``n_show`` matched patch centres side by side with connecting colours."""
    if len(corr) == 0:
        return
    uv = patch_center_pixels(grid_hw, image_hw).numpy()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(corr), size=min(n_show, len(corr)), replace=False)
    si = corr.src_idx.cpu().numpy()[pick]
    di = corr.dst_idx.cpu().numpy()[pick]
    colors = plt.cm.turbo(np.linspace(0, 1, len(pick)))

    fig, ax = plt.subplots(2, 1, figsize=(11, 4.2), constrained_layout=True)
    for a, im, idx, name in ((ax[0], img_src, si, "source"), (ax[1], img_dst, di, "target")):
        a.imshow(_to_img(im))
        a.scatter(uv[idx, 0], uv[idx, 1], c=colors, s=42, edgecolors="white", linewidths=0.8)
        a.set_ylabel(name, fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    if title:
        ax[0].set_title(title, fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def similarity_figure(
    feats: Dict[str, torch.Tensor], query_idx: int, grid_hw: Tuple[int, int],
    img_dst: np.ndarray, true_idx: int, path: str, title: str = "",
) -> None:
    """Cosine of one query patch against every patch of the target frame, per representation.

    A representation with real spatial selectivity puts a tight peak on the geometric
    match (marked); a collapsed one is uniformly bright everywhere.
    """
    gh, gw = grid_hw
    names = list(feats)
    fig, axes = plt.subplots(len(names) + 1, 1, figsize=(9, 1.35 * (len(names) + 1)),
                             constrained_layout=True)
    axes[0].imshow(_to_img(img_dst))
    ty, tx = divmod(true_idx, gw)
    H, W = img_dst.shape[1], img_dst.shape[2]
    axes[0].scatter([(tx + 0.5) * W / gw], [(ty + 0.5) * H / gh], marker="x", c="red", s=70)
    axes[0].set_ylabel("target", fontsize=8)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    if title:
        axes[0].set_title(title, fontsize=10)

    for a, name in zip(axes[1:], names):
        f = feats[name]
        q = f["src"][query_idx]
        d = f["dst"]
        q = q / q.norm().clamp_min(1e-6)
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sim = (d @ q).reshape(gh, gw).float().cpu().numpy()
        im = a.imshow(sim, cmap="magma", aspect="auto", vmin=sim.min(), vmax=sim.max())
        a.scatter([tx], [ty], marker="x", c="cyan", s=55)
        a.set_ylabel(name.replace("lingbot_", ""), fontsize=7)
        a.set_xticks([]); a.set_yticks([])
        a.text(0.995, 0.06, f"peak {sim.max():.3f} / at-match {sim[ty, tx]:.3f} / spread {sim.std():.3f}",
               transform=a.transAxes, ha="right", fontsize=6.5, color="white")
    fig.savefig(path, dpi=130)
    plt.close(fig)
