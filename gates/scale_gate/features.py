"""Inference-legal features for the non-oracle scale predictor.

The contract (plan 4.1) is enforced here rather than by convention: this module reads
**only** the frozen LingBot cache, and the loader deliberately never receives the LiDAR
depth, the ground-truth poses, the sequence id or the dataset name. Anything a target
could leak through simply is not in scope.

Three feature blocks, switchable so the plan's baselines B / C / combined can be built
from one implementation:

``depth``  log-depth quantiles / mean / std and confidence statistics per frame,
           aggregated over the clip by mean and std.
``pose``   predicted relative-translation magnitudes and predicted focal length --
           both come from LingBot, both are available at deployment.
``image``  mean-pooled pre-GCT DINO tokens, aggregated by mean and std over the clip.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
#: Keys a feature builder must never touch, asserted in tests.
FORBIDDEN_KEYS = ("gt_pose_c2w", "gt_K_native", "depth", "valid", "sequence", "dataset")


def depth_features(z) -> np.ndarray:
    """Per-frame log-depth and confidence statistics -> ``[T, D]``."""
    dep = z["pred_depth"].astype(np.float32)
    conf = z["pred_depth_conf"].astype(np.float32)
    out = []
    for t in range(dep.shape[0]):
        d, c = dep[t].ravel(), conf[t].ravel()
        m = np.isfinite(d) & (d > 0)
        ld = np.log(d[m]) if m.any() else np.zeros(1, np.float32)
        cm = c[m] if m.any() else np.zeros(1, np.float32)
        out.append(np.concatenate([
            np.quantile(ld, QUANTILES),
            [ld.mean(), ld.std(), float(m.mean())],
            np.quantile(cm, QUANTILES),
            [cm.mean(), cm.std(), float((cm >= 1.5).mean())],
        ]))
    return np.asarray(out, np.float32)


def pose_features(z) -> np.ndarray:
    """Predicted trajectory and intrinsics statistics -> ``[T, D]`` (broadcast clip-wise)."""
    c2w = z["pred_pose_c2w"].astype(np.float64)
    K = z["pred_K"].astype(np.float64)
    t = c2w[:, :3, 3]
    step = np.linalg.norm(np.diff(t, axis=0), axis=1) if len(t) > 1 else np.zeros(1)
    span = float(np.linalg.norm(t[-1] - t[0])) if len(t) > 1 else 0.0
    path = float(step.sum())
    # log1p keeps these finite for a near-static clip without discarding the magnitude.
    clip = np.array([
        np.log1p(path), np.log1p(span), np.log1p(step.mean()), np.log1p(step.std()),
        np.log1p(step.max()), span / max(path, 1e-9),
        np.log(max(K[:, 0, 0].mean(), 1e-6)), np.log(max(K[:, 1, 1].mean(), 1e-6)),
        float(K[:, 0, 0].std()), float(K[:, 0, 2].mean()), float(K[:, 1, 2].mean()),
    ], np.float32)
    return np.tile(clip, (len(c2w), 1))


def image_features(z) -> np.ndarray:
    """Mean-pooled pre-GCT encoder tokens -> ``[T, C]``."""
    f = z["pooled_pre_gct"].astype(np.float32)
    return f if f.ndim == 2 and f.shape[1] > 1 else np.zeros((len(z["pred_depth"]), 0), np.float32)


BLOCKS = {"depth": depth_features, "pose": pose_features, "image": image_features}


def clip_feature(z, blocks: Sequence[str]) -> np.ndarray:
    """Aggregate the selected per-frame blocks over the clip with mean and std.

    Mean+std is permutation-aware but temporally simple, which is what the plan asks
    for in the first version -- no transformer, no recurrence.
    """
    parts = [BLOCKS[b](z) for b in blocks]
    parts = [p for p in parts if p.size and p.shape[1] > 0]
    if not parts:
        raise ValueError(f"no usable feature blocks among {list(blocks)}")
    per_frame = np.concatenate(parts, axis=1)
    v = np.concatenate([per_frame.mean(0), per_frame.std(0)])
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_matrix(cache_dir: str, clip_ids: Sequence[str],
                 blocks: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """``[N, D]`` feature matrix for the given clips, skipping any that fail to load."""
    import os
    X, kept = [], []
    for cid in clip_ids:
        p = os.path.join(cache_dir, "lingbot", f"{cid}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=False)
        try:
            X.append(clip_feature(z, blocks))
            kept.append(cid)
        except (ValueError, KeyError):
            continue
    return (np.stack(X) if X else np.zeros((0, 0), np.float32)), kept
