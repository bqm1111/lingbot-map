"""Depth conventions, the per-frame MoGe index, and the frozen LingBot gate.

Three conventions are fixed here once, because Gate 7A showed how easy they are to
confuse:

* **LingBot depth** is optical-axis *z* on the processed lattice, in **canonical** units.
  It becomes metric only after multiplication by the G51 scalar, which must multiply the
  pose translations by the same factor;
* **MoGe-2 depth** is also optical-axis *z* (``points[..., 2]``, asserted inside
  ``moge_gauge.calibrated``) and is **already metric**;
* a voxel's distance along a ray is Euclidean, not *z*. Anything that compares a voxel to
  a depth map converts explicitly.

The MoGe cache produced by Gates 5.1/5.2 is keyed by *clip*. Gate 7B needs it keyed by
*frame*, because the stream is deduplicated. :func:`moge_frame_index` builds that map and
:func:`load_moge_frame` reads it; where two clips share a frame the cached MoGe arrays are
bit-identical (asserted in ``tests/gate7b``), because MoGe is deterministic and sees only
the image and one scalar FOV.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

#: Frozen LingBot acceptance gate, unchanged since Gate 0.
CONF_THRESHOLD = 1.5
MIN_DEPTH_M = 1.0
MAX_DEPTH_M = 60.0

#: Per-frame G51-B MoGe caches written by ``tools/gate7b/cache_moge_b.py``.
MOGE_B_ROOT = "/media/SSD1/MINH_DATASETS/lingbot_gate7b/moge_b"
#: The Gate-5.2 KITTI-360 cache is already the calibrated-FOV (B) variant, verified by
#: reproducing ``artifacts/gate5_2/scales_B.csv`` to 9e-6 relative.
MOGE_CLIP_CACHE = {
    "kitti360": "/media/SSD1/MINH_DATASETS/lingbot_gate5_2/moge/B",
}
#: The Gate-5.1 dense caches for these two are the **A** variant (MoGe's own FOV), proven
#: by reproducing ``scales_G51-A_*.csv`` to 2.5e-5 while missing G51-B by 4-18 %. They are
#: therefore NOT used as a metric gauge here; the B variant is regenerated per frame.
MOGE_CLIP_CACHE_WRONG_VARIANT = {
    "semantickitti": "/media/SSD1/MINH_DATASETS/lingbot_gate5_moge/kitti",
    "occ3d": "/media/SSD1/MINH_DATASETS/lingbot_gate5_moge/occ3d",
}


def unpack_mask(packed: np.ndarray, shape) -> np.ndarray:
    """``np.packbits`` pads to a byte boundary; slice before reshaping."""
    shape = tuple(int(x) for x in shape)
    n = int(np.prod(shape))
    return np.unpackbits(np.asarray(packed, np.uint8))[:n].reshape(shape).astype(bool)


def lingbot_accepts(depth_canonical: np.ndarray, conf: np.ndarray, scale: float,
                    conf_threshold: float = CONF_THRESHOLD,
                    min_depth_m: float = MIN_DEPTH_M,
                    max_depth_m: float = MAX_DEPTH_M) -> Tuple[np.ndarray, np.ndarray]:
    """``(accepted, metric_depth)`` under a LingBot gate. Exactly Gate 6's frozen rule
    when called with the frozen constants."""
    d = float(scale) * np.asarray(depth_canonical, np.float64)
    ok = (np.asarray(conf, np.float64) >= conf_threshold) & np.isfinite(d) \
        & (d > min_depth_m) & (d < max_depth_m)
    return ok, d


def moge_frame_path(dataset: str, key: str, root: str = MOGE_B_ROOT) -> str:
    return os.path.join(root, dataset, f"{key}.npz")


def moge_frame_index(dataset: str, repo_root: str) -> Dict[str, Tuple[str, int]]:
    """``frame_key -> (clip npz path, slot)`` for the clip-keyed B caches."""
    from gates.gate6 import frames as G6F
    root = MOGE_CLIP_CACHE.get(dataset)
    if root is None:
        return {}
    out: Dict[str, Tuple[str, int]] = {}
    for rec in G6F.read_manifest(dataset, repo_root):
        p = os.path.join(root, rec.clip_id + ".npz")
        if not os.path.exists(p):
            continue
        for i, k in enumerate(rec.keys):
            out.setdefault(k, (p, i))
    return out

def load_moge_frame(dataset: str, key: str, index: Optional[Dict] = None,
                    root: str = MOGE_B_ROOT):
    """``(depth_z_metric, valid_mask, fov_x_deg)`` for one frame, or ``None``."""
    p = moge_frame_path(dataset, key, root)
    if os.path.exists(p):
        with np.load(p) as z:
            return (z["moge_depth"].astype(np.float32),
                    unpack_mask(z["moge_mask"], z["mask_shape"]),
                    float(z["fov_x_deg"]))
    if index and key in index:
        path, slot = index[key]
        with np.load(path) as z:
            return (z["moge_depth"][slot].astype(np.float32),
                    unpack_mask(z["moge_mask"], z["mask_shape"])[slot],
                    float(z["fov_x_deg"][slot]))
    return None


__all__ = ["CONF_THRESHOLD", "MIN_DEPTH_M", "MAX_DEPTH_M", "MOGE_B_ROOT",
           "MOGE_CLIP_CACHE", "MOGE_CLIP_CACHE_WRONG_VARIANT", "unpack_mask",
           "lingbot_accepts", "moge_frame_index", "load_moge_frame", "moge_frame_path"]
