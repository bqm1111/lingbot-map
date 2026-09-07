"""Gate 8D: which privileged future frames a target may use.

Gate 8C-1 built its target from the anchor plus 20 *stream* frames, and the stream is a
stride-5 decimation of the KITTI-360 recording, so 19 of every 20 sweeps inside the horizon
were discarded. Gate 8D keeps the horizon identical and uses every native sweep inside it.

The horizon is a statement about *time*, not about sampling, which is why widening the
sampling is not widening the privilege: the last frame the target may see is unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from gates.gate8d import protocol as P


def native_window(anchor_native: int, future_natives: Sequence[int],
                  available: Dict[int, object], stride: int = P.NATIVE_STRIDE) -> List[int]:
    """Every native frame from the anchor to the last stream frame in the horizon.

    ``future_natives`` is Gate 8C-1's stride-5 list; its maximum defines the horizon. The
    returned list is the dense fill of the same interval, restricted to frames that have a
    pose, so a missing pose shortens nothing -- it simply drops that sweep.
    """
    fut = [int(f) for f in future_natives if int(f) >= 0]
    if not fut:
        return [int(anchor_native)] if int(anchor_native) in available else []
    lo, hi = int(anchor_native), max(fut)
    dense = [f for f in range(lo, hi + 1, max(1, int(stride))) if f in available]
    return dense


def coverage(anchor_native: int, future_natives: Sequence[int],
             available: Dict[int, object]) -> Dict[str, int]:
    dense = native_window(anchor_native, future_natives, available)
    sparse = [int(f) for f in [anchor_native] + list(future_natives)
              if int(f) in available]
    return {"n_dense": len(dense), "n_sparse": len(sparse),
            "horizon_native_frames": (max([int(f) for f in future_natives] or [0])
                                      - int(anchor_native)),
            "gain": len(dense) / max(len(sparse), 1)}


__all__ = ["native_window", "coverage"]
