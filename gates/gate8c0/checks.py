"""The invariants Stage 5 asserts, as callable predicates.

Each returns ``(ok, detail)`` so the audit tool can report *why* a sample failed and the
deliberate-failure tests in ``tests/gate8c0`` can prove the check actually fires. Keeping
them here rather than inline in the tool is what makes the injection tests possible.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

FUTURE_FRAMES = 20


def causal_input_only(t: int, input_frames: Sequence[int]) -> Tuple[bool, dict]:
    """The causal input may contain frames 0..t and nothing later."""
    a = np.asarray(input_frames)
    bad = a[a > t]
    return bool(len(bad) == 0 and a.max(initial=-1) == t and a.min(initial=0) == 0), \
        {"t": int(t), "max": int(a.max(initial=-1)), "min": int(a.min(initial=0)),
         "n_future_in_input": int(len(bad)),
         "why": "input_frames must be exactly 0..t" if len(bad) else ""}


def future_target_window(t: int, target_frames: Sequence[int], n_seq: int = None,
                         n_future: int = FUTURE_FRAMES) -> Tuple[bool, dict]:
    """The privileged target may contain only t+1 .. t+n_future, clipped to the sequence."""
    a = np.asarray(target_frames)
    lo, hi = int(a.min(initial=t + 1)), int(a.max(initial=t + 1))
    ok = lo == t + 1 and hi <= t + n_future and len(a) == len(set(a.tolist()))
    if n_seq is not None:
        ok = ok and hi <= n_seq - 1
    return bool(ok), {"t": int(t), "lo": lo, "hi": hi, "n": int(len(a)),
                      "expected_lo": t + 1, "max_allowed": t + n_future}


def no_future_leak(input_frames: Sequence[int], target_frames: Sequence[int]) -> Tuple[bool, dict]:
    """Input and privileged-target frame sets must be disjoint."""
    i, f = set(int(x) for x in input_frames), set(int(x) for x in target_frames)
    inter = sorted(i & f)
    return bool(not inter), {"n_overlap": len(inter), "overlap": inter[:8]}


def integrated_once(n_integrated: int, n_frames: int) -> Tuple[bool, dict]:
    return bool(n_integrated == n_frames), {"n_integrated": int(n_integrated),
                                            "n_frames": int(n_frames)}


def scale_anchor_window(observed_indices: Sequence[int], n_anchor: int = 5) -> Tuple[bool, dict]:
    """``ScaleState`` may only have observed its first ``n_anchor`` frames."""
    a = sorted(int(x) for x in observed_indices)
    return bool(len(a) <= n_anchor and a == list(range(len(a)))), \
        {"n_observed": len(a), "indices": a[:8], "n_anchor": n_anchor}


def frame_identity_consistent(rgb_key: str, trident_key: str, pose_native: int,
                              target_anchor: int, native_of_anchor: int) -> Tuple[bool, dict]:
    """RGB, Trident cache, pose row and target must all name the same physical frame."""
    ok = (rgb_key == trident_key) and int(pose_native) == int(native_of_anchor)
    return bool(ok), {"rgb_key": rgb_key, "trident_key": trident_key,
                      "pose_native": int(pose_native), "target_anchor": int(target_anchor),
                      "native_of_anchor": int(native_of_anchor)}


def cache_keys_cannot_collide(keys_by_drive: Dict[str, Sequence[str]]) -> Tuple[bool, dict]:
    """No cache key may appear under two drives."""
    seen, dup = {}, {}
    for d, ks in keys_by_drive.items():
        for k in ks:
            if k in seen and seen[k] != d:
                dup.setdefault(k, []).append(d)
            seen[k] = d
    return bool(not dup), {"n_collisions": len(dup), "examples": list(dup)[:8]}


def grid_declaration(dims, voxel_size, origin, frame: str, expect) -> Tuple[bool, dict]:
    got = (tuple(int(x) for x in dims), float(voxel_size),
           tuple(float(x) for x in origin), str(frame))
    return bool(got == expect), {"got": got, "expect": expect}


def partitions_disjoint(train, val, heldout) -> Tuple[bool, dict]:
    s = [set(train), set(val), set(heldout)]
    inter = [sorted(a & b) for a, b in ((s[0], s[1]), (s[0], s[2]), (s[1], s[2]))]
    return bool(not any(inter)), {"train_val": inter[0], "train_heldout": inter[1],
                                  "val_heldout": inter[2]}


def unknown_handled_consistently(gt_occ, gt_valid, loss_mask) -> Tuple[bool, dict]:
    """A voxel excluded by the official mask must not be supervised."""
    sup_invalid = int((loss_mask & ~gt_valid).sum())
    return bool(sup_invalid == 0), {"n_supervised_but_invalid": sup_invalid,
                                    "n_valid": int(gt_valid.sum()),
                                    "n_supervised": int(loss_mask.sum())}


__all__ = ["causal_input_only", "future_target_window", "no_future_leak", "integrated_once",
           "scale_anchor_window", "frame_identity_consistent", "cache_keys_cannot_collide",
           "grid_declaration", "partitions_disjoint", "unknown_handled_consistently",
           "FUTURE_FRAMES"]
