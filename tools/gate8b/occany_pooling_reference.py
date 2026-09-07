#!/usr/bin/env python
"""Run OccAny's *official* ``apply_majority_pooling`` on an array and save the result.

Executed by the Gate 8B test inside the OccAny environment, so the reproduction in
``gate8b/pooling.py`` is checked against the real function, not a transcription of it.

    <occany_env python> tools/gate8b/occany_pooling_reference.py in.npy out.npz
"""
from __future__ import annotations
import importlib.util, os, sys, types
import numpy as np

OCCANY = "/home/minh/workspace/OccAny"


def load_official():
    """Import ``apply_majority_pooling`` while stubbing the heavy, irrelevant imports."""
    sys.path.insert(0, OCCANY)
    for name in ("cv2", "occany.utils.image_util", "occany.utils.cropping", "einops",
                 "depth_anything_3", "depth_anything_3.utils", "depth_anything_3.utils.geometry",
                 "occany.utils.fusion"):
        if name not in sys.modules:
            try:
                __import__(name)
            except Exception:
                m = types.ModuleType(name)
                m.__getattr__ = lambda *_a, **_k: (lambda *a, **k: None)
                sys.modules[name] = m
    spec = importlib.util.spec_from_file_location(
        "occany_helpers", os.path.join(OCCANY, "occany", "utils", "helpers.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.apply_majority_pooling


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    x = np.load(src)
    fn = load_official()
    n_classes, other, empty = 20, 19, 0
    out = {
        "geometry_dilation": fn(x.copy(), n_classes, other, empty, is_geometry_only=True),
        "geometry_majority": fn(x.copy(), n_classes, other, empty, is_geometry_only=True,
                                use_dilation=False),
        "semantic_separate": fn(x.copy(), n_classes, other, empty, is_geometry_only=False),
    }
    np.savez(dst, **out)
    print("ok", {k: v.shape for k, v in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
