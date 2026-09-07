"""OccAny's *official* scene-completion metric and post-processing, called directly.

The brief requires OccAny's own evaluator and pooling rather than a reimplementation of
them, so this module loads them from the checkout at ``/home/minh/workspace/OccAny`` by
file path. ``occany/metrics/ssc.py`` has no heavy dependencies and imports cleanly; the
pooling functions live in ``occany/utils/helpers.py``, whose module-level imports do not
resolve here, so :mod:`gate8b.pooling` supplies them -- that reproduction was verified
bit-for-bit against the official function inside the OccAny environment in Gate 8B and is
re-checked by ``tests/gate8c1``.

What this module cannot do is run the released OccAny *model*; see
``artifacts/gate8c1/occany_reproduction.md`` for exactly which dependencies are missing.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Dict, Sequence, Tuple

import numpy as np

OCCANY_ROOT = "/home/minh/workspace/OccAny"
SSC_PATH = os.path.join(OCCANY_ROOT, "occany", "metrics", "ssc.py")
#: OccAny's published five-frame single-camera scene-completion IoU.
PUBLISHED_5FRAME = {"semantickitti": {"precision": 0.3679, "recall": 0.4670, "sc_iou": 0.2591},
                    "occ3d": {"precision": 0.3609, "recall": 0.4039, "sc_iou": 0.2355}}


def available() -> bool:
    return os.path.exists(SSC_PATH)


def _load():
    spec = importlib.util.spec_from_file_location("occany_ssc", SSC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def official_sc_counts(pred_label: np.ndarray, target_label: np.ndarray, n_classes: int,
                       empty_class: int) -> Tuple[int, int, int]:
    """``(tp, fp, fn)`` from OccAny's ``SSCMetrics.get_score_completion``.

    ``pred_label`` and ``target_label`` are integer label volumes on the evaluation grid
    with ``255`` marking ignore, exactly as OccAny's own evaluator expects.
    """
    mod = _load()
    sm = mod.SSCMetrics(n_classes=n_classes,
                        class_names=[str(i) for i in range(n_classes)],
                        other_class=n_classes - 1, ignore_other_class_in_mIoU=False,
                        empty_class=empty_class)
    p = np.asarray(pred_label)[None] if pred_label.ndim == 3 else np.asarray(pred_label)
    t = np.asarray(target_label)[None] if target_label.ndim == 3 else np.asarray(target_label)
    tp, fp, fn = sm.get_score_completion(p.copy(), t.copy())
    return int(tp), int(fp), int(fn)


def rates(tp: int, fp: int, fn: int) -> Dict[str, float]:
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "sc_iou": tp / max(tp + fp + fn, 1)}


__all__ = ["available", "official_sc_counts", "rates", "PUBLISHED_5FRAME", "OCCANY_ROOT"]
