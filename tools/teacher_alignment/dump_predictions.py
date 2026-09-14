#!/usr/bin/env python
"""R/S predictions on the feature study's frozen eight-anchor comparison set.

    python tools/teacher_alignment/dump_predictions.py --seed 0 --device cuda:0
"""
from __future__ import annotations
import os, shutil, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.teacher_alignment import study as ST                                   # noqa: E402

if __name__ == "__main__":
    ST.require_data_root()
    src = os.path.join(ST.FEATURE_STUDY_ART, "prediction_set.json")
    dst = os.path.join(ST.ART, "prediction_set.json")
    if not os.path.exists(dst):
        shutil.copy(src, dst)                     # the same deterministic set, unchanged
    from tools.feature_study import dump_predictions as DP                        # noqa: E402
    raise SystemExit(DP.main())
