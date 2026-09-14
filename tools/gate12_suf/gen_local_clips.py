#!/usr/bin/env python
"""Stride-1 local-reset L clips for the training drives: the unchanged
tools/local_registration_audit/run_local_lingbot.py, redirected to the Gate 12 data root with anchor
stride 1 (the local-registration audit ran stride 3). Same checkpoint, preprocessing, reset and
causal MoGe-2 clip scale. Anchors already produced at stride 3 must come out bit-identical (verify).

    python tools/gate12_suf/gen_local_clips.py --drive 2013_05_28_drive_0007_sync --device cuda:0 [--shard i --shards n]
"""
from __future__ import annotations
import os, runpy, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g12                                                                                  # noqa: E402
sys.path.insert(0, g12.REPO_ROOT)
from tools.local_registration_audit import common as LR                                     # noqa: E402

LR.DATA_ROOT = g12.DATA_ROOT            # clip_path -> <DATA_ROOT>/local_lingbot/<drive>/<t>.npz
LR.ANCHOR_STRIDE = 1

if __name__ == "__main__":
    assert "--drive" in sys.argv and sys.argv[sys.argv.index("--drive") + 1] in g12.TRAIN_DRIVES, "training drives only"
    sys.argv = [os.path.join(g12.REPO_ROOT, "tools", "local_registration_audit", "run_local_lingbot.py")] + sys.argv[1:]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except SystemExit as e:
        raise SystemExit(e.code)
