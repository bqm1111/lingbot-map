#!/usr/bin/env python
"""Trident-H token extraction for one added drive, run in the Trident environment with the unchanged extractor
(``tools/feature_first_3d_semantics/extract_features.py``). Two redirections only: tokens are written to the cold tier
(``storage.cold_token_dir()``) and the extractor's JSON record to this study's prep directory.

    cd /home/minh/workspace/third_party/Trident && PYTHONPATH=$REPO:$PWD HF_HUB_OFFLINE=1 ../trident_env/bin/python \
        $REPO/tools/semantic_field/scaling/prepare_tokens.py --drive 2013_05_28_drive_0005_sync --device cuda:0 [--shard i --num-shards n]
"""
from __future__ import annotations

import os
import runpy
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)

from tools.feature_first_3d_semantics import common as FF          # noqa: E402
from tools.semantic_field.scaling.config import ARTIFACT_DIR       # noqa: E402
from tools.semantic_field.scaling.registration import ADDED_DRIVES, register_added_drives   # noqa: E402

COLD_TOKEN_DIR = "/media/minh/MINH1/lingbot_b4_scaling/cold/trident_tokens"   # == storage.cold_token_dir() (not imported: torch-free env path)


def main() -> None:
    drive = sys.argv[sys.argv.index("--drive") + 1]
    if drive not in ADDED_DRIVES:
        raise SystemExit(f"{drive} is not an added drive; existing tokens are never re-extracted")
    register_added_drives()
    FF.TOKEN_DIR = COLD_TOKEN_DIR
    FF.ART = os.path.join(ARTIFACT_DIR, "prep", "tokens")
    extractor = os.path.join(REPO, "tools", "feature_first_3d_semantics", "extract_features.py")
    sys.argv = [extractor] + sys.argv[1:]
    runpy.run_path(extractor, run_name="__main__")


if __name__ == "__main__":
    main()
