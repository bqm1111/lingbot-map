#!/usr/bin/env python
"""Trident-H tokens for the five non-front cameras of the surround setting (CAM_FRONT is already cached).

The unchanged extractor runs with its frame list redirected to ``surround.FRAME_LIST``; tokens land in the shared cache.

Budget note: the cache is ~13.7 MB per frame, so the 24,095 non-front frames need ~330 GB. Check free space before
launching -- the first attempt filled SSD1 and every shard died with ENOSPC.

    cd /home/minh/workspace/third_party/Trident && PYTHONPATH=$REPO:$PWD ../trident_env/bin/python \
        $REPO/tools/like_for_like/extract_surround_tokens.py --dataset occ3d --shard 0 --shards 4 --device cuda:0
"""
from __future__ import annotations

import os
import runpy
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from tools.like_for_like import surround as SR        # noqa: E402
from tools.zero_shot_eval import config as ZC         # noqa: E402


def main() -> None:
    ZC.FRAME_LIST = SR.FRAME_LIST                     # a plain path; ``.format(dataset=...)`` leaves it unchanged
    extractor = os.path.join(REPO, "tools", "zero_shot_eval", "teacher", "extract_tokens.py")
    sys.argv = [extractor] + sys.argv[1:]
    runpy.run_path(extractor, run_name="__main__")


if __name__ == "__main__":
    main()
