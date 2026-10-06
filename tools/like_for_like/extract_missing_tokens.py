#!/usr/bin/env python
"""Trident-H tokens for the forward-window frames the frozen evaluation never needed.

The unchanged extractor (``tools/zero_shot_eval/teacher/extract_tokens.py``) is run with its frame list redirected to this
package's list of MISSING frames (``frames.write_frame_list``); tokens land in the same shared cache, so a frame is never
extracted twice.

    cd /home/minh/workspace/third_party/Trident && PYTHONPATH=$REPO:$PWD ../trident_env/bin/python \
        $REPO/tools/like_for_like/extract_missing_tokens.py --dataset occ3d --shard 0 --shards 4 --device cuda:0
"""
from __future__ import annotations

import os
import runpy
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from tools.like_for_like import config as LC          # noqa: E402
from tools.zero_shot_eval import config as ZC         # noqa: E402


def main() -> None:
    ZC.FRAME_LIST = LC.FRAME_LIST                     # the extractor reads C.FRAME_LIST.format(dataset=...) at call time
    extractor = os.path.join(REPO, "tools", "zero_shot_eval", "teacher", "extract_tokens.py")
    sys.argv = [extractor] + sys.argv[1:]
    runpy.run_path(extractor, run_name="__main__")


if __name__ == "__main__":
    main()
