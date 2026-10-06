"""The frames our heads need under OccAny's forward protocol, in the token extractor's schema.

Most are already cached by the frozen evaluation (they are other samples' targets); only the remainder is extracted.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from tools.like_for_like import config as LC
from tools.occany_comparison import samples as OS
from tools.zero_shot_eval import config as ZC


def window_frames(sample) -> List[Dict[str, str]]:
    """The five frames of the forward window, in the clip's order (target first)."""
    return [{"key": sample.frame_keys[j], "path": sample.frame_paths[j]} for j in sample.window(LC.PROTOCOL)]


def samples(dataset: str) -> List:
    return [s for s in OS.build_samples(dataset) if s.protocol_ok(LC.PROTOCOL)]


def write_frame_list(dataset: str) -> Dict:
    """Write every frame of the forward windows, marking which tokens are still missing."""
    seen, frames = set(), []
    ss = samples(dataset)
    for s in ss:
        for f in window_frames(s):
            if f["key"] not in seen:
                seen.add(f["key"]); frames.append(f)
    missing = [f for f in frames if not os.path.exists(ZC.token_path(dataset, f["key"]))]
    rec = {"dataset": dataset, "protocol": LC.PROTOCOL.name, "n_samples": len(ss), "n_frames": len(frames),
           "n_missing_tokens": len(missing), "frames": missing}          # the extractor only needs the missing ones
    LC.ensure_dirs()
    json.dump(rec, open(LC.FRAME_LIST.format(dataset=dataset), "w"))
    return {k: v for k, v in rec.items() if k != "frames"}


__all__ = ["window_frames", "samples", "write_frame_list"]
