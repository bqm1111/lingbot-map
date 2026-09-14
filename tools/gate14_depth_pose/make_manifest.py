#!/usr/bin/env python
"""Per-drive clip manifest for the MUSt3R environment: for every local-reset L clip (stride-3 anchors,
the Gate 9 / Gate 13 clips), the anchor t, its stream indices t-4..t, frame keys and native image paths.
The MUSt3R process reads only this file and the listed images, so it never imports this repository.

    python tools/gate14_depth_pose/make_manifest.py --drives 2013_05_28_drive_0006_sync [...]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g14                                                                                   # noqa: E402
sys.path.insert(0, g14.REPO_ROOT)
from core.datasets import kitti360_splits as SRC, sources as S                              # noqa: E402
from core.run._common import REPO_ROOT                                                       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--drives", nargs="+", required=True); a = ap.parse_args()
    os.makedirs(g14.MANIFEST_DIR, exist_ok=True)
    for d in a.drives:
        seg = [s for s in S.segments(SRC.source_of(d), REPO_ROOT) if s.name == d][0]
        clips = []
        for f in sorted(glob.glob(os.path.join(g14.LR_ROOT, d, "*.npz"))):
            with np.load(f) as z:
                t = int(z["t"]); idx = [int(x) for x in z["stream_indices"]]; keys = [str(k) for k in z["keys"]]
            assert idx == [t - 4, t - 3, t - 2, t - 1, t], (d, t, idx)
            assert keys == [seg.frames[j].key for j in idx], (d, t)
            clips.append({"t": t, "stream_indices": idx, "keys": keys, "image_paths": [seg.frames[j].path for j in idx]})
        json.dump({"drive": d, "n_clips": len(clips), "window": "t-4..t (causal)", "clips": clips}, open(g14.clip_manifest(d), "w"), indent=1)
        print(d, len(clips), "clips; first", clips[0]["image_paths"][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
