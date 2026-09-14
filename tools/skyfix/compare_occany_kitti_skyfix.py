#!/usr/bin/env python
"""Run the SemanticKITTI head-to-head against OccAny for a *variant* manifest.

``tools/gate8c1/compare_occany_kitti.py`` writes ``occany_kitti_headtohead.json`` and
``fig_occany_kitti_3d.png`` into ``artifacts/gate8c1/`` -- the frozen records of the gate as
it shipped. This wrapper runs the identical code with ``ART`` redirected, so a variant (the
ceiling fix, say) can be compared without moving anything the audit depends on.

    G8C1_MANIFEST=$PWD/artifacts/gate8c1/frozen_manifest_sky.json \
      python tools/skyfix/compare_occany_kitti_skyfix.py \
        --out output/skyfix/figures --seed 2 --n 6 --device cuda:1

Every other flag is forwarded untouched; see that module's --help.
"""
from __future__ import annotations
import argparse, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", required=True, help="directory for the json and png")
    a, rest = ap.parse_known_args()
    out = os.path.abspath(a.out)
    if os.path.abspath(os.path.join(ROOT, "artifacts", "gate8c1")) == out:
        raise SystemExit("--out is the frozen artifact directory; pick another")
    os.makedirs(out, exist_ok=True)
    sys.path.insert(0, ROOT)
    import tools.gate8c1.compare_occany_kitti as M
    print(f"manifest {M.MANIFEST}\noutput   {out}", flush=True)
    M.ART = out                                    # the module reads ART at call time
    sys.argv = ["compare_occany_kitti"] + rest
    return M.main()


if __name__ == "__main__":
    raise SystemExit(main())
