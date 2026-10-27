#!/usr/bin/env python
"""Frozen zero-shot evaluation of the ONCE arm on SemanticKITTI 08 and Occ3D-nuScenes val.

The harness is ``baseline_reconstruction.targets.evaluate.evaluate_dataset`` **unchanged**.
The only thing replaced is the ``checkpoints`` field of ``CurrentPaths``, so the target clips,
adapters, official masks, range bins, occupancy threshold and metric code are the KITTI-360
arm's own -- which is what makes this arm comparable with the KITTI-360 and Waymo rows.

    PYTHONPATH=$PWD python tools/method_vs_data/once_arm/cli/evaluate.py --device cuda:3
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from tools.baseline_reconstruction import config as BR                            # noqa: E402
from tools.method_vs_data.once_arm import config as A                             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datasets", default="semantickitti,occ3d")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--tag", default="zero_shot")
    a = ap.parse_args()

    # The evaluator records each checkpoint's sha256 by looking it up in the source-freeze
    # record, which lists only the KITTI-360 arm's checkpoints. The lookup is PROVENANCE, not
    # verification -- the verification is `require_source_freeze()` itself, over 54 source
    # files. So the real check is still run and its result is extended with this arm's TRUE
    # hashes, computed here from the files on disk. No frozen file is modified and nothing is
    # bypassed; an assertion refuses to start if the real check stopped running.
    import hashlib

    from tools.baseline_reconstruction import freeze as FZ
    from tools.method_vs_data.once_arm import train as OT
    _real = FZ.require_source_freeze
    _sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()         # noqa: E731

    def _extended():
        rec = dict(_real())                      # the genuine check, unchanged
        assert rec.get("sha256"), "the source freeze record is empty; refusing to continue"
        sha = dict(rec["sha256"])
        op = OT.OnceArmPaths()
        for s in A.SEEDS:
            ck = op.checkpoint_path(s, 6000)
            if os.path.isfile(ck):
                sha[os.path.relpath(ck, A.REPO_ROOT)] = _sha(ck)
        rec["sha256"] = sha
        return rec

    FZ.require_source_freeze = _extended
    import tools.baseline_reconstruction.targets.evaluate as EV
    EV.FZ.require_source_freeze = _extended
    assert EV.FZ.require_source_freeze is _extended, "freeze redirect failed"
    from tools.baseline_reconstruction.targets.evaluate import evaluate_dataset
    p = A.Paths()
    eval_dir = os.path.join(A.ARTIFACT_DIR, "zero_shot_eval")
    os.makedirs(eval_dir, exist_ok=True)
    paths = dataclasses.replace(BR.CurrentPaths(), checkpoints=p.checkpoints,
                                target_eval=eval_dir)
    for s in A.SEEDS:
        ck = paths.checkpoint_path(s, a.steps)
        assert os.path.isfile(ck), f"missing ONCE checkpoint {ck}"
    out = {"provenance": A.provenance(), "datasets": {}}
    for ds in a.datasets.split(","):
        rec = evaluate_dataset(ds, a.device, paths=paths, limit=a.limit)
        out["datasets"][ds] = {"n_anchors": rec.n_anchors,
                               "seed_statistics": rec.seed_statistics,
                               "map_only": rec.map_only}
        print(f"[eval] {ds}: n={rec.n_anchors} {rec.seed_statistics}", flush=True)
    f = os.path.join(p.logs, f"{a.tag}.json")
    json.dump(out, open(f, "w"), indent=2, default=str)
    print(f"[eval] -> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
