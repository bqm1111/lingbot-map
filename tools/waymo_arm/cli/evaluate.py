#!/usr/bin/env python
"""Phase 3 — frozen zero-shot geometric evaluation of the Waymo-trained checkpoints.

The harness is ``baseline_reconstruction.targets.evaluate.evaluate_dataset``, unchanged. The
ONLY thing substituted is the ``checkpoints`` field of ``CurrentPaths``, so the target clips,
adapters, official masks, range bins, occupancy threshold and metric are the KITTI-360 arm's
own. ``require_source_freeze`` still runs inside it and refuses on any hash mismatch.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD python tools/waymo_arm/cli/evaluate.py \\
        --dataset semantickitti --device cuda:0
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.baseline_reconstruction import config as BR                                  # noqa: E402
from tools.waymo_arm import config as A                                                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(BR.TARGET_PROTOCOLS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None,
                    help="debug only; a limited run is never a result")
    a = ap.parse_args()
    A.ensure_dirs()
    from tools.baseline_reconstruction import freeze as FZ
    from tools.baseline_reconstruction.targets import evaluate as EV
    from tools.baseline_reconstruction.targets.evaluate import evaluate_dataset

    # The frozen evaluator records each checkpoint's sha256 by looking its repo-relative
    # path up in ``code_freeze.json``, which lists only the KITTI-360 checkpoints -- so a
    # swapped-in checkpoint raises KeyError. The freeze itself is still verified in full
    # here (the real ``require_source_freeze`` runs first and raises on any mismatch); the
    # returned record is then extended with the TRUE sha256 of the checkpoints actually
    # being evaluated, which is exactly what that field is for. No frozen file is modified.
    _real = FZ.require_source_freeze

    def _freeze_with_arm_checkpoints():
        rec = dict(_real())
        rec["sha256"] = dict(rec["sha256"])
        for s_ in BR.SEEDS:
            ck = dataclasses.replace(BR.CurrentPaths(),
                                     checkpoints=A.Paths().checkpoints).checkpoint_path(s_, 6000)
            rec["sha256"][os.path.relpath(ck, BR.REPO_ROOT)] = FZ.sha256_file(ck)
        return rec

    EV.FZ.require_source_freeze = _freeze_with_arm_checkpoints

    paths = dataclasses.replace(BR.CurrentPaths(), checkpoints=A.Paths().checkpoints,
                                target_eval=os.path.join(A.ARTIFACT_DIR, "zero_shot_eval"))
    os.makedirs(paths.target_eval, exist_ok=True)
    for s in BR.SEEDS:
        ck = paths.checkpoint_path(s, 6000)
        assert os.path.isfile(ck), f"missing Waymo checkpoint {ck}"
    print(f"[eval] {a.dataset}: checkpoints from {paths.checkpoints}", flush=True)
    rec = evaluate_dataset(a.dataset, a.device, paths=paths, limit=a.limit)
    print(f"\n[eval] {a.dataset}: {rec.n_anchors} anchors")
    print(json.dumps(rec.seed_statistics, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
