#!/usr/bin/env python
"""Gate: this arm's training loop must be the frozen loop.

``tools/waymo_arm/train.py`` repeats the loop of the hash-frozen
``baseline_reconstruction/current/train.py`` with the scene list as an argument. If the two
ever diverge, a Waymo-vs-KITTI-360 difference could be the trainer instead of the data, so
both are run on the SAME KITTI-360 drives, the same seed and the same step budget.

Bitwise equality is NOT the criterion, because the frozen loop is not bitwise reproducible
against itself: MEASURED on this machine, two runs of it at one seed differ on 98 of 100
steps with max |dloss| 1.3e-4 and a different final checkpoint sha256, since bf16 autocast
convolutions pick non-deterministic cuDNN algorithms. The criterion is therefore:

* **exact** on everything that is the data -- the anchor/crop draw digest, the target
  digest, the anchor counts and the initial parameter digest;
* **within the frozen loop's own run-to-run envelope** on the loss trajectory, which this
  gate measures in the same session by running the frozen loop twice.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD python \\
        tools/waymo_arm/cli/check_trainer_equivalence.py --steps 100 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.baseline_reconstruction import config as BR                                  # noqa: E402
from tools.baseline_reconstruction.current.train import \
    train_one_seed as frozen_train                                                      # noqa: E402
from tools.waymo_arm import config as A                                                 # noqa: E402
from tools.waymo_arm import train as WT                                                 # noqa: E402

KEYS = ("loss", "focal", "dice", "iou", "lr", "step")


def _series(log) -> list:
    return [{k: r[k] for k in KEYS if k in r} for r in log if "val" not in r]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    A.ensure_dirs()
    scratch = os.path.join(A.Paths().logs, "trainer_gate")
    os.makedirs(scratch, exist_ok=True)

    print("[gate] frozen loop on KITTI-360 (run 1 of 2, for its own envelope) ...", flush=True)
    ref = frozen_train(a.seed, a.device, steps=a.steps, tag_prefix="smoke")
    print("[gate] frozen loop again, same seed ...", flush=True)
    ref2 = frozen_train(a.seed, a.device, steps=a.steps, tag_prefix="smoke")
    print("[gate] this arm's loop on the SAME KITTI-360 data ...", flush=True)
    got = WT.train_one_seed(a.seed, a.device, list(BR.TRAIN_DRIVES), [BR.VAL_DRIVE],
                            paths=WT.K360ArmPaths(), steps=a.steps, tag_prefix="smoke",
                            record_dir=scratch)

    rs, r2s, gs = _series(ref.log), _series(ref2.log), _series(got.log)
    same_len = len(rs) == len(gs) == len(r2s)

    def envelope(x, y):
        if not same_len:
            return None
        return {k: max(abs(p[k] - q[k]) for p, q in zip(x, y) if k in p and k in q)
                for k in ("loss", "focal", "dice", "iou")}

    self_env = envelope(rs, r2s)            # the frozen loop against itself
    cross_env = envelope(rs, gs)            # frozen against this arm's loop
    within = (same_len and self_env is not None
              and all(cross_env[k] <= max(self_env[k], 1e-12) * 2.0 + 1e-9
                      for k in cross_env))
    out = {"version": A.VERSION, "steps": a.steps, "seed": a.seed,
           "criterion": ("data digests exact; loss trajectory within twice the frozen "
                         "loop's own run-to-run envelope"),
           "frozen_self_envelope": self_env, "cross_envelope": cross_env,
           "loss_within_envelope": bool(within),
           "n_train_anchors": [ref.n_train, got.n_train],
           "n_val_anchors": [ref.n_val, got.n_val],
           "init_digest": [ref.init_state_digest, got.init_state_digest],
           "draw_digest": [ref.draw_digest, got.draw_digest],
           "target_digest": [ref.target_digest, got.target_digest],
           "final_sha256": [ref.final_checkpoint_sha256, got.final_checkpoint_sha256],
           "n_logged": [len(rs), len(r2s), len(gs)],
           "frozen_self_bitwise_equal": bool(ref.final_checkpoint_sha256
                                             == ref2.final_checkpoint_sha256)}
    out["data_exact"] = bool(
        ref.n_train == got.n_train and ref.n_val == got.n_val
        and ref.init_state_digest == got.init_state_digest
        and ref.draw_digest == got.draw_digest
        and ref.target_digest == got.target_digest)
    out["passed"] = bool(out["data_exact"] and within)
    path = os.path.join(A.Paths().logs, "check_trainer_equivalence.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"\n[{'PASS' if out['passed'] else 'FAIL'}] trainer equivalence over {len(rs)} steps")
    print(f"  data exact (draws, targets, init, counts): {out['data_exact']}")
    print(f"  frozen loop bitwise reproducible against itself: "
          f"{out['frozen_self_bitwise_equal']}")
    fmt = lambda e: json.dumps({k: f"{v:.3e}" for k, v in (e or {}).items()})
    print(f"  max |delta| frozen-vs-frozen : {fmt(self_env)}")
    print(f"  max |delta| frozen-vs-thisarm: {fmt(cross_env)}")
    print(f"-> {path}")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
