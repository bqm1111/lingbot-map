#!/usr/bin/env python
"""Run the Waymo arm end to end as its dependencies become available.

Order, with each step skipped when its output already exists:

  1. wait for the laser sidecars (re-download; the only step this cannot do itself)
  2. occupancy targets        tools/waymo_arm/cli/build_targets.py
  3. hard-map inputs          tools/waymo_arm/cli/build_inputs.py
  4. train seeds 0,1,2        tools/waymo_arm/train.py
  5. zero-shot evaluation     the FROZEN harness, pointed at this arm's checkpoints

Step 5 reuses ``baseline_reconstruction.targets.evaluate.evaluate_dataset`` unchanged: the
only thing replaced is the ``checkpoints`` field of ``CurrentPaths``, so the target clips,
adapters, masks, range bins, threshold and metric are the KITTI-360 arm's own.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD python tools/waymo_arm/cli/run_chain.py \\
        --device cuda:0 --steps 6000
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.baseline_reconstruction import config as BR                                  # noqa: E402
from tools.waymo_arm import config as A                                                 # noqa: E402

PY_EXE = sys.executable
CLI = os.path.dirname(os.path.abspath(__file__))


def _run(script: str, args: list, env_extra: dict = None) -> int:
    env = dict(os.environ, PYTHONPATH=A.REPO_ROOT, PYTHONDONTWRITEBYTECODE="1",
               **(env_extra or {}))
    cmd = [PY_EXE, os.path.join(CLI, script)] + [str(x) for x in args]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env).returncode


def wait_for_sidecars(target: int, poll_s: int = 300, timeout_h: float = 12.0) -> int:
    p = A.Paths()
    t0 = time.time()
    while True:
        n = len([f for f in os.listdir(p.waymo_lasers) if f.endswith(".npz")]) \
            if os.path.isdir(p.waymo_lasers) else 0
        if n >= target:
            print(f"[chain] sidecars complete: {n}/{target}", flush=True)
            return n
        if (time.time() - t0) / 3600 > timeout_h:
            print(f"[chain] sidecar wait timed out at {n}/{target}; continuing with what "
                  f"is on disk", flush=True)
            return n
        print(f"[chain] sidecars {n}/{target}; waiting {poll_s}s "
              f"({(time.time() - t0) / 60:.0f} min elapsed)", flush=True)
        time.sleep(poll_s)


def evaluate(dataset: str, device: str, limit: int = None) -> dict:
    """The frozen harness with this arm's checkpoints swapped in, and nothing else."""
    from tools.baseline_reconstruction.targets.evaluate import evaluate_dataset
    paths = dataclasses.replace(BR.CurrentPaths(), checkpoints=A.Paths().checkpoints,
                                target_eval=os.path.join(A.ARTIFACT_DIR, "zero_shot_eval"))
    os.makedirs(paths.target_eval, exist_ok=True)
    for s in BR.SEEDS:
        ck = paths.checkpoint_path(s, 6000)
        assert os.path.isfile(ck), f"missing Waymo checkpoint {ck}"
    rec = evaluate_dataset(dataset, device, paths=paths, limit=limit)
    return {"dataset": dataset, "n_anchors": rec.n_anchors,
            "seed_statistics": rec.seed_statistics, "map_only": rec.map_only}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--expect-sidecars", type=int, default=0,
                    help="0 = every cached segment")
    ap.add_argument("--skip-wait", action="store_true")
    ap.add_argument("--datasets", default="semantickitti,occ3d",
                    help="comma separated; pass '' to train only and evaluate separately")
    ap.add_argument("--eval-device", default=None)
    a = ap.parse_args()

    A.ensure_dirs()
    p = A.Paths()
    want = a.expect_sidecars or len(A.segments(p))
    out = {"version": A.VERSION, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "steps": []}

    if not a.skip_wait:
        n = wait_for_sidecars(want)
        out["steps"].append({"step": "wait_sidecars", "n_sidecars": n, "wanted": want})

    rc = _run("build_targets.py", ["--device", a.device, "--tag", "build_targets_full"])
    out["steps"].append({"step": "build_targets", "returncode": rc})

    rc = _run("build_inputs.py", ["--device", a.device, "--tag", "build_inputs_full"])
    out["steps"].append({"step": "build_inputs", "returncode": rc})

    from tools.waymo_arm import train as WT
    segs = A.segments(p)
    sp = A.split(segs)
    print(f"\n[chain] {len(sp['train'])} train / {len(sp['val'])} val segments", flush=True)
    for seed in [int(s) for s in a.seeds.split(",")]:
        ck = WT.WaymoArmPaths().checkpoint_path(seed, a.steps or 6000)
        if os.path.isfile(ck):
            print(f"[chain] seed {seed} already trained: {ck}", flush=True)
            out["steps"].append({"step": f"train_seed{seed}", "skipped": True})
            continue
        rec = WT.train_one_seed(seed, a.device, sp["train"], sp["val"], steps=a.steps)
        out["steps"].append({"step": f"train_seed{seed}", "n_train": rec.n_train,
                             "n_val": rec.n_val, "minutes": rec.seconds / 60,
                             "final_sha256": rec.final_checkpoint_sha256,
                             "firewall": rec.firewall})

    for ds in [x for x in a.datasets.split(",") if x]:
        try:
            out["steps"].append({"step": f"eval_{ds}",
                                 **evaluate(ds, a.eval_device or a.device)})
        except Exception as exc:                                           # noqa: BLE001
            out["steps"].append({"step": f"eval_{ds}",
                                 "error": f"{type(exc).__name__}: {exc}"})
        print(f"[chain] {ds} evaluated", flush=True)

    path = os.path.join(p.logs, "run_chain.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"\n[chain] done -> {path}")
    for s in out["steps"]:
        print("  " + json.dumps(s, default=float)[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
