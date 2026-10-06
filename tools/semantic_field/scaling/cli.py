"""Command line for the scaling study.

    # launch (or resume: the same command continues from runs/<method>_<pool>_seed<s>/resume.pt if it exists)
    python -m tools.semantic_field.scaling.cli train --method B4 --pool expanded --seed 0 --device cuda:0
    # stop cleanly after a step (a resume checkpoint is written at that step)
    python -m tools.semantic_field.scaling.cli train --method B3 --pool current --seed 1 --device cuda:1 --stop-after-step 5000
"""
from __future__ import annotations

import argparse

from tools.semantic_field.scaling.config import METHODS, POOLS, SEEDS, LoaderConfig
from tools.semantic_field.scaling.run import run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="launch or resume one run")
    tr.add_argument("--method", choices=METHODS, required=True)
    tr.add_argument("--pool", choices=POOLS, required=True)
    tr.add_argument("--seed", type=int, choices=SEEDS, required=True)
    tr.add_argument("--device", default="cuda:0")
    tr.add_argument("--stop-after-step", type=int, default=None)
    tr.add_argument("--host-cache-gib", type=float, default=LoaderConfig().host_cache_bytes / 1024 ** 3)
    a = ap.parse_args()
    if a.command == "train":
        rec = run(a.method, a.pool, a.seed, a.device, a.stop_after_step, LoaderConfig(int(a.host_cache_gib * 1024 ** 3)))
        print({k: rec[k] for k in ("tag", "completed_steps", "total_steps", "seconds")})


if __name__ == "__main__":
    main()
