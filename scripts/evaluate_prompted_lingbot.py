#!/usr/bin/env python
"""Evaluate every metric anchor over the full prompt grid on cached predictions.

    python scripts/evaluate_prompted_lingbot.py \
        --cache-dir outputs/prompted_lingbot/cache \
        --output-dir outputs/prompted_lingbot/evaluation \
        --split-file configs/prompted_lingbot/splits.json --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompted_lingbot.anchors import BASELINES
from prompted_lingbot.evaluation import evaluate_cache_dir
from prompted_lingbot.prompts import standard_configs


def write_rows(rows, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{tag}_rows.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=1)
    keys = sorted(set().union(*(r.keys() for r in rows)))
    csv_path = os.path.join(out_dir, f"{tag}_rows.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(k) is None else str(r.get(k, "")) for k in keys) + "\n")
    return json_path, csv_path


def summarise(rows):
    """Mean over sequences for every (config, anchor)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        groups[(r["config"], r["anchor"])].append(r)
    out = []
    numeric = set()
    for rs in groups.values():
        for r in rs:
            numeric |= {k for k, v in r.items() if isinstance(v, (int, float, np.floating))}
    for (cfg, anchor), rs in sorted(groups.items()):
        row = {"config": cfg, "anchor": anchor, "n_sequences": len(rs),
               "causal": rs[0].get("causal", True)}
        for k in sorted(numeric):
            vals = [r[k] for r in rs if isinstance(r.get(k), (int, float, np.floating))
                    and np.isfinite(r[k])]
            if vals:
                row[k] = float(np.mean(vals))
                row[k + "__sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True, nargs="+")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tag", default="baselines")
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--anchors", nargs="*", default=None)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-point-cloud", action="store_true")
    ap.add_argument("--learned-checkpoint", default=None,
                    help="best.pt from train_prompted_corrector.py; adds the "
                         "learned_corrector anchor to the grid")
    args = ap.parse_args()

    names = None
    if args.split_file and args.split:
        splits = json.load(open(args.split_file))
        names = set(splits[args.split])

    cfgs = standard_configs()
    if args.configs:
        cfgs = {k: v for k, v in cfgs.items() if k in set(args.configs)}
    anchors = args.anchors or list(BASELINES.keys())

    rows = []
    for cache_dir in args.cache_dir:
        got = evaluate_cache_dir(cache_dir, configs=cfgs, anchor_names=anchors, names=names,
                                 workers=args.workers, point_cloud=not args.no_point_cloud,
                                 seed=args.seed, learned_checkpoint=args.learned_checkpoint)
        print(f"{cache_dir}: {len(got)} rows")
        rows.extend(got)
    if not rows:
        raise SystemExit("no rows produced -- check --cache-dir / --split")

    jp, cp = write_rows(rows, args.output_dir, args.tag)
    summ = summarise(rows)
    with open(os.path.join(args.output_dir, f"{args.tag}_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(f"wrote {jp}\n      {cp}\n      {len(summ)} (config, anchor) summaries")


if __name__ == "__main__":
    main()
