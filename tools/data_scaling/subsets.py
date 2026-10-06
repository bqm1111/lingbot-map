"""Deterministic nested training subsets: D_small subset-of D_medium subset-of D_large subset-of D_all.

Nesting is guaranteed by construction: every added drive's anchors are permuted ONCE with a fixed seed, and each
scale takes a prefix of that permutation. A prefix of a fixed permutation is always contained in a longer prefix.
Permuting rather than taking a contiguous block keeps each subset spread over the whole drive, so a small subset is
not a short stretch of road with its own weather and architecture.

D_small is exactly the frozen pool (drives 0003 / 0007 / 0010, every anchor), so the smallest point of the learning
curve is the published baseline rather than a re-sample of it.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np

from tools.baseline_reconstruction import config as BR
from tools.baseline_reconstruction.current.dataset import anchor_ids
from tools.data_scaling import config as G19


def usable_anchors(drive: str, paths: BR.CurrentPaths) -> List[int]:
    """Anchors with BOTH an input and a target. Some drives have more targets than inputs; the intersection is used."""
    have_input = set(anchor_ids(drive, paths))
    tdir = os.path.dirname(paths.target_path(drive, 0))
    have_target = {int(f[:-4]) for f in os.listdir(tdir) if f.endswith(".npz")}
    return sorted(have_input & have_target)


def build_pools(paths: BR.CurrentPaths = BR.CurrentPaths()) -> Dict[str, Dict]:
    frozen: List[Tuple[str, List[int]]] = [(d, usable_anchors(d, paths)) for d in G19.FROZEN_TRAIN_DRIVES]
    n_frozen = sum(len(a) for _, a in frozen)

    rng = np.random.default_rng(G19.SEED_PERMUTATION)
    permuted: Dict[str, List[int]] = {}
    for d in G19.ADDED_DRIVES:
        a = np.asarray(usable_anchors(d, paths))
        permuted[d] = a[rng.permutation(len(a))].tolist()        # one permutation per drive, fixed forever
    n_added_total = sum(len(v) for v in permuted.values())

    pools: Dict[str, Dict] = {}
    for name, target in G19.SCALES.items():
        need = max(0, min(target, n_frozen + n_added_total) - n_frozen)
        take: Dict[str, List[int]] = {}
        if need:
            # proportional to each added drive's size, so the class/geography mix of the added data is preserved
            share = {d: len(v) / n_added_total for d, v in permuted.items()}
            counts = {d: int(round(need * s)) for d, s in share.items()}
            # fix rounding drift deterministically, largest drive first
            drift = need - sum(counts.values())
            for d in sorted(permuted, key=lambda x: -len(permuted[x])):
                if drift == 0:
                    break
                step = 1 if drift > 0 else -1
                if 0 <= counts[d] + step <= len(permuted[d]):
                    counts[d] += step; drift -= step
            take = {d: permuted[d][:counts[d]] for d in permuted if counts[d] > 0}
        drives = {d: sorted(a) for d, a in frozen}
        for d, a in take.items():
            drives[d] = sorted(a)
        n = sum(len(v) for v in drives.values())
        pools[name] = {"name": name, "n_anchors": n, "n_frozen": n_frozen,
                       "n_added": n - n_frozen, "drives": {d: v for d, v in drives.items()},
                       "per_drive_counts": {d: len(v) for d, v in drives.items()},
                       "permutation_seed": G19.SEED_PERMUTATION,
                       "epochs_at_6000_steps": G19.epochs_of(n),
                       "steps_for_fixed_epochs": G19.steps_for_fixed_epochs(n, n_frozen)}
    return pools


def write_pools(paths: BR.CurrentPaths = BR.CurrentPaths()) -> Dict[str, Dict]:
    pools = build_pools(paths)
    os.makedirs(os.path.join(G19.ARTIFACT_DIR, "pools"), exist_ok=True)
    order = list(G19.SCALES)
    for i, name in enumerate(order):                              # nesting is a property, so it is asserted
        cur = pools[name]
        for d, a in cur["drives"].items():
            for smaller in order[:i]:
                prev = pools[smaller]["drives"].get(d, [])
                missing = set(prev) - set(a)
                assert not missing, f"{name} does not contain {smaller} on {d}: {sorted(missing)[:5]}"
        json.dump(cur, open(G19.pool_path(name), "w"), indent=1)
    summary = {n: {k: v for k, v in p.items() if k != "drives"} for n, p in pools.items()}
    json.dump(summary, open(os.path.join(G19.ARTIFACT_DIR, "pools", "summary.json"), "w"), indent=1)
    return pools


def load_pool(name: str) -> List[Tuple[str, List[int]]]:
    d = json.load(open(G19.pool_path(name)))
    return [(drive, ids) for drive, ids in sorted(d["drives"].items())]


if __name__ == "__main__":
    p = write_pools()
    print("%-10s %8s %8s %8s %10s %14s" % ("pool", "anchors", "frozen", "added", "epochs@6k", "steps@fixed-ep"))
    for n, v in p.items():
        print("%-10s %8d %8d %8d %10.2f %14d" % (n, v["n_anchors"], v["n_frozen"], v["n_added"],
                                                 v["epochs_at_6000_steps"], v["steps_for_fixed_epochs"]))
    print("\nper-drive counts:")
    for n, v in p.items():
        print("  %-10s %s" % (n, {k[-9:-5]: c for k, c in sorted(v["per_drive_counts"].items())}))
