#!/usr/bin/env python
"""Train the ONCE arm's CompletionUNet, seeds 0/1/2, under the frozen recipe.

    PYTHONPATH=$PWD python tools/method_vs_data/once_arm/cli/train.py --device cuda:3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from tools.method_vs_data.once_arm import anchors as AN, config as A, train as T   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", default=",".join(str(s) for s in A.SEEDS))
    ap.add_argument("--steps", type=int, default=None, help="default: the recipe's 6,000")
    ap.add_argument("--tag", default="train")
    a = ap.parse_args()

    A.ensure_dirs()
    p = A.Paths()
    man = AN.load_or_build(p)
    sp = A.split(A.sequences())
    print(f"[train] ONCE arm {A.VERSION} status={A.STATUS}")
    print(f"[train] train sequences {sp['train']}  val {sp['val']}")
    print(f"[train] manifest {man['digest'][:12]}: {man['n_train']} train / {man['n_val']} "
          f"val anchors over {man['independent_source_sequences']} independent sequences")
    print(f"[train] PREVALENCE BAND: {A.PREVELANCE_BAND_STATUS['verdict']} "
          f"({A.PREVELANCE_BAND_STATUS['once_measured_median']} vs band "
          f"{A.PREVELANCE_BAND_STATUS['predeclared_band']})", flush=True)

    out = {"provenance": A.provenance(), "manifest_digest": man["digest"],
           "train_sequences": sp["train"], "val_sequences": sp["val"], "seeds": {}}
    for s in [int(x) for x in a.seeds.split(",")]:
        rec = T.train_one_seed(s, a.device, steps=a.steps)
        d = asdict(rec) if hasattr(rec, "__dataclass_fields__") else dict(rec)
        out["seeds"][str(s)] = d
        print(f"[train] seed {s} done", flush=True)
    os.makedirs(p.logs, exist_ok=True)
    f = os.path.join(p.logs, f"{a.tag}.json")
    json.dump(out, open(f, "w"), indent=2, default=str)
    print(f"[train] -> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
