#!/usr/bin/env python
"""Gate 16 steps (see tools/delayed_semantic/README.md, "Gate 16").

    python tools/delayed_semantic/gate16/cli.py freeze --stage predeclaration
    python tools/delayed_semantic/gate16/cli.py ceiling --device cuda:0          # then: freeze --stage ceiling
    python tools/delayed_semantic/gate16/cli.py train --variant B3 --device cuda:1
    python tools/delayed_semantic/gate16/cli.py freeze --stage code
    python tools/delayed_semantic/gate16/cli.py evaluate --shard 0 --shards 4 --device cuda:0
    python tools/delayed_semantic/gate16/cli.py analyse | verify | report
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["freeze", "ceiling", "train", "evaluate", "analyse", "verify", "report"])
    ap.add_argument("--stage", choices=["predeclaration", "ceiling", "code"]); ap.add_argument("--variant", choices=["B3", "B3geom"])
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    from tools.delayed_semantic.gate16 import config as G, rules
    G.ensure_dirs()
    if a.step == "freeze":
        payload = json.load(open(os.path.join(G.ARTIFACT_DIR, "candidate_ceiling.json")))["selection_threshold"] if a.stage == "ceiling" else None
        print(rules.freeze(a.stage, payload)["frozen_at"])
    elif a.step == "ceiling":
        from tools.delayed_semantic.gate16 import ceiling
        print(ceiling.run(a.device)["selection_threshold"])
    elif a.step == "train":
        from tools.delayed_semantic.gate16.train import train_variant
        print({s: r["checkpoints"][-1] for s, r in train_variant(a.variant, a.device).items()})
    elif a.step == "evaluate":
        from tools.delayed_semantic.gate16 import evaluate as EV
        rules.check("ceiling"); rules.check("code"); print(EV.evaluate(a.device, a.shard, a.shards))
    elif a.step == "analyse":
        from tools.delayed_semantic.gate16 import analysis as AN
        print(AN.analyse()["decision"])
    elif a.step == "verify":
        from tools.delayed_semantic.gate16 import verify as VF
        r = VF.run(a.device); print(f"{r['n_pass']}/{r['n']}")
    elif a.step == "report":
        from tools.delayed_semantic.gate16 import report as RP
        print(RP.write_all(a.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
