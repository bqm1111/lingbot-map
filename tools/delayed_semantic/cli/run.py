#!/usr/bin/env python
"""Steps 3-6 of the pilot (see tools/delayed_semantic/README.md).

    python tools/delayed_semantic/cli/run.py reliability --device cuda:0
    python tools/delayed_semantic/cli/run.py train_cache --drive 2013_05_28_drive_0007_sync --device cuda:0
    python tools/delayed_semantic/cli/run.py train --method B1 --device cuda:0
    python tools/delayed_semantic/cli/run.py evaluate --shard 0 --shards 4 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.delayed_semantic import config as C, rules  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["reliability", "train_cache", "train", "evaluate", "analyse", "verify", "report"])
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--drive"); ap.add_argument("--method", choices=["B1", "B2", "B2s"])
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=None); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(); C.ensure_dirs(); logs = os.path.join(C.ARTIFACT_DIR, "logs")
    if a.step == "reliability":
        from tools.delayed_semantic import reliability as RL
        rec = RL.run_all(a.device); print(json.dumps(rec["gate"], indent=1))
    elif a.step == "train_cache":
        from tools.delayed_semantic import train_cache as TC
        rec = TC.build_drive(a.drive, a.device, a.shard, a.shards)
        json.dump(rec, open(os.path.join(logs, f"train_cache_{a.drive[17:21]}_s{a.shard}.json"), "w"), indent=1); print(rec)
    elif a.step == "train":
        from tools.delayed_semantic import train as TR
        rec = TR.train(a.method, a.device, a.steps); print(rec["tag"], rec["checkpoints"][-1], f"{rec['seconds']/60:.1f} min")
    elif a.step == "evaluate":
        from tools.delayed_semantic import evaluate as EV
        rules.check("decision")
        steps = C.TrainConfig().steps
        heads = {m: C.Paths().head_checkpoint(m, steps) for m in ("B1", "B2", "B2s")}
        assert all(os.path.exists(p) for p in heads.values()), heads
        rec = EV.evaluate(heads, a.device, a.shard, a.shards, a.limit); print(rec)
    elif a.step == "analyse":
        from tools.delayed_semantic import analysis as AN
        print(AN.analyse()["decision"])
    elif a.step == "verify":
        from tools.delayed_semantic import verify as VF
        rec = VF.run(a.device); print(f"{rec['n_pass']}/{rec['n']}")
    elif a.step == "report":
        from tools.delayed_semantic import report as RP
        print(RP.write_all(a.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
