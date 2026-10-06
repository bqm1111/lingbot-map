"""Delayed-target reliability for the expanded training set with the FROZEN pilot gate (``tools/delayed_semantic/reliability.py``,
thresholds ``rules.RELIABILITY_GATE``). Label-free; runs before the manifest freeze and before any training.

Same reading as the frozen gate: G1 registration pooled over the training drives (now the 7 drives), G2 coverage and G3 repeatability
on the pooled training drives, G3 also on the development drive 0006, G4 re-observation. The frozen ``gate`` pools G1 over
``config.TRAIN_DRIVES``, so that tuple is set to the 7 drives for the duration of the call only.
Required outcome (declared): pass, with delayed-loss tiers [1, 2] unchanged. Anything else is BLOCKED.

    PYTHONPATH=$PWD python -m tools.semantic_field.scaling.reliability_check --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

from core.datasets.audit import FileAudit
from tools.delayed_semantic import config as C, reliability as REL
from tools.semantic_field.scaling.config import ARTIFACT_DIR
from tools.semantic_field.scaling.registration import ADDED_DRIVES, register_added_drives

FROZEN_TIERS = [1, 2]


def run(device: str) -> Dict:
    register_added_drives()
    drives = tuple(C.TRAIN_DRIVES) + ADDED_DRIVES
    with FileAudit(patterns=C.TARGET_FORBIDDEN, strict=True) as audit:
        reg = {d: REL.registration(d, device) for d in drives + (C.DEV_DRIVE,)}
        stats = {d: REL.target_statistics(d, device=device) for d in drives + (C.DEV_DRIVE,)}
    train = REL.pooled([stats[d] for d in drives]); dev = REL.pooled([stats[C.DEV_DRIVE]])
    original = C.TRAIN_DRIVES
    C.TRAIN_DRIVES = drives
    try:
        gate = REL.gate(reg, train, dev)
    finally:
        C.TRAIN_DRIVES = original
    for r in reg.values():
        r.pop("_raw")
    rec = {"drives": list(drives), "gate": gate, "required": {"pass": True, "delayed_loss_tiers": FROZEN_TIERS},
           "status": "ok" if gate["pass"] and gate["delayed_loss_tiers"] == FROZEN_TIERS else "BLOCKED",
           "registration": reg, "per_drive": {d: {k: v for k, v in s.items() if k != "per_anchor"} for d, s in stats.items()},
           "train_pooled": train, "dev_pooled": dev, "firewall": {"opened": len(audit.opened), "violations": audit.violations}}
    out = os.path.join(ARTIFACT_DIR, "prep", "reliability_expanded.json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rec, open(out, "w"), indent=1, default=float)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
    rec = run(ap.parse_args().device)
    print(json.dumps({"status": rec["status"], "gate": rec["gate"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
