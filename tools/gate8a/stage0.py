#!/usr/bin/env python
"""Gate 8A stage 0: hash everything Gate 8A must not modify, before it modifies anything.

Frozen by the brief: LingBot-Map, MoGe-2 and the five-frame scale anchor, the Trident-H
caches, the incremental mapper and map representation, the completion U-Net architecture
and its inputs, the privileged targets and the 20-frame horizon, the training datasets, the
budget/batch/crop, the union vocabulary and the semantic KL, and the official evaluation
masks. Re-running this after the gate and diffing the JSON is the audit.

    python tools/gate8a/stage0.py
"""
from __future__ import annotations
import hashlib, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                              # noqa: E402

FROZEN = [
    "gates/gate8/mapper.py", "gates/gate8/net.py", "gates/gate8/targets.py", "gates/gate8/vocab.py",
    "gates/gate8/sources.py", "gates/gate8/feed.py", "gates/gate8/losses.py",
    "gates/gate7b/scale.py", "gates/gate7b/depth.py", "gates/gate7b/voxmap.py", "gates/gate7b/rays.py",
    "gates/gate6/grids.py", "gates/gate6/metrics.py", "gates/gate6/targets.py", "gates/gate6/vocab.py",
    "configs/gate8/completion.yaml",
    "artifacts/gate8/gate8_results.json", "artifacts/gate8/gate8_manifest.json",
    "artifacts/gate8/train_completion.json",
    "artifacts/gate8/checkpoints/completion_best.pt",
    "artifacts/gate8/checkpoints/completion_last.pt",
    "reports/gate8/gate8_report.md",
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    out = {"note": "hashes of everything Gate 8A must not change", "files": {}}
    for rel in FROZEN:
        p = os.path.join(REPO_ROOT, rel)
        out["files"][rel] = {"sha256": sha256(p), "bytes": os.path.getsize(p)} \
            if os.path.exists(p) else None
    p = os.path.join(REPO_ROOT, "artifacts", "gate8a", "stage0_audit.json")
    prev = json.load(open(p)) if os.path.exists(p) else None
    if prev:
        drift = [k for k, v in out["files"].items()
                 if prev["files"].get(k) and v and v["sha256"] != prev["files"][k]["sha256"]]
        out["drift_since_first_run"] = drift
        for k in drift:
            print(f"  CHANGED since stage 0: {k}")
    write_json(p, out)
    print(f"stage 0: {sum(v is not None for v in out['files'].values())}/{len(FROZEN)} "
          f"frozen files hashed -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
