#!/usr/bin/env python
"""Gate 8B stage 0: hash everything Gate 8B must not modify. Re-run at the end; any drift
is printed. The two additive edits the gate *does* make (the source registry in
``gate8/sources.py`` and the ``sequence`` argument of the KITTI-360 target loader) are
declared in the report and are the only expected differences from Gate 8A's audit.

    python tools/gate8b/stage0.py
"""
from __future__ import annotations
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT                                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402

FROZEN = ["gates/gate8/mapper.py", "gates/gate8/net.py", "gates/gate8/targets.py", "gates/gate8/vocab.py", "gates/gate8/feed.py",
          "gates/gate8/losses.py", "gates/gate8a/regions.py", "gates/gate8a/scores.py", "gates/gate8a/baselines.py",
          "gates/gate8a/sampler.py", "gates/gate8a/boot.py", "gates/gate7b/scale.py", "gates/gate7b/depth.py",
          "gates/gate7b/voxmap.py", "gates/gate7b/rays.py", "gates/gate6/grids.py", "gates/gate6/metrics.py", "gates/gate6/vocab.py",
          "configs/gate8a/cellB_uniform_focal.yaml", "configs/gate8a/frozen_selection.yaml",
          "artifacts/gate8a/gate8a_results.json", "artifacts/gate8a/frozen_manifest.json",
          "artifacts/gate8a/checkpoints/cellB_uniform_focal_last.pt",
          "artifacts/gate8a/eval_kitti360_locked.json", "reports/gate8a/gate8a_report.md",
          "artifacts/gate8/gate8_results.json", "checkpoints/lingbot-map/204754b/lingbot-map.pt"]
DECLARED_EDITS = ["gates/gate8/sources.py", "gates/gate6/targets.py", "sscbench_kitti360/adapter.py",
                  "tools/gate8a/evaluate.py"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    out = {"frozen": {}, "declared_additive_edits": {}}
    for rel in FROZEN:
        p = os.path.join(REPO_ROOT, rel)
        out["frozen"][rel] = {"sha256": sha256(p), "bytes": os.path.getsize(p)} if os.path.exists(p) else None
    for rel in DECLARED_EDITS:
        out["declared_additive_edits"][rel] = sha256(os.path.join(REPO_ROOT, rel))
    p = os.path.join(ART, "stage0_audit.json")
    prev = json.load(open(p)) if os.path.exists(p) else None
    if prev:
        out["drift_since_first_run"] = [k for k, v in out["frozen"].items()
                                        if prev["frozen"].get(k) and v and v["sha256"] != prev["frozen"][k]["sha256"]]
        for k in out["drift_since_first_run"]:
            print("  CHANGED since stage 0:", k)
    os.makedirs(ART, exist_ok=True); write_json(p, out)
    print(f"stage 0: {sum(v is not None for v in out['frozen'].values())}/{len(FROZEN)} frozen files hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
