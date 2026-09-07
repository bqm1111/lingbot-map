#!/usr/bin/env python
"""Gate 8C-0: hash every configuration and artifact this gate produced or must not change.

Gate 8C-0 is a read-only audit of Gates 8-8B, so the frozen list is longer than the
produced list on purpose. Re-running prints anything that drifted.
"""
from __future__ import annotations
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT                                               # noqa: E402
from gates.scale_gate.config import write_json                                         # noqa: E402

FROZEN = ["gates/gate8/mapper.py", "gates/gate8/net.py", "gates/gate8/targets.py", "gates/gate8/losses.py",
          "gates/gate8/vocab.py", "gates/gate8/sources.py", "gates/gate8b/sources.py", "gates/gate8b/clips.py",
          "gates/gate8a/regions.py", "gates/gate8a/scores.py", "gates/gate6/grids.py", "gates/gate6/metrics.py",
          "gates/gate6/targets.py", "sscbench_kitti360/adapter.py",
          "configs/gate8a/cellB_uniform_focal.yaml", "configs/gate8a/frozen_selection.yaml",
          "configs/gate8b/fold_semantickitti.yaml", "configs/gate8b/fold_occ3d.yaml",
          "configs/gate8b/frozen_fold_semantickitti.yaml", "configs/gate8b/frozen_fold_occ3d.yaml",
          "artifacts/gate8/gate8_results.json", "artifacts/gate8a/gate8a_results.json",
          "artifacts/gate8b/gate8b_results.json", "artifacts/gate8b/gate8b_manifest.json",
          "reports/gate8a/gate8a_report.md", "reports/gate8b/gate8b_report.md",
          "artifacts/gate8a/checkpoints/cellB_uniform_focal_last.pt",
          "artifacts/gate8b/checkpoints/fold_semantickitti_best.pt",
          "artifacts/gate8b/checkpoints/fold_occ3d_best.pt"]
PRODUCED = ["configs/gate8c0/audit.yaml", "gates/gate8c0/transforms.py", "gates/gate8c0/oracle.py",
            "gates/gate8c0/checks.py"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    out = {"frozen": {}, "produced": {}, "artifacts": {}}
    for rel in FROZEN:
        p = os.path.join(REPO_ROOT, rel)
        out["frozen"][rel] = {"sha256": sha256(p), "bytes": os.path.getsize(p)} \
            if os.path.exists(p) else None
    for rel in PRODUCED:
        out["produced"][rel] = sha256(os.path.join(REPO_ROOT, rel))
    for f in sorted(os.listdir(ART)):
        if f.endswith((".json", ".csv", ".png")) and f != "hashes.json":
            out["artifacts"][f] = sha256(os.path.join(ART, f))
    p = os.path.join(ART, "hashes.json")
    prev = json.load(open(p)) if os.path.exists(p) else None
    if prev:
        drift = [k for k, v in out["frozen"].items()
                 if prev["frozen"].get(k) and v and v["sha256"] != prev["frozen"][k]["sha256"]]
        out["drift_since_first_run"] = drift
        for k in drift:
            print("  CHANGED:", k)
    write_json(p, out)
    print(f"hashes: {sum(v is not None for v in out['frozen'].values())}/{len(FROZEN)} frozen, "
          f"{len(out['produced'])} produced modules, {len(out['artifacts'])} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
