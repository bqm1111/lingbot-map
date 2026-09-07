#!/usr/bin/env python
"""Hash the frozen artifacts Gate 8 must not disturb, and record provenance."""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                       # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    git = lambda *a: subprocess.run(["git", "-C", REPO_ROOT, *a], capture_output=True, text=True).stdout.strip()
    rec = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "git": {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                   "status": git("status", "--short").splitlines()},
           "pre_existing_dirty": {r: sha(os.path.join(REPO_ROOT, r)) for r in
                                  ("lingbot_map/models/gct_stream.py", "research/sem_bypass/model.py")},
           "gate7b_artifacts": {}, "gate6_artifacts": {}, "gate7a_artifacts": {}, "frozen_models": {}}
    for g in ("gate6", "gate7a", "gate7b"):
        d = os.path.join(REPO_ROOT, "artifacts", g)
        for rel in sorted(os.listdir(d)):
            p = os.path.join(d, rel)
            if os.path.isfile(p):
                rec[f"{g}_artifacts"][rel] = sha(p)
    rec["frozen_models"]["lingbot"] = {"path": "checkpoints/lingbot-map/204754b/lingbot-map.pt",
                                       "sha256": sha(os.path.join(REPO_ROOT, "checkpoints/lingbot-map/204754b/lingbot-map.pt"))}
    tp = os.path.join(REPO_ROOT, "artifacts", "gate6", "teacher_provenance.json")
    rec["frozen_models"]["trident"] = json.load(open(tp)) if os.path.exists(tp) else None
    rec["frozen_models"]["moge"] = {"hf_repo": "Ruicheng/moge-2-vitl",
                                    "hf_revision": "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"}
    write_json(os.path.join(REPO_ROOT, "artifacts", "gate8", "stage0_audit.json"), rec)
    print(f"gate8 stage0: commit {rec['git']['commit'][:12]}, "
          f"{sum(len(rec[k]) for k in rec if k.endswith('_artifacts'))} prior artifacts hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
