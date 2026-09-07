#!/usr/bin/env python
"""Hash everything Gate 7B must not disturb, before Gate 7B touches anything."""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                       # noqa: E402

OUT = "artifacts/gate7b/stage0_audit.json"
PRE_EXISTING_DIRTY = ("lingbot_map/models/gct_stream.py", "research/sem_bypass/model.py")
REPORTS = ("reports/gate6/frozen_trident_semantic_lifting.md",
           "reports/gate7a/completion_reachability.md")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", REPO_ROOT, *a], capture_output=True,
                          text=True).stdout.strip()


def collect() -> dict:
    rec = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "git": {"commit": git("rev-parse", "HEAD"),
                   "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                   "status": git("status", "--short").splitlines()},
           "pre_existing_dirty": {}, "reports": {}, "gate6": {"predictions": {},
                                                              "artifacts": {}},
           "gate7a": {"artifacts": {}}}
    for rel in PRE_EXISTING_DIRTY:
        p = os.path.join(REPO_ROOT, rel)
        rec["pre_existing_dirty"][rel] = {"sha256": sha256_file(p),
                                          "bytes": os.path.getsize(p)}
    for rel in REPORTS:
        p = os.path.join(REPO_ROOT, rel)
        if os.path.exists(p):
            rec["reports"][rel] = sha256_file(p)
    for ds in ("semantickitti", "occ3d", "kitti360"):
        mp = os.path.join(REPO_ROOT, "artifacts", "gate6",
                          f"prediction_manifest_{ds}.json")
        man = json.load(open(mp))
        rec["gate6"]["predictions"][ds] = {"manifest_sha256": sha256_file(mp),
                                           "rollup_sha256": man["rollup_sha256"],
                                           "n_files": man["n_files"],
                                           "prediction_root": man["prediction_root"]}
    for gate in ("gate6", "gate7a"):
        d = os.path.join(REPO_ROOT, "artifacts", gate)
        for rel in sorted(os.listdir(d)):
            p = os.path.join(d, rel)
            if os.path.isfile(p):
                rec[gate]["artifacts"][rel] = sha256_file(p)
    for gate, pin_rel in (("gate6", "artifacts/gate6/precommit_pin.json"),
                          ("gate7a", "artifacts/gate7a/precommit_pin.json")):
        pin = json.load(open(os.path.join(REPO_ROOT, pin_rel)))
        got = sha256_file(os.path.join(REPO_ROOT, pin["path"]))
        rec[gate]["precommit"] = {"path": pin["path"], "pinned": pin["sha256"],
                                  "observed": got, "matches": got == pin["sha256"]}
    return rec


def main() -> int:
    rec = collect()
    write_json(os.path.join(REPO_ROOT, OUT), rec)
    print(OUT)
    for g in ("gate6", "gate7a"):
        print(f"  {g} precommit matches: {rec[g]['precommit']['matches']}  "
              f"({len(rec[g]['artifacts'])} artifacts hashed)")
    for ds, d in rec["gate6"]["predictions"].items():
        print(f"  {ds:14s} rollup {d['rollup_sha256'][:16]}")
    for rel, h in rec["reports"].items():
        print(f"  report {os.path.basename(rel)} {h[:16]}")
    return 0 if all(rec[g]["precommit"]["matches"] for g in ("gate6", "gate7a")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
