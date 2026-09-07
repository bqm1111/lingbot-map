#!/usr/bin/env python
"""Hash everything Gate 7A must not disturb, before Gate 7A touches anything.

Records the working tree, the two pre-existing dirty files, every Gate-6 prediction
manifest and rollup, the Gate-6 count blocks and the Gate-6 precommit pin. The Gate-7A
tests re-run this and assert the record is unchanged.

    python tools/gate7a/stage0.py
"""
from __future__ import annotations

import hashlib, json, os, subprocess, sys, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                       # noqa: E402

OUT = "artifacts/gate7a/stage0_audit.json"

PRE_EXISTING_DIRTY = ("lingbot_map/models/gct_stream.py", "research/sem_bypass/model.py")
GATE6_DATASETS = ("semantickitti", "occ3d", "kitti360")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", REPO_ROOT, *args], capture_output=True,
                          text=True).stdout.strip()


def collect() -> dict:
    rec = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "git": {"commit": git("rev-parse", "HEAD"),
                   "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                   "status": git("status", "--short").splitlines()},
           "pre_existing_dirty": {}, "gate6": {"predictions": {}, "artifacts": {}}}
    for rel in PRE_EXISTING_DIRTY:
        p = os.path.join(REPO_ROOT, rel)
        rec["pre_existing_dirty"][rel] = {"sha256": sha256_file(p),
                                          "bytes": os.path.getsize(p)}
    for ds in GATE6_DATASETS:
        mp = os.path.join(REPO_ROOT, "artifacts", "gate6",
                          f"prediction_manifest_{ds}.json")
        man = json.load(open(mp))
        rec["gate6"]["predictions"][ds] = {
            "manifest_sha256": sha256_file(mp),
            "rollup_sha256": man["rollup_sha256"],
            "n_files": man["n_files"], "prediction_root": man["prediction_root"]}
    for rel in sorted(os.listdir(os.path.join(REPO_ROOT, "artifacts", "gate6"))):
        p = os.path.join(REPO_ROOT, "artifacts", "gate6", rel)
        if os.path.isfile(p):
            rec["gate6"]["artifacts"][rel] = sha256_file(p)
    pin = json.load(open(os.path.join(REPO_ROOT, "artifacts", "gate6",
                                      "precommit_pin.json")))
    got = sha256_file(os.path.join(REPO_ROOT, pin["path"]))
    rec["gate6"]["precommit"] = {"path": pin["path"], "pinned": pin["sha256"],
                                 "observed": got, "matches": got == pin["sha256"]}
    return rec


def main() -> int:
    rec = collect()
    write_json(os.path.join(REPO_ROOT, OUT), rec)
    print(f"{OUT}")
    print(f"  gate6 precommit matches its pin: {rec['gate6']['precommit']['matches']}")
    for ds, d in rec["gate6"]["predictions"].items():
        print(f"  {ds:14s} rollup {d['rollup_sha256'][:16]} ({d['n_files']} files)")
    for rel, d in rec["pre_existing_dirty"].items():
        print(f"  dirty {rel} {d['sha256'][:16]}")
    return 0 if rec["gate6"]["precommit"]["matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
