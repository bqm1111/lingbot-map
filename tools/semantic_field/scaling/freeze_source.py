"""Freeze the source side of the scaling study, once, before any scaling training step.

Refuses unless: every added drive's preparation stages finished (exit 0) for exactly its kept anchors, the frozen reliability gate
passed with tiers [1, 2] (``prep/reliability_expanded.json``), and every file B4 and B3 read exists for every anchor of both pools.
Then it writes ``manifests/source_pool_expanded.json`` (pool, budgets, per-drive retention, storage layout digest, cache-identity
digest) and freezes stage ``source`` (declaration, manifest, code, frozen configs) with ``study_freeze.freeze``.

Cache identity: sha256 of every file both methods read (evidence, inputs, delayed targets, provenance, B3 student / query caches)
for all 9,066 anchors; the per-file list is written to ``manifests/cache_sha256.json`` and its digest goes into the manifest.

    PYTHONPATH=$PWD python -m tools.semantic_field.scaling.freeze_source
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List

from tools.semantic_field.scaling import loaders, study_freeze
from tools.semantic_field.scaling.config import ARTIFACT_DIR, OriginalBudgets, StudyPaths, stretched_steps
from tools.semantic_field.scaling.registration import ADDED_DRIVES
from tools.semantic_field.scaling.source_pool import SourcePool, check_membership, current_pool

PREP = os.path.join(ARTIFACT_DIR, "prep")
STAGES = ("clips", "inputs", "link_tokens", "evidence", "provenance", "targets", "train_cache")


def _last_exit(log: str) -> int | None:
    if not os.path.exists(log):
        return None
    lines = [l for l in open(log, errors="replace").read().splitlines() if l.startswith("exit ")]
    return int(lines[-1].split()[1]) if lines else None


def retention(draft: Dict) -> Dict[str, Dict]:
    """Per added drive: eligible (7-drive manifest), kept (geography), and file counts per cache kind; stage exits."""
    geo = json.load(open(os.path.join(ARTIFACT_DIR, "verification", "geography_check.json")))["drives"]
    out = {}
    for d in ADDED_DRIVES:
        kept = draft["added_anchors"][d]; row = {"eligible": geo[d]["eligible"], "kept": len(kept), "stage_exit": {}}
        for s in STAGES:
            row["stage_exit"][s] = _last_exit(os.path.join(ARTIFACT_DIR, "logs", f"{s}_{d[17:21]}.log"))
        missing = {m: sum(not os.path.exists(f) for t in kept for f in loaders.files_of(m)((d, t))) for m in ("B4", "B3")}
        row["missing_files"] = missing
        out[d] = row
    return out


def build_expanded(draft: Dict) -> SourcePool:
    added = tuple((d, int(t)) for d in draft["added_drives"] for t in draft["added_anchors"][d])
    pool = SourcePool("expanded", current_pool().anchors + added)
    check_membership(pool.anchors)
    return pool


def cache_files(pool: SourcePool) -> List[str]:
    return sorted({f for k in pool.anchors for m in ("B4", "B3") for f in loaders.files_of(m)(k)})


def main() -> None:
    paths = StudyPaths()
    draft = json.load(open(os.path.join(paths.manifests, "source_pool_expanded.DRAFT.json")))
    ret = retention(draft)
    problems = [f"{d}: {r}" for d, r in ret.items() if any(v != 0 for v in r["stage_exit"].values()) or any(r["missing_files"].values())]
    rel = json.load(open(os.path.join(PREP, "reliability_expanded.json"))) if os.path.exists(os.path.join(PREP, "reliability_expanded.json")) else None
    if rel is None or rel["status"] != "ok":
        problems.append(f"reliability: {None if rel is None else rel['status']}")
    if problems:
        raise SystemExit("BLOCKED, not freezing:\n" + "\n".join(problems))
    pool = build_expanded(draft)
    files = cache_files(pool)
    digests = study_freeze.hash_many(files)
    cache_path = os.path.join(paths.manifests, "cache_sha256.json")
    json.dump(digests, open(cache_path, "w"), indent=0)
    budgets = {m: stretched_steps(OriginalBudgets().of(m), len(pool.anchors)) for m in ("B4", "B3")}
    manifest = {"study_version": draft.get("study_version", "b4_source_scaling/2026-09-15"), "rule": draft["rule"],
                "added_drives": draft["added_drives"], "added_anchors": draft["added_anchors"], "n_current": len(current_pool().anchors),
                "n_added": len(pool.anchors) - len(current_pool().anchors), "n_expanded": len(pool.anchors),
                "current_pool_sha256": current_pool().digest(), "expanded_pool_sha256": pool.digest(), "budgets": budgets,
                "retention": ret, "reliability_gate": rel["gate"], "cache_files": len(files),
                "cache_sha256_digest": hashlib.sha256(open(cache_path, "rb").read()).hexdigest(),
                "storage_layout_sha256": study_freeze.sha256(os.path.join(PREP, "storage_layout.json")),
                "geography_check_sha256": draft["geography_check_sha256"]}
    json.dump(manifest, open(os.path.join(paths.manifests, "source_pool_expanded.json"), "w"), indent=1)
    rec = study_freeze.freeze("source", study_freeze.SOURCE_FILES + [os.path.relpath(cache_path, study_freeze.REPO_ROOT)],
                              {"budgets": budgets, "n_expanded": len(pool.anchors), "expanded_pool_sha256": pool.digest()})
    print(json.dumps({"frozen_at": rec["frozen_at"], "budgets": budgets, "n_expanded": len(pool.anchors), "cache_files": len(files)}, indent=1))


if __name__ == "__main__":
    main()
