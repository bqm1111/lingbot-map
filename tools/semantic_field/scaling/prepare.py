"""Cache preparation for the added drives, one stage at a time, with the unchanged producers.

Stage order (each needs the previous ones; tokens and clips are independent of each other):
    tokens      Trident-H token maps           (``prepare_tokens.py`` in the Trident env; then ``link``)
    clips       local-reset LingBot + MoGe-2   (run_local_lingbot.py, stride 1, kept anchors only)
    inputs      hard maps                      (gate12_suf/build_inputs.py)
    evidence    Gate 15 evidence lifting       (semantic_completion/evidence/build.py::build_drive)
    provenance  evidence frame bits            (delayed_semantic/provenance.py::build_drive)
    targets     delayed targets, 10 s, TRAINING ONLY (delayed_semantic/targets.py::build_drive)
    train_cache B3 student / query features    (delayed_semantic/train_cache.py::build_drive, variant "" only)
Only anchors kept by the expanded-pool manifest get clips, so every later stage (which enumerates the drive's input files)
covers exactly the kept anchors. Each stage writes ``artifacts/semantic_field_scaling/prep/<stage>_<drive>[_s<i>].json``.

    PYTHONPATH=$PWD python -m tools.semantic_field.scaling.prepare --stage clips --drive 2013_05_28_drive_0005_sync --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time
from typing import Dict

from tools.semantic_field.scaling.config import ARTIFACT_DIR, REPO_ROOT
from tools.semantic_field.scaling.registration import ADDED_DRIVES, default_manifest_path, kept_stream_indices, register_added_drives

STAGES = ("link_tokens", "clips", "inputs", "evidence", "provenance", "targets", "train_cache")
PREP_DIR = os.path.join(ARTIFACT_DIR, "prep")
#: cache kind(s) placed before a stage writes, with the MB per kept anchor reserved by the free-space check (evidence: measured, see below)
STAGE_KINDS = {"clips": [("clips", 1.0)], "inputs": [("inputs", 0.2)], "evidence": [("evidence", None)], "provenance": [("provenance", 0.3)],
               "targets": [("delayed_targets", 0.4)], "train_cache": [("train_cache_student", 2.5), ("train_cache_query", 0.4)]}
EVIDENCE_SAFETY_FACTOR = 1.10       # over the measured mean of the added drives built so far (0005: 35.6, 0000: 44.4 MB / anchor)
EVIDENCE_FALLBACK_MB = 50.0          # just above the largest measured drive mean (0010: 48.7 MB / anchor)


def evidence_mb_per_anchor() -> float:
    """Reserve for one evidence file: the measured mean over added drives already built (x safety factor), else the fallback."""
    from tools.semantic_field.scaling import storage
    sizes, counts = 0, 0
    for key, p in storage._load_layout()["placements"].items():
        rec = os.path.join(PREP_DIR, f"evidence_{key.split('/')[1][17:21]}_s0of1.json")
        if key.startswith("evidence/") and os.path.exists(rec) and os.path.isdir(p["target"]):
            files = [os.path.join(p["target"], f) for f in os.listdir(p["target"]) if f.endswith(".npz")]
            sizes += sum(os.path.getsize(f) for f in files); counts += len(files)
    return EVIDENCE_SAFETY_FACTOR * sizes / counts / 1e6 if counts else EVIDENCE_FALLBACK_MB


def _run_script(path: str, argv) -> None:
    sys.argv = [path] + list(argv)
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise RuntimeError(f"{path} exited with {exc.code}") from exc


def run_stage(stage: str, drive: str, device: str, shard: int, shards: int, hot_root: str | None) -> Dict:
    from tools.semantic_field.scaling import storage
    if drive not in ADDED_DRIVES:
        raise SystemExit(f"{drive} is not an added drive")
    register_added_drives()
    kept = kept_stream_indices(default_manifest_path())[drive]
    for kind, mb_per_anchor in STAGE_KINDS.get(stage, []):
        mb = evidence_mb_per_anchor() if mb_per_anchor is None else mb_per_anchor
        placement = storage.place(kind, drive, hot_root, need_gb=len(kept) * mb / 1e3 if shard == 0 else 0.0)
        rec_reserved = {"kind": kind, "reserved_gb": round(len(kept) * mb / 1e3, 1), "target": placement.target}
        print(f"[prepare] placed {rec_reserved}", flush=True)
    t0 = time.time(); rec: Dict = {"stage": stage, "drive": drive, "shard": shard, "shards": shards, "kept_anchors": len(kept)}
    if stage == "link_tokens":
        rec["token_links"] = storage.link_tokens(drive)
    elif stage == "clips":
        from core.datasets import kitti360_splits as SRC
        from tools.local_registration_audit import common as LR
        sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "gate12_suf")); import g12
        LR.DATA_ROOT = g12.DATA_ROOT; LR.ANCHOR_STRIDE = 1                    # exactly gen_local_clips.py's redirection
        all_anchors, keep = SRC.anchors, set(kept)
        SRC.anchors = lambda d, repo_root, **kw: [a for a in all_anchors(d, repo_root, **kw) if d != drive or a.stream_index in keep]
        _run_script(os.path.join(REPO_ROOT, "tools", "local_registration_audit", "run_local_lingbot.py"),
                    ["--drive", drive, "--device", device, "--shard", str(shard), "--shards", str(shards)])
    elif stage == "inputs":
        _run_script(os.path.join(REPO_ROOT, "tools", "gate12_suf", "build_inputs.py"), ["--drive", drive, "--device", device, "--shard", str(shard), "--shards", str(shards)])
    elif stage == "evidence":
        from tools.semantic_completion.evidence import build as G15B
        rec["result"] = G15B.build_drive(drive, device, shard, shards)
    elif stage == "provenance":
        from tools.delayed_semantic import provenance as PV
        r = PV.build_drive(drive, device, shard, shards); r.pop("rows", None); rec["result"] = r
    elif stage == "targets":
        from tools.delayed_semantic import targets as TG
        r = TG.build_drive(drive, device, shard, shards); rec["result"] = {k: v for k, v in r.items() if k != "rows"}; rec["per_anchor"] = r["rows"]
    elif stage == "train_cache":
        from tools.delayed_semantic import train_cache as TC
        rec["result"] = TC.build_drive(drive, device, shard, shards, variants=("",))
    rec["seconds"] = time.time() - t0
    os.makedirs(PREP_DIR, exist_ok=True)
    json.dump(rec, open(os.path.join(PREP_DIR, f"{stage}_{drive[17:21]}_s{shard}of{shards}.json"), "w"), indent=1, default=str)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES, required=True); ap.add_argument("--drive", choices=ADDED_DRIVES, required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--hot-root", default=None, help="one of storage.HOT_ROOTS (default: the first)")
    a = ap.parse_args()
    rec = run_stage(a.stage, a.drive, a.device, a.shard, a.shards, a.hot_root)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("per_anchor", "result")}, default=str))


if __name__ == "__main__":
    main()
