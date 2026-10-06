"""Resource accounting: GPU device-seconds from ``/usr/bin/time -v`` logs, storage from disk usage, totals from the ledger.

Rule: allocated device time is counted once per GPU while a job holds it. Jobs that share a GPU at the same time are entered as
one ledger row for that GPU (wall time of the shared period), never once per job.

    PYTHONPATH=$PWD python -m tools.semantic_field.scaling.resources add --category preparation --gpu 0 --logs 'logs/tokens_0005_s0of2.log'
    PYTHONPATH=$PWD python -m tools.semantic_field.scaling.resources summary
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
from typing import Dict, List

from tools.semantic_field.scaling.config import ARTIFACT_DIR, ResourceCaps, StudyPaths

_ELAPSED = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)")


def elapsed_seconds(log_path: str) -> float:
    """Wall time printed by ``/usr/bin/time -v`` at the end of a log."""
    m = _ELAPSED.findall(open(log_path, errors="replace").read())
    if not m:
        raise ValueError(f"{log_path}: no /usr/bin/time elapsed line (job unfinished or not wrapped)")
    parts = [float(p) for p in m[-1].split(":")]
    return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


def busy_interval(log_path: str) -> tuple[float, float]:
    """(start, end) epoch seconds of a job: end = the log's last modification; start = end - /usr/bin/time elapsed for a finished
    job, or the log's creation time for a killed job (no elapsed line). The killed case slightly under-counts (after the last write)."""
    end = os.path.getmtime(log_path)
    try:
        return end - elapsed_seconds(log_path), end
    except ValueError:
        st = os.stat(log_path)
        birth = getattr(st, "st_birthtime", None)
        if birth is None:                                   # Linux: statx birth time via `stat -c %W`
            birth = float(subprocess.run(["stat", "-c", "%W", log_path], capture_output=True, text=True, check=True).stdout.strip())
        if birth <= 0:
            raise ValueError(f"{log_path}: killed job and no creation time available")
        return birth, end


def union_seconds(intervals: List[tuple[float, float]]) -> float:
    """Total covered time of possibly overlapping intervals (jobs sharing one GPU are counted once)."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s if cur_e is not None else 0.0)


def add_gpu_jobs(category: str, gpu: int, logs: List[str], note: str = "") -> Dict:
    """One ledger row for one GPU: the union of the logged jobs' busy intervals (sequential or concurrent)."""
    seconds = union_seconds([busy_interval(p) for p in logs])
    row = {"category": category, "item": note or ", ".join(os.path.basename(p) for p in logs), "devices": [f"cuda:{gpu}"],
           "gpu_seconds": round(seconds), "source": [os.path.relpath(p, ARTIFACT_DIR) for p in logs], "storage_bytes": 0,
           "accounting": "union of busy intervals on this GPU"}
    with open(StudyPaths().ledger, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def du_bytes(path: str) -> int:
    """Disk usage in bytes. ``du`` exits 1 when a file vanishes mid-walk (a writer renaming tmp -> final); its total is still printed."""
    proc = subprocess.run(["du", "-sbL", path], capture_output=True, text=True)
    if not proc.stdout.strip():
        raise RuntimeError(f"du {path} failed: {proc.stderr.strip()[:300]}")
    return int(proc.stdout.split()[0])


def summary() -> Dict:
    rows = [json.loads(l) for l in open(StudyPaths().ledger)]
    by_cat: Dict[str, float] = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0.0) + r["gpu_seconds"] / 3600
    # new study caches only; directories MOVED to the cold tier (``relocated/``) are existing data, reported separately
    new_storage = {p: du_bytes(p) for p in ("/media/welf/SSD2/MINH_DATASETS/lingbot_b4_scaling/hot", "/media/SSD1/MINH_DATASETS/lingbot_b4_scaling/hot",
                                            "/media/minh/MINH1/lingbot_b4_scaling/cold") if os.path.exists(p)}
    relocated = "/media/minh/MINH1/lingbot_b4_scaling/relocated"
    caps = ResourceCaps()
    return {"gpu_hours_by_category": {k: round(v, 3) for k, v in by_cat.items()}, "gpu_hours_total": round(sum(by_cat.values()), 3),
            "gpu_hours_cap": caps.gpu_hours, "new_storage_gb": {k: round(v / 1e9, 2) for k, v in new_storage.items()},
            "new_storage_gb_total": round(sum(new_storage.values()) / 1e9, 2), "storage_cap_gb": caps.extra_storage_gb,
            "relocated_existing_data_gb": round(du_bytes(relocated) / 1e9, 2) if os.path.exists(relocated) else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add"); add.add_argument("--category", required=True); add.add_argument("--gpu", type=int, required=True)
    add.add_argument("--logs", nargs="+", required=True); add.add_argument("--note", default="")
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "add":
        logs = sorted({p for pat in a.logs for p in glob.glob(os.path.join(ARTIFACT_DIR, pat) if not os.path.isabs(pat) else pat)})
        print(add_gpu_jobs(a.category, a.gpu, logs, a.note))
    else:
        print(json.dumps(summary(), indent=1))


if __name__ == "__main__":
    main()
