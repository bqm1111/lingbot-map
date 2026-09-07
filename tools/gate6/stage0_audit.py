#!/usr/bin/env python
"""Gate 6 stage 0 — repository, environment and frozen-artifact audit.

Records the state everything downstream depends on, **before** a line of Gate-6 code
touches a dataset: the working tree, the commit, library versions, and SHA-256 hashes of
every manifest, scale table and configuration Gate 6 reuses. Bulk caches (tens of
thousands of NPZs) are fingerprinted by a deterministic index hash over
``(relative path, size)`` rather than by reading ~4 GB per run; the per-file contents are
already pinned by the gates that wrote them.

    python tools/gate6/stage0_audit.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gates.scale_gate.config import REPO_ROOT, write_json          # noqa: E402

# Files Gate 6 reuses unchanged. Every one is fully hashed.
PINNED_FILES = [
    "manifests/gate5_2/val.jsonl",
    "manifests/occ3d_zeroshot/val.jsonl",
    "artifacts/scale_gate/manifests/val.jsonl",
    "artifacts/gate5_1/scales_G51-B_kitti.csv",
    "artifacts/gate5_1/scales_G51-B_occ3d.csv",
    "artifacts/gate5_2/scales_B.csv",
    "configs/gate5/moge_metric_gauge.yaml",
    "configs/gate5_1/calibrated_gauge.yaml",
    "configs/gate5_2/kitti360_transfer.yaml",
    "configs/occ3d_zeroshot/frozen_transfer.yaml",
    "configs/scale_gate/semantickitti.yaml",
    "configs/depth_gate/refine.yaml",
    "checkpoints/lingbot-map/204754b/lingbot-map.pt",
]

# Bulk caches, fingerprinted by index.
PINNED_TREES = {
    "lingbot_cache_kitti": "artifacts/scale_gate/cache/lingbot",
    "lingbot_cache_occ3d": "/media/SSD1/MINH_DATASETS/lingbot_occ3d_zeroshot/cache_lingbot",
    "lingbot_cache_kitti360": "/media/SSD1/MINH_DATASETS/lingbot_gate5_2/cache_lingbot",
}

# User modifications that Gate 6 must preserve untouched.
PRESERVE = ["lingbot_map/models/gct_stream.py", "research/sem_bypass/model.py"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_index_hash(root: str):
    if not os.path.isdir(root):
        return {"exists": False}
    entries = []
    for dirpath, _, names in os.walk(root):
        for n in sorted(names):
            p = os.path.join(dirpath, n)
            entries.append((os.path.relpath(p, root), os.path.getsize(p)))
    entries.sort()
    h = hashlib.sha256()
    for rel, size in entries:
        h.update(f"{rel}\0{size}\0".encode())
    return {"exists": True, "n_files": len(entries),
            "bytes": sum(s for _, s in entries), "index_sha256": h.hexdigest()}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=120).stdout.strip()
    except Exception as e:                                     # pragma: no cover
        return f"<{type(e).__name__}: {e}>"


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("torch", "numpy", "scipy", "cv2", "open_clip", "mmcv", "mmengine", "mmseg",
              "timm", "PIL"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception:
            out[m] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["gpus"] = [torch.cuda.get_device_name(i)
                       for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return out


def main() -> int:
    audit = {
        "git": {
            "status_short": sh("git status --short"),
            "commit": sh("git rev-parse HEAD"),
            "branch": sh("git branch --show-current"),
            "describe": sh("git log -1 --format='%H %ad %s' --date=iso"),
        },
        "environment": versions(),
        "preserved_user_modifications": {
            p: {"sha256": sha256_file(os.path.join(REPO_ROOT, p)),
                "bytes": os.path.getsize(os.path.join(REPO_ROOT, p))}
            for p in PRESERVE if os.path.exists(os.path.join(REPO_ROOT, p))},
        "pinned_files": {}, "pinned_trees": {},
    }
    for rel in PINNED_FILES:
        p = rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)
        audit["pinned_files"][rel] = ({"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
                                      if os.path.exists(p) else {"missing": True})
    for name, root in PINNED_TREES.items():
        p = root if os.path.isabs(root) else os.path.join(REPO_ROOT, root)
        audit["pinned_trees"][name] = {"root": root, **tree_index_hash(p)}

    out = os.path.join(REPO_ROOT, "artifacts", "gate6", "stage0_audit.json")
    write_json(out, audit)
    print(json.dumps({k: audit[k] for k in ("git", "environment")}, indent=1)[:1400])
    print("\npinned files:")
    for k, v in audit["pinned_files"].items():
        print(f"  {k:55s} {v.get('sha256', 'MISSING')[:16]}")
    print("pinned trees:")
    for k, v in audit["pinned_trees"].items():
        print(f"  {k:22s} n={v.get('n_files')} bytes={v.get('bytes')} "
              f"{str(v.get('index_sha256'))[:16]}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
