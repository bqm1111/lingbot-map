"""Config loading and run provenance for the scale gate."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))  # gates/scale_gate/ -> repo root


class Config(dict):
    """A dict with dotted access and a stable content hash."""

    def __getattr__(self, k):                                  # noqa: D105
        try:
            v = self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc
        return Config(v) if isinstance(v, dict) else v

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self, sort_keys=True).encode()).hexdigest()[:16]

    def path(self, *parts: str) -> str:
        """Resolve a repo-relative path from the config."""
        return os.path.join(REPO_ROOT, *parts)


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Config:
    with open(path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)) as fh:
        raw = yaml.safe_load(fh) or {}
    for k, v in (overrides or {}).items():                     # dotted overrides
        node, *rest = k.split(".")
        cur = raw
        while rest:
            cur = cur.setdefault(node, {})
            node, *rest = rest
        cur[node] = v
    return Config(raw)


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                          # noqa: BLE001
        return None


def collect_provenance(cfg: Config, tool: str, extra: Optional[Dict] = None) -> Dict[str, Any]:
    """Everything needed to reproduce or invalidate a run."""
    import torch
    ck = os.path.join(REPO_ROOT, cfg["lingbot"]["checkpoint"])
    prov = {
        "tool": tool,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty_files": (_git("status", "--porcelain") or "").splitlines(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": cfg["experiment"]["seed"],
        "config_hash": cfg.hash,
        "checkpoint": cfg["lingbot"]["checkpoint"],
        "checkpoint_sha256": file_sha256(ck) if os.path.exists(ck) else None,
        "argv": sys.argv,
    }
    if extra:
        prov.update(extra)
    return prov


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
