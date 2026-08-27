"""Run the existing ``tools/`` scripts against a 1024-channel pre-GCT token cache.

``semantic_sidecar.models.build_sidecar`` defaults ``in_channels`` to ``TOKEN_CHANNELS``
(2048 -- the aggregator's ``frame ‖ global`` width), and both ``train_semantic_sidecar``
and ``infer_semantic_sidecar`` call it without that argument.  SemBypass feeds 1024-wide
pre-GCT encoder tokens instead.

Rather than edit those tracked scripts (which would touch the artifacts of the earlier
study), this shim rebinds the ``build_sidecar`` symbol **inside the tool's own module
namespace** to one that reads the true width from the feature-cache manifest.  All
training, inference and evaluation logic is therefore the unmodified original code, and
the previous 9.1M results remain exactly reproducible with the untouched scripts.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from typing import List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
TOOLS = os.path.join(REPO_ROOT, "tools")
for p in (REPO_ROOT, TOOLS):
    if p not in sys.path:
        sys.path.insert(0, p)


def _cache_token_channels(cache_root: str) -> Optional[int]:
    """Read ``token_channels`` from any scene manifest under ``cache_root``."""
    root = cache_root if os.path.isabs(cache_root) else os.path.join(REPO_ROOT, cache_root)
    if not os.path.isdir(root):
        return None
    for name in sorted(os.listdir(root)):
        m = os.path.join(root, name, "manifest.json")
        if os.path.exists(m):
            try:
                return int(json.load(open(m))["token_channels"])
            except (KeyError, ValueError):
                continue
    return None


#: Tools that construct a sidecar and therefore need the width patch. Track building
#: and evaluation touch only geometry, teacher features and maps, so they run unpatched.
SIDECAR_TOOLS = frozenset({"train_semantic_sidecar", "infer_semantic_sidecar"})


def run_tool(module_name: str, argv: List[str], in_channels: Optional[int] = None,
             cache_root: Optional[str] = None) -> None:
    """Invoke ``tools/<module_name>.py::main``, patching ``build_sidecar`` where needed.

    Args:
        module_name: e.g. ``"train_semantic_sidecar"``.
        argv: command-line arguments *excluding* the program name.
        in_channels: token width; inferred from ``cache_root`` when omitted.
        cache_root: feature cache to infer the width from.
    """
    mod = importlib.import_module(module_name)
    patch = module_name in SIDECAR_TOOLS
    if patch:
        if in_channels is None:
            if cache_root is None:
                raise ValueError("pass in_channels or cache_root")
            in_channels = _cache_token_channels(cache_root)
            if in_channels is None:
                raise RuntimeError(f"no cached manifest under {cache_root}; cache features first")
        if not hasattr(mod, "build_sidecar"):
            raise RuntimeError(f"{module_name} should import build_sidecar; the shim would be silent")
        original = mod.build_sidecar

        def build_sidecar(cfg, num_layers, in_channels=in_channels):  # noqa: A002
            return original(cfg, num_layers, in_channels)

        mod.build_sidecar = build_sidecar

    old_argv = sys.argv
    sys.argv = [f"tools/{module_name}.py", *argv]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
        if patch:
            mod.build_sidecar = original
