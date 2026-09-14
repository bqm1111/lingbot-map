"""The training recipe file: a verbatim copy of Gate 12's condition-H config with a provenance header."""
from __future__ import annotations

import hashlib
import os

from tools.baseline_reconstruction import config as C


def write_training_config(paths: C.CurrentPaths = C.CurrentPaths()) -> str:
    """Copy ``artifacts/gate12_suf/training_config_H.yaml`` into this experiment, once."""
    if os.path.exists(paths.training_config):
        return paths.training_config
    src = open(paths.recipe_source).read()
    sha = hashlib.sha256(src.encode()).hexdigest()
    header = (f"# {C.CURRENT_BASELINE_NAME}\n"
              f"# Verbatim copy of {os.path.relpath(paths.recipe_source, C.REPO_ROOT)} (sha256 {sha}),\n"
              f"# the frozen Gate 12 condition-H / Gate 13 condition-M recipe. Nothing below was changed.\n"
              f"# experiment: {C.PROTOCOL_VERSION}\n")
    os.makedirs(os.path.dirname(paths.training_config), exist_ok=True)
    open(paths.training_config, "w").write(header + src)
    return paths.training_config


__all__ = ["write_training_config"]
