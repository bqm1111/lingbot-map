"""Put the repository root on ``sys.path``.

The editable install of ``lingbot-map`` points at a directory that no longer exists,
so scripts under ``tools/`` must add the repo root themselves (see CLAUDE.md).
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
