#!/usr/bin/env python
"""Write reports/waymo_supervision_pilot.md from the pilot's artifacts."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.waymo_pilot import report as R                                     # noqa: E402

if __name__ == "__main__":
    p = R.write()
    print(f"wrote {p} ({os.path.getsize(p)} bytes)")
