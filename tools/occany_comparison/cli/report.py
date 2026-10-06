#!/usr/bin/env python
"""Step 4: analysis.json, results_table.md, figures, verification.md and report.md (CPU only).

    python tools/occany_comparison/cli/report.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.occany_comparison.report import write_all                # noqa: E402


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    print(write_all())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
