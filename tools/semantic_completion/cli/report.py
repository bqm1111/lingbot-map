#!/usr/bin/env python
"""Write report.md and code_freeze.json from analysis.json and verification.md (run after cli/analyse.py).

    python tools/semantic_completion/cli/report.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.semantic_completion import config as C                      # noqa: E402
from tools.semantic_completion.analysis import load_analysis                          # noqa: E402
from tools.semantic_completion.reports.final import write_code_freeze, write_report  # noqa: E402


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    res = load_analysis()
    m = re.search(r"\*\*(\d+) / (\d+) checks pass", open(os.path.join(C.ARTIFACT_DIR, "verification.md")).read())
    out = write_report(res, {"n_pass": int(m.group(1)), "n_total": int(m.group(2))})
    fz = write_code_freeze(res)
    print("wrote", out, "| decision", res["promotion"]["decision"], "| frozen", len(fz["sha256"]), "files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
