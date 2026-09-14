#!/usr/bin/env python
"""Aggregate the 0006 evaluation, apply the frozen promotion rule, write tables, results_table.md, figures and verification.md.

    python tools/semantic_completion/cli/analyse.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.semantic_completion import analysis as A, config as C       # noqa: E402
from tools.semantic_completion.reports import figures as FIG, report as REP, verification as VER  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--skip-figures", action="store_true"); ap.add_argument("--skip-verification", action="store_true")
    a = ap.parse_args()
    A.analyse()
    res = A.load_analysis()
    print("results:", REP.write_results_table(res), "| decision", res["promotion"]["decision"])
    if not a.skip_figures:
        print("figures:", FIG.write_figures(a.device)["examples"])
    if not a.skip_verification:
        print("verification:", VER.write_verification(a.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
