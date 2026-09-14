#!/usr/bin/env python
"""Phase 1: reconstruct and verify the historical post-ceiling-fix zero-shot baseline.

Reads only existing records (eval JSONs, logs, manifests, checkpoints, cached targets) and
rebuilds a few KITTI-360 targets to identify the exact sky-rule variant. Never opens a
target dataset. Writes historical_baseline_audit.md / historical_baseline.json /
historical_protocol.yaml / historical_provenance.csv under artifacts/baseline_reconstruction.

    python tools/baseline_reconstruction/cli/audit_history.py --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.baseline_reconstruction import config as C                                  # noqa: E402
from tools.baseline_reconstruction.historical import records as R, verify_ceiling_fix as V  # noqa: E402
from tools.baseline_reconstruction.reports import historical as REP                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0", help="GPU for the target-rule reproduction check; 'none' skips it")
    a = ap.parse_args()
    C.ensure_dirs()
    results = R.load_variant_results()
    checks = V.run_all(None if a.device == "none" else a.device)
    V.write(checks, os.path.join(C.ARTIFACT_DIR, "logs", "historical_verification.json"))
    for c in checks:
        print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}")
    out = REP.write_all(results, checks)
    print("wrote", out["audit"])
    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
