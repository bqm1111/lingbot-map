#!/usr/bin/env python
"""Phase 2c: freeze the source side. Run once, after source_results.md shows the reproduction PASS.

    python tools/baseline_reconstruction/cli/freeze_source.py

Refuses if the source reproduction verdict failed or if a freeze already exists. Writes
code_freeze.json and source_freeze.md. Target evaluation is unlocked only by this file.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.baseline_reconstruction import config as C, freeze as FZ                 # noqa: E402
from tools.baseline_reconstruction.reports.tables import markdown_table              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    paths = C.CurrentPaths()
    summary = json.load(open(os.path.join(paths.source_eval, "source_summary.json")))
    if not summary["reproduction"]["pass"]:
        raise SystemExit("source reproduction verdict is FAIL: diagnose before freezing (see source_results.md)")
    rec = FZ.write_source_freeze(paths, summary["reproduction"], reason="0006 result confirmed against Gate 13 M within the predeclared tolerance")
    md = [f"# Source freeze -- {C.CURRENT_BASELINE_NAME}\n",
          f"Frozen at **{rec['frozen_at']}** (git HEAD `{rec['git_head'][:12]}`, {rec['git_status_dirty_files']} uncommitted paths in the working tree). "
          "From this moment no code, config, checkpoint, mapper setting, threshold, adapter or evaluation script may change; "
          "`require_source_freeze()` re-hashes every file below before any target evaluation and refuses to run on a mismatch.\n",
          "## Reason\n", rec["reason"] + "\n", "## Frozen decisions\n",
          markdown_table(["decision", "value"], [[k, v] for k, v in rec["frozen_decisions"].items() if k != "target_protocols"]),
          "\n## Source result at freeze time (full-region 0006 IoU)\n",
          markdown_table(["seed", "this run", "Gate 13 M", "abs diff"], [[s, f"{v:.4f}", f"{rec['source_summary']['gate13_M_full_iou'][s]:.4f}", f"{rec['source_summary']['per_seed_abs_diff'][s]:.4f}"]
                                                                       for s, v in rec["source_summary"]["ours_full_iou"].items()]),
          f"\nmean abs diff {rec['source_summary']['mean_abs_diff']:.4f} (tolerance {rec['source_summary']['tolerance']}) -> PASS\n",
          f"\n## Hashed files ({len(rec['sha256'])})\n",
          markdown_table(["file", "sha256"], [[f, h[:16]] for f, h in rec["sha256"].items()]),
          "\nRender-only code excluded from the hash set (may be edited without affecting any prediction or metric): " + ", ".join(rec["excluded_from_freeze_render_only"]) + "\n",
          "\nTarget datasets opened before this freeze: **no** (every source-side stage ran under a strict file-access audit; see verification.md).\n"]
    open(os.path.join(C.ARTIFACT_DIR, "source_freeze.md"), "w").write("\n".join(md))
    print(f"frozen {len(rec['sha256'])} files at {rec['frozen_at']} -> {FZ.FREEZE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
