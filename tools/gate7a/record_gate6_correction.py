#!/usr/bin/env python
"""Record exactly what changed in the Gate-6 report, and prove no number moved.

Gate 7A's first task was to correct six reporting issues in the Gate-6 report. The fixes
were made in the **generating source** -- the template and the table generator -- and the
report was regenerated from the unchanged artifacts. This records the before/after hashes
of all three files and re-verifies that every Gate-6 numerical artifact is byte-identical
to its stage-0 record, so the corrections provably touched prose and table *rendering*
only.

    python tools/gate7a/record_gate6_correction.py
"""
from __future__ import annotations

import hashlib, json, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT, write_json                       # noqa: E402

REPORT = "reports/gate6/frozen_trident_semantic_lifting.md"
TEMPLATE = "reports/gate6/_frozen_trident_semantic_lifting.template.md"
GENERATOR = "tools/gate6/report_tables.py"

# Measured immediately before the corrections were applied, in this session.
BEFORE = {
    REPORT: ("14fbf23e12968bb537332199bd89e65d7d71f14339077e4f359a11651bfa1bf2", 956),
    TEMPLATE: ("2fa98446d5af9ee55d57eec9eaf4e36a3761d261c2fb0348f8503a10383e69c6", None),
    GENERATOR: ("d31155d1063282798bc378c680e9eec637455298b70642ce408ce400252bc1fe", None),
}

CORRECTIONS = [
    {"what": "§14 no longer claims that every bootstrap interval excludes zero. The "
             "statement is now generated from the artifacts and names the exception, the "
             "Occ3D-nuScenes TP-conditioned accuracy difference, with its interval; the "
             "surrounding prose now reads that drop as not distinguishable from zero on "
             "that benchmark.",
     "where": "`report_tables.t_bootstrap_zero_note` (new generator) + template §14"},
    {"what": "The 'Official SSC metrics' table now reports **pooled** binary IoU, "
             "precision and recall alongside its pooled semantic metrics, with every "
             "column labelled by aggregation. The values are computed from the stored "
             "counts, not transcribed.",
     "where": "`report_tables.t_main`"},
    {"what": "The reproduction/sanity table keeps the **mean-per-clip** binary IoU the "
             "frozen gates were pinned with, labels it as such, and shows the pooled "
             "value beside it for reference.",
     "where": "`report_tables.t_sanity`"},
    {"what": "`COVERAGE_DOMINATES` is now explicitly scoped to the population of valid "
             "ground-truth **occupied** voxels. §1 and §9 state that the partition "
             "excludes false-positive occupied predictions entirely and therefore does "
             "not establish that expanding occupancy would raise IoU.",
     "where": "template §1, §2, §9"},
    {"what": "§12 states the near-field reversal outright: in SemanticKITTI's 0–10 m band "
             "at B-D, naming error (0.264) exceeds coverage miss (0.256). A generated "
             "table shows the 0–10 m band of all three benchmarks.",
     "where": "`report_tables.t_near_range_note` (new generator) + template §12"},
    {"what": "The Grounded-SAM-2 comparator now carries its limits **before** the table: "
             "one dataset, one drive, no paired confidence interval, and a vocabulary "
             "confound (OccAny synonym lists against Gate 6's single deterministic "
             "phrase per class).",
     "where": "`report_tables.t_occany_section`"},
    {"what": "Trident's spatial encoder is named as **DINO v1 ViT-B/16, not DINOv2**, in "
             "the provenance table and in the operating-point listing, not only in the "
             "deviation note.",
     "where": "`report_tables.t_provenance` + template §4"},
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    stage0 = json.load(open(os.path.join(REPORT_DIR := os.path.join(
        REPO_ROOT, "artifacts", "gate7a"), "stage0_audit.json")))
    changed = []
    for rel, want in stage0["gate6"]["artifacts"].items():
        p = os.path.join(REPO_ROOT, "artifacts", "gate6", rel)
        if not os.path.exists(p) or sha256_file(p) != want:
            changed.append(rel)

    now = {k: sha256_file(os.path.join(REPO_ROOT, k))
           for k in (REPORT, TEMPLATE, GENERATOR)}
    rec = {
        "report": REPORT,
        "sha256_before": BEFORE[REPORT][0], "sha256_after": now[REPORT],
        "lines_before": BEFORE[REPORT][1],
        "lines_after": sum(1 for _ in open(os.path.join(REPO_ROOT, REPORT))),
        "template": {"path": TEMPLATE, "sha256_before": BEFORE[TEMPLATE][0],
                     "sha256_after": now[TEMPLATE]},
        "generator": {"path": GENERATOR, "sha256_before": BEFORE[GENERATOR][0],
                      "sha256_after": now[GENERATOR]},
        "corrections": CORRECTIONS,
        "gate6_artifacts_changed": ("none — all "
                                    f"{len(stage0['gate6']['artifacts'])} byte-identical"
                                    if not changed else ", ".join(changed)),
        "gate6_numerical_artifacts_untouched": not changed,
        "method": ("the fixes were made in the template and the generator and the report "
                   "was regenerated; no number was edited in the rendered Markdown"),
    }
    write_json(os.path.join(REPORT_DIR, "gate6_report_correction.json"), rec)
    print("artifacts/gate7a/gate6_report_correction.json")
    print(f"  report {rec['sha256_before'][:16]} -> {rec['sha256_after'][:16]}")
    print(f"  lines  {rec['lines_before']} -> {rec['lines_after']}")
    print(f"  gate6 artifacts changed: {rec['gate6_artifacts_changed']}")
    return 0 if not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
