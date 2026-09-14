"""Render the Phase-1 deliverables: historical_baseline_audit.md / .json / _protocol.yaml / _provenance.csv."""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import asdict
from typing import Dict, List

import yaml

from tools.baseline_reconstruction import config as C
from tools.baseline_reconstruction.historical import provenance as P, records as R
from tools.baseline_reconstruction.historical.verify_ceiling_fix import Check
from tools.baseline_reconstruction.reports.tables import markdown_table, write_csv

REMEMBERED = {"semantickitti": [22.71, 12.74], "occ3d": [30.74, 30.89]}   # the values the brief told us not to trust


def _variant_rows(results: List[R.HistoricalResult], variant: str, dataset: str) -> List[List[str]]:
    rows = []
    for proto in ("past5", "stream", "occany_fwd"):
        rs = R.select(results, variant, dataset, proto)
        if not rs:
            continue
        ious = [100 * r.sc_iou for r in rs]
        med = sorted(rs, key=lambda r: r.sc_iou)[len(rs) // 2]
        rows.append([variant, proto, " / ".join(f"{x:.2f}" for x in ious), f"{sorted(ious)[len(ious)//2]:.2f}",
                     f"{sum(ious)/len(ious):.2f}", f"{100*med.precision:.2f}", f"{100*med.recall:.2f}", f"{100*med.ssc_miou:.2f}",
                     str(rs[0].n_clips), os.path.dirname(rs[0].source_file)])
    return rows


def write_all(results: List[R.HistoricalResult], checks: List[Check], out_dir: str = C.ARTIFACT_DIR,
              paths: C.HistoricalPaths = C.HistoricalPaths()) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    verified = all(c.passed for c in checks)
    canon = {ds: P.canonical_result(results, ds, paths) for ds in C.HISTORICAL_TARGETS}
    timeline = P.fix_timeline(paths, results)
    prov = P.provenance_rows(results, paths, verified)
    logs = R.parse_result_logs(paths.sky_eval_logs)

    # ---- machine-readable ------------------------------------------------------------
    baseline_json = {
        "name": "Gate 8C-1 sky-fix baseline (post ceiling fix) -- canonical historical zero-shot baseline",
        "protocol_of_record": C.HISTORICAL_PROTOCOL, "verified": verified,
        "written": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "results": {ds: asdict(c) for ds, c in canon.items()},
        "remembered_values_checked": {ds: {"remembered": REMEMBERED[ds], "verified_median": canon[ds].median_pct} for ds in canon},
        "result_log_lines": logs, "timeline": asdict(timeline),
        "manifests": {v: R.manifest_summary(os.path.join(paths.gate_artifacts, f)) for v, f in R.VARIANTS.items()},
        "prose_only_numbers": [asdict(p) for p in R.PROSE_ONLY],
        "all_records": [asdict(r) for r in results],
        "verification": [asdict(c) for c in checks],
    }
    json.dump(baseline_json, open(os.path.join(out_dir, "historical_baseline.json"), "w"), indent=1, default=float)
    yaml.safe_dump(P.historical_protocol_yaml(), open(os.path.join(out_dir, "historical_protocol.yaml"), "w"), sort_keys=False, width=110)
    write_csv(os.path.join(out_dir, "historical_provenance.csv"), prov)

    # ---- markdown --------------------------------------------------------------------
    md = [f"# Historical baseline audit -- the post-ceiling-fix zero-shot baseline ({_dt.date.today().isoformat()})\n",
          "Every number below is read from a file on this machine (path given). Nothing was rerun on a target dataset; "
          "the ceiling-fix verification (§5) recomputes stored predictions and rebuilds KITTI-360 targets only.\n",
          "## 1. Verified post-ceiling-fix baseline\n",
          markdown_table(["Dataset", "Verified metric", "Value (median of 3 seeds)", "Per seed 0/1/2", "Mean ± sd", "Anchors", "Provenance"],
                         [[ds, c.metric, f"**{c.median_pct:.2f} %**", " / ".join(f"{c.per_seed_pct[s]:.2f}" for s in (0, 1, 2)),
                           f"{c.mean_pct:.2f} ± {c.sd_pct:.2f}", c.n_anchors, "; ".join(c.source_files)] for ds, c in canon.items()]),
          "\nPrecision / recall / SSC mIoU of the median seed: " +
          "; ".join(f"{ds}: P {c.precision_pct_median_seed:.2f} / R {c.recall_pct_median_seed:.2f} / mIoU {c.ssc_miou_pct_median_seed:.2f}" for ds, c in canon.items()) + ".\n",
          "### Remembered values, checked\n",
          markdown_table(["Dataset", "Remembered", "Verified", "Verdict"],
                         [["SemanticKITTI", "22.71 %", f"{canon['semantickitti'].median_pct:.2f} %", "correct (median seed 2; seeds 22.47 / 23.62 / 22.71)"],
                          ["Occ3D-nuScenes", "30.74 %", f"{canon['occ3d'].median_pct:.2f} %",
                           "**wrong**: 30.74 appears nowhere in the repository. The pre-fix past5 median is 30.89 (seeds 30.89 / 31.27 / 30.70) and the "
                           "post-fix matched-forward single-seed number is 30.70; the post-fix past5 median is 33.88 (seeds 33.49 / 35.03 / 33.88)."],
                          ["SemanticKITTI (pre-fix)", "12.74 %", f"{sorted(100*r.sc_iou for r in R.select(results,'prefix','semantickitti','past5'))[1]:.2f} %", "correct, but it is the *pre-fix* headline"],
                          ["Occ3D (pre-fix)", "30.89 %", f"{sorted(100*r.sc_iou for r in R.select(results,'prefix','occ3d','past5'))[1]:.2f} %", "correct, pre-fix headline"]]),
          "\n## 2. Every recorded evaluation of the two model versions\n",
          "`sky` = adopted post-fix model (`frozen_manifest_sky.json`); `prefix` = shipped pre-fix model (`frozen_manifest.json`). "
          "Metric = `semantic.completion.binary_iou` of each eval JSON (what the gate prints as `SC_IoU`).\n",
          markdown_table(["variant", "protocol", "SC IoU % seeds 0/1/2", "median", "mean", "P (median seed)", "R (median seed)", "SSC mIoU", "anchors", "record dir"],
                         [r for ds in C.HISTORICAL_TARGETS for v in ("sky", "prefix") for r in [[ds] + x for x in _variant_rows(results, v, ds)]]),
          "\n`RESULT` lines printed by the sky runs (`output/skyfix/*.log`): " + json.dumps(logs) + ".\n",
          "### Numbers that exist only in prose (their JSON records were overwritten by the rejected `sky2` run)\n",
          markdown_table(["label", "dataset", "value %", "source", "protocol", "comparability to canonical"],
                         [[p.label, p.dataset, p.value_pct, p.source, p.protocol, p.comparability_to_canonical] for p in R.PROSE_ONLY]),
          "\n`artifacts/gate8c1/occany_headtohead.json` (threshold −0.21875 = sky2 seed 0) and `occany_kitti_headtohead.json` (threshold −0.28125 = sky2 seed 2) "
          "hold the **sky2** results (28.12 / 22.93), not the adopted model's; the adopted model's head-to-head numbers (33.28 / 22.83) survive only in "
          "`ceiling_fix.md`. This audit records the correction here and leaves the files untouched.\n",
          "## 3. The ceiling fix: before, the change, the first valid evaluation, what superseded it\n",
          markdown_table(["event", "record"], [[k, v] for k, v in asdict(timeline).items()]),
          "\nThe fix has two halves; the ablation in `reports/the_ceiling_bug.md` §5 shows the padding half alone is *worse* (11.60) and the supervision half carries the "
          "whole 12.74 → 22.71 gain. The plain-column sky rule (no neighbourhood) trained the adopted model; the neighbourhood-7 extension (`sky2`) was rejected "
          "(Occ3D 33.28 → 28.12 on seed 0). Commit `647bcf3` later fixed the rule carving through measured returns; **no model was retrained after it**, so the "
          "canonical checkpoints rest on targets from the pre-`647bcf3` plain-column rule (verified bit-exactly in §5).\n",
          "## 4. The pipeline that produced the canonical numbers\n",
          markdown_table(["field", "value"], [[k, v] for k, v in asdict(P.HistoricalPipeline()).items()]),
          "\n## 5. Verification that the recovered result is the fixed pipeline\n",
          markdown_table(["#", "check", "passed"], [[i + 1, c.name, "PASS" if c.passed else "FAIL"] for i, c in enumerate(checks)]),
          "\nTop-layer (+4.3 m) supervised fraction on 40 anchors of 0006, from check 3: " + json.dumps(
              {k: round(v["top_layer_valid_pct"], 2) for k, v in checks[2].detail.items() if isinstance(v, dict) and "top_layer_valid_pct" in v}) +
          " % -- note the *current* Gate 12/13 builder (neighbourhood-7 rule + 647bcf3) supervises far more sky than the plain rule that trained the canonical model.\n",
          "\nDetails: `historical_baseline.json` → `verification`. Highlights: " + "; ".join(
              f"{c.name.split(':')[0]}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.detail.items()
                                                       if not isinstance(v, (list, dict))}, default=str) for c in checks if any(not isinstance(v, (list, dict)) for v in c.detail.values())) + ".\n",
          "## 6. Provenance table\n",
          markdown_table(["Field"] + [r["dataset"] for r in prov], [[k] + [r[k] for r in prov] for k in prov[0] if k != "dataset"]),
          "\n## 7. Canonical baseline decision\n",
          f"**Canonical historical baseline = the sky-fix model under Protocol A `past5`, full official anchor sets, median of three seeds: "
          f"SemanticKITTI {canon['semantickitti'].median_pct:.2f} %, Occ3D-nuScenes {canon['occ3d'].median_pct:.2f} % SC IoU.** "
          "Chosen for validity, not magnitude: it is the only post-fix evaluation on the complete official anchor sets with the gate's own masks and metric, "
          "and it is the headline `reports/METHOD.md` §8.1 adopts. The OccAny head-to-head numbers are partially comparable (frame subsets, OccAny's label "
          "volumes and scorer) and the `sky2` numbers are a rejected model. The `stream` and `occany_fwd` protocols were never run for the sky model.\n",
          "Caveat carried forward: the historical model is **not** geometry-only (32-channel input with Trident-H semantics, residual lock, long-stream LingBot "
          "geometry, per-seed source-selected thresholds), so any comparison with a geometry-only model is at best partially comparable -- see "
          "`historical_comparability.md` once the current baseline exists.\n"]
    open(os.path.join(out_dir, "historical_baseline_audit.md"), "w").write("\n".join(md))
    return {"audit": os.path.join(out_dir, "historical_baseline_audit.md")}


__all__ = ["write_all", "REMEMBERED"]
