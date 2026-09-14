"""report.md: the compact summary tables, the answers to the brief's questions, the decision and the one next action."""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Dict

from tools.baseline_reconstruction import config as C
from tools.baseline_reconstruction.reports.tables import markdown_table, pct


def decide(source_pass: bool, zs: Dict[str, Dict], hist: Dict, cis: Dict) -> Dict:
    """The predeclared decision rule (predeclaration.md), applied mechanically."""
    per = {}
    for ds in C.HISTORICAL_TARGETS:
        z = zs[ds]; h = hist["results"][ds]
        cur, cur_sd = 100 * z["seed_statistics"]["iou"]["mean"], 100 * z["seed_statistics"]["iou"]["sd"]
        per[ds] = {"current_mean": cur, "current_sd": cur_sd, "historical_median": h["median_pct"], "map_only": 100 * z["map_only"]["full"]["iou"],
                   "beats_map_only": cur > 100 * z["map_only"]["full"]["iou"], "at_or_above_historical": cur + cur_sd >= h["median_pct"],
                   "ci": cis.get(ds)}
    if not source_pass:
        return {"decision": "STOP", "per_target": per, "reason": "source reproduction failed"}
    if not all(v["beats_map_only"] for v in per.values()):
        return {"decision": "C", "per_target": per, "reason": "completion does not beat its own map-only reference on " + ", ".join(ds for ds, v in per.items() if not v["beats_map_only"])}
    if all(v["at_or_above_historical"] for v in per.values()):
        return {"decision": "A", "per_target": per, "reason": "current mean SC IoU at or above the canonical historical median (within one seed sd) on both targets"}
    return {"decision": "B", "per_target": per, "reason": "below the historical median on " + ", ".join(ds for ds, v in per.items() if not v["at_or_above_historical"]) +
            "; the historical system carried semantics, 32 input channels, a residual lock, source-selected thresholds and different targets (historical_comparability.md)"}


def write_report(source: Dict, zs: Dict[str, Dict], hist: Dict, comp: Dict, cis: Dict, verification: Dict) -> str:
    src_mean = sum(source["reproduction"]["ours_full_iou"].values()) / 3
    dec = decide(source["reproduction"]["pass"], zs, hist, cis)
    per = dec["per_target"]
    hsk, hoc = hist["results"]["semantickitti"], hist["results"]["occ3d"]
    names = {"semantickitti": "SemanticKITTI", "occ3d": "Occ3D-nuScenes"}
    rows_hist = [[names[ds], "SC IoU (occupancy, official mask), past5, median of 3 seeds", f"**{hist['results'][ds]['median_pct']:.2f} %**",
                  f"{'; '.join(hist['results'][ds]['source_files'])}; frozen_manifest_sky.json; verified in historical_baseline_audit.md"] for ds in C.HISTORICAL_TARGETS]
    rows_cur = [["KITTI-360 0006 (full region)", f"{100*src_mean:.2f} %", "n/a", "not comparable (source-side; different target build/threshold than the historical source number)"]] + \
               [[names[ds], f"**{per[ds]['current_mean']:.2f} %** (median {100*zs[ds]['seed_statistics']['iou']['median']:.2f}, sd {per[ds]['current_sd']:.2f})",
                 f"{per[ds]['historical_median']:.2f} %", comp["classes"][ds]["class"]] for ds in C.HISTORICAL_TARGETS]
    ratio_cur = per["semantickitti"]["current_mean"] / per["occ3d"]["current_mean"]; ratio_hist = hsk["median_pct"] / hoc["median_pct"]
    sk_map, oc_map = per["semantickitti"]["map_only"], per["occ3d"]["map_only"]
    hist_map = {ds: 100 * [r for r in hist["all_records"] if r["variant"] == "sky" and r["dataset"] == ds and r["protocol"] == "past5"][0]["mapper_only_iou"] for ds in C.HISTORICAL_TARGETS}
    sk_seeds = " / ".join(f"{hsk['per_seed_pct'][str(s)]:.2f}" for s in C.SEEDS)
    oc_seeds = " / ".join(f"{hoc['per_seed_pct'][str(s)]:.2f}" for s in C.SEEDS)
    src_seeds = " / ".join(f"{100*v:.2f}" for v in source["reproduction"]["ours_full_iou"].values())
    next_action = ("**Freeze the Current Geometry-Only Baseline (L + LB-M + H, CompletionUNet-3ch, seeds 0/1/2, step 6000, threshold 0; "
                   "code_freeze.json) as the geometry-only control and return to semantic completion.**" if dec["decision"] in ("A", "B", "D")
                   else "**Keep semantics paused and diagnose the single identified cross-dataset geometry/evaluation mismatch: " + dec["reason"] + ".**")
    md = [f"# Baseline reconstruction -- historical post-ceiling-fix baseline and the current geometry-only baseline ({_dt.date.today().isoformat()})\n",
          "## Verified historical post-ceiling-fix baseline\n",
          markdown_table(["Dataset", "Verified metric", "Value", "Provenance"], rows_hist),
          "\nThe remembered Occ3D value **30.74 % was wrong** (it appears nowhere; the pre-fix median is 30.89 %, the post-fix median 33.88 %). "
          "The remembered SemanticKITTI value 22.71 % is correct. Full trace: `historical_baseline_audit.md`.\n",
          "## Current clean baseline\n",
          markdown_table(["Dataset", "Current IoU (mean of 3 seeds)", "Historical IoU (median of 3)", "Comparability"], rows_cur),
          "\n95 % clip-block-bootstrap CIs of the current mean: " + "; ".join(f"{names[ds]} [{100*cis[ds]['lo']:.2f}, {100*cis[ds]['hi']:.2f}]" for ds in C.HISTORICAL_TARGETS if cis.get(ds)) + ".\n",
          "## Answers\n",
          f"1. **Verified SemanticKITTI post-ceiling-bug baseline:** {hsk['median_pct']:.2f} % SC IoU (seeds {sk_seeds}; mean {hsk['mean_pct']:.2f}), 163 anchors, past5.\n"
          f"2. **Verified Occ3D-nuScenes post-ceiling-bug baseline:** {hoc['median_pct']:.2f} % SC IoU (seeds {oc_seeds}; mean {hoc['mean_pct']:.2f}), 1182 anchors, past5.\n"
          "3. **Produced by:** `artifacts/gate8c1/frozen_manifest_sky.json` → `sky_seed{0,1,2}_last.pt` (configs/gate8c1_sky/seed*.yaml, replicate padding, plain-column open-sky targets at "
          "`lingbot_gate8c1_sky`, thresholds −0.125/−0.4375/−0.3125), evaluated by `core/run/eval_target.py --mode past5`; records `output/skyfix/eval/*.json` (2026-09-08 / 09-10).\n"
          "4. **Remembered numbers wrong?** Yes: Occ3D 30.74 % does not exist (30.89 pre-fix; 33.88 post-fix). SemanticKITTI 22.71 % is right. Also: two `artifacts/gate8c1/occany_*headtohead.json` files hold the rejected sky2 run, not the adopted model.\n"
          f"5. **Current clean KITTI-360 baseline:** 0006 full-region SC IoU {100*src_mean:.2f} % mean (seeds {src_seeds}); "
          f"reproduces Gate 13 M (0.1345) within the predeclared tolerance (mean |Δ| {source['reproduction']['mean_abs_diff']:.4f}). `source_results.md`.\n"
          f"6. **New untouched zero-shot results:** SemanticKITTI {per['semantickitti']['current_mean']:.2f} % (map-only {sk_map:.2f}), Occ3D-nuScenes {per['occ3d']['current_mean']:.2f} % (map-only {oc_map:.2f}); one evaluation each, nothing changed afterwards. `zero_shot_results.md`.\n"
          f"7. **Directly comparable?** No -- *partially comparable* on both targets: same anchors, masks, grids and metric, but the pipelines differ in every row of `historical_comparability.md` "
          "(geometry mode, scale rule, 32-ch semantic input vs 3-ch, residual lock, threshold rule, training targets, seed statistic).\n"
          f"8. **Which differences drive the gap?** Cannot be attributed to any single one without an ablation that does not exist. The like-for-like quantity is the map-only IoU "
          f"(no network): historical {hist_map['semantickitti']:.2f} / {hist_map['occ3d']:.2f} vs current {sk_map:.2f} / {oc_map:.2f} (SemanticKITTI / Occ3D), which isolates the geometry-mode + scale-rule change in the observed map. "
          f"Everything beyond that (semantics, 32 channels, lock, threshold, targets) acts on the completion only. SemanticKITTI/Occ3D ratio: current {ratio_cur:.2f} vs historical {ratio_hist:.2f}.\n"
          "9. **Which baseline for semantic-completion experiments?** The current one -- it is the only geometry-only, local-reset, fully frozen and hash-verified system; the historical model is not geometry-only.\n",
          "## Reading the result (observations, not attributions)\n",
          f"* SemanticKITTI moved **up** ({hsk['median_pct']:.2f} → {per['semantickitti']['current_mean']:.2f}) and Occ3D **down** ({hoc['median_pct']:.2f} → {per['occ3d']['current_mean']:.2f}); "
          f"the map-only rows barely moved ({hist_map['semantickitti']:.2f} → {sk_map:.2f}; {hist_map['occ3d']:.2f} → {oc_map:.2f}), so the observed geometry the two systems hand the network is of "
          "essentially equal quality on both targets and the divergence lives in the completion.\n"
          f"* Operating point: on Occ3D the current model is more precise and less complete than the historical one (P {100*zs['occ3d']['seed_statistics']['precision']['mean']:.1f} / R {100*zs['occ3d']['seed_statistics']['recall']['mean']:.1f} "
          f"vs {hoc['precision_pct_median_seed']:.1f} / {hoc['recall_pct_median_seed']:.1f}); on SemanticKITTI it gains recall at equal precision "
          f"(P {100*zs['semantickitti']['seed_statistics']['precision']['mean']:.1f} / R {100*zs['semantickitti']['seed_statistics']['recall']['mean']:.1f} vs {hsk['precision_pct_median_seed']:.1f} / {hsk['recall_pct_median_seed']:.1f}). "
          "Threshold 0 versus the historical negative thresholds pushes exactly this way on a benchmark that rewards precision (Occ3D, camera-visible structure) and is neutral-to-favourable "
          "where free space is scored (SemanticKITTI).\n"
          "* The direction of the asymmetry matches predeclared risk 1 (`predeclaration.md`): the neighbourhood-7 open-sky rule of the current targets was, in the historical single-seed ablation, "
          "a wash on SemanticKITTI and −5.2 points on Occ3D. It is the first documented candidate; it was **not** acted on.\n"
          f"* Seed spread on Occ3D is larger than before (sd {per['occ3d']['current_sd']:.2f} vs {hoc['sd_pct']:.2f}; seed 2 at {100*zs['occ3d']['per_seed']['2']['full']['iou']:.2f}); the bootstrap CI of the mean "
          f"[{100*cis['occ3d']['lo']:.2f}, {100*cis['occ3d']['hi']:.2f}] does not exclude the historical median, while the SemanticKITTI CI [{100*cis['semantickitti']['lo']:.2f}, {100*cis['semantickitti']['hi']:.2f}] lies entirely above it.\n"
          "* Both targets: the completion clears its own map-only reference by a wide margin (no collapse), and the source result reproduced -- the baseline is internally trustworthy.\n",
          "## Decision\n",
          f"**Decision {dec['decision']}** -- {dec['reason']}.\n",
          markdown_table(["target", "current mean", "map-only", "beats map-only", "historical median", "at/above historical (within sd)"],
                         [[names[ds], f"{v['current_mean']:.2f}", f"{v['map_only']:.2f}", v["beats_map_only"], f"{v['historical_median']:.2f}", v["at_or_above_historical"]] for ds, v in per.items()]),
          f"\nVerification: {verification['n_pass']} / {verification['n_total']} checks pass (`verification.md`).\n",
          "## Next action\n", next_action + "\n",
          "## Files\n",
          "`historical_baseline_audit.md`, `historical_baseline.json`, `historical_protocol.yaml`, `historical_provenance.csv`, `predeclaration.md`, `training_config.yaml`, "
          "`source_results.md`, `source_freeze.md`, `code_freeze.json`, `zero_shot_results.md`, `evaluation_semantickitti.yaml`, `evaluation_occ3d.yaml`, `historical_comparability.md`, "
          "`verification.md`, `tables/*.csv`, `figures/*.png`; reproduction commands in `tools/baseline_reconstruction/README.md`.\n"]
    out = os.path.join(C.ARTIFACT_DIR, "report.md"); open(out, "w").write("\n".join(md))
    json.dump(dec, open(os.path.join(C.ARTIFACT_DIR, "logs", "decision.json"), "w"), indent=1, default=float)
    return out


__all__ = ["decide", "write_report"]
