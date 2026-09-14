"""Render historical_comparability.md, tables/historical_vs_current.csv and tables/historical_comparison.csv."""
from __future__ import annotations

import json
import os
from typing import Dict

from tools.baseline_reconstruction import config as C
from tools.baseline_reconstruction.metrics.comparability import classify_comparability, property_table
from tools.baseline_reconstruction.reports.tables import markdown_table, pct, write_csv


def load_historical() -> Dict:
    return json.load(open(os.path.join(C.ARTIFACT_DIR, "historical_baseline.json")))


def write_comparability(zs: Dict[str, Dict], source_mean_iou: float) -> Dict:
    hist = load_historical(); props = property_table()
    anchors_match = {ds: zs[ds]["n_anchors"] == hist["results"][ds]["n_anchors"] for ds in zs}
    cls = classify_comparability(props, anchors_match)
    write_csv(os.path.join(C.ARTIFACT_DIR, "tables", "historical_vs_current.csv"),
              [{"property": p.name, "historical_post_ceiling_fix_baseline": p.historical, "current_geometry_only_baseline": p.current, "kind": p.kind, "differs": p.differs} for p in props])
    comp_rows = [{"dataset": "KITTI-360 0006 (source, full region)", "current_iou_pct": f"{100*source_mean_iou:.2f}", "historical_iou_pct": "n/a (historical source metric used a different target build and threshold)",
                  "comparability": "not comparable", "current_stat": "mean of 3 seeds", "historical_stat": ""}]
    for ds in C.HISTORICAL_TARGETS:
        h = hist["results"][ds]; z = zs.get(ds)
        comp_rows.append({"dataset": ds, "current_iou_pct": f"{100*z['seed_statistics']['iou']['mean']:.2f}" if z else "", "current_median_pct": f"{100*z['seed_statistics']['iou']['median']:.2f}" if z else "",
                          "historical_iou_pct": f"{h['median_pct']:.2f}", "historical_mean_pct": f"{h['mean_pct']:.2f}", "comparability": cls[ds]["class"],
                          "current_stat": "mean of 3 seeds (median alongside)", "historical_stat": "median of 3 seeds",
                          "current_map_only_pct": f"{100*z['map_only']['full']['iou']:.2f}" if z else "", "historical_map_only_pct": f"{100*[r for r in hist['all_records'] if r['variant']=='sky' and r['dataset']==ds and r['protocol']=='past5'][0]['mapper_only_iou']:.2f}"})
    write_csv(os.path.join(C.ARTIFACT_DIR, "tables", "historical_comparison.csv"), comp_rows)
    md = ["# Historical vs current baseline -- comparability audit\n",
          "The classification is mechanical (`metrics/comparability.py`): evaluation-side differences decide *not comparable*; "
          "pipeline-side differences decide *partially* vs *fully* comparable. Magnitudes never enter the classification.\n",
          "## Property table\n",
          markdown_table(["Property", "Historical post-ceiling-fix baseline", "Current geometry-only baseline", "differs"],
                         [[p.name, p.historical, p.current, "yes" if p.differs else "no"] for p in props]),
          "\n## Classification\n",
          markdown_table(["dataset", "class", "anchor set identical", "evaluation differences", "pipeline differences"],
                         [[ds, f"**{v['class']}**", v["anchor_set_identical"], ", ".join(v["evaluation_differences"]) or "none", ", ".join(v["pipeline_differences"])] for ds, v in cls.items()]),
          "\n## Numbers side by side\n",
          markdown_table(["dataset", "current mean (median)", "historical median (mean)", "current map-only", "historical map-only", "class"],
                         [[r["dataset"], f"{r['current_iou_pct']} ({r.get('current_median_pct','')})", f"{r['historical_iou_pct']} ({r.get('historical_mean_pct','')})",
                           r.get("current_map_only_pct", ""), r.get("historical_map_only_pct", ""), r["comparability"]] for r in comp_rows]),
          "\n**What the comparison can and cannot say.** Both targets are *partially comparable*: the anchors, masks, grids, cameras and the SC IoU "
          "computation are identical, so the numbers measure the same thing on the same voxels -- but the two systems differ in every pipeline row "
          "marked above. A difference in IoU therefore cannot be attributed to local-reset geometry alone; the closest fair control for the geometry "
          "question does not exist in the historical record (there is no long-stream geometry-only 3-channel model), and this experiment did not build "
          "one because its purpose is a clean forward baseline, not a retrospective ablation.\n",
          "The map-only rows are the one like-for-like quantity: they use no network, so they isolate what the geometry mode + scale rule changed in the "
          "observed map itself (historical map-only comes from the sky-model eval JSONs, `baselines.mapper_native`).\n"]
    out = os.path.join(C.ARTIFACT_DIR, "historical_comparability.md"); open(out, "w").write("\n".join(md))
    return {"path": out, "classes": cls, "rows": comp_rows, "historical": hist}


__all__ = ["load_historical", "write_comparability"]
