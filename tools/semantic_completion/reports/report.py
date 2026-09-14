"""results_table.md and report.md from analysis.json, verification and figures."""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from tools.baseline_reconstruction.reports.tables import markdown_table
from tools.semantic_completion import config as C
from tools.semantic_completion.evaluation.regions import PRIMARY, SEMANTIC_REGIONS, SSC_REGIONS

NEXT = {
    "A": "Freeze the semantic propagation method (code_freeze.json) and run the untouched zero-shot semantic-completion evaluation on SemanticKITTI and Occ3D-nuScenes.",
    "B": "Keep geometry and the teacher interface fixed and change only the propagation supervision/representation: train the head with a distillation target that removes the teacher's shared feature direction (centred, text-projected class-logit distillation on the same evidence voxels), then re-run this source gate unchanged.",
    "C": "Replace plain cosine distillation with class-logit distillation in the frozen text space (KL between softmax(40·f·text) of head and teacher on the same evidence voxels, hidden-block masking unchanged) and re-run this source gate unchanged; do not return to future voxel targets.",
    "D": "Fix the semantic observation interface (the lattice/teacher association on directly observed voxels) before attempting propagation; do not modify occupancy geometry.",
    "INVALID": "Occupancy changed: find and remove the path by which the semantic module altered the frozen geometry before interpreting any semantic number.",
}


def _m(x) -> str:
    return "–" if x is None else f"{100 * x:.2f}"


def _mean(res, r, m, key="miou"):
    return float(np.mean([res["regions"][r][m][s][key] for s in res["seeds"]]))


def write_results_table(res: Dict) -> str:
    seeds = res["seeds"]; names = res["class_names"]; P = res["promotion"]
    rows = []
    for r in SEMANTIC_REGIONS + SSC_REGIONS:
        for m in ("B0", "B1"):
            v = [res["regions"][r][m][s] for s in seeds]
            rows.append([r + (" **(primary)**" if r == PRIMARY else ""), m, " / ".join(_m(x["miou"]) for x in v), _m(np.mean([x["miou"] for x in v])),
                         _m(np.mean([x["mean_class_accuracy"] for x in v])), _m(np.mean([x["overall_accuracy"] for x in v])),
                         f"{np.mean([x['gt_classes_predicted'] for x in v]):.1f}/{v[0]['n_gt_classes']}", f"{np.mean([x['n_classes_ge_1pct_of_predictions'] for x in v]):.1f}",
                         f"{names[v[0]['dominant_pred_class']]} {_m(np.mean([x['dominant_pred_fraction'] for x in v]))}", f"{v[0]['n_voxels']:,}"])
    t = res["teacher_observed"]
    rows.append(["oracle_observed", "teacher readout", _m(t["miou"]), _m(t["miou"]), _m(t["mean_class_accuracy"]), _m(t["overall_accuracy"]),
                 f"{t['gt_classes_predicted']}/{t['n_gt_classes']}", t["n_classes_ge_1pct_of_predictions"], f"{names[t['dominant_pred_class']]} {_m(t['dominant_pred_fraction'])}", f"{t['n_voxels']:,}"])
    dist = [[name, f"{res['distance']['B1'][seeds[0]][b]['n_voxels']:,}"] + [_m(np.mean([res["distance"][m][s][b]["miou"] or 0 for s in seeds])) for m in ("B0", "B1")] +
            [_m(np.mean([res["distance"][m][s][b]["overall_accuracy"] for s in seeds])) for m in ("B0", "B1")] for b, name in enumerate(C.DISTANCE_NAMES)]
    ho = res["heldout"]
    md = ["# Gate 15 results -- KITTI-360 0006 (590 anchors, final heads, seeds 0/1/2)\n",
          "mIoU in %, per seed and mean. Semantic-only regions score GT-occupied voxels; SSC regions score all labelled voxels with the frozen occupancy.\n",
          markdown_table(["region", "method", "mIoU seeds 0/1/2", "mIoU mean", "mean class acc", "overall acc", "GT classes predicted", "classes ≥1 % of preds", "dominant pred class", "voxels (seed 0)"], rows),
          "\n## Promotion rule (frozen)\n",
          markdown_table(["criterion", "value", "pass"],
                         [["P1 Δ mIoU completed_unobserved (B1−B0)", f"per seed {', '.join(_m(v) for v in P['P1']['diff_per_seed'].values())}; mean {_m(P['P1']['mean'])}; CI [{_m(P['P1']['ci95']['lo'])}, {_m(P['P1']['ci95']['hi'])}] (need ≥ +2.00, all > 0, CI lo > 0)", P["P1"]["pass"]],
                          ["P2 Δ SSC mIoU ssc_full", f"per seed {', '.join(_m(v) for v in P['P2']['diff_per_seed'].values())}; mean {_m(P['P2']['mean'])}; CI [{_m(P['P2']['ci95']['lo'])}, {_m(P['P2']['ci95']['hi'])}]", P["P2"]["pass"]],
                          ["P3 breadth / collapse", f"{P['P3']['classes_with_gain']} classes gain (need ≥ 5); B1 dominant {_m(P['P3']['b1_dominant_fraction'])} (limit {_m(P['P3']['dominant_limit'])}); B1 classes ≥ 1 % {P['P3']['b1_classes_ge_1pct']:.1f} (need ≥ 5)", P["P3"]["pass"]],
                          ["P4 occupancy identical", "; ".join(f"seed {s}: {v['identical']}" for s, v in res["occupancy"].items()), P["P4"]["pass"]],
                          ["D teacher poor on observed", f"accuracy {_m(t['overall_accuracy'])}, mIoU {_m(t['miou'])} (D if < 40 / < 15)", P["D_teacher_poor"]]]),
          f"\n**Decision {P['decision']}.**\n",
          "\n## Propagation distance (completed_unobserved, mean over seeds)\n",
          markdown_table(["distance to nearest evidence", "voxels (seed 0)", "B0 mIoU", "B1 mIoU", "B0 acc", "B1 acc"], dist),
          "\n## Label-free held-out evidence (hidden (8,8,4) blocks, cosine to the teacher at hidden evidence voxels)\n",
          markdown_table(["seed", "hidden voxels", "cos B1", "cos B0 (nearest visible)", "class agreement B1", "class agreement B0"],
                         [[s, f"{v['n_hidden']:,}", f"{v['cos_B1']:.4f}", f"{v['cos_B0_nearest_visible']:.4f}", _m(v["agree_B1"]), _m(v["agree_B0"])] for s, v in ho.items()]),
          "\n## Per-class IoU on completed_unobserved (mean over seeds)\n",
          markdown_table(["class", "B0", "B1", "B1 − B0", "GT share %", "B1 pred share %"],
                         [[names[c], _m(np.mean([res["regions"][PRIMARY]["B0"][s]["per_class_iou"][c] for s in seeds])), _m(np.mean([res["regions"][PRIMARY]["B1"][s]["per_class_iou"][c] for s in seeds])),
                           _m(np.mean([res["regions"][PRIMARY]["B1"][s]["per_class_iou"][c] - res["regions"][PRIMARY]["B0"][s]["per_class_iou"][c] for s in seeds])),
                           _m(res["regions"][PRIMARY]["B1"][seeds[0]]["gt_frequency"][c - 1]), _m(np.mean([res["regions"][PRIMARY]["B1"][s]["pred_frequency"][c - 1] for s in seeds]))]
                          for c in range(1, C.N_CLASSES + 1) if res["regions"][PRIMARY]["B0"][seeds[0]]["per_class_iou"][c] is not None])]
    out = os.path.join(C.ARTIFACT_DIR, "results_table.md"); open(out, "w").write("\n".join(md))
    return out


__all__ = ["NEXT", "write_results_table"]
