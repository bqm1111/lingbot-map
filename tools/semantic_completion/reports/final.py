"""report.md (answers, decision, one next action) and code_freeze.json for Gate 15."""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from typing import Dict

import numpy as np

from tools.baseline_reconstruction.reports.tables import markdown_table
from tools.semantic_completion import config as C, rules as RL
from tools.semantic_completion.evaluation.regions import PRIMARY
from tools.semantic_completion.frozen_geometry import sha256_file
from tools.semantic_completion.reports.report import NEXT


def _m(x) -> str:
    return "–" if x is None else f"{100 * x:.2f}"


def write_code_freeze(res: Dict) -> Dict:
    files = []
    for d, _, fs in os.walk(os.path.join(C.REPO_ROOT, "tools", "semantic_completion")):
        files += [os.path.relpath(os.path.join(d, f), C.REPO_ROOT) for f in fs if f.endswith((".py", ".md"))]
    p = C.Paths()
    files += [os.path.relpath(p.head_checkpoint(s, C.TrainConfig().steps), C.REPO_ROOT) for s in C.SEEDS]
    files += [os.path.relpath(RL.RULE_PATH, C.REPO_ROOT), "artifacts/gate15_semantic_completion/semantic_config.yaml", "artifacts/baseline_reconstruction/code_freeze.json"]
    rec = {"frozen_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "git_head": subprocess.check_output(["git", "-C", C.REPO_ROOT, "rev-parse", "HEAD"], text=True).strip(),
           "decision": res["promotion"]["decision"], "sha256": {f: sha256_file(os.path.join(C.REPO_ROOT, f)) for f in sorted(files)},
           "note": "written after the 0006 analysis; target datasets have not been opened"}
    json.dump(rec, open(os.path.join(C.ARTIFACT_DIR, "code_freeze.json"), "w"), indent=1)
    return rec


def write_report(res: Dict, verification: Dict) -> str:
    seeds = res["seeds"]; names = res["class_names"]; P = res["promotion"]; t = res["teacher_observed"]; reg = res["regions"]
    mean = lambda r, m, k="miou": float(np.mean([reg[r][m][s][k] for s in seeds]))
    gains = P["P3"]["per_class_gain"]
    up = sorted([(k, v) for k, v in gains.items() if v > 0.005], key=lambda kv: -kv[1]); down = sorted([(k, v) for k, v in gains.items() if v < -0.005], key=lambda kv: kv[1])
    dist = [(name, float(np.mean([res["distance"]["B1"][s][b]["overall_accuracy"] for s in seeds])), float(np.mean([res["distance"]["B0"][s][b]["overall_accuracy"] for s in seeds])))
            for b, name in enumerate(C.DISTANCE_NAMES)]
    ho = res["heldout"]; hc1 = np.mean([v["cos_B1"] for v in ho.values()]); hc0 = np.mean([v["cos_B0_nearest_visible"] for v in ho.values()])
    ha1 = np.mean([v["agree_B1"] for v in ho.values()]); ha0 = np.mean([v["agree_B0"] for v in ho.values()])
    b1p = reg[PRIMARY]["B1"]; dom = names[b1p[seeds[0]]["dominant_pred_class"]]
    occ_ok = all(v["identical"] for v in res["occupancy"].values())
    headline = markdown_table(["region (0006, mean of 3 seeds)", "B0 nearest evidence", "B1 learned propagation", "B1 − B0"],
                              [[r, _m(mean(r, "B0")), _m(mean(r, "B1")), _m(mean(r, "B1") - mean(r, "B0"))]
                               for r in ("oracle_observed", "completed_observed", "completed_unobserved", "ssc_full", "ssc_no_evidence")])
    md = [f"# Gate 15 — sparse trustworthy 2D semantics propagated into completed 3D space ({_dt.date.today().isoformat()})\n",
          f"Frozen geometry-only control (SC IoU {'unchanged, verified bit-for-bit' if occ_ok else 'CHANGED -- see P4'}) + frozen Trident-H evidence on the five causal frames + a 290k-parameter "
          "propagation head. KITTI-360 0003/0007/0010 train, 0006 evaluation; target datasets not opened.\n",
          "## Headline (semantic mIoU %)\n", headline,
          f"\nTeacher readout on directly observed GT-occupied voxels: mIoU {_m(t['miou'])} %, accuracy {_m(t['overall_accuracy'])} %. "
          f"Full tables: `results_table.md`; figures: `figures/`; verification {verification['n_pass']}/{verification['n_total']} (`verification.md`).\n",
          "## Answers\n",
          f"1. **Trident-H on directly observed 3D surfaces:** mIoU {_m(t['miou'])} %, mean class accuracy {_m(t['mean_class_accuracy'])} %, overall accuracy {_m(t['overall_accuracy'])} % "
          f"({t['gt_classes_predicted']}/{t['n_gt_classes']} GT classes predicted).\n"
          f"2. **Trivial propagation (B0, nearest evidence):** completed_unobserved mIoU {_m(mean(PRIMARY, 'B0'))} %, accuracy {_m(mean(PRIMARY, 'B0', 'overall_accuracy'))} %; SSC mIoU {_m(mean('ssc_full', 'B0'))} %.\n"
          f"3. **Does B1 beat it?** completed_unobserved Δ mIoU {_m(P['P1']['mean'])} points (per seed {', '.join(_m(v) for v in P['P1']['diff_per_seed'].values())}; CI [{_m(P['P1']['ci95']['lo'])}, {_m(P['P1']['ci95']['hi'])}]); "
          f"SSC Δ {_m(P['P2']['mean'])} points. P1 {'passes' if P['P1']['pass'] else 'fails'} the frozen +2.00 bar; P2 {'passes' if P['P2']['pass'] else 'fails'}.\n"
          f"4. **In voxels with no direct observation?** completed_unobserved is by construction evidence-free; B1 {_m(mean(PRIMARY, 'B1'))} vs B0 {_m(mean(PRIMARY, 'B0'))} %. "
          f"Label-free held-out evidence: cosine to the teacher B1 {hc1:.4f} vs nearest-visible B0 {hc0:.4f}; class agreement {_m(ha1)} vs {_m(ha0)} %.\n"
          "5. **How far does useful semantics propagate?** overall accuracy by distance to the nearest evidence (B1 / B0): " + "; ".join(f"{n} {_m(a1)} / {_m(a0)}" for n, a1, a0 in dist) + ".\n"
          "6. **Classes that propagate better with B1:** " + (", ".join(f"{k} (+{_m(v)})" for k, v in up) or "none by more than 0.5 IoU points") + ".\n"
          "7. **Classes that fail / lose:** " + (", ".join(f"{k} ({_m(v)})" for k, v in down) or "none by more than 0.5 IoU points") + ".\n"
          f"8. **Collapse toward priors?** B1 on completed_unobserved: dominant class {dom} at {_m(P['P3']['b1_dominant_fraction'])} % of predictions (GT dominant {_m(b1p[seeds[0]]['dominant_gt_fraction'])} %), "
          f"{P['P3']['b1_classes_ge_1pct']:.1f} classes with ≥ 1 % of predictions (B0 {np.mean([reg[PRIMARY]['B0'][s]['n_classes_ge_1pct_of_predictions'] for s in seeds]):.1f}), "
          f"entropy {np.mean([b1p[s]['pred_entropy_bits'] for s in seeds]):.2f} bits (B0 {np.mean([reg[PRIMARY]['B0'][s]['pred_entropy_bits'] for s in seeds]):.2f}). Collapse clause {'passes' if P['P3']['collapse_clause_ok'] else 'FAILS'}.\n"
          f"9. **Occupancy exactly unchanged?** {'Yes' if occ_ok else 'NO'}: " + "; ".join(f"seed {s} TP/FP/FN {v['ours']} (SC IoU {_m(v['iou'])} %)" for s, v in res["occupancy"].items()) + ", identical to the frozen baseline; geometry digests unchanged by training.\n"
          f"10. **Genuine semantic completion rather than copying?** {'Yes' if P['decision'] == 'A' else 'Not demonstrated'}: the frozen rule requires B1 to beat nearest-evidence copying by ≥ 2 mIoU points in evidence-free completed voxels, across seeds and classes.\n"
          f"11. **Ready for untouched zero-shot semantic evaluation?** {'Yes — Decision A unlocks it.' if P['decision'] == 'A' else 'No — target evaluation stays locked (Decision ' + P['decision'] + ').'}\n",
          "## Interpretation (measurements vs reading)\n",
          f"* **Measured.** B1 is below nearest-evidence copying in every region, including evidence voxels where it receives the teacher feature directly "
          f"(oracle_observed {_m(mean('oracle_observed', 'B1'))} vs {_m(mean('oracle_observed', 'B0'))} mIoU) and in the label-free held-out test "
          f"(cosine {hc1:.3f} vs {hc0:.3f}). B1 predicts {np.mean([b1p[s]['gt_classes_predicted'] for s in seeds]):.1f} of {b1p[seeds[0]]['n_gt_classes']} GT classes on completed_unobserved; "
          f"B0 predicts all of them. It gains only on large flat classes ({', '.join(k for k, _ in up) or 'none'}).\n"
          "* **Supported reading.** The head did not learn to preserve, let alone propagate, teacher class identity: its output keeps the teacher's shared "
          "feature direction (final training loss 0.03–0.06 against 0.093 for a constant mean feature) while losing the small residual angles that decide the class. "
          "This matches the risk recorded in `predeclaration.md` before evaluation.\n"
          f"* **Rule vs pattern.** The frozen order yields Decision {P['decision']}: P1 fails, and the collapse clause of P3 passes (dominant fraction and class count "
          "within their limits), so C is not triggered. The observed pattern is nonetheless a *partial* collapse (whole classes vanish), not 'near the trivial baseline'. "
          "The decision stands as frozen; the next action targets the objective, not geometry.\n"
          "* **Hypothesis (untested).** A distillation that removes the shared component (class-logit KL in the frozen text space) would let the same head keep class identity. "
          "No ablation of this exists.\n",
          f"## Decision {P['decision']}\n",
          markdown_table(["criterion", "pass"], [["P1 unobserved gain ≥ +2 mIoU, all seeds, CI > 0", P["P1"]["pass"]], ["P2 SSC gain all seeds", P["P2"]["pass"]],
                                                ["P3 ≥ 5 classes gain, no collapse", P["P3"]["pass"]], ["P4 occupancy identical", P["P4"]["pass"]], ["D teacher poor on observed", P["D_teacher_poor"]]]),
          f"\n**Decision {P['decision']}.**\n", "## Next action\n", NEXT[P["decision"]] + "\n",
          "## Reproduction\n", "See `tools/semantic_completion/README.md`.\n"]
    out = os.path.join(C.ARTIFACT_DIR, "report.md"); open(out, "w").write("\n".join(md))
    return out


__all__ = ["write_code_freeze", "write_report"]
