"""Source report: results CSV, figures and ``source_report.md`` from ``source_analysis.json`` (no metric is recomputed)."""
from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tools.semantic_field.config.schema import ARTIFACT_DIR, Paths  # noqa: E402
from tools.semantic_field.evaluation.analysis import PRIMARY, REPORT_REGIONS  # noqa: E402

COL = {"B0": "#8c8c8c", "B3": "#1f77b4", "B4": "#d62728"}


def results_csv(a: Dict) -> str:
    p = os.path.join(ARTIFACT_DIR, "source_results.csv")
    with open(p, "w") as fh:
        fh.write("region,voxels,B0_miou,B3_miou,B3_s0,B3_s1,B3_s2,B4_miou,B4_s0,B4_s1,B4_s2,B4_minus_B0,ci_lo,ci_hi,B4_minus_B3,ci3_lo,ci3_hi,B0_acc,B3_acc,B4_acc\n")
        for r, v in a["regions"].items():
            d0, d3 = v["B4_minus_B0"], v["B4_minus_B3"]
            fh.write(",".join(str(x) for x in [r, v["voxels"], f"{v['B0']:.3f}", f"{v['B3']:.3f}", *[f"{x:.3f}" for x in v["B3_per_seed"]], f"{v['B4']:.3f}",
                                                *[f"{x:.3f}" for x in v["B4_per_seed"]], f"{d0['delta']:.3f}", f"{d0['ci95'][0]:.3f}", f"{d0['ci95'][1]:.3f}",
                                                f"{d3['delta']:.3f}", f"{d3['ci95'][0]:.3f}", f"{d3['ci95'][1]:.3f}", f"{v['accuracy']['B0']:.3f}",
                                                f"{v['accuracy']['B3']:.3f}", f"{v['accuracy']['B4']:.3f}"]) + "\n")
    return p


def figures(a: Dict) -> List[str]:
    F = Paths().figures; os.makedirs(F, exist_ok=True); out = []
    def save(fig, name):
        p = os.path.join(F, name); fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); out.append(p)
    regs = [PRIMARY, "completed_geo_unobserved", "never_observed", "oracle_observed", "completed_observed", "ssc_full"]
    fig, ax = plt.subplots(figsize=(13, 4.5)); x = np.arange(len(regs))
    for i, m in enumerate(("B0", "B3", "B4")):
        ax.bar(x + (i - 1) * 0.27, [a["regions"][r][m] for r in regs], 0.27, color=COL[m], label=m)
        if m != "B0":
            for j, r in enumerate(regs):
                ax.scatter([j + (i - 1) * 0.27] * 3, a["regions"][r][f"{m}_per_seed"], s=8, color="k", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([REPORT_REGIONS[r] for r in regs], rotation=15, fontsize=8); ax.set_ylabel("mIoU %"); ax.legend()
    ax.set_title("KITTI-360 0006: semantic completion by region (dots = seeds)"); save(fig, "1_source_regions.png")
    pc = a["per_class_primary"]; names = list(pc)
    fig, ax = plt.subplots(figsize=(15, 4.5)); x = np.arange(len(names))
    for i, m in enumerate(("B0", "B3", "B4")):
        ax.bar(x + (i - 1) * 0.27, [pc[n][m] for n in names], 0.27, color=COL[m], label=m)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_ylabel("IoU %"); ax.legend(); ax.set_title("per-class IoU, completed_unobserved")
    save(fig, "2_source_per_class.png")
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, r in zip(axs[:2], ("primary_candidate_supported", "primary_candidate_unsupported")):
        v = a["regions"][r]; ax.bar(["B0", "B3", "B4"], [v["B0"], v["B3"], v["B4"]], color=[COL[m] for m in ("B0", "B3", "B4")])
        ax.set_title(f"{REPORT_REGIONS[r]}\n{v['voxels']:,} voxels", fontsize=9); ax.set_ylabel("mIoU %")
    md = a["masked_diagnostic"]; ks = list(md)
    axs[2].bar(range(len(ks)), [md[k]["teacher_class_agreement"] for k in ks], color=[COL["B0"], COL["B3"], COL["B4"]])
    axs[2].bar(range(len(ks)), [md[k]["gt_accuracy"] for k in ks], fill=False, edgecolor="k", hatch="//")
    axs[2].set_xticks(range(len(ks))); axs[2].set_xticklabels(["nearest", "B3", "B4"]); axs[2].set_ylabel("% (solid: teacher agreement, hatched: GT acc)")
    axs[2].set_title("masked-evidence reconstruction (hidden cuboids)", fontsize=9)
    save(fig, "3_candidate_support_and_masked.png")
    d = a["distance"]; bins = list(d)
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in ("B0", "B3", "B4"):
        ax.plot(bins, [d[b][m] for b in bins], "o-", color=COL[m], label=m)
    ax.set_ylabel("mIoU %"); ax.set_xlabel("distance to nearest evidence"); ax.legend(); ax.set_title("completed_unobserved by distance to evidence")
    save(fig, "4_source_distance.png")
    return out


def report_md(a: Dict) -> str:
    rows = ["| region | voxels | B0 | B3 (seeds) | B4 (seeds) | B4 − B0 [CI] | B4 − B3 [CI] |", "|---|---:|---:|---:|---:|---|---|"]
    for r, v in a["regions"].items():
        rows.append(f"| {v['label']} | {v['voxels']:,} | {v['B0']:.2f} | {v['B3']:.2f} ({' / '.join(f'{x:.2f}' for x in v['B3_per_seed'])}) | "
                    f"{v['B4']:.2f} ({' / '.join(f'{x:.2f}' for x in v['B4_per_seed'])}) | {v['B4_minus_B0']['delta']:+.2f} [{v['B4_minus_B0']['ci95'][0]:+.2f}, "
                    f"{v['B4_minus_B0']['ci95'][1]:+.2f}] | {v['B4_minus_B3']['delta']:+.2f} [{v['B4_minus_B3']['ci95'][0]:+.2f}, {v['B4_minus_B3']['ci95'][1]:+.2f}] |")
    pc = ["| class | GT share % | B0 | B3 | B4 | B4 pred share % |", "|---|---:|---:|---:|---:|---:|"] + \
         [f"| {n} | {v['gt_share']:.2f} | {v['B0']:.2f} | {v['B3']:.2f} | {v['B4']:.2f} | {v['B4_pred_share']:.2f} |" for n, v in a["per_class_primary"].items()]
    dist = ["| distance | voxels | B0 | B3 | B4 |", "|---|---:|---:|---:|---:|"] + [f"| {b} | {v['voxels']:,} | {v['B0']:.2f} | {v['B3']:.2f} | {v['B4']:.2f} |" for b, v in a["distance"].items()]
    md = ["| predictor | hidden voxels | teacher-class agreement % | GT accuracy % | GT mIoU % |", "|---|---:|---:|---:|---:|"] + \
         [f"| {k} | {v['hidden_voxels']:,} | {v['teacher_class_agreement']:.2f} | {v['gt_accuracy']:.2f} | {v['gt_miou']:.2f} |" for k, v in a["masked_diagnostic"].items()]
    S = a["criteria"]
    crit = ["| criterion | value | pass |", "|---|---|---|",
            f"| S1 B4 − B0 ≥ +2.0 | {S['S1']['gain']:+.2f} | {S['S1']['pass']} |",
            f"| S2 all seeds > 0, CI lo > 0 | seeds {[round(g, 2) for g in S['S2']['seed_gains']]}, CI [{S['S2']['ci95'][0]:+.2f}, {S['S2']['ci95'][1]:+.2f}] | {S['S2']['pass']} |",
            f"| S3 B4 − B3 ≥ +1.0, CI lo > 0 | {S['S3']['gain_vs_B3']:+.2f}, CI [{S['S3']['ci95'][0]:+.2f}, {S['S3']['ci95'][1]:+.2f}] | {S['S3']['pass']} |",
            f"| S4 ≥ 8 classes improve | {S['S4']['classes_improved_vs_B0']} | {S['S4']['pass']} |",
            f"| S5 no collapse | dominant {S['S5']['dominant_fraction']:.3f} (≤ {S['S5']['limit']:.3f}), classes ≥1 % {S['S5']['classes_ge_1pct']:.1f}, entropy {S['S5']['entropy_pred']:.2f} vs GT {S['S5']['entropy_gt']:.2f} | {S['S5']['pass']} |",
            f"| S6 occupancy identical | | {S['S6']['pass']} |"]
    return "\n".join(["# Source results tables (KITTI-360 0006, 590 anchors; mIoU %)\n", "## Regions\n", *rows, "\n## Per class (completed_unobserved)\n", *pc,
                      "\n## Distance to nearest evidence (completed_unobserved)\n", *dist, "\n## Masked-evidence reconstruction\n", *md,
                      "\n## Promotion rule\n", *crit, f"\n**{a['decision']}**\n"])


def write_all() -> Dict:
    a = json.load(open(os.path.join(ARTIFACT_DIR, "source_analysis.json")))
    open(os.path.join(ARTIFACT_DIR, "source_tables.md"), "w").write(report_md(a))
    return {"csv": results_csv(a), "figures": figures(a)}


__all__ = ["results_csv", "figures", "report_md", "write_all"]
