#!/usr/bin/env python
"""results_table.md + config.json for the region-semantic-consistency study.

    python tools/region_semantic_consistency/summarize.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools.region_semantic_consistency import common as RC                                # noqa: E402
from core.datasets import kitti360_splits as SRC                                          # noqa: E402

UNITS = ["voxel"] + list(RC.SCALES)
SHORT = {d: d[17:21] for d in list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]}; SHORT["train_pooled"] = "train pooled"


def res(an, d):
    return an["train_pooled"]["results"] if d == "train_pooled" else an["drives"][d]["results"]


def main() -> int:
    an = json.load(open(os.path.join(RC.ART, "analysis.json"))); an2 = json.load(open(os.path.join(RC.ART, "analysis_allocc.json")))
    ver = json.load(open(os.path.join(RC.ART, "verification.json")))
    val = SRC.VAL_DRIVE; L = ["# Region semantic consistency -- results\n"]
    L.append(f"Regions: cubic blocks of side s voxels (0.2 m) on the raw-LiDAR occupied grid, 26-connected components inside a block; "
             f"s = {RC.SCALES}. Groups: interleaved frame-disjoint A/B/C ({an['groups']}). Units carrying both A and B are scored; "
             f"R1 = within-anchor shuffle stratified by observation count; R2 = local shuffle within (z bin {RC.LOCAL_Z_BIN} vox, xy cell {RC.LOCAL_XY_CELL} vox).\n")
    for pop, A in (("eligible population (unobserved at t -> future-observed -> LiDAR-occupied)", an), ("all future-observed LiDAR-occupied voxels", an2)):
        L.append(f"## {pop}\n")
        for d in (val, "train_pooled"):
            R = res(A, d); nel = A["drives"][d]["n_eligible"] if d != "train_pooled" else A["train_pooled"]["n_eligible"]
            L.append(f"**{SHORT[d]}** -- {nel:,} voxels in population\n")
            L.append("| representation | A&B units | coverage A&B (% pop.) | coverage any (%) | top-1 A-B [CI] | cosine | JS | A-C | B-C | 3-way | R0 prior | R1 shuffle | R2 local | true-R1 | true-R2 | classes >=200 | dominant share | entropy | median occ. voxels | median extent (m) | mean frames A+B |")
            L.append("|---|" + "---|" * 20)
            for u in UNITS:
                x = R[u]; s = x["size"]; ci = x["AB_top1_ci95_blocks"]
                L.append(f"| {u} | {x['pairs']['AB']['n']:,} | {100*x['coverage_of_eligible']['AB']:.2f} | {100*x['coverage_of_eligible']['any_evidence']:.1f} | "
                         f"{x['pairs']['AB']['top1']:.3f} [{ci[0]:.2f},{ci[1]:.2f}] | {x['pairs']['AB']['cosine']:.3f} | {x['pairs']['AB']['js']:.3f} | "
                         f"{x['pairs']['AC']['top1']:.3f} | {x['pairs']['BC']['top1']:.3f} | {x['three_way']['consistent']:.3f} ({x['three_way']['n']}) | "
                         f"{x['R0_class_prior_top1']:.3f} | {x['controls']['R1']['top1']:.3f} | {x['controls']['R2']['top1']:.3f} | "
                         f"{x['pairs']['AB']['top1']-x['controls']['R1']['top1']:+.3f} | {x['pairs']['AB']['top1']-x['controls']['R2']['top1']:+.3f} | "
                         f"{x['n_classes_support_ge_200']} | {x['dominant_share']:.2f} ({x['dominant_class']}) | {x['entropy_nats']:.2f} | "
                         f"{s['median_occupied_voxels']:.0f} | {s['median_extent_m']:.1f} | {s['mean_frames_AB']:.2f} |")
            m = {u: R[u]["matched_voxels_with_own_AB"] for u in RC.SCALES}
            L.append(f"\nMatched (voxels with their own A&B evidence, n = {list(m.values())[0]['n']:,}): own top-1 {list(m.values())[0]['own_top1']:.3f}, region-inherited "
                     + ", ".join(f"{u} {v['region_top1']:.3f}" for u, v in m.items()) + f"; own cosine {list(m.values())[0]['own_cosine']:.3f}, region "
                     + ", ".join(f"{u} {v['region_cosine']:.3f}" for u, v in m.items()) + "\n")
    R = res(an, val)
    L.append("## Repeatability by number of contributing frames (A+B), drive 0006, eligible population\n")
    L.append("| representation | " + " | ".join(R["voxel"]["by_frames_AB"]) + " | " + " | ".join(k for k in R["xlarge"]["by_frames_AB"] if k not in R["voxel"]["by_frames_AB"]) + " |")
    names = list(R["xlarge"]["by_frames_AB"]); L.append("|---|" + "---|" * len(names))
    for u in UNITS:
        L.append(f"| {u} | " + " | ".join(f"{R[u]['by_frames_AB'][k]['top1']:.3f} (n={R[u]['by_frames_AB'][k]['n']})" if k in R[u]["by_frames_AB"] else "-" for k in names) + " |")
    L.append("\n## Repeatability by observations pooled (A+B), drive 0006\n")
    names = list(R["xlarge"]["by_observations_AB"]); L.append("| representation | " + " | ".join(names) + " |"); L.append("|---|" + "---|" * len(names))
    for u in UNITS:
        L.append(f"| {u} | " + " | ".join(f"{R[u]['by_observations_AB'][k]['top1']:.3f} (n={R[u]['by_observations_AB'][k]['n']})" if k in R[u]["by_observations_AB"] else "-" for k in names) + " |")
    L.append("\n## Per drive (A-B top-1, R1 in brackets)\n")
    ds = list(SRC.TRAIN_DRIVES) + [val]
    L.append("| representation | " + " | ".join(SHORT[d] for d in ds) + " |"); L.append("|---|" + "---|" * len(ds))
    for u in UNITS:
        L.append(f"| {u} | " + " | ".join(f"{res(an, d)[u]['pairs']['AB']['top1']:.3f} [{res(an, d)[u]['controls']['R1']['top1']:.3f}] n={res(an, d)[u]['pairs']['AB']['n']}" for d in ds) + " |")
    L.append("\n## Per class (A-B top-1; class = argmax of the A vector; classes with >= 200 units at voxel level)\n")
    for d in (val, "train_pooled"):
        R = res(an, d); names = sorted(R["voxel"]["per_class"], key=lambda n: -R["voxel"]["per_class"][n]["n"])
        L.append(f"**{SHORT[d]}**\n"); L.append("| class | " + " | ".join(UNITS) + " |"); L.append("|---|" + "---|" * len(UNITS))
        for n in names:
            L.append(f"| {n} | " + " | ".join((f"{R[u]['per_class'][n]['top1']:.3f} (n={R[u]['per_class'][n]['n']})" if n in R[u]["per_class"] else "<200") for u in UNITS) + " |")
        L.append("")
    L.append("## Observed semantic quality against benchmark labels\n")
    L.append("Not measured: the gate's protocol never reads a KITTI-360 human semantic label on the source drives (the SSCBench-KITTI-360 label "
             "was shown to contradict its own LiDAR in Gate 8C-0), and SemanticKITTI / Occ3D are untouched in this step by constraint.\n")
    L.append(f"Verification: {ver['n_pass']}/{ver['n_total']} checks pass.\n")
    open(os.path.join(RC.ART, "results_table.md"), "w").write("\n".join(L) + "\n")
    cfg = {"study": "region_semantic_consistency", "regions": {"definition": "cubic block partition of the raw-LiDAR occupied grid (gt_occ & gt_valid), 26-connected components within a block",
           "block_side_vox": RC.SCALES, "voxel_m": 0.2, "semantic_input": "none"},
           "populations": {"eligible": "~observed & fut_observed & gt_occ & gt_valid", "all_future_observed": "fut_observed & gt_occ & gt_valid"},
           "evidence": {"eligible": RC.EVIDENCE_DIR, "all_future_observed": RC.EVIDENCE_DIR + "_allocc",
                        "source": "tools/future_semantic_completion/target_consistency/build_evidence.py (per future frame, frozen mapper)"},
           "grouping": {"name": RC.GROUPING, "groups": an["groups"]},
           "aggregation": "p_group(unit) = sum_i w_i p_i / sum_i w_i over observations of that group at the unit's voxels (mapper evidence weights)",
           "controls": {"R0": "sum_k q_k r_k over the retained A/B class distributions", "R1": f"within-anchor permutation of B among A&B units, stratified by A+B observation count (edges {RC.SIZE_EDGES})",
                        "R2": f"permutation within (z bin {RC.LOCAL_Z_BIN} vox, xy cell {RC.LOCAL_XY_CELL} vox) strata"},
           "frozen": {"gate8c1_manifest": "artifacts/gate8c1/frozen_manifest_sky.json",
                      "mapper_code_digests": next(c["detail"] for c in ver["checks"] if c["name"] == "lingbot_and_trident_frozen")},
           "env": RC.environment()}
    RC.write_json(os.path.join(RC.ART, "config.json"), cfg)
    print("\n".join(L[:14]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
