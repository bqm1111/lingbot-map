#!/usr/bin/env python
"""results_table.md + config.json for the target-consistency study, from the stage JSONs.

    python tools/future_semantic_completion/target_consistency/summarize.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from tools.future_semantic_completion.target_consistency import common as TC             # noqa: E402
from tools.future_semantic_completion import study as ST                                  # noqa: E402
from core.datasets import kitti360_splits as SRC                                          # noqa: E402

SHORT = {d: d[17:21] for d in list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]}; SHORT["train_pooled"] = "train pooled"


def main() -> int:
    an = json.load(open(os.path.join(TC.ART, "analysis.json")))
    nb = json.load(open(os.path.join(TC.ART, "neighbourhood_support.json")))
    cd = json.load(open(os.path.join(TC.ART, "causal_diagnostic.json")))
    ver = json.load(open(os.path.join(TC.ART, "verification.json")))
    builds = {b["drive"]: b for b in (json.load(open(p)) for p in sorted(
        os.path.join(TC.ART, "evidence_build", f) for f in os.listdir(os.path.join(TC.ART, "evidence_build"))))}
    val = SRC.VAL_DRIVE
    L = ["# Target consistency -- results\n"]
    L.append("## Evidence structure (frozen future mapper, one throwaway mapper per future frame)\n")
    L.append("| drive | anchors | eligible voxels | with any teacher evidence | existing target | 1 frame | 2 frames | >=3 frames | interleaved A&B | interleaved A&B&C | contiguous A&B | contiguous A&B&C |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in list(SRC.TRAIN_DRIVES) + [val, "train_pooled"]:
        S = an["support"][d]
        if d == "train_pooled":
            h = sum(np.asarray(an["support"][dd]["frames_per_voxel_hist"]) for dd in SRC.TRAIN_DRIVES)
        else:
            h = np.asarray(S["frames_per_voxel_hist"])
        hs = h.sum()
        L.append(f"| {SHORT[d]} | {S['n_anchors']} | {S['n_eligible']:,} | {S['n_any_evidence']:,} ({100*S['n_any_evidence']/S['n_eligible']:.1f} %) | "
                 f"{S['n_existing_target']:,} | {100*h[1]/hs:.1f} % | {100*h[2]/hs:.1f} % | {100*h[3:].sum()/hs:.1f} % | "
                 f"{S['interleaved_AB']:,} ({100*S['interleaved_AB']/S['n_eligible']:.2f} %) | {S['interleaved_ABC']:,} | "
                 f"{S['contiguous_AB']:,} ({100*S['contiguous_AB']/S['n_eligible']:.2f} %) | {S['contiguous_ABC']:,} |")
    L.append("\nRaw single-group agreement on voxels carrying both groups (no filtering), interleaved grouping:\n")
    L.append("| drive | A vs B (n) | A vs C (n) | B vs C (n) |")
    L.append("|---|---|---|---|")
    for d in list(SRC.TRAIN_DRIVES) + [val, "train_pooled"]:
        cells = []
        for k in ("AB", "AC", "BC"):
            v = an["pairwise_raw_agreement"][f"interleaved:{k}:{d}"]
            cells.append(f"{v['top1']:.3f} / cos {v['cosine']:.3f} ({v['n']:,})")
        L.append(f"| {SHORT[d]} | " + " | ".join(cells) + " |")
    for g in ("interleaved", "contiguous"):
        L.append(f"\n## Filters, {g} grouping (A = {an['groupings'][g]['A']}, B = {an['groupings'][g]['B']}, C = {an['groupings'][g]['C']})\n")
        for d in (val, "train_pooled"):
            L.append(f"**{SHORT[d]}** -- eligible {an['support'][d]['n_eligible']:,}\n")
            L.append("| filter | retained | % eligible | validated (has C) | top-1 repeat. [block CI] | cosine | JS | shuffled top-1 | class-prior top-1 | majority (C) | classes >=500 | top-1 class share | entropy |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for fname, r in an["results"][g].items():
                rr = r[d]
                if rr["n_validated"] < 20:
                    L.append(f"| {fname} | {rr['n_retained']:,} | {100*rr['coverage_of_eligible']:.2f} | {rr['n_validated']} | (insufficient C support) | | | | | | {rr['n_classes_support_ge_500']} | {rr['top1_class_share'] or 0:.3f} | {rr['entropy_nats'] or 0:.2f} |")
                    continue
                ci = rr["top1_ci95_blocks"]
                L.append(f"| {fname} | {rr['n_retained']:,} | {100*rr['coverage_of_eligible']:.2f} | {rr['n_validated']:,} | "
                         f"{rr['top1_repeatability']:.3f} [{ci[0]:.2f}, {ci[1]:.2f}] | {rr['cosine']:.3f} | {rr['js_divergence']:.3f} | "
                         f"{rr['shuffled_top1']:.3f} | {rr['class_prior_top1']:.3f} | {rr['majority_rate_C']:.3f} | "
                         f"{rr['n_classes_support_ge_500']} | {rr['top1_class_share']:.3f} | {rr['entropy_nats']:.2f} |")
            L.append("")
    R = an["results"]["interleaved"]
    L.append("## Per drive, interleaved grouping (top-1 repeatability vs C; shuffled reference in brackets)\n")
    fs = ["any_AB", "F0_existing_rule", "F1_agree", "F2_tau0.60", "F3_gam0.90", "F4_tau0.60_gam0.90"]
    L.append("| filter | " + " | ".join(SHORT[d] for d in list(SRC.TRAIN_DRIVES) + [val]) + " |")
    L.append("|---|" + "---|" * 4)
    for f in fs:
        L.append(f"| {f} | " + " | ".join(f"{R[f][d]['top1_repeatability']:.3f} [{R[f][d]['shuffled_top1']:.3f}] n={R[f][d]['n_validated']}"
                                          for d in list(SRC.TRAIN_DRIVES) + [val]) + " |")
    L.append("\n## Class distribution of the retained set, drive 0006 (interleaved)\n")
    names = sorted(R["any_AB"][val]["class_distribution"], key=lambda n: -R["any_AB"][val]["class_distribution"][n])[:10]
    L.append("| filter | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    for f in fs:
        L.append(f"| {f} | " + " | ".join(f"{100*R[f][val]['class_distribution'].get(n, 0):.1f}" for n in names) + " |")
    L.append("\n## Per-class repeatability, drive 0006 (classes with >= 500 validated voxels)\n")
    for f in ("any_AB", "F0_existing_rule", "F1_agree"):
        pc = R[f][val]["per_class_repeatability"]
        L.append(f"- {f}: " + ", ".join(f"{n} {v['top1']:.2f} (n={v['n']})" for n, v in sorted(pc.items(), key=lambda x: -x[1]['n'])))
    L.append("\n## Confidence vs repeatability, drive 0006 (interleaved)\n")
    for f in ("any_AB", "F1_agree"):
        bc = R[f][val]["by_confidence"]
        L.append(f"- {f}: " + ", ".join(f"{k} {v['top1']:.2f} (n={v['n']})" for k, v in bc.items()))
    L.append("\n## Where does other-frame evidence land? (nearest evidence from a different future frame, per single-frame observation)\n")
    L.append("| drive | single-frame share of evidence pairs | " + " | ".join(nb["names"]) + " |")
    L.append("|---|---|" + "---|" * len(nb["names"]))
    for d, v in nb["drives"].items():
        L.append(f"| {SHORT[d]} | {v['single_frame_fraction']:.3f} | " + " | ".join(
            f"{100*x['fraction']:.1f} % / {x['top1_agreement']:.2f}" if x["top1_agreement"] is not None else f"{100*x['fraction']:.1f} %"
            for x in v["nearest_other_frame"].values()) + " |")
    L.append("\n(cells: share of observations whose nearest other-frame evidence is at that distance / top-1 agreement with it)\n")
    L.append("## Causal-map diagnostic, drive 0006 (nearest causal voxel with teacher evidence vs the A+B target; diagnostic only)\n")
    L.append("| filter | n | top-1 | cosine | " + " | ".join(f"d {k}" for k in cd["filters"]["F1_agree"]["by_distance_vox"]) + " |")
    L.append("|---|---|---|---|" + "---|" * len(cd["filters"]["F1_agree"]["by_distance_vox"]))
    for f, v in cd["filters"].items():
        L.append(f"| {f} | {v['n']:,} | {v['top1_agreement']:.3f} | {v['cosine']:.3f} | " + " | ".join(
            f"{x['top1']:.2f} ({100*x['fraction']:.0f} %)" for x in v["by_distance_vox"].values()) + " |")
    L.append("\nVerification: " + f"{ver['n_pass']}/{ver['n_total']} checks pass.\n")
    open(os.path.join(TC.ART, "results_table.md"), "w").write("\n".join(L) + "\n")
    cfg = {"study": "target_consistency", "eligible_rule": "~observed & fut_observed & gt_occ & gt_valid (no teacher-present requirement)",
           "future_horizon_frames": SRC.FUTURE_FRAMES, "groupings": an["groupings"], "primary_grouping": "interleaved",
           "grouping_rationale": "contiguous early/middle/late leaves 0.1-0.4 % of eligible voxels with A&B support and ~0 with A&B&C; "
                                 "interleaved offsets keep every frame in exactly one group and consecutive frames in different groups",
           "candidate_target": "p_AB = (s_A + s_B) / (w_A + w_B), the mapper's evidence-weighted fusion of groups A and B",
           "validation": "argmax(p_AB) == argmax(p_C), cosine(p_AB, p_C), JS(p_AB, p_C) on retained voxels with w_C > 0",
           "filters": an["filters"], "tau_grid": TC.TAUS, "gamma_grid": TC.GAMMAS, "sem_w_min": TC.SEM_W_MIN,
           "min_class_support": an["min_class_support"], "threshold_selection_drive": val,
           "evidence_root": TC.EVIDENCE_DIR, "evidence_builds": {d: {k: b[k] for k in ("n_written", "max_abs_dp", "max_abs_dw")} for d, b in builds.items()},
           "frozen": {"gate8c1_manifest": os.path.relpath(ST.FROZEN_MANIFEST, ST.REPO_ROOT),
                      "trident_cache": "/media/SSD1/MINH_DATASETS/lingbot_gate6/semantics/kitti360 (1777 files, newest 2026-09-02)",
                      "mapper_code_digests": next(c["detail"] for c in ver["checks"] if c["name"] == "frozen_teacher_and_mapper")},
           "previous_reference": an["previous_reference"], "env": ST.environment()}
    ST.write_json(os.path.join(TC.ART, "config.json"), cfg)
    print("\n".join(L[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
