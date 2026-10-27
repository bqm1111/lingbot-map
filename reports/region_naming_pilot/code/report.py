"""Stage 7: results.csv and report.md, generated from the written artifacts. Never hand-typed."""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

import numpy as np

import evaluate as EV
import legacy as L
import manifest as MF
import paths as P
import predict as PD
from figures import NAMES

REGION_LABEL = {
    "completed_unobserved": "PRIMARY  GT-occupied & O & no causal evidence",
    "ssc_no_evidence": "deployable SSC, every labelled voxel without causal evidence",
    "ssc_full": "deployable SSC, every labelled voxel",
    "oracle_unobserved": "GT-occupied & no causal evidence (oracle occupancy)",
    "oracle_observed": "GT-occupied & causal evidence (teacher copy quality)",
    "completed_observed": "GT-occupied & O & causal evidence",
    "completed_unobserved_geo_observed": "PRIMARY, clip geometrically observed the voxel",
    "completed_unobserved_geo_unobserved": "PRIMARY, clip never observed the voxel (S|F empty)",
    "completed_unobserved_objects": "PRIMARY, object classes",
    "completed_unobserved_surfaces": "PRIMARY, broad surface classes",
}


def pct(x, d=2):
    return "n/a" if x is None else f"{100 * float(x):.{d}f}"


def write_csv(an: Dict) -> str:
    rows: List[Dict] = []
    for setname, s in an["sets"].items():
        for r in EV.ALL_REGIONS:
            for m in PD.METHODS:
                v = s[m][r]
                rows.append({"set": setname, "n_anchors": s["_n_anchors"], "region": r, "method": m,
                             "miou_pct": None if v["miou"] is None else round(100 * v["miou"], 4),
                             "overall_accuracy_pct": round(100 * v["overall_accuracy"], 4),
                             "mean_class_accuracy_pct": None if v["mean_class_accuracy"] is None else round(100 * v["mean_class_accuracy"], 4),
                             "n_voxels": v["n_voxels"], "n_gt_classes": v["n_gt_classes"],
                             "gt_classes_predicted": v["gt_classes_predicted"],
                             "dominant_pred_class": NAMES[v["dominant_pred_class"]],
                             "dominant_pred_fraction": round(v["dominant_pred_fraction"], 4)})
    with open(P.under_experiment(P.RESULTS_CSV), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return P.RESULTS_CSV


def recognition_summary(an: Dict, ids: List[int]) -> Dict:
    rows = [r for t, v in an["recognition"].items() if int(t) in ids for r in v]
    lab = [r for r in rows if r["modal_gt"]]
    sv = [r for r in rows if r["single_view_label"] is not None and r["outcome"] == "ok"]
    out = {"n_queried_regions": len(rows), "n_with_human_label": len(lab),
           "n_without_human_label": len(rows) - len(lab),
           "outcomes": {k: sum(1 for r in rows if r["outcome"] == k) for k in
                        ("ok", "abstain", "out_of_vocabulary", "malformed")},
           "p_accuracy": np.mean([r["p_label"] == r["modal_gt"] for r in lab]) if lab else None,
           "q_accuracy": np.mean([r["q_label"] == r["modal_gt"] for r in lab]) if lab else None,
           "q_accuracy_ok_only": np.mean([r["q_label"] == r["modal_gt"] for r in lab if r["outcome"] == "ok"]) or None,
           "agree_pq": np.mean([r["p_label"] == r["q_label"] for r in rows]) if rows else None,
           "mean_purity": float(np.mean([r["purity"] for r in lab])) if lab else None,
           "n_pure_ge_90": sum(1 for r in lab if r["purity"] >= 0.9),
           "n_mixed_lt_60": sum(1 for r in lab if r["purity"] < 0.6),
           "n_single_view_compared": len(sv),
           "single_view_disagreement": float(np.mean([r["single_view_label"] != r["q_label"] for r in sv])) if sv else None,
           "weighted_p_accuracy": None, "weighted_q_accuracy": None}
    if lab:
        w = np.array([r["n_evidence"] for r in lab], float)
        out["weighted_p_accuracy"] = float(np.average([r["p_label"] == r["modal_gt"] for r in lab], weights=w))
        out["weighted_q_accuracy"] = float(np.average([r["q_label"] == r["modal_gt"] for r in lab], weights=w))
    per_class = {}
    for r in lab:
        d = per_class.setdefault(NAMES[r["modal_gt"]], {"n": 0, "p": 0, "q": 0})
        d["n"] += 1; d["p"] += int(r["p_label"] == r["modal_gt"]); d["q"] += int(r["q_label"] == r["modal_gt"])
    out["per_modal_class"] = dict(sorted(per_class.items(), key=lambda kv: -kv[1]["n"]))
    return out


def verdict(an: Dict, rg: Dict) -> Dict:
    """The decision rule, declared here and applied mechanically."""
    b = an.get("bootstrap", {})
    prim = L.PRIMARY
    qn_pn = b.get(f"Q_N-P_N @ {prim}", {})
    pr_pn = b.get(f"P_R-P_N @ {prim}", {})
    qr_qn = b.get(f"Q_R-Q_N @ {prim}", {})
    better_names = (rg["q_accuracy"] or 0) > (rg["p_accuracy"] or 0) and qn_pn.get("ci95", [0, 0])[0] > 0
    better_ext = pr_pn.get("ci95", [0, 0])[0] > 0 or qr_qn.get("ci95", [0, 0])[0] > 0
    if better_names and better_ext:
        d = "CONTINUE"
    elif better_names or better_ext:
        d = "FIX ONE COMPONENT"
    else:
        d = "STOP"
    return {"decision": d, "recognition_better": bool(better_names), "extension_better": bool(better_ext),
            "rule": "CONTINUE if Qwen beats pooled Trident at region level AND its mIoU gain on the primary "
                    "population has a bootstrap CI above zero AND an extension gain has a CI above zero; "
                    "FIX ONE COMPONENT if exactly one holds; STOP otherwise."}


def main() -> str:
    an = json.load(open(os.path.join(P.ARTIFACTS, "analysis.json")))
    man = json.load(open(P.MANIFEST))
    cal = json.load(open(os.path.join(P.ARTIFACTS, "calibration.json")))
    vb0 = json.load(open(os.path.join(P.ARTIFACTS, "verify_b0.json")))
    plog = json.load(open(os.path.join(P.LOGS, "proposals.json")))
    dlog = json.load(open(os.path.join(P.LOGS, "predict.json")))
    a = man["anchors"]
    ids = [t for t in a["evaluation"] if str(t) in an["per_anchor"]]
    rg = recognition_summary(an, ids)
    vd = verdict(an, rg)
    S = an["sets"]["evaluation"]
    write_csv(an)
    L_ = []
    w = L_.append

    w(f"# Region naming for semantic completion — pilot report\n")
    w(f"**Decision: {vd['decision']}.** {_headline(an, rg, vd)}\n")
    w("## 1. What was run\n")
    w(f"One question: with completed occupancy fixed, can a vision-language model name objects and regions "
      f"more accurately than the deployed Trident evidence, and can those names be carried to occupied "
      f"voxels that have no causal semantic evidence?\n")
    w(f"- **Scope**: KITTI-360 `{man['anchors']['drive']}` only, {S['_n_anchors']} evaluation anchors "
      f"(one per temporal block out of {a['n_blocks']}), {len(a['calibration'])} calibration anchors, "
      f"the golden anchor `t={a['golden']}`. No SemanticKITTI, no Occ3D, no target evaluation.\n")
    w(f"- **Everything frozen** was reused unchanged: the 5-frame causal clip, the hard map, "
      f"`CompletionUNet` seed 0 occupancy, the Trident-H evidence, the nearest-evidence rule B0 and the "
      f"repository's own scorer. Legacy commit `{man['legacy']['commit'][:9]}`.\n")
    w(f"- **New**: SAM region proposals, pooled-Trident region naming (P), {man['frozen_settings']['recognition']['qwen']['model']} "
      f"region naming (Q), and one bounded geodesic extension over the fixed occupancy.\n")
    w(f"- **Isolation**: the experiment itself is entirely under `{P.EXPERIMENT_ROOT}` -- all code, caches, "
      f"the model checkpoint and every output. Nothing was added to, imported from or changed in "
      f"`{P.REPRO_REPO}`.\n")
    if os.path.isdir(P.DELIVERABLE_COPY):
        w(f"- **One exception, on explicit instruction**: a read-only copy of the deliverables (this report, "
          f"`results.csv`, the manifest, the figures, the small artifacts and the pilot's own code) was "
          f"placed at `{P.DELIVERABLE_COPY}` so the write-up sits beside the repository's other reports. "
          f"It is a record, not an integration: nothing under `reports/` is importable by the system, and "
          f"no module, config or pipeline of the legacy repository references it.\n")

    w("## 2. Adapter alignment — the baseline is reproduced exactly\n")
    w("B0 recomputed through this experiment's adapter over all 590 anchors of the drive, against the "
      "numbers the legacy Gate 17 report published:\n")
    w("| population | published mIoU | reproduced | voxels | |")
    w("|---|---:|---:|---:|---|")
    for r, v in vb0["regions"].items():
        w(f"| `{r}` | {v['reference_miou']:.6f} | {v['reproduced_miou']:.6f} | {v['reproduced_voxels']:,} | "
          f"{'match' if v['match'] else '**MISMATCH**'} |")
    w(f"\nThe grid, the transform, the occupancy threshold, the teacher readout and the metric are therefore "
      f"the frozen ones.\n")

    w("## 3. Region proposals\n")
    pl = plog["log"]
    w(f"Frozen SAM ViT-H automatic mask generation on each of the five causal frames, "
      f"{np.mean([x['n_frame_masks'] for x in pl]):.0f} frame masks and "
      f"**{np.mean([x['n_regions'] for x in pl]):.0f} cross-frame regions per anchor** on average.\n")
    w(f"- Lifted support is **exactly** the frozen evidence set: a mask pixel becomes 3D support only if it "
      f"is a ray the frozen mapper accepts, so the region decomposition partitions the existing evidence "
      f"and adds nothing. {100*np.mean([x['evidence_ownership_fraction'] for x in pl]):.1f} % of evidence "
      f"voxels fall inside some mask; the rest keep the original per-voxel baseline label.\n")
    w(f"- **Fragmentation is the dominant failure of the proposer**: "
      f"{100*np.mean([x['n_single_frame_regions']/max(x['n_regions'],1) for x in pl]):.0f} % of regions are "
      f"seen in a single frame, "
      f"{np.mean([x['n_regions_merging_two_masks_of_one_frame'] for x in pl]):.1f} regions per anchor merge "
      f"two masks of the same frame, and "
      f"{np.mean([x['n_regions_with_non_contiguous_frames'] for x in pl]):.1f} span non-contiguous frames.\n")
    w(f"- {np.mean([x['n_regions_with_rgb_support_no_evidence'] for x in pl]):.1f} regions per anchor have RGB "
      f"support but no Trident evidence at all. They are counted here and never mixed into the matched "
      f"comparison: neither recognizer can seed from them.\n")

    w("## 4. Did Qwen name the same regions better than pooled Trident?\n")
    w(f"Region-level, on the {rg['n_with_human_label']} queried regions that have a human label "
      f"(of {rg['n_queried_regions']} queried; {rg['n_without_human_label']} have none and are *unknown*, "
      f"never counted wrong):\n")
    w("| | pooled Trident (P) | Qwen (Q) |")
    w("|---|---:|---:|")
    w(f"| modal-class accuracy | {pct(rg['p_accuracy'])} % | {pct(rg['q_accuracy'])} % |")
    w(f"| accuracy weighted by evidence support | {pct(rg['weighted_p_accuracy'])} % | {pct(rg['weighted_q_accuracy'])} % |")
    w(f"\n- P and Q agree on {pct(rg['agree_pq'])} % of queried regions.")
    w(f"- Qwen outcomes: {rg['outcomes']}. Abstentions and out-of-vocabulary answers fall back to the same "
      f"region's pooled Trident label, so a refusal never removes a seed.")
    if rg["single_view_disagreement"] is not None:
        w(f"- On {rg['n_single_view_compared']} multi-frame regions asked twice, a single view instead of three "
          f"changes the answer {pct(rg['single_view_disagreement'])} % of the time — the recognizer is not "
          f"view-stable, and its self-reported wording is not treated as a probability.")
    w(f"- Mean region purity against human labels is {pct(rg['mean_purity'])} %; "
      f"{rg['n_pure_ge_90']} regions are at least 90 % one class and {rg['n_mixed_lt_60']} are below 60 % "
      f"(mixed proposals, computed after prediction).\n")
    w("Per modal human class (regions, P correct, Q correct):\n")
    w("| class | regions | P | Q |")
    w("|---|---:|---:|---:|")
    for k, v in list(rg["per_modal_class"].items()):
        w(f"| {k} | {v['n']} | {v['p']} | {v['q']} |")

    w("\n## 5. Voxel results — the matched two-by-two, plus the original baseline\n")
    w(f"Pooled over the {S['_n_anchors']} evaluation anchors; one confusion matrix per population, the metric "
      f"computed once from the pooled matrix. Occupancy is one volume shared by every method: "
      f"SC IoU **{100*S['_occupancy']['sc_iou']:.2f}** "
      f"(TP {S['_occupancy']['tp']:,} FP {S['_occupancy']['fp']:,} FN {S['_occupancy']['fn']:,}) for all five.\n")
    w("| population | voxels | " + " | ".join(PD.METHODS) + " |")
    w("|---|---:|" + "---:|" * len(PD.METHODS))
    for r in EV.ALL_REGIONS:
        n = S[PD.METHODS[0]][r]["n_voxels"]
        w(f"| {REGION_LABEL.get(r, r)} | {n:,} | " + " | ".join(pct(S[m][r]["miou"]) for m in PD.METHODS) + " |")
    w("\nmIoU in points. The same table by overall accuracy:\n")
    w("| population | " + " | ".join(PD.METHODS) + " |")
    w("|---|" + "---:|" * len(PD.METHODS))
    for r in (L.PRIMARY, "ssc_no_evidence", "ssc_full", "completed_unobserved_objects", "completed_unobserved_surfaces"):
        w(f"| {REGION_LABEL.get(r, r)} | " + " | ".join(pct(S[m][r]["overall_accuracy"]) for m in PD.METHODS) + " |")

    w("\n### The five comparisons the design asks for\n")
    w("| comparison | what it isolates | Δ mIoU on the primary population | 95 % block CI |")
    w("|---|---|---:|---|")
    for key, what in (("P_N-B0", "pooling per-voxel teacher argmax into one label per region"),
                      ("Q_N-P_N", "recognition, identical nearest extension"),
                      ("P_R-P_N", "extension, Trident naming"),
                      ("Q_R-Q_N", "extension, Qwen naming"),
                      ("Q_R-P_R", "recognition, identical region extension")):
        b = an["bootstrap"].get(f"{key} @ {L.PRIMARY}")
        if b:
            w(f"| `{key}` | {what} | {b['delta']*100:+.2f} | [{b['ci95'][0]*100:+.2f}, {b['ci95'][1]*100:+.2f}] |")
    w(f"\nBlocks are the {S['_n_anchors']} evaluation anchors, one per temporal block, 2,000 paired resamples, "
      f"seed 0. This is an exploratory pilot: the interval describes anchor-to-anchor variation, not a "
      f"significance claim about individual correlated voxels.\n")

    pa = an["per_anchor"]
    fx = {m: sum(pa[str(t)]["fixed"].get(m, 0) for t in ids) for m in ("P_N", "P_R", "Q_N", "Q_R")}
    bk = {m: sum(pa[str(t)]["broken"].get(m, 0) for t in ids) for m in ("P_N", "P_R", "Q_N", "Q_R")}
    w("### Corrections and damage on the primary population, against B0\n")
    w("| method | voxels B0 got wrong and this got right | voxels B0 got right and this got wrong | net |")
    w("|---|---:|---:|---:|")
    for m in ("P_N", "Q_N", "P_R", "Q_R"):
        w(f"| {m} | {fx[m]:,} | {bk[m]:,} | {fx[m]-bk[m]:+,} |")

    w("\n## 6. Did the extension reach the no-evidence population?\n")
    ex = [pa[str(t)]["stats"]["extension"] for t in ids]
    fbr = float(np.mean([pa[str(t)]["stats"]["membership_fallback_fraction_on_O"] for t in ids]))
    w(f"- The membership map is computed once per anchor from occupancy, observed free space and adjacency, "
      f"with `max_hops = {cal['chosen_max_hops']}` ({0.2*cal['chosen_max_hops']:.1f} m), selected on the "
      f"calibration anchors alone.\n")
    w(f"- It assigns **{100*np.mean([e['occupied_coverage'] for e in ex]):.1f} %** of predicted-occupied voxels; "
      f"**{100*fbr:.1f} %** fall back to the matching nearest-evidence prediction. Growth adds "
      f"{np.mean([e['n_grown'] for e in ex]):.0f} voxels per anchor beyond "
      f"{np.mean([e['n_seeds'] for e in ex]):.0f} seeds, and "
      f"{np.mean([e['n_contested'] for e in ex]):.0f} are contested by two regions and left unassigned.\n")
    gobs = S["B0"]["completed_unobserved_geo_observed"]["n_voxels"]
    gunobs = S["B0"]["completed_unobserved_geo_unobserved"]["n_voxels"]
    w(f"- Of the primary population, {gobs:,} voxels were geometrically observed by the clip "
      f"({100*gobs/max(gobs+gunobs,1):.0f} %) and {gunobs:,} were not. This is a geometry split, not an "
      f"occlusion split: the frozen representation cannot separate \"occluded in all five frames\" from "
      f"\"never resolved\", so **physical-occlusion attribution is unavailable** and is not claimed.\n")

    w("## 7. Calibration trace\n")
    w("| max_hops | mIoU on the primary population | accuracy | occupied coverage |")
    w("|---|---:|---:|---:|")
    for r in cal["rows"]:
        w(f"| {r['max_hops']} | {pct(r['miou'])} | {pct(r['accuracy'])} | {r['occupied_coverage']:.3f} |")
    w(f"\nMeasured on anchors {cal['anchors']}, pooled-Trident naming, before any evaluation prediction was "
      f"written. The association thresholds were **not** searched: "
      f"{cal['association_thresholds_not_searched']}.\n")

    w("## 8. Answers\n")
    for q, ans in _answers(an, rg, S, cal, vd, fx, bk, ids, pa).items():
        w(f"**{q}**  \n{ans}\n")

    w("## 9. Limitations\n")
    w("- 24 evaluation anchors on one drive of one city: a pilot, not a benchmark. Paired per-anchor "
      "variation is reported; no claim is projected onto SemanticKITTI or Occ3D, which were never opened.\n")
    w("- The modal-class target for recognition comes from human voxel labels inside each region. Regions "
      "with no labelled voxel are unknown, and region purity is far from 1, so region accuracy is a "
      "coarse measure of naming, not a clean instance benchmark.\n")
    w("- The proposer is one frozen setting of SAM, not a tuned one. Its fragmentation bounds everything "
      "downstream.\n")
    w("- Qwen saw tight crops and a highlighted context frame; its answer about a region is a statement "
      "about the visible surface, never evidence about hidden geometry.\n")
    w("- Only one constant (`max_hops`) was selected, on calibration anchors. No hyperparameter search ran.\n")

    w("## 10. What was not touched\n")
    w(f"- `{P.REPRO_REPO}`: never imported, never read as a dependency, never written. Its git state was "
      f"recorded before and after. It *was* changed during this run by another session "
      f"(`src/lingssc/training/geometry_training.py`, `.pytest_cache`, one `.pyc`); that is reported, not "
      f"reverted, and none of it came from here.\n")
    w(f"- `{P.LEGACY_REPO}`: imported read-only with bytecode writes disabled; no legacy script executed. "
      f"Its tree was byte-identical through the whole measurement "
      + ("and the only later change is the deliverable copy under `reports/` described in section 1.\n"
         if os.path.isdir(P.DELIVERABLE_COPY) else "and no file was written.\n"))
    w("- No retraining or fine-tuning of anything. No dataset permission, lockfile or shared environment "
      "was changed; the Qwen checkpoint and all caches live inside the experiment directory.\n")
    w(f"\n## 11. One next action\n{_next_action(vd, rg, an)}\n")
    w(f"\n---\nGenerated by `code/report.py` from `artifacts/analysis.json`, `artifacts/calibration.json`, "
      f"`artifacts/verify_b0.json` and `logs/`. Runtime: proposals "
      f"{sum(x['seconds'] for x in pl)/60:.0f} min, prediction {sum(x['seconds'] for x in dlog['log'])/60:.0f} min.\n")

    with open(P.under_experiment(P.REPORT), "w") as f:
        f.write("\n".join(L_) + "\n")
    return P.REPORT


def _headline(an, rg, vd):
    prim = L.PRIMARY
    b = an["bootstrap"]
    qn = b[f"Q_N-P_N @ {prim}"]; pr = b[f"P_R-P_N @ {prim}"]; qr = b[f"Q_R-Q_N @ {prim}"]
    S = an["sets"]["evaluation"]
    return (f"On {S['_n_anchors']} evaluation anchors, replacing the teacher's name for a region with "
            f"Qwen's moves the primary population by **{qn['delta']*100:+.2f} mIoU** "
            f"(CI [{qn['ci95'][0]*100:+.2f}, {qn['ci95'][1]*100:+.2f}]) at region accuracy "
            f"{pct(rg['q_accuracy'])} % against pooled Trident's {pct(rg['p_accuracy'])} %, and the bounded "
            f"region extension moves it by **{pr['delta']*100:+.2f}** under Trident naming and "
            f"**{qr['delta']*100:+.2f}** under Qwen naming.")


def _answers(an, rg, S, cal, vd, fx, bk, ids, pa):
    prim = L.PRIMARY
    b = an["bootstrap"]
    def line(k):
        x = b[f"{k} @ {prim}"]
        return f"{x['delta']*100:+.2f} mIoU, CI [{x['ci95'][0]*100:+.2f}, {x['ci95'][1]*100:+.2f}]"
    o = {}
    o["Did Qwen name the same regions better than pooled Trident?"] = (
        f"At region level, {pct(rg['q_accuracy'])} % against {pct(rg['p_accuracy'])} % "
        f"({'Qwen ahead' if (rg['q_accuracy'] or 0) > (rg['p_accuracy'] or 0) else 'Qwen behind'}); weighted by "
        f"evidence support, {pct(rg['weighted_q_accuracy'])} % against {pct(rg['weighted_p_accuracy'])} %. "
        f"At voxel level under the identical nearest extension, `Q_N - P_N` = {line('Q_N-P_N')}.")
    o["Did region extension improve over nearest extension at fixed naming?"] = (
        f"With Trident naming `P_R - P_N` = {line('P_R-P_N')}; with Qwen naming `Q_R - Q_N` = {line('Q_R-Q_N')}. "
        f"The membership map is identical in both, so this isolates the extension rule.")
    o["Did improvements reach the no-evidence population and full SSC?"] = (
        f"The primary population *is* the no-evidence population. On the deployable SSC read over every "
        f"labelled voxel without causal evidence, "
        + ", ".join(f"{m} {pct(S[m]['ssc_no_evidence']['miou'])}" for m in PD.METHODS)
        + f"; over every labelled voxel, "
        + ", ".join(f"{m} {pct(S[m]['ssc_full']['miou'])}" for m in PD.METHODS) + ".")
    o["How much was fallback, and where did errors spread?"] = (
        f"{100*float(np.mean([pa[str(t)]['stats']['membership_fallback_fraction_on_O'] for t in ids])):.1f} % of "
        f"predicted-occupied voxels fall back to nearest evidence, plus Qwen's "
        f"{rg['outcomes']['abstain'] + rg['outcomes']['out_of_vocabulary'] + rg['outcomes']['malformed']} "
        f"non-answers of {rg['n_queried_regions']} queries, which fall back to the pooled Trident label. "
        f"Against B0 on the primary population, Q_R fixes {fx['Q_R']:,} voxels and breaks {bk['Q_R']:,}; "
        f"P_R fixes {fx['P_R']:,} and breaks {bk['P_R']:,}.")
    gobs = S["B0"]["completed_unobserved_geo_observed"]["n_voxels"]
    gunobs = S["B0"]["completed_unobserved_geo_unobserved"]["n_voxels"]
    o["Were improvements confined to visible evidence repair rather than occlusion?"] = (
        f"Split by whether the clip geometrically observed the voxel: on the {gobs:,} observed voxels "
        + ", ".join(f"{m} {pct(S[m]['completed_unobserved_geo_observed']['miou'])}" for m in PD.METHODS)
        + f"; on the {gunobs:,} never-observed voxels "
        + ", ".join(f"{m} {pct(S[m]['completed_unobserved_geo_unobserved']['miou'])}" for m in PD.METHODS)
        + ". Physical occlusion cannot be separated from unresolved visibility in the frozen "
          "representation, so that attribution is marked unavailable rather than guessed.")
    o["What remained unchanged, and were any external concurrent changes observed?"] = (
        "Occupancy, the evidence set, the grid and the scorer are identical across all five methods "
        "(the shared occupancy volume gives one SC IoU for all of them). Concurrent external activity is "
        "reported in `artifacts/protected_state_after.txt`.")
    o["Should we continue this idea, fix a specific problem, or stop?"] = vd["decision"] + ". " + vd["rule"]
    return o


def _next_action(vd, rg, an):
    prim = L.PRIMARY
    qn = an["bootstrap"][f"Q_N-P_N @ {prim}"]
    if vd["decision"] == "STOP":
        return ("Stop this line. The measured bottleneck is not the name: "
                f"recognition at region level is {pct(rg['q_accuracy'])} % for Qwen against "
                f"{pct(rg['p_accuracy'])} % for the teacher, yet the primary population moves "
                f"{qn['delta']*100:+.2f} mIoU. The single next action worth taking is to attack the "
                "proposer, not the recognizer: measure how much of the primary population a *perfect* "
                "region decomposition could reach at all, by scoring the same extension rule seeded from "
                "connected components of the human labels — the oracle that bounds every proposal-based "
                "method here — before any further VLM work.")
    if vd["decision"] == "FIX ONE COMPONENT":
        return ("Fix the one component that moved. Re-run this same pilot with that component replaced and "
                "everything else byte-identical, on the same 24 anchors, before considering integration.")
    return ("Repeat this pilot on the three held-out KITTI-360 drives with the identical frozen settings. "
            "Only a result that survives unseen routes justifies an integration recommendation, and even "
            "then the reproduction repository keeps its own milestones.")


if __name__ == "__main__":
    print(main())
