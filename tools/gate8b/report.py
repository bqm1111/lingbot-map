#!/usr/bin/env python
"""Render reports/gate8b/gate8b_report.md from artifacts/gate8b/gate8b_results.json.

    python tools/gate8b/report.py --write
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT                                               # noqa: E402

OUT = os.path.join(REPO_ROOT, "reports", "gate8b", "gate8b_report.md")
T = ("semantickitti", "occ3d", "kitti360")
NICE = {"semantickitti": "SemanticKITTI 08", "occ3d": "Occ3D-nuScenes val", "kitti360": "KITTI-360 (drive 0006)"}


def f(x, n=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "**no**"
    if isinstance(x, str):
        return x
    return f"{x:.{n}f}"


def ci(c, key="binary_iou"):
    if not c:
        return "—"
    d = c[key]
    return f"{d['difference']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]"


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def provenance(res):
    rows = []
    for t in T:
        fd = res["folds"][t]
        rows.append([NICE[t], ", ".join(fd["train"]), ", ".join(fd["val"]), t, fd["status"],
                     fd.get("kitti360_partition", "—")])
    return table(["target (held out)", "completion-training sources", "source-validation domains",
                  "never read before freezing", "status", "KITTI-360 partition"], rows)


def frozen_table(res):
    rows = []
    for t in T:
        fz = res["folds"][t].get("frozen")
        if not fz:
            rows.append([NICE[t], "—", "—", "—", "—", "—"]); continue
        sel = res["folds"][t]["selection"]["selected"]
        rows.append([NICE[t], fz["selected_candidate"], f"`{fz['checkpoint_sha256'][:16]}…`",
                     f(sel["macro_ap"]), f"{fz['global_threshold_final_logodds']:+.4f}",
                     f"{fz['mapper_calibrated_threshold_logodds']:+.4f}"])
    return table(["fold", "selected checkpoint", "sha256", "macro source AP", "global τ (final log-odds)",
                  "mapper τ"], rows)


def candidates_table(res):
    rows = []
    for t in T:
        s = res["folds"][t].get("selection")
        if not s:
            continue
        for r in s["candidates"]:
            srcs = list(r["sources"])
            rows.append([NICE[t], r["candidate"]] +
                        [f"{NICE.get(x, x)}: AP {r['sources'][x]['average_precision']:.4f} "
                         f"(prev {r['sources'][x]['prevalence']:.4f}, ×{r['sources'][x]['ap_over_prevalence']:.2f})"
                         for x in srcs] + [f(r["macro_ap"]), f(r["ap_exceeds_prevalence_on_both"])])
    return table(["fold", "candidate", "source A", "source B", "macro AP", "AP > prev on both"], rows)


def tf_table(res, setting):
    rows = []
    for t in T:
        sb = res["settings"][t][setting]
        if not sb:
            rows.append([NICE[t]] + ["—"] * 11); continue
        for reg in ("completion_full", "completion_edit", "mapper_full"):
            s = sb["threshold_free"][reg]
            rows.append([NICE[t] if reg == "completion_full" else "", reg, s["n_voxels"], f(s["prevalence"]),
                         f(s["average_precision"]), f(s["ap_over_prevalence"], 2), f(s["auroc"]), f(s["brier"]),
                         f(s["ece"]), f(s["at_zero"]["iou"]), f(s["best_iou"]["threshold"], 3), f(s["best_iou"]["iou"])])
    return table(["target", "region", "voxels", "prevalence", "AP", "AP/prev", "AUROC", "Brier", "ECE",
                  "IoU @ 0", "oracle τ", "IoU @ oracle τ"], rows)


ORDER = ["completion", "completion_edit_region", "mapper_native", "mapper_dilate", "mapper_calibrated",
         "frozen_5frame_raw", "frozen_5frame_dil", "all_valid_occupied", "editable_fill",
         "random_editable_mean", "completion_occany_dilation", "completion_occany_majority",
         "mapper_native_occany_dilation", "mapper_native_occany_majority", "mapper_dilate_occany_dilation"]


def op_table(res, setting, t):
    sb = res["settings"][t][setting]
    if not sb:
        return "(missing)"
    m = sb["methods"]; rows = []
    for k in ORDER:
        if k not in m:
            continue
        x = m[k]
        rows.append([k, f(x["binary_iou"]), f(x.get("precision")), f(x.get("recall")), f(x.get("density")),
                     f(x.get("pred_over_gt_volume"), 2),
                     (f"sd {x['binary_iou_sd']:.4f}, range [{x['binary_iou_range'][0]:.4f}, {x['binary_iou_range'][1]:.4f}]"
                      if k == "random_editable_mean" else ci(sb["comparisons"].get(f"completion vs {k}")))])
    return table(["method", "SC IoU", "precision", "recall", "predicted density", "pred/GT volume",
                  "Δ vs completion (paired 95% CI)"], rows)


def sem_table(res, setting, t):
    sb = res["settings"][t][setting]
    if not sb:
        return "(missing)"
    rows = []
    for k, s in sb["semantic"].items():
        rows.append([k, f(s.get("ssc_miou")), f(s.get("binary_iou")), f(s.get("tp_accuracy")),
                     f(s.get("coverage_miss")), f(s.get("naming_error")),
                     ci(sb["comparisons"].get(f"SSC completion vs {k}"), "ssc_miou")])
    nc = sb.get("newly_completed") or {}
    extra = ""
    if nc:
        extra = (f"\n\nTP-conditioned semantic accuracy of the completion: **all** true positives "
                 f"{f(sb['semantic'].get('completion', {}).get('tp_accuracy'))}; **causally observed** "
                 f"true positives {f(nc.get('observed_semantic_accuracy'))} (n = {nc.get('n_observed_true_positives', 0):,}); "
                 f"**newly completed** true positives {f(nc.get('semantic_accuracy'))} "
                 f"(n = {nc.get('n_new_true_positives', 0):,}).")
    return table(["method", "SSC mIoU", "SC IoU", "TP-cond. sem. acc.", "coverage miss", "naming error",
                  "Δ SSC mIoU (paired 95% CI)"], rows) + extra


def classwise(res, t):
    sb = res["settings"][t]["stream"]
    if not sb:
        return "(missing)"
    sem = sb["semantic"]
    keys = [k for k in ("completion", "mapper_native", "mapper_dilate", "frozen_5frame_dil") if k in sem]
    names = sorted({n for k in keys for n in sem[k].get("per_class_iou", {})})
    return table(["class"] + keys, [[n] + [f(sem[k]["per_class_iou"].get(n)) for k in keys] for n in names])


def teacher_table(res):
    rows = []
    for t in T:
        x = res["teacher"].get(t)
        if not x:
            rows.append([NICE[t], "—", "—", "—", "—"]); continue
        rows.append([NICE[t], x["n_anchors"], f"{x['accuracy_future_visible_occupied']:.4f} (n = {x['n_future_visible_gt_occupied']:,})",
                     f"{x['accuracy_on_causally_observed_subset']:.4f} (n = {x['n_causally_observed_subset']:,})",
                     f"{x['accuracy_on_newly_visible_subset']:.4f} (n = {x['n_newly_visible_subset']:,})"])
    return table(["target", "anchors", "teacher accuracy, all future-visible GT-occupied voxels",
                  "…on the causally observed subset", "…on the newly visible subset"], rows)


def occany_table(res):
    out = []
    for t in ("semantickitti", "occ3d"):
        sb = res["settings"][t]["clips"]; pub = res["occany"]["published_5frame_single_camera"][t]
        proto = res["occany"]["protocol"][t]
        rows = [["**OccAny (published, 5 frames, single camera)**", f"{100*pub['precision']:.2f}", f"{100*pub['recall']:.2f}",
                 f"{100*pub['sc_iou']:.2f}", "OccAny's own post-processing as published"]]
        if sb:
            m = sb["methods"]
            for k, lbl in (("completion", "ours: completion, raw (primary)"),
                           ("completion_occany_dilation", "ours: completion + OccAny geometry pooling (3×3×3 max-pool)"),
                           ("completion_occany_majority", "ours: completion + OccAny geometry pooling (3×3×3 vote)"),
                           ("mapper_native", "ours: five-frame map, no completion, raw"),
                           ("mapper_native_occany_dilation", "ours: five-frame map + OccAny max-pool"),
                           ("mapper_dilate", "ours: five-frame map + fixed 0.4 m dilation"),
                           ("frozen_5frame_raw", "Gate 6 frozen five-frame G51-B, raw"),
                           ("frozen_5frame_dil", "Gate 6 frozen five-frame G51-B + 0.4 m dilation")):
                if k in m:
                    x = m[k]
                    rows.append([lbl, f"{100*x['precision']:.2f}", f"{100*x['recall']:.2f}", f"{100*x['binary_iou']:.2f}", ""])
        out.append(f"**{NICE[t]}** — *{proto['label']}*\n\n" + table(["method", "precision %", "recall %", "SC IoU %", "note"], rows)
                   + "\n\nProtocol audit:\n\n" + table(["item", "OccAny", "ours", "match"],
                                                      [[i["item"], i["occany"], i["ours"], "yes" if i["match"] else "**no**"]
                                                       for i in proto["items"]]))
    return "\n\n".join(out)


def decision_table(res, setting):
    rows = []
    for t in T:
        d = res["decision"][t][setting]
        if not d:
            rows.append([NICE[t]] + ["—"] * 10); continue
        g = d["geometry"]
        rows.append([NICE[t], f(g["ap_over_prevalence"], 2), f(g["auroc"]), f(g["ap_meaningfully_above_prevalence"]),
                     f(g["beats_incremental_mapper"]), f(g["beats_calibrated_mapper"]), f(g["beats_strongest_dilation"]),
                     f(g["beats_editable_fill"]), f(g["beats_matched_density_random"]), f(g["pred_over_gt_volume"], 2),
                     "**PASS**" if g["passed"] else "FAIL"])
    return table(["target", "AP/prev", "AUROC", "AP > 1.5×prev", "beats mapper", "beats calibrated mapper",
                  "beats strongest dilation", "beats editable fill", "beats matched random", "pred/GT vol",
                  "geometry"], rows)


def sem_decision_table(res, setting):
    rows = []
    for t in T:
        d = res["decision"][t][setting]
        if not d:
            rows.append([NICE[t]] + ["—"] * 6); continue
        s = d["semantics"]
        rows.append([NICE[t], f(s["completion_ssc_miou"]), f"{s['strongest_non_completion']} ({f(s['strongest_non_completion_ssc_miou'])})",
                     f(s["beats_it"]), f(s["newly_completed_semantic_accuracy"]), f(s["new_tp_carry_information"]),
                     "**PASS**" if s["passed"] else "FAIL"])
    return table(["target", "completion SSC mIoU", "strongest non-completion baseline", "beats it (CI > 0)",
                  "new-TP semantic acc.", "carries information", "semantics"], rows)


def cost_table(res):
    rows = []
    for t in T:
        tr = res["folds"][t].get("training") or {}
        for setting in ("stream", "clips"):
            sb = res["settings"][t][setting]
            rows.append([NICE[t], setting, f"{tr.get('seconds', 0)/60:.1f} min" if tr else "—",
                         f"{tr.get('peak_gpu_gib', 0):.2f} GiB" if tr else "—",
                         f"{sb['seconds']/60:.1f} min" if sb else "—", f"{sb['peak_gpu_gib']:.1f} GiB" if sb else "—",
                         sb["n"] if sb else "—"])
    return table(["fold", "setting", "training wall", "training peak GPU", "target eval wall", "eval peak GPU",
                  "anchors / clips"], rows)


def k360_in_domain(res):
    """What the new folds say about KITTI-360 as a *source-validation* domain: the models
    trained on KITTI-360 drives, scored on the KITTI-360 val drive before any target was opened."""
    lines = []
    for t in ("semantickitti", "occ3d"):
        sel = res["folds"][t].get("selection")
        if not sel:
            continue
        win = sel["selected"]["candidate"]
        cand = [c for c in sel["candidates"] if c["candidate"] == win][0]
        other = [x for x in cand["sources"] if x != "kitti360"][0]
        k, o = cand["sources"]["kitti360"], cand["sources"][other]
        mp = [c for c in sel["candidates"] if c["candidate"] == "mapper"][0]["sources"]["kitti360"]
        lines.append(f"* fold **{NICE[t]}** (trained on KITTI-360 drives 0003/0007/0010 + "
                     f"{res['folds'][t]['train'][0]}): on the KITTI-360 val drive 0006 the selected "
                     f"checkpoint scores AP {k['average_precision']:.4f} against prevalence "
                     f"{k['prevalence']:.4f} (×{k['ap_over_prevalence']:.2f}, AUROC {k['auroc']:.4f}; the "
                     f"mapper alone ×{mp['ap_over_prevalence']:.2f}), while on {NICE.get(other, other)} in the "
                     f"same fold it scores ×{o['ap_over_prevalence']:.2f} (AUROC {o['auroc']:.4f}).")
    return "\n".join(lines)


def clips_note(res):
    d = res["decision"]["occ3d"]["clips"]["geometry"]; c = res["settings"]["occ3d"]["clips"]["comparisons"]
    if d["passed"] or not all(d[k] for k in ("beats_incremental_mapper", "beats_calibrated_mapper",
                                            "beats_strongest_dilation", "beats_editable_fill",
                                            "beats_matched_density_random")):
        return ""
    return (f"On Occ3D in the matched setting the completion clears every declared baseline with paired "
            f"intervals excluding zero (e.g. vs matched-density random {ci(c['completion vs random_editable_s0'])}, "
            f"vs the 0.4 m dilation {ci(c['completion vs mapper_dilate'])}) and fails the mechanical rule only "
            f"on the occupied-volume ratio, {d['pred_over_gt_volume']:.2f} against the 2.0 line; the streaming "
            f"setting, at 1.91, passes it. The rule is applied as written rather than bent.")


def occany_note(res):
    sk = res["settings"]["semantickitti"]["clips"]["methods"]; oc = res["settings"]["occ3d"]["clips"]["methods"]
    pub = res["occany"]["published_5frame_single_camera"]
    return (f"Read as a reference comparison only. On Occ3D our raw five-frame completion "
            f"({100*oc['completion']['binary_iou']:.2f} % SC IoU) sits above OccAny's published "
            f"{100*pub['occ3d']['sc_iou']:.2f} %, and on SemanticKITTI far below it "
            f"({100*sk['completion']['binary_iou']:.2f} % vs {100*pub['semantickitti']['sc_iou']:.2f} %); neither "
            f"number is exact against OccAny's, because OccAny's five frames run *forward* from the target and "
            f"ours *backward*, and because the metric scale comes from a different source. On SemanticKITTI our "
            f"five-frame map with the fixed 0.4 m dilation and **no completion** "
            f"({100*sk['mapper_dilate']['binary_iou']:.2f} %) beats the completion, which is the transfer failure "
            f"of Section 3 seen again. No number in this section supports a claim over OccAny.")


def verdict(res):
    oc = res["outcome"]; dec = res["decision"]
    g = {t: dec[t]["stream"]["geometry"] for t in T if dec[t]["stream"]}
    lifts = ", ".join(f"{NICE[t]} ×{g[t]['ap_over_prevalence']:.2f} (AUROC {g[t]['auroc']:.3f})" for t in T if t in g)
    o = oc.get("outcome", "incomplete")
    if o.startswith("A"):
        v = (f"Only KITTI-360 fails: the completion keeps a substantial ranking lift on the two new held-out "
             f"targets while KITTI-360 stays at chance ({lifts}).")
        rec = ("Audit the KITTI-360 side before touching the completion representation: its map export, "
               "coordinate conventions, label construction and history statistics. Do not modify the model.")
    elif o.startswith("B"):
        v = (f"Every held-out fold collapses toward chance ({lifts}): the completion representation is "
             "source-specific, not KITTI-360-specific.")
        rec = ("A subsequent, source-validated map-state canonicalisation gate. Do not implement it here, and do "
               "not scale the network or change the architecture.")
    elif o.startswith("C"):
        v = f"SC occupancy transfers on every held-out target ({lifts}) but SSC mIoU stays below the strongest frozen semantic baseline."
        rec = "Preserve the geometry module; redesign the semantic teacher / representation in a bounded gate."
    elif o.startswith("D"):
        v = f"Both SC occupancy and SSC mIoU transfer on every held-out target with paired intervals excluding zero ({lifts})."
        rec = "A focused semantic / open-vocabulary refinement, not architecture scaling."
    else:
        near = oc.get("near_chance", {})
        fails = [NICE[t] for t in T if near.get(t)]
        holds = [NICE[t] for t in T if t in g and not near.get(t)]
        v = (f"Transfer is target-dependent, not KITTI-360-specific: the completion collapses to chance on "
             f"{' and '.join(fails)} but keeps a real ranking lift on {', '.join(holds)} ({lifts}); "
             f"geometry passes the declared baselines on {', '.join(NICE[t] for t in T if t in g and g[t]['passed']) or 'no target'}.")
        rec = ("A source-validated map-state canonicalisation gate (Outcome B's remedy), because the failure "
               "reproduces on SemanticKITTI -- a target whose grid, frame and stride the KITTI-360 training "
               "source shares exactly -- so it is a property of the map state handed to the head, not of one "
               "benchmark's export. Run the cheap KITTI-360 export/label audit (Outcome A's remedy) first, since "
               "KITTI-360 also fails as an in-domain source-validation set. Do not scale the network, change "
               "the architecture, add datasets or redesign semantics.")
    return v, rec


TEMPLATE = '''# Gate 8B — leave-one-dataset-out transfer evaluation

**Verdict.** {verdict}

**Recommended next action.** {recommend}

**Outcome class.** {outcome}

Every fold is *leave-one-dataset-out transfer with no target-domain training or adaptation of the
completion module*. None is claimed as whole-system zero-shot: LingBot-Map's published training
mixture includes KITTI-360, and the training data of MoGe-2 and Trident-H were not audited here.

## 1. Provenance — three folds

{provenance}

The two new folds were built with Gate 8A's frozen method (`configs/gate8a/cellB_uniform_focal.yaml`:
uniform crop sampler, focal BCE + Dice, 0.5 × future-teacher KL, 6 000 steps, batch 4, crop 128×128×32,
seed 0, candidates {{best, last}}, macro-average source AP for the checkpoint, one global final-logit
threshold from source validation). A test asserts the fold configs differ from cell B only in their
source lists and never name the target. The two training sources are drawn with equal probability.

## 2. Frozen configuration per fold

{frozen}

Selection candidates (source-validation domains only; the target of each fold was not read until the
frozen file existed):

{candidates}

### KITTI-360 as an in-domain source-validation set

The two new folds put KITTI-360 on the *training* side. Before either target was opened, their
selected checkpoints were scored on the KITTI-360 val drive (a source-validation domain here, so this
is not a held-out number):

{k360_in_domain}

## 3. Primary setting — causal all-past streaming

Scale fixed from the first five anchor frames, every frame integrated exactly once, persistent map
state, every anchor scored with all causal history to that time, no future frame on the prediction
path. Threshold-free first:

{tf_stream}

At each fold's locked source-selected threshold (paired scene-aware bootstrap, Gate 6 units, 10 000
draws, seed 0):

### SemanticKITTI 08 (target of the new SK fold)

{op_stream_sk}

### Occ3D-nuScenes val (target of the new Occ3D fold)

{op_stream_occ}

### KITTI-360 (Gate 8A fold, reused)

{op_stream_k360}

## 4. Matched five-frame setting

`ScaleState` and map state reset for every official clip; exactly the clip's five real frames from
one camera, LingBot run on those five alone (per-clip caches), scale fixed from those five (the pinned
G51-B scalar), each frame integrated once, query and completion after frame five, nothing carried
between clips. Reported separately: **OccAny and our streaming method do not share a temporal input
budget, and the numbers below are not to be read against Section 3.**

{tf_clips}

### SemanticKITTI 08

{op_clips_sk}

### Occ3D-nuScenes val

{op_clips_occ}

### KITTI-360

{op_clips_k360}

## 5. OccAny reference comparison (matched setting only)

The published OccAny five-frame single-camera numbers appear only here. The post-processing columns
apply OccAny's own `apply_majority_pooling` (separate mode) to our predictions, untuned; the
reproduction is checked bit-for-bit against the official function in the OccAny environment. Raw
completion remains the primary result. In geometry-only mode OccAny's default is `use_dilation=True`
— a 3×3×3 max-pool, i.e. a one-voxel dilation — so both that and the true vote are shown.

{occany}

{occany_note}

## 6. Geometry-transfer decision per target

Primary setting:

{geo_stream}

Matched five-frame setting (secondary):

{geo_clips}

{clips_note}

## 7. Semantic-transfer decision per target

Primary setting:

{sem_dec_stream}

### Semantics, primary setting

**SemanticKITTI 08**

{sem_stream_sk}

**Occ3D-nuScenes val**

{sem_stream_occ}

**KITTI-360**

{sem_stream_k360}

### Classwise IoU, primary setting

SemanticKITTI 08:

{cls_sk}

Occ3D-nuScenes val:

{cls_occ}

KITTI-360:

{cls_k360}

### The teacher itself

Future-Trident target accuracy against the benchmark's semantic ground truth on future-visible
GT-occupied voxels (evaluation-only use of semantic labels):

{teacher}

## 8. Runtime, memory and training cost

{cost}

Caches built for the new source `k360_train` (drives 0003/0007/0010, 1 385 stream frames, 1 289
labelled anchors): targets 6 min (201 MB by byte-range), LingBot stream 0.9–2.1 min per drive at
10 FPS, MoGe-B and scale candidates on the same pass, Trident-H ≈ 1.6–1.8 s/frame on two GPUs, samples
≈ 0.5 s each.

## 9. Reproduction

```bash
cd /home/minh/workspace/lingbot-map_fork
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1 GATE8_GPUS="1 2 3"
PY=/home/minh/anaconda3/envs/cu128/bin/python
$PY tools/gate8b/stage0.py
$PY tools/gate8b/fetch_k360_labels.py --drives 0003 0007 0010
tools/gate8b/run_caches.sh
$PY tools/gate8b/build_samples.py --source kitti360 --anchor-stride 10
tools/gate8b/run_all.sh                      # remaining samples, then both folds concurrently
$PY tools/gate8b/eval_target.py --fold kitti360 --setting clips
for D in semantickitti occ3d kitti360; do $PY tools/gate8b/teacher_accuracy.py --dataset $D; done
$PY tools/gate8b/aggregate.py && $PY tools/gate8b/figures.py && $PY tools/gate8b/report.py --write
$PY -m pytest tests/gate6 tests/gate7a tests/gate7b tests/gate8 tests/gate8a tests/gate8b -q
```

Seeds: 0 everywhere (training, crop and source sampling, the bootstrap); random editable baselines use
seeds 0–4 with a per-clip offset. Hashes of every selected checkpoint and every Gate 8B config are in
`artifacts/gate8b/gate8b_manifest.json`; the frozen surface is hashed in
`artifacts/gate8b/stage0_audit.json`.

## 10. Tests

{tests}

## 11. Deviations and failures

{deviations}
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    res = json.load(open(os.path.join(ART, "gate8b_results.json")))
    v, rec = verdict(res)
    tests = open(os.path.join(ART, "test_summary.txt")).read().strip() \
        if os.path.exists(os.path.join(ART, "test_summary.txt")) else "(not recorded)"
    dev = open(os.path.join(REPO_ROOT, "reports", "gate8b", "_deviations.md")).read()
    md = TEMPLATE.format(
        verdict=v, recommend=rec, outcome=res["outcome"].get("outcome"),
        provenance=provenance(res), frozen=frozen_table(res), candidates=candidates_table(res),
        tf_stream=tf_table(res, "stream"), tf_clips=tf_table(res, "clips"),
        op_stream_sk=op_table(res, "stream", "semantickitti"), op_stream_occ=op_table(res, "stream", "occ3d"),
        op_stream_k360=op_table(res, "stream", "kitti360"),
        op_clips_sk=op_table(res, "clips", "semantickitti"), op_clips_occ=op_table(res, "clips", "occ3d"),
        op_clips_k360=op_table(res, "clips", "kitti360"), occany=occany_table(res),
        geo_stream=decision_table(res, "stream"), geo_clips=decision_table(res, "clips"),
        sem_dec_stream=sem_decision_table(res, "stream"),
        sem_stream_sk=sem_table(res, "stream", "semantickitti"), sem_stream_occ=sem_table(res, "stream", "occ3d"),
        sem_stream_k360=sem_table(res, "stream", "kitti360"),
        cls_sk=classwise(res, "semantickitti"), cls_occ=classwise(res, "occ3d"), cls_k360=classwise(res, "kitti360"),
        teacher=teacher_table(res), cost=cost_table(res), tests=tests, deviations=dev,
        k360_in_domain=k360_in_domain(res), clips_note=clips_note(res), occany_note=occany_note(res))
    if a.write:
        open(OUT, "w").write(md); print("wrote", OUT, len(md), "chars")
    else:
        print(md[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
