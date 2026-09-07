#!/usr/bin/env python
"""Generate every numeric table of the Gate-6 report from the artifacts.

The report is written with ``{{PLACEHOLDER}}`` markers; this substitutes generated tables
into them, so no number in the report is transcribed by hand.

    python tools/gate6/report_tables.py --write
"""
from __future__ import annotations

import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gates.scale_gate.config import REPO_ROOT                                   # noqa: E402
from gates.gate6 import vocab                                                   # noqa: E402

ART = os.path.join(REPO_ROOT, "artifacts", "gate6")
TEMPLATE = os.path.join(REPO_ROOT, "reports", "gate6",
                        "_frozen_trident_semantic_lifting.template.md")
OUT = os.path.join(REPO_ROOT, "reports", "gate6", "frozen_trident_semantic_lifting.md")
DS_LABEL = {"semantickitti": "SemanticKITTI 08", "occ3d": "Occ3D-nuScenes",
            "kitti360": "SSCBench-KITTI-360"}
ORDER = ["semantickitti", "occ3d", "kitti360"]


def j(name):
    p = os.path.join(ART, name)
    return json.load(open(p)) if os.path.exists(p) else None


def have(ds):
    return j(f"summary_{ds}.json") is not None and j(f"analysis_{ds}.json") is not None


def f(x, n=4):
    return "—" if x is None else f"{x:.{n}f}"


def t_main():
    rows = ["| benchmark | clips | cond | binary IoU (pooled) | prec (pooled) | rec (pooled) | SSC mIoU (pooled) | TP-cond. acc | TP-bal. recall | coverage miss | naming error | correct |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ds in ORDER:
        if not have(ds):
            continue
        s = j(f"summary_{ds}.json")
        for cond in ("B-R", "B-D"):
            c = s["conditions"][cond]
            d = c["decomposition"]
            t = c["tp_conditioned"]
            b = "**" if cond == "B-D" else ""
            rows.append(
                f"| {DS_LABEL[ds] if cond=='B-R' else ''} | {s['n_clips'] if cond=='B-R' else ''} "
                f"| {b}{cond}{b} | {b}{f(c['binary_iou_pooled'])}{b} "
                f"| {f(c['binary_precision'])} | {f(c['binary_recall'])} "
                f"| {b}{f(c['ssc_miou'])}{b} | {b}{f(t['top1_accuracy'])}{b} "
                f"| {f(t['balanced_recall'])} | {b}{f(d['coverage_miss_fraction'])}{b} "
                f"| {b}{f(d['naming_error_fraction'])}{b} | {f(d['correct_fraction'])} |")
    return "\n".join(rows)


def t_decomposition():
    rows = ["| benchmark | cond | valid GT occupied voxels | coverage miss | naming error | correct | miss : naming |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for ds in ORDER:
        if not have(ds):
            continue
        s = j(f"summary_{ds}.json")
        for cond in ("B-R", "B-D"):
            d = s["conditions"][cond]["decomposition"]
            r = (d["coverage_miss_fraction"] / d["naming_error_fraction"]
                 if d["naming_error_fraction"] else float("inf"))
            b = "**" if cond == "B-D" else ""
            rows.append(f"| {DS_LABEL[ds] if cond=='B-R' else ''} | {b}{cond}{b} "
                        f"| {d['n_valid_gt_occupied']:,} "
                        f"| {b}{d['coverage_miss_fraction']:.4f}{b} "
                        f"| {b}{d['naming_error_fraction']:.4f}{b} "
                        f"| {d['correct_fraction']:.4f} | {r:.1f} : 1 |")
    return "\n".join(rows)


def t_sanity():
    pinned = {("semantickitti", "B-D"): 0.1584, ("occ3d", "B-D"): 0.2070,
              ("kitti360", "B-D"): 0.1342, ("kitti360", "B-R"): 0.0376}
    rows = ["| benchmark | condition | frozen value (mean-per-clip) | Gate 6 (mean-per-clip) | \\|Δ\\| | within 0.0005 | Gate 6 (pooled, for reference) |",
            "|---|---|---:|---:|---:|---|---:|"]
    for (ds, cond), exp in sorted(pinned.items(), key=lambda kv: ORDER.index(kv[0][0])):
        if not have(ds):
            rows.append(f"| {DS_LABEL[ds]} | {cond} | {exp:.4f} | not produced | — | — | — |")
            continue
        c = j(f"summary_{ds}.json")["conditions"][cond]
        got = c["binary_iou_mean_per_clip"]
        rows.append(f"| {DS_LABEL[ds]} | {cond} | {exp:.4f} | {got:.4f} | "
                    f"{abs(got-exp):.5f} | {'yes' if abs(got-exp) <= 5e-4 else '**NO**'} "
                    f"| {c['binary_iou_pooled']:.4f} |")
    return "\n".join(rows)


def t_permutation():
    rows = ["| benchmark | metric | real | permutation mean | permutation p95 | permutations beating real | real percentile | passes |",
            "|---|---|---:|---:|---:|---:|---:|---|"]
    for ds in ORDER:
        if not have(ds):
            continue
        pc = j(f"analysis_{ds}.json")["permutation_control"]["B-D"]
        for m, lab in (("ssc_miou", "full SSC mIoU"),
                       ("tp_balanced_recall", "TP-conditioned balanced recall")):
            d = pc[m]
            rows.append(f"| {DS_LABEL[ds] if m=='ssc_miou' else ''} | {lab} | "
                        f"**{d['real']:.4f}** | {d['permutation_mean']:.4f} | "
                        f"{d['permutation_p95']:.4f} | {d['n_permutations_beating_real']} / 100 | "
                        f"{d['real_percentile']:.0f} | "
                        f"{'yes' if d['real_exceeds_p95'] else '**no**'} |")
    return "\n".join(rows)


def t_bootstrap():
    rows = ["| benchmark | metric | B-R | B-D | B-D − B-R | 95% CI | excludes 0 |",
            "|---|---|---:|---:|---:|---|---|"]
    names = {"ssc_miou": "full SSC mIoU", "tp_accuracy": "TP-conditioned accuracy",
             "coverage_miss_fraction": "coverage-miss fraction",
             "naming_error_fraction": "naming-error fraction"}
    for ds in ORDER:
        if not have(ds):
            continue
        a = j(f"analysis_{ds}.json")
        first = True
        for m, lab in names.items():
            b = a["bootstrap"][m]
            p = b["paired_difference"]
            rows.append(f"| {DS_LABEL[ds] if first else ''} | {lab} | {b['point']['B-R']:.4f} "
                        f"| {b['point']['B-D']:.4f} | {p['point']:+.4f} "
                        f"| [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}] "
                        f"| {'yes' if p['excludes_zero'] else 'no'} |")
            first = False
    return "\n".join(rows)


def t_bootstrap_zero_note():
    """Generated, not asserted: which paired differences actually exclude zero."""
    names = {"ssc_miou": "full SSC mIoU", "tp_accuracy": "TP-conditioned accuracy",
             "coverage_miss_fraction": "coverage-miss fraction",
             "naming_error_fraction": "naming-error fraction"}
    incl = []
    n_tot = 0
    for ds in ORDER:
        if not have(ds):
            continue
        a = j(f"analysis_{ds}.json")
        for m, lab in names.items():
            b = a["bootstrap"][m]["paired_difference"]
            n_tot += 1
            if not b["excludes_zero"]:
                incl.append((DS_LABEL[ds], lab, b["ci"]))
    if not incl:
        return f"All {n_tot} reported differences exclude zero."
    parts = ", ".join(f"{d} {lab} ([{ci[0]:+.4f}, {ci[1]:+.4f}])" for d, lab, ci in incl)
    only = "" if len(incl) > 1 else ""
    return (f"All reported differences except {parts} exclude zero "
            f"({n_tot - len(incl)} of {n_tot}).")


def t_near_range_note():
    """The 0-10 m band of each benchmark, generated from the count blocks."""
    rows = ["| benchmark | 0–10 m valid GT occupied | coverage miss | naming error | correct | larger at 0–10 m |",
            "|---|---:|---:|---:|---:|---|"]
    for ds in ORDER:
        if not have(ds):
            continue
        d = j(f"summary_{ds}.json")["conditions"]["B-D"]["by_distance"]["0-10m"]
        tot = d["coverage_miss"] + d["naming_error"] + d["correct"]
        if tot == 0:
            continue
        mi, na, co = (d["coverage_miss"] / tot, d["naming_error"] / tot, d["correct"] / tot)
        which = "**naming**" if na > mi else ("coverage" if mi > na else "tied")
        rows.append(f"| {DS_LABEL[ds]} | {tot:,} | {mi:.4f} | {na:.4f} | {co:.4f} "
                    f"| {which} |")
    return "\n".join(rows)


def t_support():
    rows = ["| benchmark | population | voxels | top-1 accuracy | balanced recall |",
            "|---|---|---:|---:|---:|"]
    for ds in ORDER:
        if not have(ds):
            continue
        c = j(f"summary_{ds}.json")["conditions"]["B-D"]
        for key, lab in (("tp_conditioned_reconstruction_support", "reconstruction support"),
                         ("tp_conditioned_dilation_only", "dilation-only"),
                         ("tp_conditioned", "all co-occupied")):
            d = c[key]
            rows.append(f"| {DS_LABEL[ds] if lab=='reconstruction support' else ''} | {lab} "
                        f"| {d['n']:,} | {f(d['top1_accuracy'])} | {f(d['balanced_recall'])} |")
    return "\n".join(rows)


def t_distance():
    rows = ["| benchmark | band | coverage miss | naming error | correct | semantic mIoU |",
            "|---|---|---:|---:|---:|---:|"]
    for ds in ORDER:
        if not have(ds):
            continue
        c = j(f"summary_{ds}.json")["conditions"]["B-D"]
        first = True
        for band, b in c["by_distance"].items():
            n = b["coverage_miss"] + b["naming_error"] + b["correct"]
            if not n:
                continue
            rows.append(f"| {DS_LABEL[ds] if first else ''} | {band} "
                        f"| {b['coverage_miss']/n:.3f} | {b['naming_error']/n:.3f} "
                        f"| {b['correct']/n:.3f} | {c['miou_by_distance'][band]:.4f} |")
            first = False
    return "\n".join(rows)


def t_height():
    rows = ["| benchmark | height band (m) | coverage miss | naming error | correct |",
            "|---|---|---:|---:|---:|"]
    for ds in ORDER:
        if not have(ds):
            continue
        c = j(f"summary_{ds}.json")["conditions"]["B-D"]
        first = True
        for band, b in c["by_height"].items():
            n = b["coverage_miss"] + b["naming_error"] + b["correct"]
            if not n:
                continue
            lo, hi = band[1:].split("_")
            rows.append(f"| {DS_LABEL[ds] if first else ''} | [{lo}, {hi}) "
                        f"| {b['coverage_miss']/n:.3f} | {b['naming_error']/n:.3f} "
                        f"| {b['correct']/n:.3f} |")
            first = False
    return "\n".join(rows)


def t_per_class(ds):
    if not have(ds):
        return "_not produced_"
    c = j(f"summary_{ds}.json")["conditions"]["B-D"]
    rows = ["| class | SSC IoU | TP-cond. recall | GT occupied voxels | co-occupied voxels |",
            "|---|---:|---:|---:|---:|"]
    for n, d in c["per_class"].items():
        rows.append(f"| {n} | {f(d['iou'])} | {f(d['tp_conditioned_recall'])} "
                    f"| {d['gt_voxels']:,} | {d['tp_conditioned_gt_voxels']:,} |")
    return "\n".join(rows)


def t_confusion(ds, k=8):
    p = os.path.join(ART, f"counts_{ds}_B-D.npz")
    if not os.path.exists(p):
        return "_not produced_"
    z = np.load(p, allow_pickle=False)
    names = [str(x) for x in z["names"]]
    M = z["conf"].astype(np.int64).sum(0)
    rows_ = M.sum(1)
    out = ["| ground-truth class | co-occupied voxels | most frequent predictions |",
           "|---|---:|---|"]
    for i in np.argsort(-rows_)[:k]:
        o = np.argsort(-M[i])[:3]
        s = ", ".join(f"{names[jj]} {100*M[i,jj]/rows_[i]:.1f}%" for jj in o if M[i, jj] > 0)
        out.append(f"| {names[i]} | {rows_[i]:,} | {s} |")
    return "\n".join(out)


def t_provenance():
    p = j("teacher_provenance.json")
    if p is None:
        return "_not produced_"
    r = ["| component | identity | parameters | licence | supervision | SHA-256 |",
         "|---|---|---:|---|---|---|"]
    r.append(f"| repository | Trident `{p['trident']['commit'][:12]}` "
             f"({p['trident']['commit_date'][:10]}) | — | {p['trident']['license']} "
             f"| training-free framework | `trident.py` "
             f"{p['source_file_hashes']['trident.py'][:12]} |")
    r.append(f"| semantic encoder | OpenCLIP {p['clip']['model_type']} "
             f"`{p['clip']['pretrained_tag']}` | {p['clip']['n_params']:,} "
             f"| {p['clip']['license']} | {p['clip']['supervision']} "
             f"| {p['clip']['weight_sha256'][:12]} |")
    r.append(f"| spatial encoder | **DINO v1** {p['vfm']['model']} "
             f"(ViT-B/16, patch {p['vfm']['patch_size']}) — *not DINOv2* "
             f"| {p['vfm']['n_params']:,} | {p['vfm']['license']} "
             f"| {p['vfm']['supervision']} | torch.hub `{p['vfm']['hub']}` |")
    r.append(f"| mask refinement | SAM {p['sam']['model_type']} "
             f"| {p['sam']['n_params']:,} | {p['sam']['license']} "
             f"| {p['sam']['supervision']} | {p['sam']['weight_sha256'][:12]} |")
    r.append(f"| **total** | | **{p['total_params']:,}** | | all frozen | |")
    return "\n".join(r)


def t_counts():
    r = ["| benchmark | clips | unique RGB frames | classes | prediction grid | prediction rollup SHA-256 |",
         "|---|---:|---:|---:|---|---|"]
    from gates.gate6 import frames as F
    for ds in ORDER:
        m = j(f"prediction_manifest_{ds}.json")
        v = vocab.load(ds)
        n_fr = len(F.unique_frames(ds, REPO_ROOT))
        if m is None:
            r.append(f"| {DS_LABEL[ds]} | — | {n_fr} | {len(v)} | — | not produced |")
            continue
        r.append(f"| {DS_LABEL[ds]} | {m['n_clips']} | {n_fr} | {len(v)} "
                 f"| {m['grid']} {tuple(m['dims'])} | `{m['rollup_sha256'][:16]}…` |")
    return "\n".join(r)


def t_audit():
    r = ["| benchmark | file opens observed | unique paths | forbidden accesses | prediction pinned before targets |",
         "|---|---:|---:|---:|---|"]
    for ds in ORDER:
        m = j(f"prediction_manifest_{ds}.json")
        if m is None:
            r.append(f"| {DS_LABEL[ds]} | — | — | — | — |")
            continue
        mp = os.path.join(ART, f"prediction_manifest_{ds}.json")
        sp = os.path.join(ART, f"summary_{ds}.json")
        ok = os.path.exists(sp) and os.path.getmtime(mp) <= os.path.getmtime(sp)
        r.append(f"| {DS_LABEL[ds]} | {m['audit']['n_opened']:,} | {m['audit']['n_unique']:,} "
                 f"| **{len(m['audit']['violations'])}** | {'yes' if ok else 'n/a'} |")
    return "\n".join(r)


def t_decision():
    d = j("diagnosis.json")
    if d is None:
        return "_not produced_"
    r = ["| benchmark | 1. mIoU > perm p95 | 2. TP-bal > perm p95 | 3. target-independent | 4. mapping/eval tests | passes |",
         "|---|---|---|---|---|---|"]
    tick = lambda b: "yes" if b else "**no**"
    for ds in ORDER:
        if ds not in d["datasets"]:
            continue
        c = d["datasets"][ds]["decision"]
        r.append(f"| {DS_LABEL[ds]} | {tick(c['criterion_1_miou_exceeds_p95'])} "
                 f"| {tick(c['criterion_2_tp_balanced_exceeds_p95'])} "
                 f"| {tick(c['criterion_3_prediction_target_independent'])} "
                 f"| {tick(c['criterion_4_mapping_and_evaluation_tests'])} "
                 f"| **{tick(c['dataset_passes'])}** |")
    return "\n".join(r)


def t_fp16():
    d = j("fp16_fidelity_kitti360.json")
    if d is None:
        return "_not produced_"
    return (f"| quantity | value |\n|---|---:|\n"
            f"| clips re-run in float32 | {d['n_clips']} |\n"
            f"| max abs probability error | {d['max_abs_prob_err']:.2e} |\n"
            f"| raw voxels compared | {d['total_raw_voxels']:,} |\n"
            f"| raw label disagreements | {d['total_raw_disagreements']} "
            f"({d['raw_disagreement_fraction']:.2e}) |\n"
            f"| dilated voxels compared | {d['total_dil_voxels']:,} |\n"
            f"| dilated label disagreements | {d['total_dil_disagreements']} "
            f"({d['dil_disagreement_fraction']:.2e}) |\n"
            f"| predicted occupancy identical | {'yes' if d['occupancy_identical'] else 'NO'} |")


def t_parity():
    d = j("trident_parity.json")
    if d is None:
        return "_not produced_"
    s = d["summary"]
    return (f"| quantity | value |\n|---|---:|\n"
            f"| frames tested | {s['n_frames']} across {s['n_datasets']} benchmarks |\n"
            f"| pixels compared | {s['total_pixels']:,} |\n"
            f"| pixels where argmax ≠ official label | **{s['total_tie_pixels']}** |\n"
            f"| frames with any disagreement | {s['frames_with_any_tie']} |\n"
            f"| frames re-run with the recorder removed | {s['n_pristine_checked']} |\n"
            f"| those bit-identical to the recorded run | "
            f"{'all' if s['pristine_all_identical'] else 'NOT ALL'} |\n"
            f"| max \\|Σ probabilities − 1\\| | {s['max_prob_sum_err']:.2e} |")


def t_figures():
    r = ["| benchmark | percentile of frozen B-D IoU | clip | frozen B-D binary IoU |",
         "|---|---|---|---:|"]
    for ds in ORDER:
        p = j(f"figure_selection_{ds}.json")
        if p is None:
            continue
        for i, s in enumerate(p["selected"]):
            r.append(f"| {DS_LABEL[ds] if i==0 else ''} | p{s['percentile']} "
                     f"| `{s['clip_id']}` | {s['frozen_bd_iou']:.3f} |")
    return "\n".join(r)


def t_occany_section():
    d = j("occany_comparator.json")
    if d is None:
        return ("**Not produced.** See the blocker record in §22.")
    c, t = d["comparator"], d["primary"]
    def row(s, name):
        dd = s["decomposition"]
        return (f"| {name} | {s['binary_iou_mean_per_clip']:.4f} | {s['ssc_miou']:.4f} "
                f"| {s['tp_conditioned']['top1_accuracy']:.4f} "
                f"| {s['tp_conditioned']['balanced_recall']:.4f} "
                f"| {dd['coverage_miss_fraction']:.4f} | {dd['naming_error_fraction']:.4f} "
                f"| {s['named_fraction_of_frozen_occupancy']*100:.1f} % |")
    tbl = ("| 2D readout | binary IoU (mean-per-clip) | SSC mIoU (pooled) | TP-cond. acc | TP-bal. recall "
           "| coverage miss | naming error | frozen voxels named |\n"
           "|---|---:|---:|---:|---:|---:|---:|---:|\n"
           + row(t, "**Trident-H** (primary, predeclared)") + "\n"
           + row(c, "OccAny-aligned Grounded-SAM-2 (comparator)"))
    return f"""Reproduced on **SemanticKITTI sequence 08** ({d['n_clips']} clips), the
smallest of the three benchmarks. **Both readouts are lifted through the identical frozen
geometry, on the identical clips, and scored by the identical evaluator and mask**, so the
only thing that differs is the 2D semantic readout. This was run *after* the primary
predictions were SHA-256-pinned and was not used to select, tune or modify anything.

**Read the table with three limits attached, stated before the numbers rather than after
them.** This is a **one-dataset** comparison — SemanticKITTI sequence 08 only, one drive,
{d['n_clips']} clips. **No paired confidence interval was computed for it**, so no
difference in the table is known to be distinguishable from resampling noise; every other
comparison in this report carries a 10,000-resample paired interval and this one does not.
And the two readouts were **not given the same vocabulary**: OccAny prompts with per-class
**synonym lists** while the Gate-6 protocol mandated a **single deterministic phrase per
class**, so the comparison confounds the choice of teacher with the choice of prompt. The
row below is a measurement under our protocol, not a ranking of the two teachers.

{tbl}

**The comparator is more accurate where it names, and names less.** Grounded-SAM-2 is
detection-driven: Grounding DINO proposes a median of
{d['grounding_dino_boxes_per_frame_median']:.0f} boxes per frame, SAM 2.1 turns them into
masks, and OccAny paints the highest-confidence masks first without overwriting. That
covers a median {100*d['pixels_named_per_frame_median']:.1f} % of pixels and
{c['named_fraction_of_frozen_occupancy']*100:.1f} % of the frozen occupied voxels; the rest
are left unnamed and are scored as empty, which is why its binary IoU
({c['binary_iou_mean_per_clip']:.4f}) sits below the frozen {t['binary_iou_mean_per_clip']:.4f}.
Where it does commit to a class it is right
{c['tp_conditioned']['top1_accuracy']*100:.1f} % of the time against Trident's
{t['tp_conditioned']['top1_accuracy']*100:.1f} %, and its full SSC mIoU is correspondingly
higher ({c['ssc_miou']:.4f} vs {t['ssc_miou']:.4f}).

This result does **not** favour the predeclared primary, and it is reported as measured.
Three things must be said with it, none of which are excuses and all of which are
verifiable in the code:

* **The two readouts were not given the same vocabulary.** The Gate-6 protocol mandated
  *one deterministic phrase per class, no synonyms, no alternatives evaluated* — so Trident
  ran on `bicycle`, `vegetation`, `terrain`. OccAny's own prompts are per-class **synonym
  lists**: `bicycle; bike`, `vegetation; bush; shrub; foliage`, `terrain; grass; soil`,
  `motorcycle; motorbike; scooter`. Those synonyms attack exactly the ontology-boundary
  classes §11 identifies as the dominant naming failure. The gap therefore confounds
  "Grounded-SAM vs Trident" with "synonym lists vs single phrases", and this experiment
  cannot separate them.
* **The two are not doing the same job.** A detector-driven readout answers "what did I
  detect and where"; Trident answers "which of these classes is this pixel" for every
  pixel. Declining to name 6.5 % of the occupied voxels is not free — those voxels become
  coverage misses, which is why the comparator's coverage-miss fraction
  ({c['decomposition']['coverage_miss_fraction']:.4f}) is *higher* than Trident's
  ({t['decomposition']['coverage_miss_fraction']:.4f}) even though its naming error is
  lower.
* **It is one benchmark, and it carries no interval.** SemanticKITTI 08 only, one drive,
  8 bootstrap blocks, and **no paired confidence interval was computed for the
  comparator** — so none of the differences above is known to exceed resampling noise.
  Producing one would require re-running the comparator's per-clip count blocks through
  the Gate-6 bootstrap, which was not done.

**The bottleneck diagnosis is unchanged and is now teacher-independent.** For the
comparator too, coverage miss ({c['decomposition']['coverage_miss_fraction']:.4f}) exceeds
naming error ({c['decomposition']['naming_error_fraction']:.4f}) by roughly ten to one.
Swapping the frozen 2D teacher for a stronger-when-it-commits alternative moves the SSC
mIoU by {c['ssc_miou']-t['ssc_miou']:+.4f} and leaves seven of every ten occupied voxels
still missing. That is the most useful thing the comparator says, and it argues the
recommended next experiment (§18) even more strongly than the primary result alone does.

**No claim of superiority over OccAny, in either direction, is made.** OccAny's
reconstruction, grid, evaluation mask and protocol are not reproduced here — only its
semantic-mask procedure, transplanted onto our frozen occupancy. The row above compares two
*2D readouts under our protocol*, nothing more."""


def t_occany_deviations():
    d = j("occany_comparator.json")
    if d is None:
        return """**C1 — the comparator was not produced.** See the blocker record above."""
    return """**C1 — SAM 3 is gated; the Grounded-SAM-2 branch was used.** OccAny's newer
semantic path loads `facebook/sam3`, a **manually gated** Hugging Face repository. No
credentials are configured on this machine and the brief forbids bypassing authentication
or licensing, so that branch is unavailable. The brief names Grounded-SAM2 as an accepted
alternative and that is what ran.

**C2 — OccAny's `occany/model/model_sam2.py` cannot be imported as released.** Line 26 does
`from PIL import Image` (binding the *module*) while line 521 writes
`Union[np.ndarray, Image]` and line 537 `isinstance(image, Image)`, both of which require
the *class* — which is what upstream sam2 binds (`from PIL.Image import Image`). Under
Python 3.10 the module raises `TypeError` on import. Binding the class is the only reading
under which the file runs, so it is a repair with no methodological content. **The upstream
file was not edited**: everything `model_sam2` imports before line 26 is pre-warmed, a shim
of the `PIL` package whose `Image` is the class is installed for the duration of that single
import, and it is removed immediately afterwards so no other module ever sees it.

**C3 — OccAny's vendored dust3r requires Python ≥ 3.11.** `occany/datasets/__init__.py`
unconditionally re-exports dust3r, whose `datasets/base/batched_sampler.py:290` contains
`np.c_[sample_idxs, *idxs]` — a starred expression inside a subscript, a **SyntaxError**
before Python 3.11. The entire frozen Gate-1..6 stack runs on Python 3.10, so that package
cannot be imported here at all. Only a literal constant was needed from it
(`KITTI_CLASS_PROMPTS`), so it is parsed straight out of the source with
`ast.literal_eval` — byte-identical data, no import, no edit.

**C4 — GroundingDINO's CUDA extension had to be rebuilt for this GPU.** Its `setup.py`
hardcodes `-gencode` for sm_70/75/80/86 with no PTX and ignores `TORCH_CUDA_ARCH_LIST`, so
on these sm_120 Blackwell cards the kernel fails at runtime with
`no kernel image is available for execution on the device` — and it fails *silently*,
printing to stderr while returning garbage, because `ms_deform_attn.py:331` takes the CUDA
branch whenever the tensor is on CUDA and the PyTorch reference path is unreachable. Three
`-gencode` lines for `compute_120` were added to that vendored `setup.py`, the extension
rebuilt, and **the file restored to its original contents**; only the compiled `.so`
differs from a stock checkout. This is a build-configuration change, not a method change.

**C5 — full-extent resize instead of OccAny's principal-point crop.** OccAny crops around
the principal point before rescaling, a crop tied to its own reconstruction. Using it would
misalign the readout with the frozen LingBot lattice this gate is required to keep, so the
native image is resized full-extent to OccAny's inference resolution (long side 1216)
instead. This is also the fairer comparator setup: both teachers then see the same pixels.
Everything else — prompts, `box_threshold = 0.1`, `text_threshold = 0.0`, SAM 2.1 Hiera-L,
`infer_semantic` itself and its confidence-ordered non-overwriting overlap rule — is
OccAny's own code, called unchanged.

**C6 — hard labels, so the voxel rule is a majority vote.** The readout emits integer
labels, not scores. Writing each pixel as a one-hot vector makes the frozen fusion (mean,
then argmax) exactly a majority vote among contributing pixels, so the identical lifting
code is reused. An extra channel represents "the teacher detected nothing here", making
*unnamed* a legitimate outcome rather than an arbitrary fallback class; those voxels are
scored as empty and their rate is reported explicitly above."""


TABLES = {
    "BOOTSTRAP_ZERO_NOTE": t_bootstrap_zero_note,
    "TABLE_NEAR_RANGE": t_near_range_note,
    "OCCANY_SECTION": t_occany_section, "OCCANY_DEVIATIONS": t_occany_deviations,
    "ART_SIZE": lambda: _art_size(),
    "TABLE_MAIN": t_main, "TABLE_SANITY": t_sanity,
    "TABLE_DECOMPOSITION": t_decomposition, "TABLE_PERMUTATION": t_permutation,
    "TABLE_BOOTSTRAP": t_bootstrap, "TABLE_SUPPORT": t_support,
    "TABLE_DISTANCE": t_distance, "TABLE_HEIGHT": t_height,
    "TABLE_PROVENANCE": t_provenance, "TABLE_COUNTS": t_counts, "TABLE_AUDIT": t_audit,
    "TABLE_DECISION": t_decision, "TABLE_FP16": t_fp16, "TABLE_PARITY": t_parity,
    "TABLE_FIGURES": t_figures,
    "TABLE_PERCLASS_SEMANTICKITTI": lambda: t_per_class("semantickitti"),
    "TABLE_PERCLASS_OCC3D": lambda: t_per_class("occ3d"),
    "TABLE_PERCLASS_KITTI360": lambda: t_per_class("kitti360"),
    "TABLE_CONFUSION_SEMANTICKITTI": lambda: t_confusion("semantickitti"),
    "TABLE_CONFUSION_OCC3D": lambda: t_confusion("occ3d"),
    "TABLE_CONFUSION_KITTI360": lambda: t_confusion("kitti360"),
}


def _art_size():
    tot = 0
    for dp, _dn, fn in os.walk(ART):
        for f in fn:
            tot += os.path.getsize(os.path.join(dp, f))
    return f"{tot / 2**20:.0f} MB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(TEMPLATE):
        print(f"template missing: {TEMPLATE}")
        return 1
    text = open(TEMPLATE).read()
    missing = []
    for key, fn in TABLES.items():
        tok = "{{" + key + "}}"
        if tok in text:
            text = text.replace(tok, fn())
    import re
    left = re.findall(r"\{\{([A-Z_0-9]+)\}\}", text)
    if left:
        missing = sorted(set(left))
    if a.write:
        with open(OUT, "w") as fh:
            fh.write(text)
        print(f"wrote {OUT} ({len(text)} chars)")
    if missing:
        print("UNSUBSTITUTED:", missing)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
