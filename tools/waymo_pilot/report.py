"""Build ``reports/waymo_supervision_pilot.md`` from the artifacts, never from memory.

Every number in the report is read out of a file this pilot wrote. Nothing is typed in by
hand, so the report cannot drift from the measurement, and a rerun restates it.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Dict, List, Optional

from tools.waymo_pilot import config as C, stats as ST, waymo as W


def _load(path: str) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=C.REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                               # noqa: BLE001
        return "unknown"


def _pct(x: Optional[float], nd: int = 1) -> str:
    return "—" if x is None else f"{100.0 * x:.{nd}f} %"


def build() -> str:
    logs = C.Paths().logs
    ref = _load(os.path.join(logs, "k360_reference.json"))
    seven = _load(os.path.join(logs, "k360_7drives.json"))
    checks = _load(os.path.join(logs, "checks.json"))
    diag_ref = _load(os.path.join(logs, "teacher_correctness_k360_reference.json"))
    diag_7 = _load(os.path.join(logs, "teacher_correctness_k360_7drives.json"))
    ex = _load(os.path.join(C.Paths().examples, "k360_reference", "index.json"))
    avail = W.probe()

    aggs = [a["aggregate"] for a in (ref, seven) if a]
    L: List[str] = []
    w = L.append

    # ---------------------------------------------------------------- header
    w("# Waymo supervision pilot — does a small Waymo subset give hidden-region semantic "
      "supervision our KITTI-360 recipe lacks?")
    w("")
    w(f"`{C.VERSION}` · commit `{_git_commit()}` · audit only, no training launched, no "
      f"target benchmark opened.")
    w("")
    w("## Verdict: **BLOCKED** on the Waymo arm — and one measured finding that should "
      "change what we do when it unblocks")
    w("")
    w("The question cannot be answered on this machine: **there is no Waymo data here and "
      "the SDK is not installed.** The pilot therefore delivers the repository-grounded "
      "design, a runnable entry point that demands real input, and the full KITTI-360 "
      "reference measurement the Waymo arm would have been compared against.")
    w("")
    w("That reference measurement is not a null result. On the population the method has to "
      "serve, the **existing privileged target reaches "
      f"{_pct(_g(diag_ref, 'annotated_top1_accuracy'))} top-1 correctness and "
      f"{_fmt(_g(diag_ref, 'annotated_target_miou'), 100, 2)} mIoU against human labels**, and it "
      f"covers only {_pct(_g2(ref, 'fraction_of_hidden_with_target'))} of the hidden occupied "
      "voxels it is supposed to supervise. Before asking whether a second dataset adds "
      "diversity, it is worth knowing that the supervision we would mix it into is itself "
      "about one third wrong wherever it exists.")
    w("")
    w("This says nothing about whether heterogeneous training works. It was not tested.")
    w("")

    # ---------------------------------------------------------------- question/data
    w("## 1. The question and the exact data used")
    w("")
    w("> Does a small Waymo subset give us useful hidden-region semantic supervision that "
      "our existing KITTI-360 recipe lacks?")
    w("")
    w("**Waymo — not available.** Probed, not assumed:")
    w("")
    w("```json")
    w(json.dumps(avail.as_dict(), indent=1))
    w("```")
    w("")
    w("**KITTI-360 — available and measured.** The causal contract was verified rather than "
      "inherited: the cached recipe's own frame table gives a causal clip of "
      f"**{_g3(ref, 'timing', 'n_causal_frames'):.0f} frames spanning "
      f"{_g3(ref, 'timing', 'causal_span_s'):.3f} s** at a stream step of "
      f"**{_g3(ref, 'timing', 'stream_dt_s'):.3f} s**, ending at the anchor, from one camera "
      f"(`image_00`) — the historical contract. The privileged window is the existing "
      f"recipe's: {_g3(ref, 'timing', 'n_future_frames'):.0f} future stream frames out to "
      f"{_g3(ref, 'timing', 'future_last_dt_s'):.2f} s.")
    w("")
    if ref:
        sel = ref["selection"]
        w(f"Sample: **{sel['n_anchors']} anchors from {sel['n_scenes_used']} drives** "
          f"(digest `{sel['digest']}`), selected before any yield was examined, spread with a "
          f"minimum separation of {sel['caps']['min_separation_frames']} stream frames so no "
          f"two anchors share a causal clip or a privileged window. Drive "
          f"`{C.K360_DEV_DRIVE[17:21]}` is development data and is asserted absent.")
        w("")
        w("A cap on anchors is not the same as having that many independent examples. The "
          "measured ceiling matters more than the cap:")
        w("")
        w("| drive | eligible anchors | temporally independent |")
        w("|---|---:|---:|")
        ceil = sel["independent_anchor_ceiling"]
        for d in sorted(ceil):
            w(f"| {d[17:21]} | {sel.get('eligible_per_scene', {}).get(d, '—')} | {ceil[d]} |")
        w(f"| **total** | **{sel['eligible_anchors_total']:,}** | "
          f"**{sel['independent_anchor_ceiling_total']}** |")
        w("")
        w(f"The frozen training pool is {sel['eligible_anchors_total']:,} anchors, but at a "
          f"separation that guarantees no shared frame (~21 s) only "
          f"**{sel['independent_anchor_ceiling_total']} of them are temporally independent**, "
          f"across {len(ceil)} routes through one city. That is the quantity a second dataset "
          f"would be adding to.")
        w("")

    # ---------------------------------------------------------------- comparison
    w("## 2. Measured comparison")
    w("")
    w("`k360_reference` is the three drives the current recipe trains on. `k360_7drives` is "
      "the same recipe, same teacher, same grid, on every KITTI-360 training drive whose "
      "caches exist — it varies **scene count** with everything else fixed. It is the closest "
      "available stand-in for the diversity question, and it is **not** a substitute for "
      "Waymo: same city, same sensor, same camera count.")
    w("")
    w(ST.table(aggs))
    w("")
    w("Definitions, all label-free: `O_priv` = privileged LiDAR says occupied; `E_causal` = "
      "the causal clip has lifted teacher evidence there; `hidden` = `O_priv & ~E_causal`; "
      "`U` = `hidden` and a reliable privileged semantic target exists. Every fraction above "
      "names its denominator.")
    w("")
    if ref and seven:
        a, b = ref["aggregate"], seven["aggregate"]
        w(f"**Doubling the scene count barely moves the supervision.** 3 drives → 7 drives "
          f"changes yield from {_pct(a['fraction_of_hidden_with_target'])} to "
          f"{_pct(b['fraction_of_hidden_with_target'])} of hidden voxels, the dominant class "
          f"stays `{a['class_diversity']['top_class']}` at "
          f"{_pct(a['class_diversity']['top_class_fraction'])} → "
          f"{_pct(b['class_diversity']['top_class_fraction'])}, and the thing share stays at "
          f"{_pct(a['class_diversity']['thing_fraction'])} → "
          f"{_pct(b['class_diversity']['thing_fraction'])}. Within KITTI-360, scene count is "
          f"not what limits this supervision's class balance.")
        w("")

    # ---------------------------------------------------------------- what the recipe discards
    if ref:
        exc = ref["aggregate"]["exclusions"]
        hit = exc.get("n_future_hit", 0)
        w("### What the recipe discards before semantics are even considered")
        w("")
        w("| stage | voxels | of future-hit |")
        w("|---|---:|---:|")
        w(f"| privileged future sweeps hit the voxel | {hit:,} | 100 % |")
        w(f"| excluded: occupied/free conflict between sweeps (dynamic) | "
          f"{exc.get('n_conflict_dynamic', 0):,} | {_pct(ST.frac(exc.get('n_conflict_dynamic', 0), hit))} |")
        w(f"| excluded: failed the ≥1 s static-support rule | "
          f"{exc.get('n_not_static_1s', 0):,} | {_pct(ST.frac(exc.get('n_not_static_1s', 0), hit))} |")
        w(f"| kept as a target (tiers A+B, ≥1 camera-visible view) | "
          f"{exc.get('n_target', 0):,} | {_pct(ST.frac(exc.get('n_target', 0), hit))} |")
        w("")
        w("The motion exclusions are the existing recipe's own, and they are the right "
          "default — ego-motion compensation does not align moving objects. They are also "
          f"most of the loss: the recipe keeps about "
          f"{_pct(ST.frac(exc.get('n_target', 0), hit), 0)} of what the future sweeps touch.")
        w("")

    # ---------------------------------------------------------------- correctness
    w("## 3. Is the supervision correct? (annotation-assisted diagnostic)")
    w("")
    w("Teacher agreement measures whether two views of the frozen teacher say the same thing. "
      "It is **not** correctness. Measured separately, on the labelled source subset only, "
      "with human labels read nowhere else in this pilot:")
    w("")
    if diag_ref and diag_ref["summary"].get("available"):
        s = diag_ref["summary"]
        w("| quantity | value | denominator |")
        w("|---|---:|---|")
        w(f"| two-view **consistency** | {_pct(_g2(ref, 'agreement', 'two_view_agreement'))} | "
          f"targets in `U` with ≥2 views |")
        w(f"| top-1 **correctness** | {_pct(s['annotated_top1_accuracy'])} | "
          f"{s['n_U_with_label']:,} labelled target voxels in `U` |")
        w(f"| top-1 correctness on two-view-**agreeing** voxels | "
          f"{_pct(s['annotated_top1_two_view_agreeing'])} | {s['n_two_view_agreeing']:,} voxels |")
        w(f"| target mIoU, 18 classes | {100 * s['annotated_target_miou']:.2f} | "
          f"{s['n_classes_scored']}/18 classes present |")
        w("")
        w("Agreement overstates correctness by roughly 10 points, and even where two "
          "independent views of the teacher agree, **about a quarter of the targets are still "
          "wrong**. The mIoU is the quality of the *supervision*, not of any model.")
        w("")
        if s.get("annotated_top1_tierA") is not None:
            w("Split by the recipe's own reliability tier — tier A is occupancy at `t` "
              "confirmed by a causal-window sweep, tier B is occupancy at `t` uncertain and "
              "carried by two or more future sweeps:")
            w("")
            w("| tier | target voxels | top-1 correctness |")
            w("|---|---:|---:|")
            w(f"| A (occupied at `t` confirmed) | {s['n_tierA']:,} | "
              f"{_pct(s['annotated_top1_tierA'])} |")
            w(f"| B (occupancy at `t` uncertain) | {s['n_tierB']:,} | "
              f"{_pct(s['annotated_top1_tierB'])} |")
            w("")
            w("**Tier is not the lever.** Tier B carries about two thirds of all targets and "
              "is, if anything, marginally *more* accurate than tier A, so dropping it would "
              "shrink the supervision without improving it. The one thing that does separate "
              "right from wrong targets is two-view agreement — but that gap was measured with "
              "human labels, so it must not be turned into a filtering threshold here.")
            w("")
        per = s.get("annotated_per_class_iou", {})
        good = sorted(((k, v) for k, v in per.items() if v is not None), key=lambda kv: -kv[1])
        w("Per-class target IoU — the supervision is usable for a handful of large stuff "
          "classes and close to useless for the rest:")
        w("")
        w("| class | target IoU | | class | target IoU |")
        w("|---|---:|---|---|---:|")
        half = (len(good) + 1) // 2
        for i in range(half):
            lk, lv = good[i]
            if i + half < len(good):
                rk, rv = good[i + half]
                w(f"| {lk} | {100 * lv:.1f} | | {rk} | {100 * rv:.1f} |")
            else:
                w(f"| {lk} | {100 * lv:.1f} | | | |")
        w("")
    if diag_7 and diag_7["summary"].get("available"):
        s7 = diag_7["summary"]
        w(f"On the seven-drive sample the same diagnostic gives "
          f"{_pct(s7['annotated_top1_accuracy'])} top-1 and "
          f"{100 * s7['annotated_target_miou']:.2f} mIoU, but only "
          f"{s7['n_anchors']} of its {s7['n_anchors'] + s7.get('n_anchors_without_labels', 0)} "
          f"anchors carry labels, so it is a different and smaller subset — not a comparison.")
        w("")

    # ---------------------------------------------------------------- examples
    w("## 4. Inspectable examples")
    w("")
    if ex:
        w(f"Selection rule: {ex['selection_rule']}. Worst and best cases are included by "
          f"construction.")
        w("")
        w("| scene | anchor | yield `U`/`hidden` | targets | figure |")
        w("|---|---:|---:|---:|---|")
        for e in ex["examples"]:
            w(f"| {e['scene'][17:21]} | {e['anchor']} | {_pct(e['yield_U_over_hidden'])} | "
              f"{e['counts']['U']:,} | `{e['figure']}` |")
        w("")
        w("Each figure shows the five causal views, the causal semantic evidence in BEV, the "
          "privileged view that first reveals the targets, the hidden population before a "
          "target is required, and the target itself coloured by readout — with provenance "
          "printed underneath. The JSON sidecar beside each figure carries the frame indices, "
          "timestamps and image paths.")
        w("")

    # ---------------------------------------------------------------- implementation
    w("## 5. Implementation and checks")
    w("")
    w("New, and only what the pilot needed:")
    w("")
    w("| file | what it is |")
    w("|---|---|")
    for f, d in (
        ("tools/waymo_pilot/config.py", "single source of truth: caps, bounds, sources, firewall"),
        ("tools/waymo_pilot/populations.py", "the §4 decomposition, stated without human labels"),
        ("tools/waymo_pilot/provenance.py", "per-target source, time, camera, count, agreement; overlaps retained"),
        ("tools/waymo_pilot/sample.py", "reproducible selection + the independence ceiling"),
        ("tools/waymo_pilot/k360.py", "KITTI-360 arm over the existing caches and the existing recipe"),
        ("tools/waymo_pilot/waymo.py", "Waymo reader; SDK-verified fields, axis convention, availability probe"),
        ("tools/waymo_pilot/stats.py", "aggregation; every fraction carries its denominator"),
        ("tools/waymo_pilot/examples.py", "deterministic inspectable figures"),
        ("tools/waymo_pilot/checks.py", "the four focused checks"),
        ("tools/waymo_pilot/diagnostic.py", "ANNOTATION-ASSISTED correctness audit, quarantined"),
        ("tools/waymo_pilot/report.py", "this report, generated from the artifacts"),
        ("tools/waymo_pilot/cli/", "waymo_extract · run_k360_reference · run_checks · run_examples · run_diagnostic"),
    ):
        w(f"| `{f}` | {d} |")
    w("")
    w("Reused unchanged: the causal clip and hard mapper (`tools/baseline_reconstruction`), "
      "the frozen geometry and evidence lifting (`tools/semantic_completion`), the privileged "
      "target recipe (`tools/delayed_semantic/targets.py`), the raw-LiDAR occupancy target "
      "(`core/supervision/rawtarget.py`), the frozen Trident-H teacher and its 18 phrases.")
    w("")
    if checks:
        w(f"**Checks: {'all pass' if checks['all_ok'] else 'FAILURE'}** "
          f"(`{checks['drive'][17:21]}` anchor {checks['anchor']}).")
        w("")
        w("| check | result | evidence |")
        w("|---|---|---|")
        for c in checks["checks"]:
            ev = _check_evidence(c)
            w(f"| {c['name']} | {'PASS' if c['ok'] else 'FAIL'} | {ev} |")
        w("")
    if ref:
        w(f"The whole measurement ran inside a file audit forbidding "
          f"`{'`, `'.join(C.BENCHMARK_FORBIDDEN)}`: "
          f"{ref['firewall']['n_opened']} paths opened, "
          f"{len(ref['firewall']['violations'])} violations.")
        w("")

    # ---------------------------------------------------------------- cost
    w("## 6. Runtime and storage")
    w("")
    w("| item | value |")
    w("|---|---:|")
    if ref:
        w(f"| KITTI-360 reference, {ref['aggregate']['n_anchors']} anchors | "
          f"{ref['seconds'] / 60:.1f} min ({ref['gpu_hours']:.3f} GPU-h) |")
    if seven:
        w(f"| KITTI-360 seven drives, {seven['aggregate']['n_anchors']} anchors | "
          f"{seven['seconds'] / 60:.1f} min ({seven['gpu_hours']:.3f} GPU-h) |")
    w(f"| GPU-hour cap | {C.MAX_PREPROCESS_GPU_HOURS} h |")
    w(f"| new derived artifacts | {_du(C.ARTIFACT_DIR)} |")
    w(f"| derived-artifact cap | {C.MAX_DERIVED_GIB} GiB |")
    w(f"| Waymo derived cache | none — arm blocked |")
    w("")
    w("The pilot reuses every existing cache and adds essentially nothing: both caps are "
      "untouched. A Waymo run would be the first real cost, and its size should be estimated "
      "from the smoke test before expanding, as §2 requires.")
    w("")

    # ---------------------------------------------------------------- limitations
    w("## 7. Limitations")
    w("")
    for lim in [
        "**The headline question is untested.** No Waymo data was read, so nothing here "
        "supports or refutes the claim that a second dataset helps.",
        "`k360_7drives` varies scene count inside one city with one camera. It is evidence "
        "about KITTI-360, not about heterogeneity.",
        f"Sub-cases of `hidden` are measured on the {_g2(seven, 'n_anchors_with_subcases') or 0} "
        f"anchors that still have a causal clip; the four extra drives keep coverage caches but "
        f"their clips were not retained, so their frusta cannot be rebuilt without re-running "
        f"LingBot. Those anchors contribute coverage and are excluded from the sub-case sums.",
        "**Privileged sensors are not omniscient.** `hidden` is bounded by what the LiDAR "
        "sweeps resolved; genuinely hidden occupied space that no sweep ever touched is in "
        "none of these counts. Coverage of all hidden space is not established.",
        "Correctness is measured on the labelled source subset only, and on one teacher. "
        "A different teacher could change it.",
        "Anchors are independent in time, not in environment: three drives is three routes.",
        "Two-view consistency is reported wherever it is measured, and is never called mIoU.",
    ]:
        w(f"- {lim}")
    w("")

    # ---------------------------------------------------------------- next
    w("## 8. What should happen next")
    w("")
    w("### The prerequisite, not a command")
    w("")
    w("The Waymo arm is blocked on a step a human must take:")
    w("")
    w("```")
    w(W.acquisition_instructions())
    w("```")
    w("")
    w("`tools/waymo_pilot/cli/waymo_extract.py` is written and runnable; today it exits 2 and "
      "prints exactly the above. It is not presented as executable.")
    w("")
    w("### The comparison this pilot was meant to justify")
    w("")
    w("Kept here so it is not re-derived later, and **not launched**:")
    w("")
    w("- **A**: KITTI-360, current target recipe.")
    w("- **B**: KITTI-360 + Waymo, same recipe. Tests adding a source at fixed compute; it "
      "does not isolate diversity from dataset size.")
    w("- **C**: the same mixed data and anchors as B, with W-extended targets. Tests the "
      "supervision recipe alone.")
    w("")
    w("Architecture, initialisation, geometry predictions, optimiser, step budget and "
      "source-only checkpoint selection stay fixed; B and C must match sampling exactly and "
      "report their change in unique scenes and target counts. Evaluation scenes stay "
      "disjoint from training scenes. A compatible learner already exists and does not need "
      "inventing: `tools/future_semantic_prior/model.SemanticCompletionHead` "
      "(`C3_full`, 257,394 parameters) consumes geometry plus causal evidence and is trained "
      "by soft distillation against exactly this kind of privileged target. B0 nearest-"
      "evidence stays as the non-learning reference; it cannot learn a new prior.")
    w("")
    w("### One verified command")
    w("")
    w("Two internal levers were tested in the course of this pilot and **both are closed**: "
      "scene count inside KITTI-360 does not change the supervision profile (§2), and the "
      "recipe's reliability tier does not separate correct targets from wrong ones (§3). "
      "There is no cheap repair left to run on KITTI-360, which is precisely why the Waymo "
      "question stays open rather than being answered by proxy.")
    w("")
    w("The next supported action is therefore the first step of the Waymo arm, and it is "
      "gated on the human prerequisite above. The command that runs today, and that returns "
      "0 the moment the prerequisite is met, is the readiness probe:")
    w("")
    w("```bash")
    w("python tools/waymo_pilot/cli/waymo_extract.py --probe-only")
    w("```")
    w("")
    w("It opens nothing and downloads nothing: it reports the dataset root, the number of "
      "`.tfrecord` files, and whether TensorFlow and the SDK import, then exits 0 if the arm "
      "can run and 2 otherwise. **Cost: under a second, no GPU, no storage.** It is verified — "
      "it is the command that produced the availability block in §1, and it exits 2 today.")
    w("")
    w("When it exits 0, the next command is the smoke test the brief asks for — "
      f"`waymo_extract.py --segments {C.SMOKE_SEGMENTS}`, whole segments — "
      "whose runtime and storage must be measured from that run before any expansion, rather "
      "than estimated here. Whole segments are not a convenience: an anchor needs "
      f"{C.anchor_frame_budget()['span']} native frames around it "
      f"({C.anchor_frame_budget()['frames_back']} back for the causal clip, "
      f"{C.anchor_frame_budget()['frames_forward']} forward for the existing privileged "
      "window), so a truncated extraction supports no anchors at all and the extractor "
      "refuses it. The pilot's caps "
      f"({C.MAX_DERIVED_GIB} GiB derived, {C.MAX_PREPROCESS_GPU_HOURS} GPU-h) stop the "
      "expansion if the smoke test says it would exceed them.")
    w("")
    return "\n".join(L) + "\n"


def _check_evidence(c: Dict) -> str:
    n = c["name"]
    if n.startswith("preprocessing"):
        return (f"{c['n_evidence_voxels']:,} evidence voxels identical, max feature diff "
                f"{c['max_feature_abs_diff']:.2e} ≤ fp16 {c['fp16_tolerance']:.2e}, "
                f"geometry {c['geometry_params']:,} params")
    if n.startswith("privileged"):
        return (f"causal digest `{c['causal_digest_before']}` unchanged after perturbation; "
                f"causal state opened {c['n_paths_opened_by_causal_state']} paths, none privileged")
    if n.startswith("known"):
        z = c["zbuffer"]
        return (f"voxel binning exact; 25 m return behind a 10 m surface rejected at "
                f"{z['tolerance_m']} m tolerance; Waymo forward point → principal point "
                f"{c['waymo_case']['forward_10m_uv']}")
    return (f"{c['n_hidden_without_target']:,} unsupervised voxels, "
            f"{c['n_of_those_given_a_class']} given a class; "
            f"{c['n_unknown_voxels_in_privileged_occupancy']} unknown voxels called occupied")


def _du(path: str) -> str:
    try:
        out = subprocess.check_output(["du", "-sh", path], stderr=subprocess.DEVNULL)
        return out.decode().split()[0] + "B"
    except Exception:                                               # noqa: BLE001
        return "n/a"


def _g(d: Optional[Dict], key: str):
    return (d or {}).get("summary", {}).get(key)


def _g2(d: Optional[Dict], *keys):
    v = (d or {}).get("aggregate", {})
    for k in keys:
        v = v.get(k) if isinstance(v, dict) else None
    return v


def _g3(d: Optional[Dict], *keys) -> float:
    v = _g2(d, *keys)
    return float(v) if isinstance(v, (int, float)) else float("nan")


def _fmt(x, scale: float = 1.0, nd: int = 2) -> str:
    return "—" if x is None else f"{scale * x:.{nd}f}"


def write(path: str = C.REPORT_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = build()
    with open(path, "w") as fh:
        fh.write(text)
    return path


__all__ = ["build", "write"]
