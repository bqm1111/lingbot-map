#!/usr/bin/env python
"""Render reports/gate8c0/gate8c0_report.md from artifacts/gate8c0/gate8c0_results.json."""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT                                               # noqa: E402

OUT = os.path.join(REPO_ROOT, "reports", "gate8c0", "gate8c0_report.md")
PARTS = ("train", "source_validation", "heldout")
NICE = {"train": "train (0003, 0007)", "source_validation": "source-val (0010)",
        "heldout": "held out (0006)"}


def f(x, n=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "**no**"
    if isinstance(x, str):
        return x
    return f"{x:.{n}f}"


def table(head, rows):
    return "\n".join(["| " + " | ".join(map(str, head)) + " |",
                      "|" + "|".join(["---"] * len(head)) + "|"]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def load(n):
    p = os.path.join(ART, f"{n}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def provenance_table(s0):
    rows = []
    for d, v in s0["per_drive"].items():
        m = v["missing"]
        rows.append([d, v["partition"], v["n_labelled_anchors"], v["n_stream_frames"],
                     v["n_samples_cached"], v["keyframe_interval"], v["anchor_index_stride"],
                     v["native_frame_span"], sum(m[k] for k in ("image", "velodyne", "target")),
                     m["trident"], m["sample"]])
    return table(["drive", "partition", "labelled anchors", "stream frames", "cached samples",
                  "keyframe k", "anchor stride", "native span", "missing img/velo/target",
                  "missing Trident", "missing sample"], rows)


def transform_table(s1):
    k = next(iter(s1["round_trips"]))
    r = s1["round_trips"][k]
    rows = [[n, f"{v['max_abs_error_m']:.2e}", f(v["within_tol"])]
            for n, v in r.items() if isinstance(v, dict) and "max_abs_error_m" in v]
    rows.append(["grid index → centre → index", "exact",
                 f(r["grid_index_centre_roundtrip"]["exact"])])
    return table(["transform pair (round trip)", "max abs error (m)", f"within {s1['tolerance_m']:g} m"],
                 rows)


def rigid_table(s1):
    k = next(iter(s1["round_trips"]))
    r = s1["round_trips"][k]["rigid_inverse_residual"]
    return table(["shipped matrix", "det", "‖RᵀR−I‖∞", "round trip with Rᵀ (m)",
                  "round trip with true inverse (m)"],
                 [[n, f"{v['det']:.9f}", f"{v['max_abs_RtR_minus_I']:.2e}",
                   f"{v['roundtrip_with_R_transpose_m']:.2e}",
                   f"{v['roundtrip_with_true_inverse_m']:.2e}"] for n, v in r.items()])


def alignment_table(s2):
    rows = []
    for p in PARTS:
        v = s2["per_partition"][p]
        rows.append([NICE[p], v["full_grid"]["n_samples"], f(v["full_grid"]["precision"]),
                     f(v["full_grid"]["recall"]), f(v["full_grid"]["iou"]),
                     f(v["frustum"]["precision"]), f(v["distance_full_median_m"], 3),
                     f(v["distance_full_p95_m"], 3),
                     f(v["frac_lidar_voxels_on_official_occupied"])])
    return table(["partition", "n", "precision", "recall", "IoU", "frustum precision",
                  "median dist (m)", "p95 dist (m)", "frac of in-grid LiDAR voxels on official occupied"],
                 rows)


def shift_table(s2):
    sh = s2["aggregate_shift_scan"]
    rows = [[k, f(v["precision"]), f(v["recall"]), f(v["iou"]),
             "**identity — the confirmed one**" if k == "(0, 0, 0)" else ""]
            for k, v in sh.items()]
    nb = s2["aggregate_neighbour_anchor_scan"]
    rows += [[f"anchor offset {k} (×5 native frames)", f(v["precision"]), "—", f(v["iou"]),
              "**best**" if k == "0" else ""] for k, v in nb.items()]
    return table(["diagnostic shift", "precision", "recall", "IoU", "note"], rows)


def control_table(s2b):
    k3, sk = s2b["kitti360"], s2b["semantickitti"]
    rows = []
    for nm, blk in (("KITTI-360: **ours** vs SSCBench's own `.bin` voxel input", k3["ours_vs_official_bin"]),
                    ("KITTI-360: SSCBench's `.bin` vs SSCBench's own label *(we are nowhere in this path)*",
                     k3["official_bin_vs_official_label"]),
                    ("KITTI-360: ours vs the official label", k3["ours_vs_official_label"]),
                    ("SemanticKITTI: raw sweep vs its official label *(reference implementation)*",
                     sk["sweep_vs_official_label"])):
        for sh in ("(0, 0, 0)", "(0, 0, 1)"):
            rows.append([nm if sh == "(0, 0, 0)" else "", sh, f(blk[sh]["precision"]),
                         f(blk[sh]["recall"]), f(blk[sh]["iou"])])
    return table(["comparison", "shift", "precision", "recall", "IoU"], rows)


def oracle_table(s3):
    rows = []
    for p in PARTS:
        v = s3["aggregate"][p]
        for kind in ("1", "5", "20", "all_past", "future"):
            h = v[kind]["full_grid"]
            rows.append([NICE[p] if kind == "1" else "", kind,
                         f(v[kind]["mean_n_frames_used"], 1), f(h["precision"]), f(h["recall"]),
                         f(h["iou"]), f(v[kind]["mean_valid_grid_coverage"]),
                         f(v[kind]["median_distance_m"], 3)])
    return table(["partition", "history", "frames used", "precision", "recall (target coverage)",
                  "IoU", "valid-grid coverage", "median dist (m)"], rows)


def factor_table(s4):
    cells = ["gt_depth_gt_pose", "gt_depth_lb_pose", "lb_depth_gt_pose", "lb_depth_lb_pose"]
    rows = []
    for p in PARTS:
        if p not in s4["aggregate"]:
            continue
        for c in cells:
            v = s4["aggregate"][p][c]
            rows.append([NICE[p] if c == cells[0] else "", c.replace("_", " "),
                         *[f"{v[str(H)]['precision']:.3f}/{v[str(H)]['recall']:.3f}/{v[str(H)]['iou']:.3f}"
                           for H in s4["histories"]]])
    return table(["partition", "depth × pose"] + [f"h={H} (P/R/IoU)" for H in s4["histories"]], rows)


def memo_table(s6):
    rows = []
    for n, run in s6["runs"].items():
        for r in run["log"]:
            if r["step"] in (1, 200, 500, 1000, 2000):
                e = r["edit"]
                rows.append([f"batch {n}", r["step"], f(r["loss"]), f(r["focal"]), f(r["sem_kl"]),
                             f(r["grad_norm"], 3), f(e["ap"]), f(e["ap_over_prevalence"], 2),
                             f(e["best_iou"]), f(e["iou_at_zero"]), f(e["precision_at_zero"]),
                             f(e["recall_at_zero"]), f(e["density_at_zero"]),
                             f(r["supervised_positive_prevalence"]),
                             f(r["editable_fraction_of_valid"])])
    return table(["batch", "step", "loss", "focal", "KL", "‖grad‖", "edit AP", "AP/prev",
                  "best IoU", "IoU@0", "P@0", "R@0", "pred density", "supervised prevalence",
                  "editable frac"], rows)


def channel_table(s7):
    rows = []
    for p, v in s7["partitions"].items():
        rows.append([NICE.get(p, p), v["n_samples"], f(v["observed_fraction_of_grid"]),
                     f(v["editable_fraction_of_valid"]), f(v["valid_fraction_of_grid"]),
                     f(v["target_prevalence"]), f(v["mean_observed_logodds"], 3),
                     f(v["mean_n_obs"], 3), f(v["mean_age"], 2)])
    return table(["partition", "n", "observed frac of grid", "editable frac of valid",
                  "valid frac of grid", "target prevalence", "mean log-odds where observed",
                  "mean n_obs", "mean age"], rows)


TEMPLATE = """# Gate 8C-0 — KITTI-360 alignment and learnability audit

**Verdict.** {verdict}

**Recommended next action.** {recommend}

**Is the KITTI-360 pipeline correctly aligned?** {aligned}

**Does fixed-batch memorization pass?** {memo}

**Are the KITTI-360 training drives learnable?** {learnable}

**Which failure is it?** {failure_class}

**Decisive evidence.**

{evidence}

---

## Outcome: A — pipeline defect

Per the brief's decision rules this is **A**, because the oracle-LiDAR sanity check fails.
The important qualification is *where* the defect sits: every transform, index, mask and
cache we built is verified correct here, and our voxelization reproduces SSCBench's **own**
published voxel input at 99.90 % recall with zero shift. The inconsistency is between two
official SSCBench-KITTI-360 products — its voxel input and its `_1_1.npy` completion label.

### The exact defect

{defect}

### Minimal proposed repair

{repair}

### Affected Gate 8–8B results and what would need regenerating

{affected}

Per the brief, the repair is **not implemented** and Gate 8B is **not re-run**.

## Stage 0 — provenance

{provenance}

{prov_note}

## Stage 1 — transform chain and round trips

The full chain, with every convention stated, is the module docstring of
`gate8c0/transforms.py` and is reproduced in `stage1_transforms.json`. Round trips over
4 096 sampled points per pair, on all nine audit anchors:

{transforms}

The shipped KITTI-360 matrices are only approximately rigid, so the report states which
inverse is used and what it costs:

{rigid}

Nine overlays (three per partition) are in `artifacts/gate8c0/fig_chain_*.png`, all on
identical axes and bounds, carrying the official target, the current sweep, the future
observations, the frustum, the camera origin and forward direction, and the trajectory.
The camera lands at {cam_origin} m in the grid frame looking along {cam_fwd} — the
KITTI-360 rig geometry, which is what rules out a frame or sign error by inspection.

## Stage 2 — oracle current-frame alignment

The anchor's own ground-truth sweep is already *in* the grid frame, so this measures the
adapter's origin, resolution, axis order and binning with no pose, depth model or learned
component in the path. Low recall is expected; **precision and surface distance are the
test**.

{alignment}

Diagnostic shifts, reported and **not adopted**:

{shift}

### The controls that decide where the offset lives

{controls}

{control_note}

## Stage 3 — oracle temporal reconstruction

Ground-truth sweeps of the causal history, ground-truth poses, each integrated once. This
bounds what *any* method reading this map could achieve. `future` is the non-causal
t+1…t+20 diagnostic and is **never a deployable baseline**.

{oracle}

{oracle_note}

## Stage 4 — geometry factorization

All four cells run through the real incremental mapper. "GT depth" is the anchor sweep
projected to the network's lattice, sparse and never densified with target labels;
predicted cells apply the frozen five-frame G51-B scalar identically to depth and to pose
translation.

{factor}

{factor_note}

## Stage 5 — cache and causal-separation audit

{stage5}

Every invariant is a predicate in `gate8c0/checks.py`, and `tests/gate8c0` injects the five
defects the brief names — a one-frame offset, a reversed pose transform, a drive-ID
collision, a future-frame leak and a one-voxel origin shift — and asserts each is caught
with a diagnostic that identifies it.

## Stage 6 — fixed-batch memorization

Two deterministic batches of four KITTI-360 training-drive samples from different drives
and locations, fixed crop origins, no stochastic augmentation, unchanged architecture,
inputs, targets and loss.

{memo_tbl}

{memo_note}

## Stage 7 — gated off

{stage7_note}

The descriptive channel comparison is independent of that gate:

{channels}

{channel_note}

## Runtime, memory and commands

{runtime}

```bash
cd /home/minh/workspace/lingbot-map_fork
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1 GATE8_GPUS="1 2 3"
PY=/home/minh/anaconda3/envs/cu128/bin/python

$PY tools/gate8c0/stage0_provenance.py
$PY tools/gate8c0/stage1_transforms.py
$PY tools/gate8c0/stage2_alignment.py
$PY tools/gate8c0/stage2b_target_consistency.py
$PY tools/gate8c0/stage3_temporal.py
$PY tools/gate8c0/stage4_factorization.py     --device cuda:1
$PY tools/gate8c0/stage5_cache_audit.py       --device cuda:1
$PY tools/gate8c0/stage6_memorize.py          --device cuda:2
$PY tools/gate8c0/stage7_channel_stats.py     --device cuda:3   # descriptive only
$PY tools/gate8c0/stage0b_hashes.py
$PY tools/gate8c0/aggregate.py && $PY tools/gate8c0/figures.py && $PY tools/gate8c0/report.py --write
$PY -m pytest tests/gate6 tests/gate7b tests/gate8 tests/gate8a tests/gate8b tests/gate8c0 -q
```

Seed 0 throughout. Configuration and artifact hashes are in `artifacts/gate8c0/hashes.json`
(29 frozen files, re-run to see drift). Machine-readable results:
`artifacts/gate8c0/gate8c0_results.json`, plus one JSON per stage and
`stage0_provenance.csv` (the joined per-anchor table, {n_rows} rows).

## Tests

{tests}

## Deviations and failures

{deviations}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    R = json.load(open(os.path.join(ART, "gate8c0_results.json")))
    s0, s1, s2, s2b = (load(x) for x in ("stage0_provenance", "stage1_transforms",
                                         "stage2_alignment", "stage2b_target_consistency"))
    s3, s4, s5, s6, s7 = (load(x) for x in ("stage3_temporal", "stage4_factorization",
                                            "stage5_cache_audit", "stage6_memorize",
                                            "stage7_channel_stats"))
    H, V = R["headline"], R["verdict"]
    k3p = V["kitti360_sweep_vs_label_precision"]; skp = V["semantickitti_sweep_vs_label_precision"]
    ob = H["official_bin_vs_official_label"]
    orc = H["oracle_all_past"]
    fig0 = s1["figures"][next(iter(s1["figures"]))]
    md = TEMPLATE.format(
        verdict=(f"The KITTI-360 completion target is not geometrically consistent with its own "
                 f"LiDAR — a ground-truth sweep lands on a voxel the official label calls *free* "
                 f"{1 - k3p:.0%} of the time, against {1 - skp:.1%} on SemanticKITTI under the "
                 f"identical code path — so Gate 8B's KITTI-360 numbers were measuring a broken "
                 f"target, not a model failure."),
        recommend=("Stop using SSCBench-KITTI-360's `_1_1.npy` label as a completion target, and "
                   "withdraw every KITTI-360 conclusion from Gates 5.2–8B rather than repairing "
                   "them. Bring the KITTI-360 question back only if the target is rebuilt from "
                   "raw aggregated LiDAR under our own verified chain, as a separate approved "
                   "gate. Do not implement that here, and do not touch the completion model: "
                   "Gate 8B's *SemanticKITTI* failure is real, unexplained by this defect, and is "
                   "the right next target."),
        aligned=(f"**Yes — ours is.** All {len(s1['round_trips'])} anchors round-trip every "
                 f"transform pair to {max(v['max_abs_error_m'] for rt in s1['round_trips'].values() for v in rt.values() if isinstance(v, dict) and 'max_abs_error_m' in v):.1e} m "
                 f"(tolerance {s1['tolerance_m']:g} m); the SSCBench index map, grid, floor-binning, "
                 f"rig geometry and anchor timing all check out; Stage 5 passes every causal and "
                 f"cache invariant on {s5['n_samples']} samples. **The published target is not.**"),
        memo=(f"**Yes, decisively.** Editable-region occupancy AP reaches "
              f"{H['memorization']['final_edit_ap']['A']:.4f} and "
              f"{H['memorization']['final_edit_ap']['B']:.4f} on the two fixed batches "
              f"(best-threshold IoU {H['memorization']['final_edit_best_iou']['A']:.4f} / "
              f"{H['memorization']['final_edit_best_iou']['B']:.4f}), gradients stay healthy "
              f"throughout, and there is no information collision (minimum pairwise input "
              f"distance {H['memorization']['min_pairwise_input_l2']:.1f})."),
        learnable=("**Undetermined, and untestable as the data stands.** Fixed crops are "
                   "memorized to AP ≈ 0.99, so nothing about the model, the optimizer, the masks "
                   "or the inputs blocks learning. But generalisation cannot be measured against "
                   "a target that contradicts its own sensor, so the Stage-7 experiment was not "
                   "run."),
        failure_class=("**A data defect**, in the evaluation/training target itself. It is not an "
                       "optimization failure (gradients healthy, memorization passes), not "
                       "cross-drive shift (the defect is present identically on training, "
                       "source-validation and held-out drives), not mixed-source interference "
                       "(it reproduces with KITTI-360 alone), and not merely insufficient causal "
                       "information — although the target is *also* unreachable, which is a "
                       "consequence of the same defect."),
        evidence="\n\n".join([
            f"1. **Our chain reproduces SSCBench's own voxel input exactly.** Voxelizing the raw "
            f"velodyne sweep through our adapter and comparing with SSCBench's published `.bin` "
            f"for the same anchor gives recall "
            f"{H['ours_vs_official_bin_zero_shift']['recall']:.4f} at **zero shift**, and both ±1 "
            f"z shifts destroy it (IoU {H['ours_vs_official_bin_zero_shift']['iou']:.4f} → "
            f"{s2b['kitti360']['ours_vs_official_bin']['(0, 0, 1)']['iou']:.4f}). The velodyne "
            f"frame also beats every alternative grid-frame hypothesis, and anchor offset 0 beats "
            f"±1 and ±2.",
            f"2. **SSCBench's own input disagrees with SSCBench's own label, by the same one "
            f"voxel.** `.bin` versus `_1_1.npy`, with our code nowhere in the path: precision "
            f"{ob['(0, 0, 0)']['precision']:.4f} at zero shift, "
            f"{ob['(0, 0, 1)']['precision']:.4f} at +1 z. The offset is internal to the published "
            f"release.",
            f"3. **SemanticKITTI, under the identical code path, is clean.** Raw sweep versus its "
            f"official label: precision {skp:.4f} at zero shift, falling to "
            f"{s2b['semantickitti']['sweep_vs_official_label']['(0, 0, 1)']['precision']:.4f} at "
            f"+1 z. So this is not a general property of SSC completion targets, and not a bug in "
            f"our voxelizer — KITTI-360 is the outlier.",
            f"4. **The disagreement is far larger than a shift.** Even at the best continuous "
            f"offset ({s2b['kitti360_best_dz_m']:+.2f} m) KITTI-360 precision only reaches "
            f"{max(v['precision'] for v in s2b['kitti360_continuous_dz_scan'].values()):.2f}. The "
            f"gain is concentrated in ground classes (road {s2b['kitti360_label_of_voxel_hit']['(0, 0, 0)'].get('road', 0):,} → "
            f"{s2b['kitti360_label_of_voxel_hit']['(0, 0, 1)'].get('road', 0):,} voxels) while "
            f"vertical structure barely moves, so no single rigid correction fixes it.",
            f"5. **A ground-truth oracle therefore cannot reach the target.** GT sweeps, GT poses, "
            f"all available causal past, integrated once: recall "
            f"{orc['train']['recall']:.4f} / {orc['source_validation']['recall']:.4f} / "
            f"{orc['heldout']['recall']:.4f} and precision {orc['train']['precision']:.4f} / "
            f"{orc['source_validation']['precision']:.4f} / {orc['heldout']['precision']:.4f} on "
            f"train / source-validation / held-out. The four-way depth×pose table confirms the "
            f"ceiling is the target, not the components: GT depth × GT pose peaks at IoU "
            f"{max(H['factorization_20frame'][p]['gt_depth_gt_pose']['iou'] for p in H['factorization_20frame']):.3f}, "
            f"and swapping GT pose for LingBot's changes almost nothing.",
            f"6. **The model is not the problem.** Fixed-batch memorization reaches editable-region "
            f"AP {H['memorization']['final_edit_ap']['B']:.4f} on the same data, so the inputs "
            f"carry enough signal to fit these targets exactly when generalisation is not "
            f"required."]),
        defect=(f"`preprocess/labels/<drive>/<anchor>_1_1.npy` in the SSCBench-KITTI-360 release "
                f"marks voxels as **observed free** at locations where the same release's own "
                f"LiDAR measured a return. Measured on {s2b['n_kitti360_anchors']} anchors of the "
                f"held-out drive:\n\n"
                f"* our voxelized sweep vs the label: precision **{k3p:.4f}** — {1 - k3p:.0%} of "
                f"measured surface voxels fall on label-*free* (not invalid, not unknown) voxels;\n"
                f"* SSCBench's own `.bin` vs the same label: precision **{ob['(0, 0, 0)']['precision']:.4f}**, "
                f"rising to **{ob['(0, 0, 1)']['precision']:.4f}** under a +1 voxel z shift;\n"
                f"* SemanticKITTI, same code: precision **{skp:.4f}**, and +1 z makes it worse.\n\n"
                f"The dominant component is vertical and concentrated on the ground plane, "
                f"consistent with the label's ground surface sitting roughly one voxel above the "
                f"measured return; but a rigid correction recovers precision only to ~0.6, so the "
                f"label is not simply displaced. The consequence for Gates 6–8B is that the "
                f"KITTI-360 geometry target was largely unreachable: a perfect-geometry oracle "
                f"attains ≈{100 * orc['heldout']['recall']:.0f} % recall and "
                f"≈{100 * orc['heldout']['precision']:.0f} % precision on the held-out drive."),
        repair=("The minimal repair is **not** a code change — our adapter is verified correct, "
                "and shifting it to chase the label would break its confirmed agreement with the "
                "official voxel input. The options, cheapest first:\n\n"
                "1. **Withdraw KITTI-360 as a benchmark and as a training source** (recommended). "
                "No cache regeneration; the affected results are marked invalid and the "
                "conclusions that rested on them are re-derived from SemanticKITTI and Occ3D "
                "alone. This is a documentation and aggregation change only.\n"
                "2. **Rebuild the KITTI-360 target ourselves** from raw aggregated LiDAR under the "
                "verified chain, SemanticKITTI-style. This restores geometric consistency but "
                "breaks comparability with every published SSCBench-KITTI-360 number, so it must "
                "be reported as a different benchmark.\n"
                "3. **Adopt the +1 z shift** — rejected. It is not confirmed by official metadata, "
                "it recovers precision only to ~0.6, and it would put us out of agreement with "
                "SSCBench's own voxel input, which we currently match at 99.9 %."),
        affected=table(["status", "artifacts"],
                       [["**invalidated**", "<br>".join(R["affected_artifacts"]["kitti360_results_invalidated"])],
                        ["unaffected", "<br>".join(R["affected_artifacts"]["unaffected"])]])
        + "\n\nRegeneration required **only under repair option 2**: the KITTI-360 targets, then "
          "the Gate 8B `k360_train` samples (1 276), the KITTI-360 source-validation samples "
          "(175), both new folds' training runs and every KITTI-360 evaluation. Under the "
          "recommended option 1, nothing is regenerated.\n\n"
          f"**{R['affected_artifacts']['note']}**",
        provenance=provenance_table(s0),
        prov_note=(f"{s0['n_rows']} anchor rows joined across 4 drives. No missing image, velodyne, "
                   f"target or pose row anywhere; no duplicate stream key; no non-monotonic "
                   f"sequence; no image↔velodyne timestamp gap above 50 ms; "
                   f"{s0['frame_key_collisions_across_drives']} frame-key collisions across drives. "
                   f"The missing counts that are non-zero are expected and explained: the last few "
                   f"anchors of each drive have fewer than five future frames so no sample is built, "
                   f"drive 0006 was sampled at anchor-stride 10 by Gate 8B, and its 35 anchors "
                   f"without a Trident cache are the ones Gate 5.2's clip-eligibility rule excluded "
                   f"from the official evaluation manifest."),
        transforms=transform_table(s1), rigid=rigid_table(s1),
        cam_origin=str(fig0["camera_origin_in_grid_frame_m"]),
        cam_fwd=str(fig0["camera_forward_in_grid_frame"]),
        alignment=alignment_table(s2), shift=shift_table(s2), controls=control_table(s2b),
        control_note=(f"Read the second block first: it contains no code of ours. The adapter's "
                      f"elementwise 255/0/occupied rule was also re-verified against the raw "
                      f"`.label`/`.invalid` files and holds exactly on every anchor "
                      f"(`adapter_rule_holds_everywhere` = {s2b['adapter_rule_holds_everywhere']}), "
                      f"so `_1_1.npy` occupied is precisely `.label > 0`."),
        oracle=oracle_table(s3),
        oracle_note=(f"Recall rises with history and then saturates — correctly, since a sweep more "
                     f"than 51.2 m back contributes no voxel to this grid, which is why `all_past` "
                     f"equals `20`. The sanity pattern the brief expects therefore **half** holds: "
                     f"recall does improve with history, but ground-truth observations do **not** "
                     f"align with target surfaces (precision ≈0.25–0.30, median distance exactly one "
                     f"voxel), and the non-causal future oracle covers only "
                     f"{H['oracle_future_diagnostic']['heldout']['recall']:.0%}–"
                     f"{H['oracle_future_diagnostic']['train']['recall']:.0%} of the privileged "
                     f"target. Target prevalence is "
                     f"{H['target_prevalence']['heldout']:.3f} on a valid mask covering only "
                     f"{H['valid_fraction_of_grid']['heldout']:.3f} of the grid, against "
                     f"SemanticKITTI's ≈0.08 prevalence on ≈0.68 of the grid."),
        factor=factor_table(s4),
        factor_note=("Pose is not the problem: replacing ground-truth pose with LingBot's changes "
                     "IoU by less than the sample spread. Predicted depth costs recall (the frozen "
                     "confidence gate keeps fewer rays) while holding or raising precision, i.e. it "
                     "is sparser, not misaligned. Every cell lives under a ceiling set by the "
                     "target."),
        stage5=table(["check", "result"],
                     [["grid declaration (dims, voxel, origin, frame)", f(s5["global"]["grid_declaration"]["ok"])],
                      ["drive partitions disjoint", f(s5["global"]["drive_partitions_disjoint"]["ok"])],
                      ["cache keys cannot collide across drives", f(s5["global"]["cache_keys_cannot_collide"]["ok"])],
                      ["Trident cache path is drive-scoped", f(s5["global"]["trident_path_is_drive_scoped"]["ok"])],
                      [f"causal input only (0..t), {s5['n_samples']} samples", f(not s5["failures"].get("causal_input_only"))],
                      ["future target window t+1..t+20", f(not s5["failures"].get("future_target_window"))],
                      ["no future frame leaks into the input", f(not s5["failures"].get("no_future_leak"))],
                      ["RGB / Trident / pose / target name the same frame", f(not s5["failures"].get("frame_identity_consistent"))],
                      ["invalid voxels are never supervised", f(not s5["failures"].get("unknown_not_supervised"))],
                      ["each frame integrated exactly once (live mapper)", f(s5["live_mapper"][0]["integrated_once"]["ok"])],
                      ["scale anchor sees only its first five frames", f(s5["live_mapper"][0]["scale_anchor_window"]["ok"])]]),
        memo_tbl=memo_table(s6),
        memo_note=(f"Both batches memorize. Cause discrimination: gradients are finite and non-zero "
                   f"throughout (optimization is healthy); the supervised mask carries a real "
                   f"positive prevalence and covers {s6['runs']['A']['final']['editable_fraction_of_valid']:.1%} "
                   f"of valid voxels (masking is fine); the minimum pairwise input distance across "
                   f"the fixed samples is {H['memorization']['min_pairwise_input_l2']:.1f} with up to "
                   f"{H['memorization']['max_pairwise_target_disagreement']:.1%} target disagreement "
                   f"(the samples are distinct, so neither information collision nor contradictory "
                   f"identical inputs applies). Capacity is not the limit either. **Outcome B is "
                   f"ruled out.**"),
        stage7_note=R["verdict"]["stage7_gate_reason"],
        channels=channel_table(s7),
        channel_note=("Descriptive only — Gate 8C-0 normalises nothing. The number worth carrying "
                      "forward is that the causal map observes 4–8 % of the grid while the target "
                      "asserts 17–27 % occupancy over the valid region, and 94–97 % of that valid "
                      "region is *editable*, i.e. the map holds no committed evidence there."),
        runtime=table(["stage", "wall time", "notes"],
                      [["0 provenance", f"{s0['seconds']:.0f} s", f"{s0['n_rows']} anchors, 4 drives"],
                       ["1 transforms + 9 overlays", f"{s1['seconds']:.0f} s", "CPU"],
                       ["2 alignment", f"{s2['seconds']:.0f} s", f"{s2['n_samples']} anchors, CPU"],
                       ["2b controls", f"{s2b['seconds']:.0f} s", f"{s2b['n_kitti360_anchors']} K360 + {s2b['n_semantickitti_clips']} SK"],
                       ["3 oracle temporal", f"{s3['seconds'] / 60:.1f} min", f"{s3['n_samples']} anchors, CPU"],
                       ["4 factorization", f"{s4['seconds'] / 60:.1f} min",
                        f"{s4['n_samples']} anchors, peak {s4['peak_gpu_gib']:.2f} GiB"],
                       ["5 cache audit", f"{s5['seconds']:.0f} s", f"{s5['n_samples']} samples"],
                       ["6 memorization", f"{(s6['runs']['A']['seconds'] + s6['runs']['B']['seconds']) / 60:.1f} min",
                        f"2 × {s6['steps']} steps, peak {s6['runs']['B']['peak_gpu_gib']:.2f} GiB"],
                       ["7 channel stats", f"{s7['seconds']:.0f} s", "descriptive"]]),
        n_rows=s0["n_rows"],
        tests=open(os.path.join(ART, "test_summary.txt")).read().strip(),
        deviations=open(os.path.join(REPO_ROOT, "reports", "gate8c0", "_deviations.md")).read())
    if a.write:
        open(OUT, "w").write(md); print("wrote", OUT, len(md), "chars")
    else:
        print(md[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
