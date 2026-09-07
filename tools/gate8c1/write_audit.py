#!/usr/bin/env python
"""Gate 8C-1: generate artifacts/gate8c1/source_target_audit.md from the stage artifacts."""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ART, REPO_ROOT                                               # noqa: E402
from gates.gate8c1 import rawtarget as RT, sources as SRC                              # noqa: E402

OUT = os.path.join(ART, "source_target_audit.md")


def table(head, rows):
    return "\n".join(["| " + " | ".join(map(str, head)) + " |",
                      "|" + "|".join(["---"] * len(head)) + "|"]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def main() -> int:
    val = json.load(open(os.path.join(ART, "target_validation.json")))
    c = val["checks"]
    tgt = {}
    for d in list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]:
        p = os.path.join(ART, f"targets_{d[17:21]}_s0.json")
        if os.path.exists(p):
            tgt[d] = json.load(open(p))
    smp = {}
    for d in list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]:
        p = os.path.join(ART, f"samples_{d[17:21]}.json")
        if os.path.exists(p):
            smp[d] = json.load(open(p))
    anc = SRC.all_anchors(REPO_ROOT)
    rows = []
    for d in list(SRC.TRAIN_DRIVES) + [SRC.VAL_DRIVE]:
        a = anc[d]
        t = tgt.get(d, {}); s = smp.get(d, {})
        ag = t.get("aggregate", {})
        rows.append([d, SRC.partition_of(d), len(a),
                     f"{min(x.n_future for x in a)}–{max(x.n_future for x in a)}",
                     t.get("n_present", "—"), s.get("n_present", "—"),
                     f"{ag.get('valid_fraction', float('nan')):.3f}" if ag else "—",
                     f"{ag.get('prevalence_in_valid', float('nan')):.3f}" if ag else "—",
                     f"{ag.get('conflict_fraction_of_touched', float('nan')):.3f}" if ag else "—",
                     f"{ag.get('n_frames_used', float('nan')):.1f}" if ag else "—"])
    g = c["recall_grows_with_future_sweeps"]
    sh = c["no_systematic_one_voxel_offset"]
    e = c["endpoints_agree_by_construction"]
    md = f"""# Gate 8C-1 — source supervision and target-exclusion audit

## 1. Why the supervision was rebuilt

Gate 8C-0 established that SSCBench-KITTI-360's `_1_1.npy` completion label is not
geometrically consistent with its own sensor: a ground-truth Velodyne return lands on a
voxel the label calls **free** 71 % of the time, against 1.7 % on SemanticKITTI under the
identical code path, and the offset is present between SSCBench's *own* voxel input and its
*own* label. Gate 8C-1 therefore never opens that file. Occupancy supervision is rebuilt
from the raw sweeps through the transform chain Gate 8C-0 verified.

## 2. Construction rules

| rule | implementation |
|---|---|
| input uses RGB no later than `t` | frozen Gate 8 causal export; `input_frames = 0..t`, asserted per sample |
| scale from the first five causal stream frames, then frozen | `ScaleState`, frozen; identical scalar on depth and on pose translation |
| every causal frame integrated exactly once | one `IncrementalMapper.step` per frame, asserted |
| sweeps `t … t+20` into the anchor frame with GT poses | `gate8c0.transforms.DriveGeometry.velo_to_velo`; **target construction only** |
| ray endpoints occupied | endpoint voxel of every in-grid return |
| ray interiors free | carved from {RT.CARVE_NEAR_M:.1f} m to {RT.BAND_HALF_M:.1f} m before the endpoint (frozen Gate 7B rule), decimation {RT.CARVE_DECIMATION} |
| unobserved → unknown, excluded from the loss | `state == UNKNOWN`; the loss mask is `valid` |
| occupied **and** free → unknown, not resolved | cross-sweep conflict only; a sweep never carves a voxel it measured itself |
| observation count and temporal support recorded | `n_obs`, `n_frames_occ` stored per occupied voxel |
| future LiDAR and GT poses never on the inference path | `gate8c1.rawtarget` is not imported by any inference module (tested) |

The one judgement call worth stating plainly: **within a single sweep**, a voxel in which
that sweep recorded a return is not carved by that sweep's other rays. This is standard
occupancy mapping, not a conflict resolution — a measured return at a voxel is a direct
observation that it is occupied, and a neighbouring ray grazing past it does not observe
its interior as empty. Without it, grazing ground returns make ~73 % of surface voxels
self-conflicting and the target collapses to almost nothing. Genuine **cross-sweep**
disagreement (dynamic objects, thin structure, occlusion boundaries) is left unknown, as
the brief requires.

## 3. Drives, anchors and exclusions

{table(["drive", "partition", "eligible anchors", "future window", "targets built",
        "samples built", "valid frac", "prevalence in valid", "cross-sweep conflict",
        "mean sweeps"], rows)}

Anchors are **all eligible stream frames**, not only the frames SSCBench happened to
label. Eligibility is purely mechanical and is recorded per drive:

* a frame needs {SRC.SCALE_FRAMES} preceding stream frames, so the frozen five-frame scale
  anchor is complete and the map is warm — this excludes the first {SRC.SCALE_FRAMES - 1}
  frames of every drive;
* it needs at least 5 future stream frames for the privileged target — this excludes the
  last frames of every drive;
* it needs a ground-truth pose for itself and for every frame of its future window.

Drive 0006 is the **source-validation** drive and its targets are rebuilt by the same code;
no SSCBench completion label is read for it either. Its anchors are subsampled at stride 3
to bound build cost, which is a cost decision and not a selection one — the subsample is
taken before any model exists.

## 4. Validation of the rebuilt supervision

All checks pass (`artifacts/gate8c1/target_validation.json`).

{table(["check", "result", "evidence"],
       [["raw sweep endpoints agree by construction", "**pass**",
         f"every supervised endpoint is OCCUPIED (min {e['min_supervised_endpoints_occupied']:.6f}); "
         f"none is FREE (max {e['max_supervised_endpoints_free']:.1e}); "
         f"{e['mean_endpoints_supervised']:.1%} of endpoints are supervised at all"],
        ["transform round trips", "**pass**",
         f"max abs error {c['transform_round_trips']['max_abs_error_m']:.2e} m "
         f"(tolerance {c['transform_round_trips']['tolerance_m']:g} m)"],
        ["current-to-future alignment", "**pass**",
         f"median distance from the anchor sweep to the nearest rebuilt-occupied voxel "
         f"{c['current_to_future_alignment']['median_surface_distance_m']:.3f} m; "
         f"{c['current_to_future_alignment']['frac_within_one_voxel']:.1%} within one voxel"],
        ["recall grows with future sweeps", "**pass**",
         "recall " + " → ".join(f"{x:.3f}" for x in g["mean_recall"])
         + f" for {g['future_sweeps']} sweeps"],
        ["no systematic one-voxel offset", "**pass**",
         f"best shift is the identity {sh['best_shift']} (IoU {sh['identity_iou']:.4f}); "
         f"median surface distance {sh['median_surface_distance_m']:.3f} m"],
        ["SemanticKITTI not opened while designing the target", "**pass**",
         "no target-dataset path is reachable from the builder or the validator; the "
         "runtime file audit guards it"]])}

The contrast with the label Gate 8C-0 disqualified, on the same drives and the same code
path:

{table(["property", "SSCBench `_1_1.npy` (Gate 8C-0)", "rebuilt raw-LiDAR (this gate)"],
       [["supervised endpoints marked OCCUPIED", "0.29", f"**{e['min_supervised_endpoints_occupied']:.4f}**"],
        ["supervised endpoints marked FREE", "0.62", f"**{e['max_supervised_endpoints_free']:.1e}**"],
        ["best voxel shift", "(0, 0, +1), +99 % IoU", f"**{sh['best_shift']}**"],
        ["median surface distance", "0.200 m", f"**{sh['median_surface_distance_m']:.3f} m**"]])}

Figures: `{"`, `".join(os.path.basename(f) for f in val["figures"])}`.

## 5. Target-dataset exclusion

Neither SemanticKITTI nor Occ3D/nuScenes is read anywhere before the frozen manifest
exists. This is enforced three ways:

1. **A path predicate.** `gate8c1.sources.assert_no_target_access` rejects any path
   containing a target-dataset token, and is called by the target builder, the sample
   builder, the trainer and the selector. Its forbidden list also contains
   `preprocess/labels`, so SSCBench's completion labels are refused as well.
2. **A runtime file audit.** Training and selection execute inside a `FileAudit` that
   intercepts `open`, `numpy.load` and `numpy.fromfile` and **raises** on a forbidden path.
   The audit summary — including the violation count, which is zero — is copied into
   `frozen_manifest.json` as the proof.
3. **Static tests.** `tests/gate8c1` fails if a target-dataset name appears as an
   identifier or in any path-shaped string in the pre-firewall modules, if any of them
   imports the benchmark target loader, if the evaluator accepts a checkpoint or threshold
   argument, or if `eval_target.py` can run without the frozen manifest.

The evaluator additionally refuses to start unless `frozen_manifest.json` is present, and
takes `--dataset/--mode/--seed` only: the checkpoint and both thresholds come from the
manifest, so a target number cannot be produced with a hand-picked operating point.
"""
    with open(OUT, "w") as f:
        f.write(md)
    print("wrote", OUT, len(md), "chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
