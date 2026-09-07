# Gate 8C-1 — source supervision and target-exclusion audit

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
| ray interiors free | carved from 1.0 m to 0.2 m before the endpoint (frozen Gate 7B rule), decimation 4 |
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

| drive | partition | eligible anchors | future window | targets built | samples built | valid frac | prevalence in valid | cross-sweep conflict | mean sweeps |
|---|---|---|---|---|---|---|---|---|---|
| 2013_05_28_drive_0003_sync | train | 193 | 5–20 | 193 | 193 | 0.310 | 0.047 | 0.082 | 20.4 |
| 2013_05_28_drive_0007_sync | train | 569 | 5–20 | 569 | 569 | 0.296 | 0.056 | 0.079 | 20.8 |
| 2013_05_28_drive_0010_sync | train | 596 | 5–20 | 596 | 596 | 0.336 | 0.042 | 0.076 | 20.8 |
| 2013_05_28_drive_0006_sync | source_validation | 1768 | 5–20 | 197 | — | 0.217 | 0.071 | 0.094 | 20.9 |

Anchors are **all eligible stream frames**, not only the frames SSCBench happened to
label. Eligibility is purely mechanical and is recorded per drive:

* a frame needs 5 preceding stream frames, so the frozen five-frame scale
  anchor is complete and the map is warm — this excludes the first 4
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

| check | result | evidence |
|---|---|---|
| raw sweep endpoints agree by construction | **pass** | every supervised endpoint is OCCUPIED (min 1.000000); none is FREE (max 0.0e+00); 40.2% of endpoints are supervised at all |
| transform round trips | **pass** | max abs error 1.20e-12 m (tolerance 0.001 m) |
| current-to-future alignment | **pass** | median distance from the anchor sweep to the nearest rebuilt-occupied voxel 0.000 m; 100.0% within one voxel |
| recall grows with future sweeps | **pass** | recall 0.363 → 0.489 → 0.786 → 0.948 → 1.000 for [1, 2, 5, 10, 20] sweeps |
| no systematic one-voxel offset | **pass** | best shift is the identity (0, 0, 0) (IoU 0.2007); median surface distance 0.000 m |
| SemanticKITTI not opened while designing the target | **pass** | no target-dataset path is reachable from the builder or the validator; the runtime file audit guards it |

The contrast with the label Gate 8C-0 disqualified, on the same drives and the same code
path:

| property | SSCBench `_1_1.npy` (Gate 8C-0) | rebuilt raw-LiDAR (this gate) |
|---|---|---|
| supervised endpoints marked OCCUPIED | 0.29 | **1.0000** |
| supervised endpoints marked FREE | 0.62 | **0.0e+00** |
| best voxel shift | (0, 0, +1), +99 % IoU | **(0, 0, 0)** |
| median surface distance | 0.200 m | **0.000 m** |

Figures: `fig_target_0003_00055.png`, `fig_target_0007_00418.png`, `fig_target_0010_00005.png`, `fig_target_0006_00040.png`.

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
