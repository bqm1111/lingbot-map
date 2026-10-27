# Stage A — repository audit and OccAny comparison

**Written 2026-10-19.** Every claim here was read out of code on this machine; nothing is from
memory. Paths are as checked. Companion: `artifacts/waymo_arm/predeclaration.md` (the
diagnostic and the predeclared prediction), `reports/occany_comparison_honest_report.md`
(the measured OccAny comparison).

---

## 1. The existing pipeline, traced end to end

Entry points and the order they run in. "Frozen" = weights or code hash-checked at run time.

| stage | module | trained here? |
|---|---|---|
| causal clip, 5 frames `t−4..t`, one camera | `gates/gate7b/replay.py` + `tools/occany_comparison/clips.py::infer_clip` | no — LingBot `checkpoints/lingbot-map/204754b` **frozen** |
| metric scale | `gates/gate7b/scale.py`, MoGe-2 rev `39c4d5e9…` | no — **frozen** |
| hard map `(S, F)` | `core/mapping/{mapper,rays,depth,voxmap}.py` via `baseline_reconstruction/current/geometry.py::build_local_geometry` | no — rule-based |
| geometry completion | `core/model/net.py::CompletionUNet` (cin 3, width 24, **967,322 params**) | **YES** — `baseline_reconstruction/cli/train_current.py` |
| 2D semantic teacher | `tools/semantic_completion/evidence/trident.py` (Trident-H, 18-phrase readout) | no — **frozen**, separate `trident_env` |
| semantic evidence lift | `tools/zero_shot_eval/prediction/evidence.py::lift_evidence` | no — rule-based |
| B0 semantics | `prediction/semantics.py::nearest_evidence` (EDT to nearest evidence voxel) | no — parameter-free |
| B3 semantics | `prediction/semantics.py::b3_features` + `delayed_semantic/gate16` head | **YES** — `delayed_semantic/gate16/train.py`, seeds 0/1/2 |
| scoring | `zero_shot_eval/{official,practical}/evaluator.py`, `canonical/metrics.py` | — |

So exactly **two** things are trained on KITTI-360: `CompletionUNet` (geometry) and the B3
`EvidenceAssociation` head (semantics). Everything else is frozen or rule-based.

Grid: 256×256×32 @ 0.2 m, N = 2,097,152, origin (0, −25.6, −2) in the anchor velodyne frame
(`core/datasets/kitti360_transforms.GRID`). Source drives 0003 / 0007 / 0010, dev 0006
(`baseline_reconstruction/config.TRAIN_DRIVES`), **1,358 anchors**; a scaled 8-drive variant
with **9,672 anchors** already exists at `tools/kitti360_data_scaling_7drives`.

Supervision origin:

* **occupancy** — `core/supervision/rawtarget.py`, rebuilt from raw Velodyne sweeps with GT
  poses over the anchor + 20 future stream frames. Endpoints occupied; ray interiors free,
  stopping `BAND_HALF_M` short; a voxel this sweep measured is never carved by it;
  occupied-in-one-sweep vs free-in-another stays **UNKNOWN** and is dropped from the loss;
  an explicit open-sky rule frees columns above the topmost return. SSCBench's published
  completion label is deliberately never read (Gate 8C-0).
* **semantics** — Trident-H predictions on the clip's own five images, lifted to voxels, plus
  a privileged delayed target from future frames for B3 (`tools/delayed_semantic/targets.py`).
  **No dataset 3D semantic label is used anywhere in training.**

## 2. Files the Waymo extension touches

Added, nothing existing rewritten:

| file | role |
|---|---|
| `tools/waymo_pilot/{waymo,waymo_geometry,sample}.py` | reader, calibration, stride-5 causal stream, grid, clips (pre-existing from the pilot) |
| `tools/waymo_pilot/cli/extract_laser_sidecar.py` | **new** — recovers per-point laser identity |
| `core/supervision/rawtarget_multisensor.py` | **new** — the same rule, per (frame, laser) sweep |
| `tools/waymo_arm/{config,geometry,train}.py` | **new** — paths, privileged sweeps, training with the scene list as an argument |
| `tools/waymo_arm/cli/{build_inputs,build_targets,check_equivalence,check_trainer_equivalence,diagnose}.py` | **new** |

Modified: `tools/waymo_pilot/{sample.py, cli/build_waymo_clips.py}` — added a
`min_separation` argument that **defaults to the pilot's own value**, so pilot behaviour is
unchanged without the flag. Nothing in `baseline_reconstruction/`, `semantic_completion/`,
`delayed_semantic/`, `zero_shot_eval/` or `core/model/` was edited;
`FZ.require_source_freeze()` still passes (checked).

## 3. Verified gates

| gate | result |
|---|---|
| multi-sensor carve == frozen single-sensor carve | **PASS**, bit for bit on KITTI-360 anchors (`state`, `n_obs`, `n_frames_occ`, `n_free`, `frames_used`), and on Waymo TOP-only through both paths |
| Waymo trainer == frozen trainer | **PASS** on identical KITTI-360 data: draw digest, target digest, init digest, anchor counts **exact**; losses inside the frozen loop's own envelope |
| KITTI-360 source freeze intact | **PASS** |
| occupancy target statistics comparable | valid fraction 0.5710 (Waymo) vs 0.5703 (KITTI-360 drive 0000); conflict 0.062 vs 0.087; 21 sweep-frames both |

The frozen trainer is **not** bitwise reproducible against itself on this machine (98/100
steps differ, max |Δloss| 1.5e-4, different final sha256 — bf16 autocast picks
non-deterministic cuDNN algorithms), so bitwise equality is not an admissible gate here and
the criterion is stated accordingly.

## 4. OccAny — what it actually does with Waymo

Read from `/home/minh/workspace/OccAny`.

| question | answer | source |
|---|---|---|
| Waymo version and split | `gs://waymo_open_dataset_v_1_4_2/individual_files/training/` — **the same version and split we use** | `dataset_setup/waymo/download_waymo.sh` |
| held-out scenes | exactly **two** named segments; everything else trains | `occany/datasets/waymo.py` |
| cameras | **all five**, multi-view (`WaymoSeqMultiView`) | `occany/datasets/waymo.py` |
| frame sampling | `seq_sub1_stride9.pkl` — stride 9 | same |
| modalities read | RGB + LiDAR points projected per camera (`pixels`, `pts3d`), metric scale. **No semantic label of any kind is read** | `dataset_setup/waymo/preprocess_waymo.py` |
| Waymo's share of training | 6,000 + 4,000 samples of a ~49,000-sample mixture with VKITTI (5,000), DDAD (10,000), PandaSet (10,000), ONCE (10,000) | `sh/train_occany_plus_recon.sh` |
| training objective | `--loss_type L1`, `pointmap_lambda_c 1.0`, `raymap_lambda_c 1.0`, scale-invariant depth, plus `--distill_model SAM3 --distill_criterion DistillLoss(L1)`. **Geometry and feature distillation only — no semantic classification loss** | same |
| where semantics come from | **inference time**: GroundingDINO boxes + SAM2 masks + distilled SAM3 features, lifted onto the reconstructed geometry | `occany/semantic_inference.py`, `extract_gdino_boxes_*.py` |
| monocular 5-frame setting available? | yes — `eval_helper.py` `1frame` and `5frames` KITTI settings; already run here (24.02 geometry, 3.97 SSC mIoU causal) | `occany/datasets/eval_helper.py` |

**Unknown / not verified:** how many distinct Waymo segments the stride-9 pkl actually
yields; whether the released checkpoint corresponds to this exact script; whether the paper's
numbers used `train_occany_plus_recon.sh` or one of the other four scripts in `sh/`.

## 5. What this means for the central research question

The brief asks whether Waymo-only training can make our **semantic** mIoU competitive with
OccAny's. The audit says that question has a structural answer that does not depend on how
much Waymo data we use:

1. **OccAny trains no semantics at all.** Its semantic occupancy is a frozen 2D
   open-vocabulary stack (GDINO + SAM2 + SAM3 features) applied at inference to geometry it
   did train. Its semantic mIoU is therefore a property of that 2D stack plus its geometry,
   not of its training mixture's semantic content.
2. **Our semantics are also not trained from data labels** — Trident-H is frozen and B3 only
   learns to *associate* existing evidence with unobserved voxels. Swapping KITTI-360 for
   Waymo changes the **occupancy** supervision and the B3 association statistics; it cannot
   give the teacher new classes or better 2D predictions.
3. The teacher bound is already measured: **63.9 % top-1 correct, covering 12.3 % of hidden
   occupied voxels** (`artifacts/gate17_instance_oracle`, Waymo pilot). Three separate
   attempts to move it — oracle instance grouping (+0.09, CI [−0.27, +0.34]), a VLM
   recognizer (+0.88, CI touching zero), bounded region extension (±0.02) — all failed.
   `PAPER C` additionally records B3 − B0 = **+0.25 with the CI containing zero** on
   SemanticKITTI.

So the brief's **H2 (semantic-supervision bottleneck)** is the hypothesis the existing
evidence already favours, and this experiment is best framed as *testing* it rather than
expecting H1. The measurable prediction is: Waymo changes geometry (SC IoU) materially and
semantic mIoU little, with the direction per benchmark set by camera field of view
(§4 of the predeclaration: SemanticKITTI should lose, Occ3D should gain).

That is still worth running — it converts "we think semantics are teacher-capped" into a
controlled dataset experiment with a predeclared outcome. But a plan that expects Waymo-only
training to close the semantic gap to OccAny is not supported by anything in this repository,
and the report must not be written as though it were.

## 6. Missing components, stated rather than substituted

| required by the brief | status |
|---|---|
| §5.E Waymo semantic supervision | **NOT BUILT.** Needs Trident-H token maps on Waymo images (measured 1.82 s/frame → ~10 GPU-h for 19,950 frames, ~190 GB) plus the privileged delayed-target build, then B3 training. Geometry is unaffected and complete |
| §5.C round-trip transform unit tests + convention document | **NOT BUILT.** The LiDAR→grid path is gated by the bit-exact carve equivalence, but the brief's explicit synthetic-projection tests do not exist |
| §6 E1 (~1,500) / E2 (~10,000) nested manifests | **NOT BUILT.** Current build is ~3,955 anchors over 798 segments; E2 at 10,000 needs ~13 of the 15 eligible anchors per segment |
| §10 visual sanity checks 1–8 | **NOT BUILT** |
| §9 observed vs completed-unobserved semantic mIoU | machinery exists (`tools/observability_split`, `zero_shot_eval/practical/masks.populations`) but is not wired into this arm's evaluation |
| Waymo LiDAR semantic labels as a separate named experiment | not started; correctly out of scope for the primary experiment |
