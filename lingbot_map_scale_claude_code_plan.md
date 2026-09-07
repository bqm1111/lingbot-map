# Claude Code Execution Plan: Metric Scale for LingBot-Map

## Purpose

Execute Gate 0 of the LingBot-Map semantic occupancy project: determine whether LingBot-Map's internally consistent but non-metric geometry can be converted to metric scale reliably enough for fixed-grid occupancy prediction.

This is an implementation and experimental task, not an open-ended exploration. Do not implement semantic lifting or scene completion in this task. The output must be a reproducible metric-scale pipeline, an oracle-scale diagnosis, a learned non-oracle scale baseline, and a written go/no-go report.

## Scientific question

LingBot-Map predicts per-frame depth and camera poses in a canonical sequence scale. For a short clip, denote its outputs by

```text
relative depth:       D_tilde[t]
relative translation: t_tilde[t]
rotation:             R[t]
```

We need one positive scalar `s` per short clip such that

```text
D_metric[t] = s * D_tilde[t]
t_metric[t] = s * t_tilde[t]
R_metric[t] = R[t]
```

Depth and camera translation must always receive the same scale. Never scale rotations. Never scale depth without scaling pose translations.

The task has two parts:

1. Oracle diagnosis: using training/evaluation LiDAR or metric poses, compute the best possible scalar `s_star` and measure the maximum downstream improvement obtainable from scale alone.
2. Non-oracle prediction: train a small model that predicts `s_hat` from RGB and frozen LingBot-derived information, with no LiDAR, ground-truth poses, dataset identity, or target metadata available at inference.

## Known context and constraints

- LingBot-Map is the only geometry backbone for this project.
- Keep the full LingBot-Map checkpoint frozen.
- Do not fine-tune its encoder, GCT, depth head, or camera head in the first implementation.
- Existing reference results are approximately:
  - SemanticKITTI causal occupancy IoU: `0.087`.
  - Occ3D-nuScenes causal occupancy IoU: `0.068`.
  - Oracle-scale reference results: approximately `0.081` and `0.102`, respectively.
- These numbers are reference points to reproduce and investigate. Do not hard-code them or assume that the earlier oracle implementation was correct.
- Current LingBot post-GCT tokens did not pass the previous cross-view stability gate. Do not make them the only input to the scale predictor.
- The initial scope is a short chronological clip in direct mode, with one scale per clip. Long-window scale drift is explicitly out of scope.
- SemanticKITTI sequence 08 and Occ3D-nuScenes validation caches may already exist. Reuse valid assets, but verify their coordinate conventions and provenance.
- The repository may contain uncommitted user work. Preserve unrelated changes and do not reset, clean, delete, or overwrite them.

## Non-goals

Do not do any of the following in this task:

- Add MoGe, Depth Anything, MapAnything, or another geometry backbone.
- Train an occupancy-completion network.
- Train an open-vocabulary semantic head.
- Use SemanticKITTI semantic class labels as scale inputs or supervision.
- Use test-time LiDAR or ground-truth poses in the non-oracle method.
- Claim that scale solves the complete geometry problem.
- Start a full five-dataset training run.
- Silently download large or license-gated datasets.

## Required execution behavior

1. Inspect before modifying.
2. Find and obey `AGENTS.md`, repository instructions, and existing conventions.
3. Run `git status --short` before changes and record pre-existing modifications.
4. Reuse existing dataset loaders, voxelizers, LingBot inference code, and metric utilities where correct.
5. Do not create parallel implementations of the same transformation without documenting why.
6. Make every preprocessing step deterministic and resumable.
7. Use configuration files rather than hard-coded absolute paths.
8. Store large generated caches outside Git and add only appropriate ignore rules.
9. Add unit tests for coordinate transforms and scale computation before running full experiments.
10. If a required dataset or checkpoint is unavailable, prepare validation/setup scripts and stop with exact instructions. Do not fabricate data.

---

# Phase 0: Repository and asset preflight

## 0.1 Inspect the repository

Locate and document:

- LingBot-Map checkpoint and loading code.
- Current inference entry point.
- Exact LingBot output keys, shapes, units, and coordinate frames.
- Whether predicted depth is optical-axis `z` depth or Euclidean ray distance.
- Camera-pose convention: camera-to-world or world-to-camera.
- Which frame defines the sequence origin.
- Existing keyframe/direct/windowed inference settings.
- Existing SemanticKITTI and Occ3D dataset adapters.
- Existing voxel-grid definitions and occupancy evaluator.
- Existing cached LingBot outputs.
- Existing scripts that produced the `0.087` and `0.068` IoU results.

Write:

```text
reports/scale_gate/preflight_inventory.md
```

The report must contain:

- repository commit hash;
- pre-existing Git changes;
- checkpoint path and hash;
- Python, PyTorch, CUDA and GPU versions;
- discovered dataset paths, without copying private credentials;
- output tensor definitions;
- coordinate-frame diagram or explicit transformation chain;
- reusable code identified;
- missing assets and blockers.

## 0.2 Establish one configuration schema

Add or adapt one YAML configuration for the scale gate. Prefer existing configuration infrastructure. It must expose at least:

```yaml
experiment:
  seed: 0
  output_dir: artifacts/scale_gate

lingbot:
  checkpoint: null
  mode: direct
  clip_length: 5
  frame_stride: 5
  confidence_threshold: null
  inference_resolution: null

dataset:
  name: semantickitti
  root: null
  train_sequences: ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
  val_sequences: ["08"]
  camera: image_2

cache:
  root: null
  overwrite: false
  dtype: float32

scale:
  depth_convention: auto
  minimum_valid_lidar_pixels: 200
  minimum_translation_m: 0.5
  robust_estimator: weighted_median_log_ratio
  trim_log_ratio_quantiles: [0.01, 0.99]

voxel:
  use_existing_evaluator: true
```

Defaults may be adjusted after inspecting the existing code, but every adjustment must be recorded in the preflight report. Do not silently select a depth convention.

## 0.3 Add a dry-run command

Provide one command that validates paths, checkpoint readability, sequence availability, calibration parsing and cache writability without starting GPU inference.

Expected behavior:

- exits nonzero on missing required assets;
- lists exactly which files or directories are missing;
- prints the number of valid frames and clips per split;
- performs no large writes.

---

# Phase 1: SemanticKITTI dataset preparation

SemanticKITTI is used first to debug the mechanics. Sequence 08 is validation. Training sequences are 00-07 and 09-10. Do not use semantic classes for scale supervision.

## 1.1 Validate the raw layout

Support the conventional layout without assuming it blindly:

```text
<root>/dataset/sequences/<seq>/
  image_2/*.png
  velodyne/*.bin
  calib.txt
  poses.txt
  times.txt
```

The validator must check:

- matching image/LiDAR frame indices;
- monotonically ordered timestamps;
- calibration matrices have expected dimensions;
- pose count matches or is explicitly mapped to frames;
- LiDAR files contain finite `float32` XYZI records;
- images are readable and have stable native resolution;
- no sequence leakage between train and validation manifests.

Write a machine-readable inventory:

```text
artifacts/scale_gate/data_inventory.json
```

## 1.2 Parse and document calibration

Implement calibration parsing through the repository's existing code when possible. Explicitly document:

- the transformation from Velodyne coordinates to the rectified camera coordinate;
- how `P2`/camera intrinsics are used for `image_2`;
- the convention of `poses.txt`;
- the transform from the selected image camera to the pose reference camera if they differ;
- whether translations are in metres.

Projection of a LiDAR point must follow an explicitly named transformation chain, conceptually:

```text
p_velodyne -> p_camera -> p_pixel
```

Do not treat the `P2` matrix as a pure intrinsic matrix if it contains a stereo baseline term.

## 1.3 Build deterministic clip manifests

Create chronological clips with configurable:

- clip length, initially 5 frames;
- frame stride, initially 5 native frames, corresponding to approximately 2 Hz for 10 Hz KITTI sequences;
- clip sampling stride;
- boundary behavior.

Each JSONL record must contain only relative or configured paths and explicit identifiers:

```json
{
  "dataset": "semantickitti",
  "sequence": "08",
  "clip_id": "08_000100_000120_s5",
  "frame_ids": [100, 105, 110, 115, 120],
  "image_paths": ["..."],
  "lidar_paths": ["..."],
  "calibration_path": "...",
  "poses_path": "...",
  "times_path": "..."
}
```

Write separate manifests for train, validation and smoke test. The smoke manifest should contain two clips.

## 1.4 Project LiDAR into each image

For every selected frame:

1. Read XYZI points.
2. Transform them into the selected camera frame.
3. Remove points behind the camera and non-finite points.
4. Project to the original image.
5. Apply z-buffering: when several points map to one pixel, retain the nearest valid point.
6. Transform pixel coordinates exactly according to LingBot's resize/crop/pad preprocessing.
7. Store sparse metric depth and a valid mask at LingBot output resolution.

The cache must distinguish:

- optical-axis depth;
- Euclidean range;
- original image coordinates;
- processed image coordinates.

Do not convert between these silently.

## 1.5 Visual calibration QA

Generate overlays for at least 20 deterministic frames spanning several sequences:

- RGB with projected LiDAR colored by metric depth;
- original-resolution projection;
- processed-resolution projection;
- a small table with valid point count and depth range.

Save them under:

```text
artifacts/scale_gate/qa/semantickitti_projection/
```

The experiment must not continue to full caching until these overlays are visually inspected and the report records `PASS`.

## 1.6 Dataset-preparation tests

Add tests for:

- calibration parser shapes and finite values;
- synthetic known-point projection;
- z-buffer nearest-point selection;
- resize/crop coordinate transformation;
- train/validation sequence separation;
- deterministic manifest generation;
- missing/corrupt frame reporting.

---

# Phase 2: Cache frozen LingBot outputs

## 2.1 Cache contract

Use existing caches when they satisfy the contract. Otherwise create one cache file per clip, plus an index.

Required fields:

```text
clip_id
dataset
sequence
frame_ids
processed_image_size
depth_prediction[T,H,W]
depth_confidence[T,H,W] if available
camera_to_world[T,4,4] or world_to_camera[T,4,4], explicitly named
predicted_intrinsics[T,3,3] if available
pre-GCT pooled_image_feature[T,C] if available without changing LingBot
preprocessing_metadata
checkpoint_hash
config_hash
```

Never use a generic key such as `pose` without naming its direction.

## 2.2 Inference constraints

- Set `model.eval()`.
- Use `torch.inference_mode()`.
- Verify every parameter has `requires_grad=False`.
- Process frames chronologically.
- Use direct mode for the five-frame clip.
- Do not reset state within a clip.
- Reset state between independent clips.
- Preserve the same anchor-frame policy in all experiments.
- Record runtime and peak GPU memory.

## 2.3 Cache validation

For every cache:

- all values must be finite after applying the declared validity mask;
- rotations must be close to orthonormal;
- homogeneous matrix last row must be valid;
- frame count and IDs must match the manifest;
- predicted depth must be positive where valid;
- the checkpoint and preprocessing hashes must match the current experiment.

Produce an index with success/failure status and failure reason. Resuming must skip only valid completed clips.

## 2.4 Reconstruct before scaling

Render at least five cached clips as raw LingBot point clouds and camera trajectories. Confirm:

- neighboring frames overlap plausibly;
- pose direction is not inverted;
- axes are consistent;
- depth and pose share the same canonical scale;
- points are unprojected using the correct intrinsics/depth convention.

---

# Phase 3: Oracle scale diagnosis

## 3.1 Depth-derived scale

For every clip, collect valid pairs `(d_pred, d_gt, weight)` at projected LiDAR pixels.

For scale-only alignment, compute in log space:

```text
r_i = log(d_gt_i) - log(d_pred_i)
log(s_depth_star) = weighted_median(r_i)
```

Use robust trimming only as configured. Record before and after trimming counts. Confidence may be used as a weight but must be ablated against uniform weights.

Also compute per-frame scale estimates and their dispersion. Large per-frame variation is evidence that a single clip-level scalar is insufficient or that conventions are wrong.

## 3.2 Pose-derived scale

Use ground-truth and predicted relative translations over frame pairs with sufficient metric motion:

```text
s_pose_pair = ||delta_t_gt|| / ||delta_t_pred||
```

Aggregate robustly in log space. Exclude pairs below the configured metric-motion threshold because ratios become unstable.

Do not use Sim(3)-aligned ATE as the pose-scale target; Sim(3) alignment already absorbs scale. Use the raw relative translations after converting both trajectories to the same convention and origin.

## 3.3 Depth/pose agreement test

Compute:

```text
agreement_error = abs(log(s_depth_star / s_pose_star))
```

Visualize and report its distribution. A single scale is only physically coherent when the depth-derived and pose-derived scales broadly agree.

If they disagree systematically:

1. recheck depth convention;
2. recheck pose direction and camera reference;
3. recheck image-camera versus pose-camera extrinsics;
4. recheck whether LingBot normalizes depth and translation identically;
5. do not train a scale predictor until the discrepancy is explained.

## 3.4 Joint oracle scale

Implement a one-dimensional robust objective combining normalized depth residual and translation residual. Keep the depth-only and pose-only solutions; do not replace them with only the joint solution.

The joint objective must be insensitive to the number of LiDAR pixels overwhelming the smaller number of pose pairs. Normalize each term before applying a configurable weight.

## 3.5 Apply scale correctly

For every oracle and baseline scale:

- multiply predicted depth by `s`;
- multiply every camera translation by `s` relative to the anchor origin;
- keep rotations and intrinsics unchanged;
- reconstruct and fuse point clouds again;
- voxelize with the existing evaluator and unchanged grid definition.

## 3.6 Required oracle baselines

Evaluate:

1. Raw LingBot canonical scale.
2. One global source-training median scale.
3. Per-sequence oracle depth scale.
4. Per-sequence oracle pose scale.
5. Per-sequence joint oracle scale.
6. Existing previous scale method, if different.

Where a method is not valid at deployment, mark it explicitly as `ORACLE`.

## 3.7 Metrics

Report per clip and aggregated:

- `abs(log(s_est / s_star))`;
- relative scale error;
- scaled sparse-depth AbsRel;
- depth delta-1 if compatible with the depth convention;
- pose translation error without Sim(3) scale alignment;
- occupancy IoU;
- occupancy precision;
- occupancy recall;
- predicted occupied voxel count;
- in-grid point fraction;
- out-of-grid point fraction.

Use the exact same frame selection, confidence filtering, voxelizer, grid and evaluation mask for all scale methods.

## 3.8 Bootstrap uncertainty

Use clip-level bootstrap resampling with a fixed seed to estimate 95% confidence intervals for downstream IoU differences between raw, median-scale and oracle-scale methods.

Do not claim improvement from a point estimate whose confidence interval includes no change.

## 3.9 Oracle report and decision

Write:

```text
reports/scale_gate/oracle_scale_report.md
artifacts/scale_gate/oracle_metrics_per_clip.csv
artifacts/scale_gate/oracle_summary.json
```

The report must answer:

1. Does one scalar align both depth and translation?
2. How much of the occupancy failure is attributable to scale?
3. Does oracle scale produce a statistically supported downstream improvement?
4. Why did the previous SemanticKITTI oracle number not exceed the causal result?
5. Is training a learned scale predictor justified?

Operational decision:

- If oracle scale does not improve downstream metric occupancy, retain the simplest explicit scale baseline for coordinate validity and do not invest in a complex scale model.
- If oracle scale improves occupancy, proceed to Phase 4.
- If depth and pose require inconsistent scales, stop and diagnose the representation/coordinate contract before Phase 4.

---

# Phase 4: Learned non-oracle scale predictor

Execute this phase only when the oracle report justifies it, or when the project lead explicitly requests it despite a small oracle gain.

## 4.1 Inference contract

At inference, the predictor may use only:

- input RGB frames;
- frozen pre-GCT/image features;
- frozen LingBot depth statistics;
- frozen LingBot pose/trajectory statistics;
- predicted intrinsics or declared input intrinsics, depending on the chosen setting.

It must not use:

- LiDAR;
- ground-truth depth;
- ground-truth pose;
- semantic labels;
- dataset name or one-hot dataset identity;
- filenames or sequence IDs as learnable inputs;
- statistics calculated using validation/test targets.

## 4.2 Target

Train on `log(s_star)` from the selected oracle definition. Preserve all raw oracle targets and quality indicators. Exclude or down-weight clips whose oracle target is unreliable because of too few valid pixels, insufficient motion or depth/pose disagreement.

Do not silently delete these clips; report exclusion counts and reasons.

## 4.3 Minimal feature set

Implement the following scale predictors incrementally:

### Baseline A: Global median

One scalar computed only from source-training clips.

### Baseline B: Depth-statistics MLP

Features per frame may include log-depth quantiles, mean, standard deviation, valid/confident fraction and confidence quantiles. Aggregate across time using mean and standard deviation.

### Baseline C: Image-feature MLP

Use frozen pooled pre-GCT or DINO-style image features. Aggregate across frames with a permutation-aware but temporally simple mean/std representation for the first version.

### Proposed minimal model: Combined MLP

Concatenate:

- depth-statistics features;
- frozen pooled image features;
- predicted relative-translation magnitude statistics;
- predicted focal length/FOV statistics when available.

Use a small MLP with two hidden layers, normalization and dropout. Predict one `log(s_hat)` per clip. Keep model size small enough that overfitting and inference cost are easy to measure.

Do not introduce a transformer or recurrent network until the MLP baselines are complete.

## 4.4 Training

Use:

```text
loss = SmoothL1(log(s_hat), log(s_star))
```

Requirements:

- deterministic source-training/validation split by sequence, never by individual overlapping clips;
- fixed seeds;
- early stopping on median validation log-scale error;
- checkpoint best and last;
- TensorBoard or existing experiment tracker;
- training curves and configuration snapshot;
- no LingBot gradients;
- cached LingBot features only during scale-model training.

Before full training:

1. Overfit 16 clips and demonstrate near-zero training error.
2. Run a 2-epoch smoke experiment.
3. Only then launch the configured training run.

## 4.5 Evaluation

For every learned model:

1. predict `s_hat` without accessing targets;
2. apply it to depth and camera translations;
3. rerun point-cloud fusion and voxel evaluation;
4. compare scale metrics and downstream occupancy metrics against the global-median and oracle baselines;
5. measure runtime, model size and peak memory.

The main criterion is downstream metric occupancy, not scale error alone.

## 4.6 Learned-scale decision gate

The learned model passes only if:

- it statistically outperforms the source-training global-median baseline on held-out sequences;
- it improves downstream occupancy in the direction of the oracle result;
- it does not use target-domain statistics or metadata;
- its performance is not driven by one sequence or a few outlier clips;
- scale predictions remain positive, finite and reasonably calibrated.

Prefer reporting the fraction of the oracle downstream gain recovered:

```text
recovered_oracle_fraction =
  (IoU_learned - IoU_baseline) / (IoU_oracle - IoU_baseline)
```

When the denominator is near zero, report the scale problem as downstream-insensitive rather than interpreting the ratio.

---

# Phase 5: Cross-dataset preparation and evaluation

This phase establishes whether the scale mechanism generalizes. Implement dataset adapters behind a common interface. Do not mix dataset-specific target information into model inputs.

## 5.1 Common dataset interface

Each adapter must return:

```text
clip_id
chronological RGB frames
camera intrinsics/extrinsics metadata
LiDAR or metric depth for target construction
metric poses when available
validity metadata
```

All adapters must feed the same LingBot cache schema and scale-target builder.

## 5.2 Source datasets

Recommended order:

1. SemanticKITTI for implementation debugging and in-domain validation.
2. DDAD plus VKITTI2 for a first real+synthetic source mixture.
3. PandaSet only after the two-source experiment is functioning.

Do not download these automatically. Add:

```text
docs/data_setup_scale_gate.md
tools/check_<dataset>_layout.py
```

Document official acquisition requirements, expected directories and verification commands. If local data are absent, finish the adapters/tests that can be implemented without bytes and report the missing assets.

## 5.3 Target datasets

For the strict generalization experiment:

- train the scale model using source datasets only;
- freeze the model;
- test unchanged on SemanticKITTI sequence 08 and Occ3D-nuScenes validation;
- do not recompute a target-dataset median scale;
- do not tune thresholds on the target test results.

An in-domain SemanticKITTI-trained result may be reported as a diagnostic, but never call it zero-shot on SemanticKITTI.

## 5.4 Dataset balance

When mixing sources:

- sample datasets explicitly rather than in proportion to their raw size;
- report the number of scenes, clips and valid scale targets from each source;
- prevent near-duplicate overlapping clips from leaking across splits;
- store source-dataset identity for analysis only, not as a predictor input.

---

# Required scripts or equivalent entry points

Prefer existing package conventions. If no suitable commands exist, implement equivalents of:

```bash
# Validate configuration and assets
python tools/scale_gate/preflight.py --config configs/scale_gate/semantickitti.yaml --dry-run

# Build deterministic manifests
python tools/scale_gate/prepare_manifest.py --config configs/scale_gate/semantickitti.yaml

# Generate LiDAR projection QA
python tools/scale_gate/project_lidar.py --config configs/scale_gate/semantickitti.yaml --qa-only

# Cache frozen LingBot outputs
python tools/scale_gate/cache_lingbot.py --config configs/scale_gate/semantickitti.yaml --split smoke
python tools/scale_gate/cache_lingbot.py --config configs/scale_gate/semantickitti.yaml --split train
python tools/scale_gate/cache_lingbot.py --config configs/scale_gate/semantickitti.yaml --split val

# Build oracle targets and run diagnostics
python tools/scale_gate/build_scale_targets.py --config configs/scale_gate/semantickitti.yaml
python tools/scale_gate/eval_oracle_scale.py --config configs/scale_gate/semantickitti.yaml

# Train and evaluate the minimal non-oracle scale model
python tools/scale_gate/train_scale.py --config configs/scale_gate/semantickitti.yaml --model combined_mlp
python tools/scale_gate/eval_scale.py --config configs/scale_gate/semantickitti.yaml --checkpoint <best_checkpoint>

# Run tests
pytest -q tests/scale_gate
```

If repository conventions require different paths or commands, preserve the same responsibilities and add a command mapping to the final report.

---

# Required tests

At minimum, implement tests for:

1. Known synthetic scalar recovery from depth.
2. Known synthetic scalar recovery from trajectory translations.
3. Joint scale recovery.
4. Correct scaling of translations but not rotations.
5. Correct scaling of depth.
6. Camera-to-world versus world-to-camera conversion.
7. LiDAR-to-image projection.
8. Original-to-processed image coordinate transformation.
9. Z-buffering.
10. Manifest determinism.
11. Train/validation sequence isolation.
12. Cache hash invalidation.
13. Resume behavior.
14. Inference feature contract excluding target-only fields.
15. MLP overfit test on a tiny synthetic dataset.

Tests must run without requiring a full dataset wherever synthetic fixtures suffice.

---

# Required visualizations

Generate reproducibly:

- LiDAR projection overlays before and after image preprocessing;
- raw versus oracle-scaled point clouds;
- raw versus learned-scaled point clouds;
- predicted and ground-truth camera trajectories in a common metric coordinate system;
- histogram of `log(s_star)`;
- histogram of per-frame scale dispersion;
- scatter plot `s_depth_star` versus `s_pose_star`;
- scale error versus downstream occupancy IoU;
- per-sequence IoU comparison for raw, median, learned and oracle scale.

Every plot must include units, sample count and configuration identifier.

---

# Reproducibility and artifact contract

Every run must save:

```text
config_resolved.yaml
environment.txt
git_state.txt
checkpoint_hashes.json
manifest_hashes.json
metrics_per_clip.csv
summary.json
run.log
```

Do not store absolute dataset paths inside shareable result files; replace them with configured aliases where possible.

Cache and run directories must never be silently overwritten. Use a unique run ID derived from timestamp plus config hash, or the repository's existing experiment manager.

---

# Final deliverables

Claude Code must finish by producing:

1. Working, tested dataset-preparation pipeline for SemanticKITTI.
2. Validated frozen LingBot cache pipeline.
3. Depth-, pose- and joint-oracle scale computation.
4. Correct metric application to both depth and pose translations.
5. Raw/median/oracle downstream occupancy comparison.
6. Bootstrap confidence intervals.
7. Minimal learned scale baselines if the oracle gate justifies training.
8. Cross-dataset adapter/setup documentation.
9. Tests and reproducible commands.
10. Final report:

```text
reports/scale_gate/final_scale_gate_report.md
```

The final report must contain:

- what was implemented;
- exact commands run;
- data used and excluded;
- all assumptions;
- coordinate and depth conventions;
- tables of scale, depth, pose and occupancy metrics;
- failures and unresolved discrepancies;
- comparison with the previous `0.087`/`0.068` results;
- whether scale is a major downstream bottleneck;
- whether the learned model passes;
- one explicit recommendation:
  - `PROCEED_TO_VISIBLE_OCCUPANCY`,
  - `USE_SIMPLE_SCALE_BASELINE_AND_PROCEED`,
  - `ESCALATE_TO_METRIC_HEAD_FINETUNING`, or
  - `STOP_AND_FIX_COORDINATES`.

Do not begin the next project stage automatically.

---

# Suggested implementation sequence

Follow this order and commit/report after each independently testable milestone:

1. Preflight inventory and configuration.
2. SemanticKITTI manifest and calibration parsing.
3. LiDAR projection tests and visual QA.
4. Smoke LingBot cache on two clips.
5. Full validated cache reuse/generation.
6. Synthetic oracle-scale unit tests.
7. Real depth/pose/joint oracle analysis.
8. Downstream raw/median/oracle occupancy evaluation.
9. Oracle report and decision.
10. Minimal learned scale baselines only if justified.
11. Cross-dataset setup/adapters.
12. Final report and handoff.

At every stage, prefer a small verified result over launching a larger experiment with an unverified coordinate system.