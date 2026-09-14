# KITTI-360 data scaling (seven training drives): does more source data improve semantic completion?

**Outcome.** More data gives a small, consistent, statistically supported gain in held-out
semantic completion — **but on an absolute level of about 0.5 % mIoU**, so it is practically
negligible — and no reliable occupancy gain. Formally Decision **A**, with the caveat that the
effect is ~0.1 pp on a model that barely predicts semantics at all.

“100 %” is **all eligible anchors across the seven SSCBench-KITTI-360 training drives 0000,
0002, 0003, 0004, 0005, 0007, 0010** (9,672 anchors; none excluded). Development evaluation on
drive 0006 (590 anchors) — a drive that has informed earlier project decisions. SemanticKITTI and
Occ3D-nuScenes untouched; 0009 unused.

| run | train anchors | epochs | sem mIoU, unobserved→later measured (primary) | occ IoU, unobs→later | sem mIoU observed | occ IoU observed | sem mIoU full | occ IoU full |
|---|---|---|---|---|---|---|---|---|
| Small-1 (seed 0) | 2,483 | 19.5 | 0.0049 | 0.1219 | 0.0091 | 0.0994 | 0.0065 | 0.1060 |
| Full-1 (seed 0) | 9,672 | 5.0 | **0.0062** | 0.1206 | 0.0100 | 0.1037 | 0.0074 | 0.1065 |
| Small-2 (seed 1) | 2,483 | 19.5 | 0.0048 | 0.1181 | 0.0095 | 0.1054 | 0.0067 | 0.1065 |
| Full-2 (seed 1) | 9,672 | 5.0 | **0.0054** | 0.1227 | 0.0093 | 0.1004 | 0.0069 | 0.1069 |

| paired Δ (full − small), primary region | Δ sem mIoU [95 % block CI] | Δ occ IoU [95 % block CI] |
|---|---|---|
| seed 0 | +0.0012 [+0.0008, +0.0018] | −0.0013 [−0.0023, −0.0005] |
| seed 1 | +0.0007 [+0.0005, +0.0008] | +0.0045 [+0.0038, +0.0054] |
| mean | +0.0009 | +0.0016 |

CIs resample contiguous blocks of 20 development anchors (60 stream frames, longer than the
20-frame target horizon) with identical draws for both conditions; they capture evaluation
uncertainty only. Training variance is represented by the two seeds.

## 1. What model was trained, and what changed between conditions?

The adopted Gate 8C-1 completion model, unchanged: CompletionUNet (width 24, replicate
padding), focal + Dice occupancy loss plus 0.5 × KL to the future Trident teacher, AdamW 1e-3,
cosine schedule, batch 4, 128×128×32 crops, drive-uniform sampling
(`configs/gate8c1_sky/seed0.yaml`, `core.run._gate8a_train.step_loss`). The readout is the
residual occupancy lock at logit 0, and map-teacher semantics where the causal map has evidence,
network softmax elsewhere. Within each seed both runs start from byte-identical weights and use
the same inputs, targets, optimizer, schedule, checkpoints and a common 12,100-step budget. Only
the training manifest differs (so the sampled draws differ).

## 2. Were targets already independently aligned, or was a correction required?

A correction was required, and both conditions use the corrected targets.
- **Semantic targets:** the existing ones came from the LingBot future mapper, which the
  correspondence audit showed is misregistered. They were rebuilt without the mapper: LiDAR-occupied
  voxels are projected into each future camera with GT calibration and poses, pass a LiDAR presence
  + occlusion visibility test, and sample cached Trident directly.
- **Occupancy targets:** rebuilt for all drives with the current raw-LiDAR `rawtarget.py`, whose
  conflicting-voxel fix the cached targets predate.
- **Inputs:** unchanged. They are byte-identical to the adopted samples except float-accumulated
  teacher evidence (`sem_w` ≤ 1e-3 relative, `sem_p` ≤ 1 u8 step), which is the builder's own
  run-to-run arithmetic.
- **Checks (6/6, `targets/verification.json`):** independent LiDAR projection, deterministic
  overlays, inputs unchanged, future and GT not in inputs, occupancy targets trusted, semantic
  targets registered.
- **Evaluation labels:** SSCBench human labels, read after a verified grid-origin conversion (label
  grid at the camera origin). Their occupancy is never used.

Earlier three-drive and mapper-target runs are not matched controls.

## 3. How much additional data and scene coverage did the full condition provide?

3.9× the anchors (9,672 vs 2,483) and about 2.9× the 20 m spatial cells (2,085 vs 709); every
drive keeps ~25 % in Small (blocks of 20 anchors, every fourth block). Per drive, eligible / small
/ cells full→small: 0000 2,094 / 534 / 363→137; 0002 2,769 / 700 / 522→186; 0003 193 / 60 /
83→28; 0004 2,202 / 560 / 476→159; 0005 1,249 / 320 / 178→79; 0007 569 / 149 / 276→72; 0010 596 /
160 / 187→48. At equal steps Full saw 96 % of its anchors about 5 times each; Small saw every
anchor about 19.5 times.

## 4. Did unobserved-space semantic accuracy improve, and by how much?

Yes, marginally and consistently: +0.0012 and +0.0007 mIoU (absolute), both CIs above zero. Gains
concentrate in classes the model predicts at all: car 0.014→0.021 / 0.010→0.017, other-structure
0.005→0.008, vegetation 0.036→0.041 (seed 0). But absolute semantic mIoU is 0.5–0.6 %. Every class
IoU is below 0.05; bicycle, motorcycle, person and traffic-sign are 0, and terrain, parking and pole are below 0.0005. `road` is predicted on 8.8 M voxels
against 128 k ground-truth voxels, so false occupancy dominates semantic error.

## 5. Did occupancy improve?

No reliable change. In the primary region Δ occ IoU is −0.0013 (seed 0) and +0.0045 (seed 1),
with CIs excluding zero in opposite directions. Observed space is also sign-inconsistent
(+0.0043 / −0.0050). Over the full mask the difference is +0.0005 / +0.0004.

## 6. Was the result consistent across seeds and adequately optimized?

The semantic sign is consistent across seeds; occupancy is not. Optimization is adequate for this
schedule. Over the last 2,100 steps the full runs changed by +0.0001 semantic mIoU and −0.0004 /
−0.0014 occupancy IoU. Validation loss is flat (1.733 / 1.732 full vs 1.741 / 1.738 small) as the
cosine schedule ends, so Decision D does not apply. This is a fixed-compute comparison, not an
asymptotic one.

## 7. Which decision is supported, and what single next action follows?

**A — scaling helps semantic completion**, in the narrow sense the rule defines, at a practically
irrelevant magnitude. Occupancy does not benefit, and absolute quality stays near zero. Both the
local-registration audit and Gate 9 locate the limit upstream: on matched pixels with GT pose,
LingBot depth reduces the observed map from 0.44 to 0.075 IoU. More source data cannot recover
geometry the inputs do not contain.

**Next action:** do not scale data further. Run Gate 9's recommended depth-source sufficiency test
(LingBot vs frozen MoGe-2 vs causally calibrated LingBot vs the LiDAR-scale oracle, on matched
pixels under GT pose) before any further completion training.

**Limitations.** Development evaluation on one previously used drive. Two seeds. Semantic ground
truth covers only human-labelled voxels after the grid-origin conversion. One fixed occupancy threshold. The 0003 sample-build provenance record
was lost during the run-to-run check; data unaffected.

Artifacts: `results_table.md`, `analysis.json` (paired deltas, curves, class-wise),
`experiment_spec.json` (commit, code hashes, checkpoints, init digests, manifests, budget),
`manifests/`, `runs/`, `eval/`, `targets/` (verification, overlays, sem_w note),
`figures/{learning_curves,paired_deltas,classwise_delta}.png`, `figures/qualitative/`,
`code_diff_tracked.patch`.
