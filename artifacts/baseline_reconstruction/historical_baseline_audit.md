# Historical baseline audit -- the post-ceiling-fix zero-shot baseline (2026-09-14)

Every number below is read from a file on this machine (path given). Nothing was rerun on a target dataset; the ceiling-fix verification (§5) recomputes stored predictions and rebuilds KITTI-360 targets only.

## 1. Verified post-ceiling-fix baseline

| Dataset | Verified metric | Value (median of 3 seeds) | Per seed 0/1/2 | Mean ± sd | Anchors | Provenance |
|---|---|---|---|---|---|---|
| semantickitti | SC IoU (binary occupancy, official mask) | **22.71 %** | 22.47 / 23.62 / 22.71 | 22.93 ± 0.60 | 163 | output/skyfix/eval/eval_semantickitti_past5_seed0.json; output/skyfix/eval/eval_semantickitti_past5_seed1.json; output/skyfix/eval/eval_semantickitti_past5_seed2.json |
| occ3d | SC IoU (binary occupancy, official mask) | **33.88 %** | 33.49 / 35.03 / 33.88 | 34.13 ± 0.80 | 1182 | output/skyfix/eval/eval_occ3d_past5_seed0.json; output/skyfix/eval/eval_occ3d_past5_seed1.json; output/skyfix/eval/eval_occ3d_past5_seed2.json |


Precision / recall / SSC mIoU of the median seed: semantickitti: P 34.49 / R 39.95 / mIoU 3.35; occ3d: P 75.24 / R 38.13 / mIoU 4.24.

### Remembered values, checked

| Dataset | Remembered | Verified | Verdict |
|---|---|---|---|
| SemanticKITTI | 22.71 % | 22.71 % | correct (median seed 2; seeds 22.47 / 23.62 / 22.71) |
| Occ3D-nuScenes | 30.74 % | 33.88 % | **wrong**: 30.74 appears nowhere in the repository. The pre-fix past5 median is 30.89 (seeds 30.89 / 31.27 / 30.70) and the post-fix matched-forward single-seed number is 30.70; the post-fix past5 median is 33.88 (seeds 33.49 / 35.03 / 33.88). |
| SemanticKITTI (pre-fix) | 12.74 % | 12.74 % | correct, but it is the *pre-fix* headline |
| Occ3D (pre-fix) | 30.89 % | 30.89 % | correct, pre-fix headline |


## 2. Every recorded evaluation of the two model versions

`sky` = adopted post-fix model (`frozen_manifest_sky.json`); `prefix` = shipped pre-fix model (`frozen_manifest.json`). Metric = `semantic.completion.binary_iou` of each eval JSON (what the gate prints as `SC_IoU`).

| variant | protocol | SC IoU % seeds 0/1/2 | median | mean | P (median seed) | R (median seed) | SSC mIoU | anchors | record dir |
|---|---|---|---|---|---|---|---|---|---|
| semantickitti | sky | past5 | 22.47 / 23.62 / 22.71 | 22.71 | 22.93 | 34.49 | 39.95 | 3.35 | 163 | output/skyfix/eval |
| semantickitti | prefix | past5 | 12.32 / 12.74 / 14.28 | 12.74 | 13.11 | 15.88 | 39.15 | 2.63 | 163 | artifacts/gate8c1 |
| semantickitti | prefix | stream | 11.49 / 12.93 / 12.83 | 12.83 | 12.42 | 15.66 | 41.51 | 2.57 | 163 | artifacts/gate8c1 |
| semantickitti | prefix | occany_fwd | 13.94 / 13.02 / 15.25 | 13.94 | 14.07 | 17.48 | 40.76 | 2.64 | 161 | artifacts/gate8c1 |
| occ3d | sky | past5 | 33.49 / 35.03 / 33.88 | 33.88 | 34.13 | 75.24 | 38.13 | 4.24 | 1182 | output/skyfix/eval |
| occ3d | prefix | past5 | 30.89 / 31.27 / 30.70 | 30.89 | 30.95 | 67.14 | 36.39 | 4.42 | 1182 | artifacts/gate8c1 |
| occ3d | prefix | stream | 31.22 / 31.99 / 31.98 | 31.98 | 31.73 | 66.75 | 38.04 | 4.32 | 1182 | artifacts/gate8c1 |
| occ3d | prefix | occany_fwd | 27.56 / 27.61 / 27.51 | 27.56 | 27.56 | 63.83 | 32.66 | 3.40 | 882 | artifacts/gate8c1 |


`RESULT` lines printed by the sky runs (`output/skyfix/*.log`): {"semantickitti": {"0": 22.47, "1": 23.62, "2": 22.71}, "occ3d": {"0": 33.49, "1": 35.03, "2": 33.88}}.

### Numbers that exist only in prose (their JSON records were overwritten by the rejected `sky2` run)

| label | dataset | value % | source | protocol | comparability to canonical |
|---|---|---|---|---|---|
| sky model, OccAny SSCMetrics, 161 shared frames, seeds 0/1/2 | semantickitti | 22.83 | artifacts/gate8c1/ceiling_fix.md (One model, both benchmarks; median of 22.61/23.79/22.83) | OccAny's stored labels on 161 of 163 frames, scored by OccAny SSCMetrics.get_score_completion | partially comparable: subset of frames and OccAny's label volume (matches ours on 1 of 161 frames) |
| sky model, OccAny SSCMetrics, 882 shared samples, seed 0 | occ3d | 33.28 | artifacts/gate8c1/ceiling_fix.md (One model, both benchmarks) | 882 of 1182 Occ3D anchors shared with OccAny's output, OccAny SSCMetrics | partially comparable: subset of anchors, single seed |
| sky model, matched forward (non-causal) input, seed 0 | occ3d | 30.7 | artifacts/gate8c1/ceiling_fix.md | occany_fwd offsets 0,+2,+4,+6,+8 (non-causal) | different protocol: non-causal window |
| sky2 (rejected neighbourhood rule), seed 0 | occ3d | 28.12 | artifacts/gate8c1/ceiling_fix.md addendum; artifacts/gate8c1/occany_headtohead.json (overwritten by this run) | 882 shared samples, OccAny SSCMetrics | different model: rejected variant |


`artifacts/gate8c1/occany_headtohead.json` (threshold −0.21875 = sky2 seed 0) and `occany_kitti_headtohead.json` (threshold −0.28125 = sky2 seed 2) hold the **sky2** results (28.12 / 22.93), not the adopted model's; the adopted model's head-to-head numbers (33.28 / 22.83) survive only in `ceiling_fix.md`. This audit records the correction here and leaves the files untouched.

## 3. The ceiling fix: before, the change, the first valid evaluation, what superseded it

| event | record |
|---|---|
| prefix_code_commit | 9a64832 2026-08-27 18:18:30 +0200 Add something |
| prefix_manifest_frozen_at | 2026-09-03T20:03:16+0200 |
| fix_applied | 2026-09-04 (untracked working tree): configs/gate8c1_sky/*.yaml padding_mode=replicate; rawtarget.py open-sky-is-FREE rule; targets rebuilt at /media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky; sky_seed{0,1,2}_last.pt trained 19:05-19:15 (file mtimes) |
| fix_first_tracked_commit | f7b7552 2026-09-07 13:06:33 +0200 Track the research tree in git  [tree tracked with SKY_NEIGHBOURHOOD_VOX=7, i.e. the sky2 form] |
| sky_rule_measured_return_fix_commit | 647bcf3 2026-09-07 13:29:05 +0200 Fix the open-sky rule marking measured returns as free |
| docs_commit | 0ba3b5d 2026-09-11 09:15:29 +0200 Fix ceiling bug and update document  [documentation + whitespace only; no pipeline change] |
| sky_manifest_frozen_at | 2026-09-04T19:19:47+02:00 |
| first_valid_eval_semantickitti | 2026-09-08T15:07:25+02:00 |
| first_valid_eval_occ3d | 2026-09-10T17:22:15+02:00 |
| superseding_baseline | none: Gates 9-14 (2026-09-12..14) were source-side diagnostics and produced no new zero-shot evaluation |


The fix has two halves; the ablation in `reports/the_ceiling_bug.md` §5 shows the padding half alone is *worse* (11.60) and the supervision half carries the whole 12.74 → 22.71 gain. The plain-column sky rule (no neighbourhood) trained the adopted model; the neighbourhood-7 extension (`sky2`) was rejected (Occ3D 33.28 → 28.12 on seed 0). Commit `647bcf3` later fixed the rule carving through measured returns; **no model was retrained after it**, so the canonical checkpoints rest on targets from the pre-`647bcf3` plain-column rule (verified bit-exactly in §5).

## 4. The pipeline that produced the canonical numbers

| field | value |
|---|---|
| lingbot_mode | long-stream causal replay (gates/gate7b/replay.py): one persistent KV cache per stream, reset only at a stream boundary; sliding 64-block KV window; per-frame depth/pose cached to /media/SSD1/MINH_DATASETS/lingbot_gate7b/stream |
| depth_source | LingBot-Map canonical depth x metric scale (LB-M); MoGe-2 used only as the scale gauge |
| scale_handling | per stream: exp(median(log_s of the FIRST five stream frames)) from MoGe-2 vs LingBot candidates (core/mapping/scale.py); the same frozen stream scale is forced on every past5 window map |
| pose_source | LingBot-Map streaming pose (long-stream), translation scaled by the stream scale |
| mapper | core/mapping/mapper.py IncrementalMapper: conf >= 1.5, depth 1-60 m, +-0.2 m occupied band, free carve to d-0.2 m at stride 4, log-odds accumulation, semantic accumulators on |
| voxel_representation | log-odds occupancy + w_free + n_obs + age + semantic evidence, exported to the benchmark grid |
| input_channels | 32 |
| channel_meaning | 0 logodds/4, 1 w_free/20, 2 observed, 3 1-observed, 4 n_obs/50, 5 age/100, 6 sem_w/20, 7..31 Trident-H union-vocabulary class probabilities (core/README.md) |
| semantics | present: Trident-H teacher probabilities as input; semantic KL loss w_sem = 0.5; semantic head |
| residual_lock | apply_residual: final = base + residual * [|base| < 2.0] (LOCK_LOGODDS = 2) |
| completion_architecture | core/model/net.py CompletionUNet, width 24, 32-ch input, occupancy residual + semantic logits |
| n_params | 986114 |
| input_shape | [1, 32, X, Y, Z] on the benchmark prediction grid |
| output_shape | [1, 1, X, Y, Z] occupancy residual (+ [1, U, X, Y, Z] semantic logits) |
| padding_mode | replicate (the padding half of the ceiling fix) |
| training_source | KITTI-360 drives 0003/0007/0010 (train), 0006 (validation/selection) |
| supervision | raw-Velodyne targets t..t+20 with GT poses, open sky FREE (plain column rule, SKY_MARGIN_VOX 1, no neighbourhood), targets at /media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky |
| training_steps | 6000 |
| losses | focal (gamma 2, pos-weight cap 8) + Dice, w_unknown 2, + 0.5 semantic KL; AdamW 1e-3 wd 0.01; cosine; clip 1; batch 4; crop 128x128x32 |
| checkpoint_selection | manifest rule: highest full-grid occupancy AP on 0006 -> 'last' (step 6000) for all three seeds |
| threshold_selection | per seed, final-log-odds threshold maximising full-grid IoU on 0006: -0.125 / -0.4375 / -0.3125 |
| evaluator | core/run/eval_target.py (G8C1_MANIFEST=artifacts/gate8c1/frozen_manifest_sky.json), mode past5 |
| zero_shot_window | fresh map over stream indices i-4..i, frozen stream scale, LingBot stream depth/pose |


## 5. Verification that the recovered result is the fixed pipeline

| # | check | passed |
|---|---|---|
| 1 | sky checkpoints: replicate padding, 32 input channels, sha256 == manifest; pre-fix: zeros padding | PASS |
| 2 | sky eval JSONs name the manifest checkpoints/thresholds; RESULT log lines agree; protocol past5 | PASS |
| 3 | sky targets supervise the top layer (>= 4x and >= +15 pp over the pre-fix fraction) | PASS |
| 4 | seed-2 prediction dump re-scores to the recorded SC IoU and the top layer is no longer filled (< 2 % vs 95.1 % pre-fix) | PASS |
| 5 | cached target roots are reproduced bit-exactly by exactly one rule variant each (sky = pre-647bcf3 plain column rule; sky2 = pre-647bcf3 neighbourhood 7; current = today's file) | PASS |
| 6 | no target-dataset path was opened during the audit | PASS |


Top-layer (+4.3 m) supervised fraction on 40 anchors of 0006, from check 3: {"prefix": 3.79, "sky": 20.11, "current_gate12_13_builder": 58.72} % -- note the *current* Gate 12/13 builder (neighbourhood-7 rule + 647bcf3) supervises far more sky than the plain rule that trained the canonical model.


Details: `historical_baseline.json` → `verification`. Highlights: sky targets supervise the top layer (>= 4x and >= +15 pp over the pre-fix fraction): {"ratio_sky_over_prefix": 5.3036}; seed-2 prediction dump re-scores to the recorded SC IoU and the top layer is no longer filled (< 2 % vs 95.1 % pre-fix): {"n_anchors": 163, "sc_iou_from_dump": 0.2271, "recorded_sc_iou": 0.2271, "tp": 7496311, "fp": 14241045, "fn": 11266619, "top_layer_pred_occupied_pct": 0.7297, "top_layer_gt_occupied_pct": 0.0043, "above_2m_pred_occupied_pct": 1.6929, "above_2m_gt_occupied_pct": 0.4019}; no target-dataset path was opened during the audit: {"files_opened": 4650}.

## 6. Provenance table

| Field | semantickitti | occ3d |
|---|---|---|
| verified_occupancy_iou_pct | 22.71 (median of 22.47, 23.62, 22.71; mean 22.93) | 33.88 (median of 33.49, 35.03, 33.88; mean 34.13) |
| semantic_miou_pct | 3.35 (SSC mIoU, median seed; the model carried Trident-H semantics) | 4.24 (SSC mIoU, median seed; the model carried Trident-H semantics) |
| artifact_report | output/skyfix/eval/eval_semantickitti_past5_seed0.json; output/skyfix/eval/eval_semantickitti_past5_seed1.json; output/skyfix/eval/eval_semantickitti_past5_seed2.json; reports/METHOD.md §8.1; artifacts/gate8c1/ceiling_fix.md | output/skyfix/eval/eval_occ3d_past5_seed0.json; output/skyfix/eval/eval_occ3d_past5_seed1.json; output/skyfix/eval/eval_occ3d_past5_seed2.json; reports/METHOD.md §8.1; artifacts/gate8c1/ceiling_fix.md |
| config | configs/gate8c1_sky/seed{0,1,2}.yaml | configs/gate8c1_sky/seed{0,1,2}.yaml |
| checkpoint | seed 0: artifacts/gate8c1/checkpoints/sky_seed0_last.pt sha256 ea721b21a259; seed 1: artifacts/gate8c1/checkpoints/sky_seed1_last.pt sha256 47353eacfbf7; seed 2: artifacts/gate8c1/checkpoints/sky_seed2_last.pt sha256 325d8783b937 | seed 0: artifacts/gate8c1/checkpoints/sky_seed0_last.pt sha256 ea721b21a259; seed 1: artifacts/gate8c1/checkpoints/sky_seed1_last.pt sha256 47353eacfbf7; seed 2: artifacts/gate8c1/checkpoints/sky_seed2_last.pt sha256 325d8783b937 |
| git_commit | code first tracked at f7b7552 (2026-09-07); fix applied 2026-09-04 in an untracked tree; docs at 0ba3b5d | code first tracked at f7b7552 (2026-09-07); fix applied 2026-09-04 in an untracked tree; docs at 0ba3b5d |
| evaluation_script | core/run/eval_target.py --dataset semantickitti --mode past5 (G8C1_MANIFEST=frozen_manifest_sky.json) | core/run/eval_target.py --dataset occ3d --mode past5 (G8C1_MANIFEST=frozen_manifest_sky.json) |
| ceiling_fix_active | yes, verified | yes, verified |
| exact_protocol | past5: fresh map over stream indices i-4..i; per-seed thresholds {0: -0.125, 1: -0.4375, 2: -0.3125}; 163 anchors; official voxels/<frame>.invalid == 0 (label != 255) | past5: fresh map over stream indices i-4..i; per-seed thresholds {0: -0.125, 1: -0.4375, 2: -0.3125}; 1182 anchors; mask_camera == 1, lidar mask off, ego x-index < 100 ignored (single front camera) |
| date | 2026-09-08 | 2026-09-10 |
| canonical_baseline | yes | yes |


## 7. Canonical baseline decision

**Canonical historical baseline = the sky-fix model under Protocol A `past5`, full official anchor sets, median of three seeds: SemanticKITTI 22.71 %, Occ3D-nuScenes 33.88 % SC IoU.** Chosen for validity, not magnitude: it is the only post-fix evaluation on the complete official anchor sets with the gate's own masks and metric, and it is the headline `reports/METHOD.md` §8.1 adopts. The OccAny head-to-head numbers are partially comparable (frame subsets, OccAny's label volumes and scorer) and the `sky2` numbers are a rejected model. The `stream` and `occany_fwd` protocols were never run for the sky model.

Caveat carried forward: the historical model is **not** geometry-only (32-channel input with Trident-H semantics, residual lock, long-stream LingBot geometry, per-seed source-selected thresholds), so any comparison with a geometry-only model is at best partially comparable -- see `historical_comparability.md` once the current baseline exists.
