# Baseline reconstruction -- historical post-ceiling-fix baseline and the current geometry-only baseline (2026-09-14)

## Verified historical post-ceiling-fix baseline

| Dataset | Verified metric | Value | Provenance |
|---|---|---|---|
| SemanticKITTI | SC IoU (occupancy, official mask), past5, median of 3 seeds | **22.71 %** | output/skyfix/eval/eval_semantickitti_past5_seed0.json; output/skyfix/eval/eval_semantickitti_past5_seed1.json; output/skyfix/eval/eval_semantickitti_past5_seed2.json; frozen_manifest_sky.json; verified in historical_baseline_audit.md |
| Occ3D-nuScenes | SC IoU (occupancy, official mask), past5, median of 3 seeds | **33.88 %** | output/skyfix/eval/eval_occ3d_past5_seed0.json; output/skyfix/eval/eval_occ3d_past5_seed1.json; output/skyfix/eval/eval_occ3d_past5_seed2.json; frozen_manifest_sky.json; verified in historical_baseline_audit.md |


The remembered Occ3D value **30.74 % was wrong** (it appears nowhere; the pre-fix median is 30.89 %, the post-fix median 33.88 %). The remembered SemanticKITTI value 22.71 % is correct. Full trace: `historical_baseline_audit.md`.

## Current clean baseline

| Dataset | Current IoU (mean of 3 seeds) | Historical IoU (median of 3) | Comparability |
|---|---|---|---|
| KITTI-360 0006 (full region) | 13.51 % | n/a | not comparable (source-side; different target build/threshold than the historical source number) |
| SemanticKITTI | **26.17 %** (median 26.25, sd 0.32) | 22.71 % | partially comparable |
| Occ3D-nuScenes | **31.22 %** (median 31.56, sd 1.78) | 33.88 % | partially comparable |


95 % clip-block-bootstrap CIs of the current mean: SemanticKITTI [24.61, 27.69]; Occ3D-nuScenes [29.17, 33.21].

## Answers

1. **Verified SemanticKITTI post-ceiling-bug baseline:** 22.71 % SC IoU (seeds 22.47 / 23.62 / 22.71; mean 22.93), 163 anchors, past5.
2. **Verified Occ3D-nuScenes post-ceiling-bug baseline:** 33.88 % SC IoU (seeds 33.49 / 35.03 / 33.88; mean 34.13), 1182 anchors, past5.
3. **Produced by:** `artifacts/gate8c1/frozen_manifest_sky.json` → `sky_seed{0,1,2}_last.pt` (configs/gate8c1_sky/seed*.yaml, replicate padding, plain-column open-sky targets at `lingbot_gate8c1_sky`, thresholds −0.125/−0.4375/−0.3125), evaluated by `core/run/eval_target.py --mode past5`; records `output/skyfix/eval/*.json` (2026-09-08 / 09-10).
4. **Remembered numbers wrong?** Yes: Occ3D 30.74 % does not exist (30.89 pre-fix; 33.88 post-fix). SemanticKITTI 22.71 % is right. Also: two `artifacts/gate8c1/occany_*headtohead.json` files hold the rejected sky2 run, not the adopted model.
5. **Current clean KITTI-360 baseline:** 0006 full-region SC IoU 13.51 % mean (seeds 13.65 / 13.46 / 13.43); reproduces Gate 13 M (0.1345) within the predeclared tolerance (mean |Δ| 0.0006). `source_results.md`.
6. **New untouched zero-shot results:** SemanticKITTI 26.17 % (map-only 8.82), Occ3D-nuScenes 31.22 % (map-only 10.04); one evaluation each, nothing changed afterwards. `zero_shot_results.md`.
7. **Directly comparable?** No -- *partially comparable* on both targets: same anchors, masks, grids and metric, but the pipelines differ in every row of `historical_comparability.md` (geometry mode, scale rule, 32-ch semantic input vs 3-ch, residual lock, threshold rule, training targets, seed statistic).
8. **Which differences drive the gap?** Cannot be attributed to any single one without an ablation that does not exist. The like-for-like quantity is the map-only IoU (no network): historical 8.90 / 9.99 vs current 8.82 / 10.04 (SemanticKITTI / Occ3D), which isolates the geometry-mode + scale-rule change in the observed map. Everything beyond that (semantics, 32 channels, lock, threshold, targets) acts on the completion only. SemanticKITTI/Occ3D ratio: current 0.84 vs historical 0.67.
9. **Which baseline for semantic-completion experiments?** The current one -- it is the only geometry-only, local-reset, fully frozen and hash-verified system; the historical model is not geometry-only.

## Reading the result (observations, not attributions)

* SemanticKITTI moved **up** (22.71 → 26.17) and Occ3D **down** (33.88 → 31.22); the map-only rows barely moved (8.90 → 8.82; 9.99 → 10.04), so the observed geometry the two systems hand the network is of essentially equal quality on both targets and the divergence lives in the completion.
* Operating point: on Occ3D the current model is more precise and less complete than the historical one (P 77.1 / R 34.4 vs 75.2 / 38.1); on SemanticKITTI it gains recall at equal precision (P 35.8 / R 49.3 vs 34.5 / 40.0). Threshold 0 versus the historical negative thresholds pushes exactly this way on a benchmark that rewards precision (Occ3D, camera-visible structure) and is neutral-to-favourable where free space is scored (SemanticKITTI).
* The direction of the asymmetry matches predeclared risk 1 (`predeclaration.md`): the neighbourhood-7 open-sky rule of the current targets was, in the historical single-seed ablation, a wash on SemanticKITTI and −5.2 points on Occ3D. It is the first documented candidate; it was **not** acted on.
* Seed spread on Occ3D is larger than before (sd 1.78 vs 0.80; seed 2 at 29.30); the bootstrap CI of the mean [29.17, 33.21] does not exclude the historical median, while the SemanticKITTI CI [24.61, 27.69] lies entirely above it.
* Both targets: the completion clears its own map-only reference by a wide margin (no collapse), and the source result reproduced -- the baseline is internally trustworthy.

## Decision

**Decision B** -- below the historical median on occ3d; the historical system carried semantics, 32 input channels, a residual lock, source-selected thresholds and different targets (historical_comparability.md).

| target | current mean | map-only | beats map-only | historical median | at/above historical (within sd) |
|---|---|---|---|---|---|
| SemanticKITTI | 26.17 | 8.82 | True | 22.71 | True |
| Occ3D-nuScenes | 31.22 | 10.04 | True | 33.88 | False |


Verification: 32 / 32 checks pass (`verification.md`).

## Next action

**Freeze the Current Geometry-Only Baseline (L + LB-M + H, CompletionUNet-3ch, seeds 0/1/2, step 6000, threshold 0; code_freeze.json) as the geometry-only control and return to semantic completion.**

## Files

`historical_baseline_audit.md`, `historical_baseline.json`, `historical_protocol.yaml`, `historical_provenance.csv`, `predeclaration.md`, `training_config.yaml`, `source_results.md`, `source_freeze.md`, `code_freeze.json`, `zero_shot_results.md`, `evaluation_semantickitti.yaml`, `evaluation_occ3d.yaml`, `historical_comparability.md`, `verification.md`, `tables/*.csv`, `figures/*.png`; reproduction commands in `tools/baseline_reconstruction/README.md`.
