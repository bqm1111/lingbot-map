# OccAny comparison — predeclaration

Written 2026-09-14, **before any score in this experiment was computed.** Clips were still being built.
No checkpoint, threshold, grid, mask or pooling choice is selected from the results.

## Question

Are the frozen current baseline's zero-shot numbers (SemanticKITTI 26.25 % / Occ3D 31.56 %, seed medians
on our 163 / 1,182 anchors, raw, causal) comparable to OccAny's published 25.91 % / 23.55 %?
They differ at once in sample set, post-processing, temporal direction and Occ3D benchmark placement.

## What is held fixed for both methods

| factor | setting |
|---|---|
| samples | every sample the released OccAny model scored on this machine: 805 SemanticKITTI 08 frames, 4,819 Occ3D val samples |
| ground truth | OccAny's stored `voxel_label`; checked per sample against an independent official-file loader (corrected ignore rule) |
| scorer | OccAny `SSCMetrics.get_score_completion`; pooled TP/FP/FN over samples |
| post-processing | both `raw` and `pooled` (OccAny geometry pooling = 3×3×3 max, pool-then-mask) for both methods |
| OccAny model | released checkpoint, full completion (`render_recon_gen`), its own confidence thresholds (2.5 / 1.1) |
| our model | the three frozen `geo_seed{0,1,2}_step06000.pt`, occupied := final > 0, verified by the source freeze |

## Variants

* **protocol** — `forward`: OccAny's five frames (target first, stream offsets 0/2/4/6/8); `causal`: our five
  frames ending at the target (only samples with four earlier frames).
* **placement (Occ3D)** — `occany`: OccAny's static camera extrinsic; `ours`: our timestamp-compensated chain.
  SemanticKITTI uses one shared transform (asserted equal per sample).
* **subsets** — `occany_all` (805 / 4,819), `causal_eligible` (801 / 4,219), `official_anchors` (our reported
  anchors with both windows: 161 / 900).

## Primary comparison (fixed now)

`occany_all`, `forward`, placement `occany` (Occ3D) / `shared` (SemanticKITTI), `pooled` vs `pooled`:
**Δ = seed-mean pooled IoU(ours) − pooled IoU(OccAny)**, with a paired block bootstrap 95 % CI
(2,000 draws, seed 0; blocks = Occ3D scenes, SemanticKITTI runs of 25 consecutive target frames).

The primary row is the *most like-for-like* one: identical frames in, identical samples, labels,
placement, post-processing and scorer. Reading rule: the CI excluding 0 → "higher" / "lower"; otherwise
"not distinguishable". The raw-vs-raw forward row is reported with equal prominence as a
post-processing-free check; the causal rows describe our deployed setting and are secondary.

## Remaining asymmetries (cannot be removed here, will be reported)

* OccAny's thresholds are per dataset; ours is one frozen threshold chosen on KITTI-360.
* Training data differ (OccAny: Waymo, DDAD, PandaSet, VKITTI2, ONCE; ours: KITTI-360 0003/0007/0010).
  Neither trained on the target datasets.
* Image preprocessing differs (OccAny principal-point crop to 512 wide; ours full image resized to 518).
* Three seeds vs one released model.
* These target sets have been evaluated many times in this project's history (RC5); this is a reporting
  measurement of a frozen model, not a clean held-out test.
