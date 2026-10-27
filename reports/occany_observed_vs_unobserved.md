# Where our advantage over OccAny actually comes from — observed vs unobserved surfaces

**Measured 2026-10-16.** SemanticKITTI sequence 08, the 801 frames every compared run covers.
Tooling: `tools/observability_split/`. Raw numbers: `artifacts/observability_split/analysis_semantickitti.json`.
Companion: `reports/occany_comparison_honest_report.md` (the undivided comparison).

**One-line answer.** Our net lead over OccAny is produced **entirely outside the input camera's
frustum**. On the surface the camera can actually see, and on the surface hidden directly behind it,
OccAny is equal or better in every configuration we can run. The system's measured value is *causal
memory*, not better single-view completion.

---

## 1. How the split is defined

A ray is cast through every second pixel of the input image (1226 × 370, 0.1 m steps) against the
**ground-truth** occupancy. Every voxel a ray enters up to and including the first GT-occupied voxel is
**observed** — free space the camera sees through, plus the surface it terminates on. The rest splits
into **occluded** (inside the image, behind a surface) and **out of view** (outside the image frustum).
**Unobserved** = occluded + out of view.

The mask comes from the ground truth and the camera alone. No method's output enters it, so the same
partition is applied to ours and to OccAny. Because the observed region contains free space as well as
surface, TP, FP and FN are all defined inside it and IoU is well posed — it is not a disguised recall.

Note: SemanticKITTI's shipped `.occluded` files cannot be used for this. They mark every occupied voxel
as occluded (verified: 0 of 12,445 GT surfaces are "visible" at frame 000020), so they encode
"not seen as free space", not surface visibility.

## 2. What the benchmark is actually made of

| region | valid voxels | % of valid | GT surfaces | % of GT surface |
|---|---:|---:|---:|---:|
| observed | 379,650,001 | 32.3 % | 4,356,235 | **4.7 %** |
| occluded | 434,305,851 | 37.0 % | 65,178,553 | 69.9 % |
| out of view | 359,903,589 | 30.7 % | 23,763,934 | 25.5 % |
| **unobserved** | 794,209,440 | 67.7 % | 88,942,487 | **95.3 %** |

**95.3 % of the SemanticKITTI SSC ground truth is not observable from the input image**, and a quarter
of it is not even inside the camera frustum. Any headline SSC number on this benchmark is dominated by
geometry the camera never saw. This is worth one sentence in the paper on its own.

Because the observed region holds a third of the valid voxels but only 4.7 % of the surfaces, absolute
IoU there is low for *every* method — false positives in visibly-free space dominate the denominator.
Compare methods within a region; do not compare a number across regions.

## 3. Occupancy — SC IoU by region

Geometry pooling (3 × 3 × 3 dilation), OccAny's released default, both sides:

| run | window | all | observed | occluded | out of view | unobserved |
|---|---|---:|---:|---:|---:|---:|
| **ours** | causal, 5 past | 26.34 | 8.89 | 30.28 | **28.01** | 29.58 |
| OccAny 1-frame | causal, 1 | 24.06 | **11.59** | 31.28 | 11.85 | 26.41 |
| OccAny 5-frame | forward, 5 future | 25.93 | 8.20 | **36.33** | 11.59 | **30.76** |

Raw (no pooling):

| run | window | all | observed | occluded | out of view | unobserved |
|---|---|---:|---:|---:|---:|---:|
| **ours** | causal | 26.40 | 11.29 | 27.90 | **29.98** | 28.51 |
| OccAny 1-frame | causal | 20.32 | **15.81** | 25.52 | 7.43 | 20.86 |
| OccAny 5-frame | forward | 25.21 | 11.71 | **33.73** | 8.04 | 27.62 |

Paired block bootstrap, 25-frame blocks, 2,000 resamples (`*` = CI excludes 0):

| ours − OccAny 1-frame (causal) | raw | pooled |
|---|---|---|
| all | +6.08 [+4.82, +7.33] \* | +2.28 [+1.23, +3.33] \* |
| observed | **−4.52** [−5.33, −3.81] \* | **−2.70** [−3.20, −2.29] \* |
| occluded | +2.39 [+0.79, +4.02] \* | −1.01 [−2.32, +0.43] |
| out of view | **+22.55** [+20.98, +24.28] \* | **+16.16** [+14.44, +17.98] \* |
| unobserved | +7.65 [+6.16, +9.08] \* | +3.17 [+1.82, +4.48] \* |

| ours − OccAny 5-frame (forward) | raw | pooled |
|---|---|---|
| all | +1.19 [+0.35, +2.09] \* | +0.41 [−0.34, +1.20] |
| observed | −0.43 [−1.12, +0.29] | +0.69 [+0.24, +1.19] \* |
| occluded | **−5.83** [−7.04, −4.56] \* | **−6.05** [−7.25, −4.88] \* |
| out of view | **+21.94** [+19.93, +24.14] \* | **+16.42** [+14.63, +18.51] \* |
| unobserved | +0.89 [−0.14, +1.98] | −1.19 [−2.13, −0.25] \* |

## 4. Semantics — SSC mIoU by region

Track O, semantic pooling both sides; ours is B3, mean of seeds 0–2.

| run | window | all | observed | occluded | out of view | unobserved |
|---|---|---:|---:|---:|---:|---:|
| ours, B0 | causal | 4.39 | 3.16 | 4.41 | 5.11 | 4.58 |
| **ours, B3** | causal | 4.45 | 3.47 | 4.45 | **5.24** | 4.61 |
| OccAny 1-frame | causal | 4.02 | **5.02** | 4.82 | 0.11 | 3.98 |
| OccAny 5-frame | forward | 7.29 | 4.77 | **9.40** | 1.66 | **7.98** |

Superclass mIoU:

| run | all | observed | occluded | out of view | unobserved |
|---|---:|---:|---:|---:|---:|
| **ours, B3** | 8.39 | 5.10 | 8.81 | **8.80** | 8.83 |
| OccAny 1-frame | 8.23 | **9.16** | 10.17 | 0.18 | 8.25 |
| OccAny 5-frame | 13.55 | 7.73 | **17.84** | 3.57 | **14.95** |

Paired deltas (semantic pooling):

| ours B3 − OccAny | region | SSC mIoU | superclass mIoU |
|---|---|---|---|
| 1-frame (causal) | all | +0.42 [+0.02, +0.83] \* | +0.15 [−0.50, +0.82] |
| | observed | **−1.55** [−2.02, −1.10] \* | **−4.07** [−4.65, −3.47] \* |
| | occluded | −0.37 [−0.87, +0.18] | −1.36 [−2.22, −0.47] \* |
| | out of view | **+5.13** [+4.32, +5.59] \* | **+8.61** [+7.99, +9.28] \* |
| | unobserved | +0.63 [+0.19, +1.09] \* | +0.58 [−0.16, +1.35] |
| 5-frame (forward) | all | −2.84 [−3.47, −2.06] \* | −5.16 [−5.83, −4.49] \* |
| | observed | −1.30 [−1.83, −0.87] \* | −2.64 [−3.10, −2.18] \* |
| | occluded | **−4.95** [−5.60, −3.96] \* | **−9.02** [−9.96, −8.09] \* |
| | out of view | **+3.57** [+2.58, +4.28] \* | **+5.23** [+4.10, +6.52] \* |
| | unobserved | −3.37 [−4.05, −2.45] \* | −6.12 [−6.90, −5.32] \* |

## 5. What this means

1. **The lead is out-of-frustum, and only out-of-frustum.** Against OccAny's causal configuration our
   whole occupancy gain (+2.28 pooled overall) is bought in the out-of-view region (+16.16 there,
   −2.70 on observed, −1.01 occluded). The same holds for semantics: +0.42 mIoU overall, +5.13 out of
   view, **−1.55 on exactly the surface the camera is looking at**.
2. **Mechanism.** Our five past frames observed the near-field lateral and vertical wedge that is
   outside the current image. OccAny has one frame and must hallucinate it from a novel-view generator;
   its out-of-view SSC mIoU is **0.11** — essentially nothing. This is the causal-memory advantage, and
   it is the honest name for what the system does.
3. **Inside the image OccAny is the stronger system.** On observed surface its causal 1-frame
   configuration beats us on occupancy (−4.52 raw) and on semantics (−1.55 mIoU, −4.07 superclass).
   Their 2D-grounded labels transfer to visible voxels better than our evidence does.
4. **Behind the first surface, the future window is what wins.** OccAny's 5-frame forward run beats us
   on occluded geometry by −6.05 and on occluded semantics by −4.95 mIoU; its causal 1-frame run is
   roughly a tie with us there (−1.01 / −0.37, both CIs touching zero). Driving forward reveals what is
   behind the surface; completing it from one frame does not.
5. **Consequence for the paper's claim.** "We beat OccAny on causal SemanticKITTI" survives, but the
   mechanism has to be stated: *we beat it by remembering, not by completing*. Writing it that way is
   both more defensible and more interesting than the aggregate number.

## 6. Caveats

- Frame budget still differs in the causal rows: our 5 past frames vs OccAny's 1. Their 5-frame causal
  configuration remains unmeasured, and would most plausibly close part of the out-of-view gap.
- The frustum is our input image (1226 × 370). OccAny consumes a 512 × 160 resize of the same image —
  the same horizontal field of view, within a degree vertically — so the partition is not tilted toward
  either side. It is computed at the anchor frame only, which is the point: "out of view" means out of
  view *at prediction time*.
- Occluders are GT-occupied voxels; ignore voxels (label 255) are transparent to the rays. In this GT
  the ignore region sits almost entirely behind real surfaces, so rays reaching it have already
  terminated.
- Our pooled overall SC IoU here is **26.34**, not the **25.89** in the occupancy table of the companion
  report. Those come from two pipelines: 25.89 is the `occany_comparison` causal clip rebuild, 26.34 is
  the frozen zero-shot prediction set that the `like_for_like` semantic numbers use. Raw agrees to the
  second decimal with the latter (26.40). Keep each table internally consistent and do not mix them.
- This analysis is SemanticKITTI only. Occ3D's protocol masks to `mask_camera`, which already deletes
  about a third of the GT surfaces, so the same split is not available there without redefining the
  benchmark region.

## 7. Reproduction gates (all on the shared 801 frames)

| quantity | this tool | previously published | source of the published value |
|---|---:|---:|---|
| ours, raw SC IoU | 26.40 | 26.40 | `artifacts/like_for_like/results_semantickitti_frozencheck.json` |
| ours B0, SSC mIoU / superclass | 4.39 / 8.34 | 4.39 / 8.34 | `..._semantic_frozencheck.json` |
| ours B3 (3 seeds), SSC mIoU / superclass | 4.45 / 8.39 | 4.45 / 8.39 | same |
| OccAny 1-frame geometry, pooled | 24.06 | 24.02 (814 frames) | `/media/SSD1/occany_causal/logs/score_k1frame_geom.txt` |
| OccAny 1-frame semantic | 4.02 / 8.23 / 20.32 | 3.97 / 8.15 / 20.33 (814) | `.../score_k1frame_sem.txt` |
| OccAny 5-frame geometry, pooled | 25.93 | 25.93 (805) | `artifacts/occany_comparison/report.md` |
| OccAny 5-frame semantic | 7.29 / 13.55 / 25.21 | 7.31 / 13.58 / 25.25 (796) | `.../paired_forward.txt` |

Residual differences are the 13 / 9 / 5 frames outside the shared set, not a scoring change.

## 8. Provenance

- Split and scorer — `tools/observability_split/{visibility,runs,evaluate,analysis}.py`.
- Counts and intervals — `artifacts/observability_split/{counts,results,analysis}_semantickitti.*`.
- Our predictions — `/media/SSD1/MINH_DATASETS/lingbot_zero_shot_eval/predictions/semantickitti` (frozen causal).
- OccAny predictions — the five directories listed in `analysis_semantickitti.json["runs"]`; the OccAny
  checkout was not modified.
- Metric — `tools/zero_shot_eval/canonical/metrics.occany_ssc_miou`, verified against
  `OccAny/occany/metrics/ssc.py::SSCMetrics.get_stats`; pooling from `gates/gate8b/pooling.py`,
  verified against `OccAny/occany/utils/helpers.py::apply_majority_pooling`.
