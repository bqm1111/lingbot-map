# Delayed observation supervision for causal semantic completion — source pilot (2026-09-14)

KITTI-360 only: training on 0003/0007/0010, **development** evaluation on 0006 (590 anchors, inspected in earlier gates).

Setup:
- five-frame bounded history
- frozen LingBot, geometry seed 0 and Trident-H
- one head seed

SemanticKITTI and Occ3D were not opened.

- **Rules:** `predeclaration.md`, `predeclaration_addendum_A.md`, hash-frozen before the results they govern (`frozen_*.json`).
- **Numbers:** `results_table.md`, `target_reliability.md`.
- **Verification:** 15/15 (`verification.md`).
- **Code:** `tools/delayed_semantic/` (README includes the implementation map and reproduction commands).

## Result in one table (primary region = never observed in H_t ∩ predicted occupied, mIoU %)

| | B0 nearest evidence | B0g geodesic | B1 learned, observed-only | **B2 learned + delayed** | B2s (4 s targets) |
|---|---|---|---|---|---|
| primary (Gate 15 `completed_unobserved`) | 8.80 | 8.70 | 9.13 | **9.21** | 9.12 |
| … geometry unobserved (83 % of voxels) | 7.73 | 7.70 | 8.03 | 7.99 | 7.95 |
| … later revealed, no H_t camera return | 5.97 | 5.89 | 5.94 | 6.35 | 6.26 |
| SSC mIoU, all labelled voxels | 1.55 | 1.48 | 1.67 | 1.60 | 1.59 |

**Paired deltas (block bootstrap):**
- B2 − B1: +0.08 [−0.11, +0.37]
- B2 − B0: +0.41 [+0.16, +0.78]
- B1 − B0: +0.33 [+0.19, +0.48]

**Criteria:** K1 fails, K2 fails, K3–K6 pass.

**Decision: Revise the semantic mechanism or supervision.**

## Answers

**1. What completion mechanism was implemented?**
- **Evidence association:** a ~25 k-parameter module (`model.py`).
  - For each non-evidence voxel it takes the 16 nearest observed evidence voxels.
  - An MLP scores them from frozen geometry features (CompletionUNet trunk), relative position, occupancy/free fractions along the connecting segment, provenance, and a vocabulary-free projection of the feature.
  - Output = normalised softmax-weighted mean of the candidates' observed Trident-H features, a 1024-d language-compatible vector.
- **Observed evidence is kept separate from predictions** (`memory.py`: `ObservedEvidence` with frame bits and times; `PredictedSemantics` is rejected by the memory). Evidence voxels keep their own class in every method.
- **Limitation:** the output is a convex mixture, so it cannot express a class no candidate carries.

**2. What is available at prediction time?**
- **Inputs:** only frames t-4..t.
  - their local-reset LingBot clip and MoGe clip scale
  - the frozen hard map S/F
  - frozen occupancy and trunk
  - Trident-H evidence lifted on those frames (Gate 15 rule, reproduced voxel-for-voxel by a frame-by-frame replay)
- **Enforcement:**
  - `CausalHistory` raises on future indices
  - the prediction path runs under a file audit that forbids LiDAR, poses, targets and labels
  - the stored prediction is unchanged when future frames are altered, appended or removed (test)
  - LingBot run after future images is bit-identical (verify #13)
  - predictions-at-t are stored before any later evidence is opened and are reproduced from the record (#11)

**3. What later information is used only for supervision?**
- **Views:** LiDAR sweeps of t+1..t+20 (≤ 10 s), projected into their own camera image (timestamps ≤ 4.2 ms apart), with a z-buffer occlusion test. The Trident-H token at each visible return gives a class distribution.
- **Registration:** GT poses move each view into the grid of t.
- **Static support:** LiDAR returns spanning ≥ 1 s with no cross-sweep free/occupied conflict.
  - Tier A: also hit by an H_t sweep.
  - Tier B: future sweeps only.
- **Target:** the mean view distribution on voxels without student evidence.
- **Excluded:** moving or conflicting voxels have no target but stay in evaluation.

**4. Are the targets trustworthy enough to learn from?**
- **Under the declared gate, yes, with one disclosed caveat.**
- **Label-free evidence** (`target_reliability.md`):
  - Registration median/p90: 0.027/0.073 m at +0.5 s and 0.069/0.184 m at +5 s, training drives pooled. A +0.2 m z-shift raises the median to ~0.2 m, so the metric can detect misregistration.
  - Two independent later views agree on 75 % (tier A) and 73 % (tier B), against a 36–38 % within-anchor shuffle; 0006 matches.
  - Coverage: 3.3 k targets per training anchor across 14 classes.
  - History camera views agree with the future target on 69 %.
- **Caveat:** the first implementation checked G1 per drive, including 0006. That reading fails at +5 s: p90 0.207 m on 0010 and 0.217 m on 0006, limit 0.20. It was corrected to the declared pooled rule after the failure was seen (addendum A). The pre-registered 4 s-target sensitivity run (B2s) behaves like B2, so the conclusion doesn't depend on the long-offset targets.
- **Labelled diagnostic (0006 GT, no rule):** target class accuracy is 63 % (mIoU 18–20 %). That is as accurate as the teacher on directly observed voxels (62 %), but that is the teacher's ceiling, not ground truth.
- **Weak link:** the student's lifted evidence class agrees with later views of the same voxel only 43–69 % per drive.

**5. Does delayed supervision outperform observed-only training?**
- **Not on the primary endpoint:** B2 − B1 = +0.08 [−0.11, +0.37]; B2 wins 11 of 29 blocks.
- **Small, reliable gains** where the supervision lives:
  - later-revealed voxels: +0.23 [+0.05, +0.38]
  - later revealed with no history camera return: +0.41 [+0.14, +0.83]
  - geometry-observed but semantically unsupported: +0.77 [+0.37, +1.45]
  - label-free agreement with 0006 delayed targets: 44.0 vs 42.4 %
- **Costs:** B2 lowers primary accuracy (46.0 vs 47.1 %) and SSC mIoU (−0.07 [−0.10, −0.01]). It shifts predictions from building (pred share 22 → 18 %, IoU 35.0 → 32.3) toward road, parking, sidewalk and other ground.

**6. Does learned completion outperform simple observed-memory transfer?**
- **Measurably, but far below the bar.**
  - B1 − B0 = +0.33 [+0.19, +0.48], winning 26/29 blocks.
  - B2 − B0 = +0.41 [+0.16, +0.78], against the +2.0 predeclared threshold.
- **Gains are broad:** 14 classes improve and the gain survives removing road. So this is a small effect, not one-class noise.
- **Geodesic B0g** is no better than Euclidean B0: −0.09 [−0.25, +0.04].

**7. Does the gain occur in space never observed in the available history?**
- **Yes, but it is small.** Observed-now and previously-observed regions are identical across methods by construction (15.21 / 17.38 mIoU), so every difference is in never-observed space.
- **B2's extra gain over B1** is confined to later-revealed and geometry-observed voxels. Geometrically unobserved space (83 % of the primary region) shows none: −0.04 [−0.19, +0.18].
- **Accuracy falls** from 52 % (revealed within 1 s) to 30 % (4–10 s) for every method.

**8. What is specifically gained from streaming?**
- **Nothing is demonstrated.**
  - The history is a bounded five-frame, local-reset window with no persistent state.
  - The memory is external: lifted Trident features with provenance.
  - The model does not use LingBot's evolving state.
  - Past-only evidence (17.38 mIoU) is remembered, which is memory, not completion.
- No persistent-rollout comparison was run, because the pilot did not pass.

**9. What remains unproven about zero-shot transfer and open vocabulary?**
- **Everything about transfer.** No target dataset was evaluated, and all numbers are one head seed on an already-inspected development drive.
- **Vocabulary.** The output lives in the frozen CLIP feature space, but the loss uses the fixed 18 KITTI-360 prompts. Success here would not establish open-vocabulary generalisation.
- **Hidden objects.** The mechanism cannot produce a semantic identity absent from nearby evidence, so universal hidden-object recognition is not claimed.

**10. The single most justified next action.**
- **Post-hoc diagnostic** (`diagnostic_candidate_ceiling.json`; written after the decision, not a criterion). On the primary region:
  - The nearest evidence voxel has the GT class 46.0 % of the time.
  - Any of the 16 candidates has it 57.3 % of the time.
  - Some evidence in H_t has it 97.5 % of the time.
- **Reading:**
  - B1 and B2 (46–47 % accuracy) recover almost none of the ~11-point headroom inside their own candidate sets. The binding limit is therefore not the convex-mixture ceiling, but that neither supervision teaches the scorer which candidate is right.
  - Two plausible causes:
    - the lifted evidence is often mislabelled (43–69 % agreement with later views)
    - the delayed targets are soft teacher distributions no more accurate than the teacher
  - For terrain, parking, other ground, other vehicle and small objects, the correct class is rarely among the candidates (1–16 %). Those classes need a mechanism that can express unsupported semantics.
- **Next action:** keep the delayed targets, which passed. Revise the supervision and selection signal: train the candidate scorer with a candidate-level objective, i.e. which candidate's observed class matches the later view, and measure against the 57 % candidate ceiling before any architecture growth.
- Do not scale model size, data or streaming until that ceiling is being approached.

## Implementation, verification and reproduction

- **Code and implementation map:** `tools/delayed_semantic/README.md`.
- **Tests:** `tests/delayed_semantic` (11), covering:
  - prefix invariance on a real anchor
  - the future-access guard
  - observed-vs-predicted provenance
  - never-observed masks
  - voxelisation and pose/projection round trips
  - geodesic rule
  - occupancy invariance under head training
- **Verification:** 15/15 in `verification.md`. Freeze order, identical draws, no firewall violation, B0 = Gate 15 per anchor, occupancy = Gate 15 per anchor, and target rebuild.
  - Target rebuild note: integer arrays match exactly; the fp16 distributions match within 6e-5 with identical argmax.
- **Figures:**
  - `figures/1_reliability.png`
  - `figures/2_deltas.png`
  - `figures/3_distance_reveal.png`
  - `figures/4_per_class.png`
  - `figures/5_example_p{10,50,90}_*.png`: deterministic percentile selection
  - `figures/6_target_construction.png`
- **Manifests:** `data_manifest.json`, `checkpoint_manifest.json`, `pilot_config.yaml`.
- **Operational note:** the first evaluation run was killed by a host memory guard. Its partial prediction records were deleted and fully regenerated; the final stored records are reproduced by verify #11.

**Revise the semantic mechanism or supervision.**
