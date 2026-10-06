# Predeclaration — delayed observation supervision pilot (written 2026-09-14, before any target statistic or trained model existed)

Nothing in this file may change after target statistics are computed. `rules.py` freezes the same text
as JSON together with the sha256 of the measuring code.

## Fixed setting

* Input budget: history H_t = stream frames t-4..t (five frames, ~2 s) for every student component —
  local-reset LingBot clip, frozen MoGe clip scale, frozen hard mapper, frozen CompletionUNet (geometry
  seed 0), Trident-H evidence lifted on those frames (Gate 15 rule, unchanged). No persistent state.
  This is a bounded five-frame history, not long-horizon streaming.
* Training drives 0003/0007/0010; development evaluation 0006 (already inspected in earlier gates).
* Head seed 0 only (feasibility). Three seeds only if the pilot passes.

## Delayed target (fixed before building)

For anchor t and each future stream frame j in t+1..t+20 with time_j − time_t ≤ 10 s:
1. the LiDAR sweep of native frame j (image/LiDAR timestamps < 5 ms apart) is projected into image j with
   calibration; a return is camera-visible iff it is ≥ 1 m in front, inside the image and passes a 4 px
   z-buffer (z ≤ zmin(cell) + max(0.5 m, 0.05 · zmin));
2. the frozen Trident-H token map of image j is sampled at each visible return (bilinear), cosine to the
   18 frozen text embeddings, averaged per (voxel, frame), softmax at the teacher's scale 40 → one view;
3. GT poses move the return into the grid of t; a voxel's target = mean of its future views' distributions.

Static support (LiDAR only, no labels): returns from sweeps in H_t ∪ future spanning ≥ 1 s, and no
cross-sweep occupied/free conflict (the frozen `rawtarget` measurement rule).
* Tier A ("occupied at t"): hit by an H_t sweep and by ≥ 1 future sweep.
* Tier B ("future-supported, earlier occupancy uncertain"): no H_t hit, ≥ 2 future sweeps.
* Delayed supervision set D: (A or B) ∧ ≥ 1 future view ∧ not a student evidence voxel. Stored with tier,
  view count, time to first view, whether an H_t camera saw a return there (privileged diagnostic only).
* Moving objects: no motion model exists, so single-sweep and conflicting voxels are excluded from
  targets. They remain in every evaluation region.

## Target reliability gate (label-free; computed before any training)

| id | test (training drives pooled unless stated) | pass if |
|---|---|---|
| G1 | point-to-plane residual of future sweeps (offsets 1, 5, 10, 20) against the anchor sweep, planar neighbourhoods | median ≤ 0.10 m and p90 ≤ 0.20 m at every offset |
| G2 | coverage | mean \|D\| per training anchor ≥ 2,000 and ≥ 8 classes (target argmax) with ≥ 10,000 voxels |
| G3 | cross-view repeatability: top-1 of the first two future views of the same voxel | agreement ≥ 0.65 and ≥ 0.10 above the within-anchor shuffled reference; also on 0006 ≥ 0.60 and ≥ 0.10 above its shuffle |
| G4 | reobservation consistency: class from H_t camera views of a voxel (LiDAR-projected) vs its future target class | agreement ≥ 0.65 |

* Tier B enters the delayed loss only if tier B alone meets G3 on the training drives; otherwise B2 uses tier A only.
* If G1, G2, G3 or G4 fails for tier A: **Stop — delayed targets are not reliable enough**; nothing is trained.
* Semantic GT (0006) is reported as a clearly labelled diagnostic of target accuracy and never used for these rules.

## Methods (shared evidence path, geometry, occupancy, vocabulary, regions)

* Evidence path (all methods): a student evidence voxel keeps its own observed feature.
* **B0** Gate 15 nearest observed evidence (3D Euclidean). **B0g** geodesic nearest evidence through
  predicted-occupied voxels (6-connected), B0 where unreachable. No learning.
* **B1** learned evidence association (per query voxel: k = 16 nearest observed evidence voxels; a small
  MLP scores each from frozen geometry features, relative position, path occupancy/free fractions,
  provenance and a vocabulary-free 16-d evidence projection; output = normalised softmax-weighted mean of
  the candidates' observed features). Trained with the observed-only term: hide Gate 15 blocks of evidence,
  predict the hidden voxels' teacher distribution (KL, scale 40).
* **B2** identical architecture, draws, steps and observed-only term, plus KL to the delayed target on D.
* Output is a convex combination of observed features: it cannot express a class absent from the
  candidates (stated limitation). Training vocabulary = the 18 KITTI-360 prompts (fixed-vocabulary limitation).

## Evaluation (0006, geometry seed 0, 590 anchors, semantic GT evaluation-only)

Semantic regions (GT occupancy): observed now (E ∩ frame t); previously observed only (E, not frame t);
never observed = no evidence in H_t (oracle occupancy, and ∩ predicted occupancy = **primary**
`completed_never_observed`, identical to Gate 15 `completed_unobserved`); its split into geometrically
observed (hard map S or F) vs geometrically unobserved; later revealed (∩ 0006 delayed targets) by time to
reveal. SSC mIoU under predicted occupancy (full, never-observed); occupancy IoU; per class; distance bins.
Uncertainty: paired bootstrap over contiguous 20-anchor blocks, 2,000 resamples.

## Decision criteria (B2 unless stated; primary region; frozen before trained heads are evaluated)

| id | criterion |
|---|---|
| K1 | mIoU(B2) − mIoU(B1) > 0 with 95 % block-bootstrap CI lower bound > 0 |
| K2 | mIoU(B2) − mIoU(B0) ≥ +2.0 points with CI lower bound > 0 (Gate 15 endpoint unchanged, so its threshold applies) |
| K3 | predictions on evidence voxels identical across methods (observed identity preserved) |
| K4 | ≥ 5 classes with positive IoU gain over B0, and the mIoU gain stays > 0 without the largest-gain class |
| K5 | SSC mIoU (ssc_full) of B2 ≥ B0 |
| K6 | occupancy TP/FP/FN identical to the frozen seed-0 counts in Gate 15 source evaluation |

* **Proceed to a matched persistent-streaming source study** iff K1–K6 all pass.
* **Stop: delayed targets are not reliable enough** iff the reliability gate fails.
* **Revise the semantic mechanism or supervision** otherwise.
* A gain confined to remembered/observed surfaces is a memory result, not completion. B0g is secondary.
