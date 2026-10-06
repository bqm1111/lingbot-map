# Final semantic-method experiment — source predeclaration (written 2026-09-15, before any B4 model, feature statistic or score existed)

Hash-frozen by `tools/semantic_field/freeze.py` (stage `source_predeclaration`) together with the configuration. The feature mean
μ is computed after this freeze from training-drive teacher evidence only and frozen in stage `centering` before training.

## Question

Can a lightweight generative, language-aligned 3D semantic field, trained on KITTI-360 with masked current observations and
privileged future observations, predict semantics in unobserved space better than transport of observed evidence?

## Frozen components (unchanged, reused by import)

Five causal frames, local-reset LingBot, LB-M depth, causal MoGe-2 clip scale, local poses, hard mapper, CompletionUNet
**geometry seed 0** (occupied := log-odds > 0), Trident-H token features and Gate 15 evidence lifting, the pilot's delayed targets and
reliability rules (10 s window, tiers A and B, not rebuilt), Gate 16 candidate rule (16 nearest), B0 and B3 (seeds 0–2) checkpoints.
Training drives 0003/0007/0010; development drive 0006. SemanticKITTI and Occ3D stay closed until SOURCE PASS.

## Feature-space conventions

* Raw teacher feature `f`: Trident-H CLIP ViT-H-14 token feature, **1024-d, unit L2** (evidence: normalised count-weighted mean).
* Text bank: Trident query features of the frozen phrases, 1024-d unit L2 (KITTI-360: 18 classes for training and source readout;
  each target's own frozen bank at target readout).
* **Centering**: μ = mean of raw unit evidence features over every evidence voxel of every training anchor (0003/0007/0010), no
  labels; `centre(f) = normalize(f − μ)`. Used for the B4 evidence input and the centered feature loss only.
* **Text logits** on raw unit features: `logit_k = 40 · cos(f, text_k)` (the teacher's own scale, τ = 1/40); readout = argmax,
  identical to B0 / B3 / the teacher.
* **B4 output**: 1024-d, L2-normalised, raw teacher space (open-vocabulary readout by any frozen text bank).

## B4 architecture (the only new learned method)

Dense input on the grid (C-order, velodyne frame of t, 0.2 m): frozen CompletionUNet decoder trunk d1 (24 ch, full resolution),
hard map S, F and frozen occupancy O (3 ch), projected centered evidence (Linear 1024→32 at evidence voxels, zeros elsewhere),
evidence mask (1), log1p(ray count)/5 (1) = 61 channels. Residual 3D UNet: widths 32 / 48 / 64 at 1, 1/2, 1/4 resolution,
GroupNorm + GELU, replicate padding, dilations 1/2 at the coarsest level, skip fusion; a pointwise head Linear(32 → 1024) applied at
queried voxels, L2-normalised. Budget 0.5–2 M parameters (exact count reported). The output is a free 1024-d vector: it is not a
mixture of input features. Evidence voxels keep their own observed feature at inference (the path shared by B0 / B3 / B4); B4
predicts every other voxel.

## Supervision (training drives only; no human labels)

* **A. Masked current reconstruction.** Grid-aligned cuboids of 16×16×8 voxels (3.2×3.2×1.6 m) are hidden with probability 0.5
  (the only masking mechanism; ≈ 50 % of evidence hidden). Hidden evidence is removed from the input and becomes the target:
  `L_A = KL(softmax(40 cos(f_t, text)) ‖ softmax(40 cos(f_s, text))) + (1 − cos(centre(f_t), centre(f_s)))`.
* **B. Privileged delayed observations.** Pilot delayed targets (non-evidence voxels, tiers A + B): the stored target is the
  view-averaged teacher class distribution, so `L_B = KL(q_delayed ‖ softmax(40 cos(f_s, text)))` (no 1024-d feature is stored;
  the centered feature term is therefore not available for B and is not approximated).
* `L = L_A + 1.0 · L_B`, per-term means over queried voxels.

## Training

Seeds 0, 1, 2 (seed s: `torch.manual_seed(s)`, data generator s, delayed subsample generator s + 1). 6,000 steps, 4 anchors per
step, full-grid frozen geometry forward then a uniform 128×128×32 crop, up to 8,192 masked and 4,096 delayed targets per anchor,
AdamW lr 1e-3, weight decay 0.01, cosine schedule, gradient clip 1.0. **Checkpoint: final step**, no selection.

## Source evaluation (0006, 590 anchors, GT evaluation-only)

B0, B3 (seeds 0–2), B4 (seeds 0–2) on one candidate set / occupancy per anchor. Regions: oracle_observed (G∩E), completed_observed
(G∩O∩E), **completed_unobserved (G∩O∖E, PRIMARY)**, geometry_unobserved (G∩O∖E∖(S∪F)), no_semantic_evidence (G∖E), SSC full;
candidate-supported / candidate-unsupported split of the primary region (GT class among / not among the 16 candidates); distance
to nearest evidence (0–1, 1–2, 2–4, 4–8, > 8 m); masked-evidence diagnostic (deterministic cuboid masks seeded by the anchor id,
hidden evidence voxels: nearest remaining evidence, B3 with remaining candidates, B4 with the masked input; teacher-class
agreement and GT accuracy/mIoU). Paired bootstrap: 20-anchor blocks, 2,000 resamples, seed 0, seed-pooled confusions.

## Source promotion rule (all required; B4 = mean of seeds)

| id | criterion |
|---|---|
| S1 | mIoU(B4) − mIoU(B0) ≥ +2.0 points on completed_unobserved |
| S2 | every seed's gain over B0 > 0 and bootstrap CI lower bound > 0 |
| S3 | mIoU(B4) − mIoU(B3) ≥ +1.0 point with CI lower bound > 0 |
| S4 | ≥ 8 classes with positive IoU gain over B0 on completed_unobserved |
| S5 | no collapse: dominant predicted-class fraction ≤ max(0.60, GT dominant + 0.10); ≥ 5 classes with ≥ 1 % of predictions; prediction entropy ≥ 0.75 × GT class entropy |
| S6 | occupancy TP/FP/FN identical to Gate 15 seed 0 on every anchor |

**SOURCE PASS** iff S1–S6 all pass → freeze and run one untouched zero-shot evaluation. **SOURCE FAIL** otherwise → close
semantic-completion method development; no further architecture.

## Conditional target evaluation (only after SOURCE PASS and a hash freeze)

The frozen final zero-shot protocols (Track P, Track O) and samples (801 / 4,219) are reused unchanged; B4 is added to the canonical
predictions with identical occupancy. Paper rule:
* **PAPER A**: on both datasets, central Track P B4 − B0 and B4 − B3 ≥ +2.0 with CI lower > 0 and all seeds positive,
  candidate-unsupported improves (CI lower > 0), and official SSC mIoU (B4) ≥ OccAny published (7.28 SemanticKITTI / 6.66 Occ3D).
* **PAPER B**: B4 − B0 central ≥ +2.0 with CI lower > 0 and all seeds positive on both datasets, ≥ 8 classes improve on each,
  but PAPER A fails.
* **PAPER C**: otherwise (including SOURCE FAIL).
