# Gate 15 predeclaration (2026-09-14) -- written before any trained propagation head was evaluated

**Question.** Can sparse, causal, language-aligned Trident-H evidence be propagated by a small learned head on top of the
*frozen* geometry-completion representation into occupied voxels that carry no direct observation?

## Frozen, not touched
Current Geometry-Only Baseline (`artifacts/baseline_reconstruction/code_freeze.json`): local-reset L, LB-M, hard mapper H,
CompletionUNet [S, 0, F] seeds 0/1/2 step 6000, occupied := final log-odds > 0. Trident-H frozen (token caches of the
feature-first study). LingBot, MoGe-2, mapper unchanged. Semantic loss has no path into the geometry network (trunk under no_grad).

## Evidence (only direct causal observations)
Five frames t-4..t. A lattice ray counts only if the frozen mapper accepts it (conf ≥ 1.5, 1–60 m, clip scale applied once,
anchor-identity local poses). Its endpoint world key is looked up by grid voxel centre exactly as `IncrementalMapper.query`;
an **evidence voxel** = key hit ∧ `S_H`. Feature = normalize(count-weighted mean of unit teacher features bilinearly sampled at
the lattice pixel centres). No labels, LiDAR, GT poses, future frames or target datasets (strict file audit).

## Methods
* **B0** (fixed now): label of the nearest evidence voxel, 3D Euclidean, no radius.
* **B1**: `PropagationHead` (290,400 params: 1024→32 evidence projection, 1×1 stem over [trunk 24, evidence 32, mask 1], four
  dilated residual blocks d = 1, 2, 4, 8 at width 32, 1×1 → 1024, L2-normalised), one per geometry seed.
* **Teacher ceiling** (diagnostic): the evidence readout at evidence voxels.
Readout for all: `1 + argmax cos(f, text)` over the 18 frozen Trident text embeddings.

## Training (one configuration, no sweep)
0003/0007/0010 only; 6000 steps, batch 4, uniform 128×128×32 crops of the frozen full-grid trunk; AdamW 1e-3 wd 0.01, cosine,
clip 1, bf16; final step. Loss = 1 − cos on **every evidence voxel of the crop**; the input evidence of voxels in randomly hidden
(8, 8, 4)-voxel blocks (p = 0.5) is removed so part of the loss can only be met by propagation. No voxel without evidence is ever
supervised.

## Evaluation (0006, 590 anchors; SSCBench human labels resampled to the grid = evaluation only)
Regions: oracle_observed (G∧E), oracle_unobserved (G∧¬E), completed_observed (G∧O∧E), **completed_unobserved (G∧O∧¬E, PRIMARY)**,
ssc_full (labelled, pred = class if O else empty), ssc_no_evidence. Distance bins 0–1 / 1–2 / 2–4 / 4–8 / > 8 m to the nearest
evidence voxel. Collapse diagnostics: GT vs predicted class frequency, dominant fraction, classes with ≥ 1 % of predictions,
entropy. Label-free check: held-out evidence blocks, cosine to teacher, B1 vs nearest-visible-evidence B0.

## Promotion rule and decisions
Frozen in `promotion_rule.json` (with the sha256 of the measuring code) before evaluation: P1 mean-over-seeds Δ mIoU(B1−B0) on
completed_unobserved ≥ +0.02 with all seeds > 0 and bootstrap CI lower bound > 0; P2 SSC mIoU Δ > 0 all seeds; P3 ≥ 5 classes
gain, B1 dominant fraction ≤ max(0.60, GT dominant + 0.10), ≥ 5 classes with ≥ 1 % of predictions; P4 occupancy counts identical
to the baseline. Order: invalid if P4 fails → D (teacher accuracy < 0.40 or mIoU < 0.15 on oracle_observed) → A (P1–P4) → C
(P1 fails and collapse) → B. Target datasets are opened only after Decision A and a code freeze.

**Disclosure.** The evaluator was smoke-tested on 10 anchors with an untrained 20-step head before this file; that run showed a
teacher readout of ~0.23 mIoU / 0.64 accuracy on observed voxels. The D thresholds are those chosen before it.

## Risk recorded during training, before any evaluation of a trained head (2026-09-14, step ≈150)
Training loss fell to 0.06–0.12 within 150 steps with hidden ≈ visible loss. Label-free statistic on training evidence
(`logs/teacher_common_direction.json`): a single constant output (the normalised mean teacher feature) already attains the loss
recorded there, and random pairs of evidence features have the mean cosine recorded there. Plain cosine distillation, as the
brief specifies, is therefore dominated by a shared direction, and class identity lives in small residual angles. The objective
is **not** changed; the risk is that B1 learns the common direction and collapses in readout. If that happens it is evidence for
Decision C and names the semantic-side alternative (a distillation that removes the shared component), not a reason to retune now.
