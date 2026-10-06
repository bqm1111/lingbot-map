# Gate 16 predeclaration — candidate-level delayed supervision (written 2026-09-14, before any B3 model, candidate ceiling on the reconstructed masks, or B3 score existed)

The file is hash-frozen by `tools/delayed_semantic/gate16/rules.py` (stage `predeclaration`). Two later stages freeze the numeric
threshold computed from the reconstructed candidate ceiling (`ceiling`, before any B3 is scored) and the evaluation code (`code`).

## Question

When the correct semantic identity exists among the 16 nearest observed evidence candidates, can candidate-level delayed
supervision teach the scorer to select it?

## Frozen from the delayed pilot (unchanged, reused by import)

Geometry and occupancy (five-frame local-reset LingBot, LB-M depth, hard map, CompletionUNet geometry seed 0, occupied := log-odds > 0),
Trident-H and its 18 text embeddings, evidence lifting/aggregation/memory/provenance, the delayed targets that passed the pilot's
reliability gate (10 s window, tiers A and B, `delayed_targets/`), the candidate rule (K = 16 nearest evidence voxel centres,
Euclidean, ascending, `model.nearest_candidates`), all scorer input features, the scorer architecture, TrainConfig (3,000 steps,
8 anchors/step, 2,048 hidden + 2,048 delayed queries per anchor, AdamW 1e-3, cosine, clip 1), training drives 0003/0007/0010,
development drive 0006, regions (`regions.py`). No target dataset is opened. No semantic GT enters training or any choice.

## Conditions

* **B0** nearest evidence, **B1** learned observed-only, **B2** learned + delayed final-feature KL: the pilot's checkpoints and
  predictions, head seed 0 only. They are historical single-seed controls; no multi-seed uncertainty is claimed for them.
* **B3** (new): identical scorer, identical observed-only term (hidden evidence, final-feature KL to the hidden voxel's own
  teacher distribution), and the delayed term replaced by **candidate-level selection**:
  * `p_i = softmax(40 · cos(f_i, text))` — the frozen readout distribution of candidate i (sums to 1);
  * `q_v` = the frozen delayed target distribution of voxel v (mean of its later views; sums to 1);
  * compatibility `r_i = <p_i, q_v>` ∈ [0, 1];
  * candidate target `target_i = softmax(r_i / τ)`, **τ = 1/40** (the teacher's own validated logit scale, reused; not swept);
  * `L_select = KL(target || softmax(s))` on the scorer logits s; output still `normalize(Σ softmax(s)_i f_i)`.
  B3 differs from B2 only in the form of the delayed term. Head seeds **0, 1, 2**; seed s uses the same generators as the pilot
  (`manual_seed(s)`, draws `s` and `s+1`), so B3 seed 0 has exactly B2's anchor, hiding and query draws.
* **B3geom** (the one allowed ablation, diagnostic only, not in any decision): B3 with the candidate semantic projection removed
  from the scorer input (zeros); the output mixture still uses the semantic features. Seeds 0, 1, 2.

## Candidate records and evaluation-only masks (0006)

For every primary-region voxel (GT occupied ∩ predicted occupied ∩ no evidence) the 16 candidates, their readout classes and
distributions are recomputed with the pilot rule. Evaluation only:
* **candidate-hit**: some candidate's readout class equals GT; **candidate-miss**: none does;
* **C0** nearest-candidate (candidate 0) GT-class accuracy; **C16** any-of-16 hit rate;
* **oracle selection**: GT where hit, otherwise the class of the candidate with the highest p_i[GT]; accuracy and mIoU;
* history-wide evidence ceiling: context only (pilot diagnostic), never used by B3.

## Stage 1 — selection (primary)

`N` = C0 accuracy, `O` = oracle selection accuracy (= C16 by construction), `H = O − N`, threshold `T = N + 0.5 H`, computed on the
reconstructed masks and frozen (stage `ceiling`) before any B3 prediction is scored. "B3 accuracy" = GT-class accuracy of B3's
semantic readout on the primary region (the top-weight candidate's class accuracy is also reported). `R = (acc − N) / H`.

* **S1 selection**: seed-mean B3 accuracy ≥ T (R ≥ 0.5) **and** B3 − B0 accuracy > 0 on every seed **and** the paired
  20-anchor block-bootstrap 95 % CI (2,000 resamples, seed 0; seed-pooled confusions) of the accuracy gain has lower bound > 0.
* **S2 candidate-hit**: on primary ∩ candidate-hit, B3 − B0 accuracy ≥ +2.0 points (seed mean), > 0 on every seed, CI lower > 0.
* Label-free selection metrics on 0006 delayed-target voxels, for B1, B2, B3, B3geom: top-1 / top-3 agreement with
  argmax r, mean weight on the best candidate, KL to the candidate target. GT diagnostic: accuracy of the delayed-best candidate's
  class, nearest candidate, top-weight candidate.

## Stage 2 — semantic completion

Primary: completed_unobserved (= pilot `completed_never_observed`) mIoU. Also geometry-unobserved, later-revealed, later-revealed
without an H_t camera return, geometry-observed, candidate-hit, candidate-miss, SSC full; per class; distance and reveal bins.

* **M1 semantic**: seed-mean mIoU(B3) − mIoU(B0) ≥ +1.0 point, > 0 on every seed, CI lower > 0; seed-mean mIoU(B3) > pilot B2;
  ≥ 5 classes with positive seed-mean IoU gain over B0.
* **Occ**: occupancy TP/FP/FN identical to the pilot / Gate 15 seed-0 counts on every anchor.

## Decision (exactly one, in this order)

1. Invalid (stop) if Occ fails.
2. **D** if B3's label-free top-1 agreement with the delayed best candidate exceeds B1's by ≥ 10 points **and** the B3 − B0 GT
   accuracy gain has CI upper bound < 0.
3. **A** if S1, S2 and M1 pass → next: freeze candidate-level supervision and address candidate-miss semantics.
4. **B** if S1 passes but S2 or M1 fails → next: a mechanism able to predict identity absent from local candidates.
5. **C** otherwise (S1 fails) → next: change the association signal/representation, not model size.

Report R whatever the outcome (0 < R < 0.5 is partial recovery, still C).
