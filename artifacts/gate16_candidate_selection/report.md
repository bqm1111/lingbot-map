# Gate 16 — can delayed observations teach which past semantic evidence belongs to a completed voxel? (2026-09-15)

KITTI-360 only (train 0003/0007/0010, development 0006, 590 anchors); geometry, occupancy, Trident-H, evidence memory, delayed
targets, candidate rule (16 nearest evidence voxels) and scorer are the pilot's, unchanged. Target datasets not opened; semantic
GT used for evaluation only. Rules: `predeclaration.md`, hash-frozen in three stages (predeclaration → reconstructed ceiling and
threshold → evaluation code), all before any B3 score (`frozen_*.json`). Verification 15/15 (`verification.md`); 20 tests pass.

**B3** = pilot scorer + pilot observed-only term, with B2's delayed final-feature KL replaced by candidate-level selection:
`r_i = <p_i, q_v>`, `target = softmax(r/τ)`, τ = 1/40 (teacher scale, not swept), `KL(target || softmax(scorer logits))`.
Seeds 0/1/2 (seed 0 reuses B2's exact draws). **B3geom**: semantic projection removed from the scorer input. B1/B2 are the pilot's
single-seed controls. Tables: `selection_results.md`, `results_table.md`, `candidate_ceiling.md`, `tables/`; figures: `figures/`.

## Headline (primary region = completed_unobserved)

| | C0 nearest | B0 | B1 | B2 | **B3 (3 seeds)** | B3geom | oracle |
|---|---|---|---|---|---|---|---|
| GT-class accuracy % | 46.00 | 46.01 | 47.09 | 46.03 | **47.69** (47.76 / 47.65 / 47.66) | 47.48 | 57.32 |
| recovered headroom R | 0 | 0.00 | 0.10 | 0.00 | **0.15** (0.155 / 0.146 / 0.146) | 0.13 | 1 |
| mIoU % | 8.77 | 8.80 | 9.13 | 9.21 | **9.38** (9.39 / 9.37 / 9.37) | 9.28 | 14.90 |

Threshold (frozen before scoring): T = 46.00 + 0.5 × 11.31 = **51.66 %**. B3 − B0 accuracy +1.68 [+1.50, +1.88] (positive on every
seed) — real, but S1 fails. S2 (candidate-hit accuracy +2.77 [+2.43, +3.09]) passes. M1 (mIoU +0.58 [+0.34, +0.90]; bar +1.0)
fails. Occupancy identical. D not triggered.

## Answers

1. **Nearest-candidate GT accuracy:** 46.00 % (2,743,212 primary voxels). B0's own readout: 46.01 %.
2. **Any-of-16 ceiling:** 57.32 %; 42.68 % of primary voxels are candidate-miss. Oracle selection mIoU 14.90 % (nearest 8.77 %).
   History-wide evidence (context only): 97.49 %.
3. **Headroom:** 11.31 accuracy points.
4. **Recovered by B3:** R = 0.149 (readout; 0.082 for the single top-weight candidate), 0.146–0.155 across seeds. B1 recovers 0.096,
   B2 0.003, B3geom 0.131. Far below the predeclared 0.5.
5. **Does B3 learn the delayed candidate target?** Partly. On 0006 delayed-target voxels its top-1 agrees with the best delayed
   candidate 31.5 % of the time (B1 9.0 %, B2 20.2 %, nearest 7.6 %), top-3 48 %, KL 0.30 (B1 0.99). But its weights stay nearly
   flat: mean weight on the best candidate 8.6 % against a uniform 6.25 %. The soft target is itself flat (mean top mass ≈ 0.14),
   because many candidates share a class.
6. **Does delayed candidate selection correspond to GT correctness?** Only weakly — this is the key finding. On those voxels the
   class of the candidate the delayed target ranks best is correct 53.5 % of the time, versus 48.1 % for the nearest candidate.
   Perfectly learning the target would add only ~5 points there; B3's top-weight candidate reaches 50.8 %, about half of that.
   The supervision does not point to the GT-correct candidate often enough to reach the 57 % ceiling. It is not *wrong* (D fails:
   GT accuracy rises with target agreement), just weak.
7. **Candidate-hit voxels:** B3 83.0 % accuracy (top-weight 81.9 %) vs B0 80.3 % vs oracle 100 %; mIoU 27.08 vs 23.41 vs 100.
   B3 closes 14 % of the gap to the oracle on exactly the voxels where the answer is available.
8. **Semantic mIoU:** +0.58 over B0 [+0.34, +0.90], +0.58 on every seed; +0.25 over B1 [+0.06, +0.52]; +0.17 over B2 [−0.01, +0.33];
   13 classes gain. Gains are broad but small; SSC full 1.67 vs B0 1.55.
9. **Association-limited classes** (correct candidate usually present, poorly selected): building (any-of-16 77.1 %, B3 top-weight
   correct 49.7 %, *below* nearest 58.4 %), truck (84.5 vs 67.1 %), car (47.2 vs 33.8 %). B3 moves selection toward vegetation
   (84.1 % vs nearest 78.8 %) and road at building's expense.
10. **Absence-limited classes** (correct candidate rarely among the 16): terrain (1.4 %), bicycle (4.3 %), other vehicle (13.7 %),
    parking (14.1 %), pole (14.6 %), motorcycle (14.8 %), other ground (15.8 %), traffic sign, other object, sidewalk (22.3 %),
    road (28.5 %), other structure (31.0 %). No selector can fix these.
11. **Distance / reveal:** candidate-hit rate falls from 65.8 % (0–1 m) to 33.9 % (> 8 m) and from 62.5 % (revealed within 1 s) to
    36.8 % (4–10 s). B3's accuracy gain over B0 stays ~1–2 points at every distance, and vanishes at 4–10 s (30.40 vs 30.37 %). So
    with distance the problem shifts from association to absence; association itself does not get much worse.
12. **Next bottleneck:** still association, but because of the *signal*, not the scorer's capacity. The scorer learns what the
    delayed target says (top-1 ×3.5 over B1); the delayed semantic-similarity target identifies the GT-correct candidate only
    ~5 points more often than nearest. B3geom (no semantic identity in the scorer) recovers almost as much GT headroom (0.13 vs 0.15)
    while matching the delayed target far less (top-1 13 %), so the semantic comparison adds target agreement, not correctness.
    Expressivity (42.7 % candidate-miss) is the next wall after association, not the current one.
13. **Occupancy:** exactly unchanged — TP/FP/FN identical to the pilot and Gate 15 seed 0 on every anchor; geometry never received
    gradient.

## Decision C

Candidate association is not learned to a meaningful fraction of its headroom (R = 0.15 < 0.5): the delayed semantic-similarity
signal does not tell the scorer which observed evidence belongs to a completed voxel.

**Next action:** replace the semantic-similarity candidate target with a correspondence-based association signal — mark as correct
the candidates that the later observation places on the same physical surface as the target voxel (shared later-view LiDAR
surface / depth-continuous segment), keep the 16-candidate set, scorer size and all frozen components, and re-measure against the
same frozen ceiling.

## Reproduction

`tools/delayed_semantic/README.md` → "Gate 16" (exact commands). Code: `tools/delayed_semantic/gate16/`; hashes: `code_freeze.json`;
configuration: `candidate_config.yaml`; checkpoints `checkpoints/B3*_seed{0,1,2}_step03000.pt`; prediction records
`/media/SSD1/MINH_DATASETS/lingbot_delayed_semantic/gate16/predictions_at_t`. The pilot's frozen stages still verify.
