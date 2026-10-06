# Gate 15 — sparse trustworthy 2D semantics propagated into completed 3D space (2026-09-14)

Frozen geometry-only control (SC IoU unchanged, verified bit-for-bit) + frozen Trident-H evidence on the five causal frames + a 290k-parameter propagation head. KITTI-360 0003/0007/0010 train, 0006 evaluation; target datasets not opened.

## Headline (semantic mIoU %)

| region (0006, mean of 3 seeds) | B0 nearest evidence | B1 learned propagation | B1 − B0 |
|---|---|---|---|
| oracle_observed | 16.91 | 9.72 | -7.19 |
| completed_observed | 15.73 | 9.32 | -6.41 |
| completed_unobserved | 8.96 | 6.54 | -2.42 |
| ssc_full | 1.58 | 1.14 | -0.44 |
| ssc_no_evidence | 1.39 | 1.10 | -0.29 |


Teacher readout on directly observed GT-occupied voxels: mIoU 16.91 %, accuracy 62.57 %. Full tables: `results_table.md`; figures: `figures/`; verification 18/18 (`verification.md`).

## Answers

1. **Trident-H on directly observed 3D surfaces:** mIoU 16.91 %, mean class accuracy 33.24 %, overall accuracy 62.57 % (18/18 GT classes predicted).
2. **Trivial propagation (B0, nearest evidence):** completed_unobserved mIoU 8.96 %, accuracy 46.37 %; SSC mIoU 1.58 %.
3. **Does B1 beat it?** completed_unobserved Δ mIoU -2.42 points (per seed -2.68, -2.90, -1.68; CI [-3.06, -1.74]); SSC Δ -0.44 points. P1 fails the frozen +2.00 bar; P2 fails.
4. **In voxels with no direct observation?** completed_unobserved is by construction evidence-free; B1 6.54 vs B0 8.96 %. Label-free held-out evidence: cosine to the teacher B1 0.9227 vs nearest-visible B0 0.9799; class agreement 69.03 vs 84.16 %.
5. **How far does useful semantics propagate?** overall accuracy by distance to the nearest evidence (B1 / B0): 0-1 m 50.91 / 53.15; 1-2 m 43.82 / 52.35; 2-4 m 35.41 / 46.36; 4-8 m 23.84 / 31.11; >8 m 16.27 / 27.27.
6. **Classes that propagate better with B1:** road (+11.89), parking (+3.37), other ground (+1.95).
7. **Classes that fail / lose:** building (-13.17), car (-10.31), sidewalk (-8.21), truck (-8.18), other structure (-7.47), fence (-4.67), motorcycle (-3.74), other object (-1.80), vegetation (-1.05), person (-0.89), other vehicle (-0.75).
8. **Collapse toward priors?** B1 on completed_unobserved: dominant class vegetation at 43.11 % of predictions (GT dominant 38.68 %), 6.3 classes with ≥ 1 % of predictions (B0 9.0), entropy 2.05 bits (B0 2.40). Collapse clause passes.
9. **Occupancy exactly unchanged?** Yes: seed 0 TP/FP/FN [6172202, 27026109, 12021922] (SC IoU 13.65 %); seed 1 TP/FP/FN [6500344, 30096511, 11693780] (SC IoU 13.46 %); seed 2 TP/FP/FN [6129120, 27435715, 12065004] (SC IoU 13.43 %), identical to the frozen baseline; geometry digests unchanged by training.
10. **Genuine semantic completion rather than copying?** Not demonstrated: the frozen rule requires B1 to beat nearest-evidence copying by ≥ 2 mIoU points in evidence-free completed voxels, across seeds and classes.
11. **Ready for untouched zero-shot semantic evaluation?** No — target evaluation stays locked (Decision B).

## Interpretation (measurements vs reading)

* **Measured.** B1 is below nearest-evidence copying in every region, including evidence voxels where it receives the teacher feature directly (oracle_observed 9.72 vs 16.91 mIoU) and in the label-free held-out test (cosine 0.923 vs 0.980). B1 predicts 7.7 of 18 GT classes on completed_unobserved; B0 predicts all of them. It gains only on large flat classes (road, parking, other ground).
* **Supported reading.** The head did not learn to preserve, let alone propagate, teacher class identity: its output keeps the teacher's shared feature direction (final training loss 0.03–0.06 against 0.093 for a constant mean feature) while losing the small residual angles that decide the class. This matches the risk recorded in `predeclaration.md` before evaluation.
* **Rule vs pattern.** The frozen order yields Decision B: P1 fails, and the collapse clause of P3 passes (dominant fraction and class count within their limits), so C is not triggered. The observed pattern is nonetheless a *partial* collapse (whole classes vanish), not 'near the trivial baseline'. The decision stands as frozen; the next action targets the objective, not geometry.
* **Hypothesis (untested).** A distillation that removes the shared component (class-logit KL in the frozen text space) would let the same head keep class identity. No ablation of this exists.

## Decision B

| criterion | pass |
|---|---|
| P1 unobserved gain ≥ +2 mIoU, all seeds, CI > 0 | False |
| P2 SSC gain all seeds | False |
| P3 ≥ 5 classes gain, no collapse | False |
| P4 occupancy identical | True |
| D teacher poor on observed | False |


**Decision B.**

## Next action

Keep geometry and the teacher interface fixed and change only the propagation supervision/representation: train the head with a distillation target that removes the teacher's shared feature direction (centred, text-projected class-logit distillation on the same evidence voxels), then re-run this source gate unchanged.

## Reproduction

See `tools/semantic_completion/README.md`.
