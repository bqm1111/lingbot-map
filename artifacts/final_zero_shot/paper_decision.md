# Paper decision (applied mechanically from the rule frozen in `practical_evaluation_predeclaration.md`)

| criterion | SemanticKITTI | Occ3D-nuScenes |
|---|---|---|
| central B3 − B0 (completed, no evidence; Track P) | +0.25 [−0.08, +0.57], all seeds > 0 | +0.54 [+0.29, +0.86], all seeds > 0 |
| generalises (Δ > 0, CI lower > 0, every seed) | **no** | yes |
| meaningful (Δ ≥ +1.0) | no | no |
| Track O SSC mIoU (B3) ≥ OccAny published | 4.43 < 7.28 | 3.80 < 6.66 |

The rule is PAPER C if B3 does not generalise on both datasets, so the decision is **PAPER C**.

What the evidence does support, for whatever paper-level decision follows:
- **B3 over B0.** B3 is never worse than B0 in any semantic population, seed or dataset. The gains are sub-point, roughly the size seen on the KITTI-360 dev drive.
- **Semantics are capped by the teacher.** Voxels with direct Trident-H evidence reach only 26 / 20 mIoU. Ground classes are misread: SemanticKITTI terrain IoU is 0.01, and *other-ground* / *other_flat* are over-predicted.
- **Geometry is the strong result.** The causal single-camera geometry beats OccAny's released (non-causal) geometry under the same evaluator:
  - SemanticKITTI: +1.4 Track P, strongest within 20 m.
  - Occ3D: +6.8 Track P and +8.4 official vs pooled.

**PAPER C**

**Next action:** close the candidate-transfer semantic-completion branch now. Make one deliberate paper-level decision: whether to write the causal single-camera zero-shot geometry result, with these semantic numbers as a reported limitation.
