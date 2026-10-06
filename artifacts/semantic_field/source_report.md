# Final semantic-method experiment — generative 3D semantic field (B4), source report (2026-09-15)

**Outcome: SOURCE FAIL.** The frozen promotion rule fails on S1, S2, S3 and S4. SemanticKITTI and Occ3D were never opened.

**Setup.**
- KITTI-360 only: training drives 0003/0007/0010, development drive 0006 (590 anchors).
- Frozen geometry seed 0, Trident-H, evidence lifting and the pilot's delayed targets.
- Rules in `source_predeclaration.md`, hash-frozen in three stages, all before any B4 checkpoint or score. The training-drive mean μ has its own freeze.
- Verification: **10/10** (`verification.md`).
- Tables: `source_tables.md`, `source_results.csv`. Figures: `figures/1–4`.

**B4.**
- A 1,357,008-parameter residual 3D UNet on the frozen CompletionUNet trunk, the hard map, occupancy, and projected centered sparse evidence.
- A pointwise head generates a free 1024-d, L2-normalised feature in Trident-H space. It is not a mixture of observed features.
- Training: cuboid masked-evidence reconstruction (text KL + centered feature cosine) plus privileged delayed-view class distributions (text KL).
- Seeds 0/1/2, 6,000 steps each, final checkpoint.

## Source results (0006, mIoU %, B3 and B4 = mean of three seeds)

| region | voxels | B0 | B3 | B4 (seeds) | B4 − B0 [95 % CI] | B4 − B3 [95 % CI] |
|---|---:|---:|---:|---:|---|---|
| **completed_unobserved (PRIMARY)** | 2,743,212 | 8.80 | 9.38 | **8.22** (7.93 / 8.33 / 8.42) | **−0.57 [−1.37, +0.23]** | **−1.15 [−2.13, −0.20]** |
| geometry_unobserved | 2,267,211 | 7.73 | 8.16 | 7.55 | −0.18 [−0.88, +0.56] | −0.61 [−1.34, +0.21] |
| no_semantic_evidence (GT occupancy) | 7,834,276 | 8.39 | 8.85 | 7.63 | −0.76 [−1.46, −0.13] | −1.22 [−1.90, −0.61] |
| completed_unobserved, later revealed | 559,021 | 7.06 | 7.29 | 6.92 | −0.14 [−0.82, +0.46] | −0.38 [−1.10, +0.22] |
| **candidate-supported** | 1,572,285 | 23.41 | 27.08 | **10.30** | −13.13 [−15.80, −11.52] | −16.79 [−19.37, −14.04] |
| **candidate-unsupported** | 1,170,927 | 0.00 | 0.08 | **3.59** | **+3.60 [+3.25, +3.95]** | +3.51 [+3.15, +3.87] |
| oracle_observed / completed_observed | — | 16.91 / 14.86 | same | same | 0 (shared evidence path) | 0 |
| SSC full | 558,778,307 | 1.55 | 1.67 | 1.35 | −0.21 [−0.41, −0.05] | −0.33 [−0.56, −0.14] |

| criterion | value | pass |
|---|---|---|
| S1 B4 − B0 ≥ +2.0 | −0.57 | no |
| S2 all seeds > 0, CI lower > 0 | seeds −0.87 / −0.47 / −0.37; CI [−1.37, +0.23] | no |
| S3 B4 − B3 ≥ +1.0, CI lower > 0 | −1.15 [−2.13, −0.20] | no |
| S4 ≥ 8 classes improve | 6 | no |
| S5 no collapse | dominant 0.40 ≤ 0.60; 7 classes ≥ 1 %; entropy 2.15 ≥ 0.75 × 2.71 | yes |
| S6 occupancy identical | every anchor | yes |

## Answers

**1. Can B4 generate semantic identities absent from nearby evidence?**
- **Yes.** On candidate-unsupported voxels, where no candidate carries the GT class, B4 reaches 3.59 mIoU against 0.00 for B0 and 0.08 for B3 (+3.60 [+3.25, +3.95], all seeds about 3.6).
- **But the generated identities are mostly context priors.** On the primary region B4 over-predicts road (33 % of predictions vs 11 % of GT) and car. Minority classes (fence, truck, motorcycle, person, other object, terrain) and thin structure (sidewalk, other structure) nearly vanish.

**2. Does masked semantic reconstruction work?**
- **As a training objective, yes.** Over training, agreement with the hidden teacher class rose from 0.67 to 0.86, and the centered feature loss fell from 0.97 to 0.34, so it did not collapse onto the shared direction as Gate 15 did.
- **On 0006 held-out masked evidence it only partly transfers:**
  - Teacher-class agreement: B4 70.7 %, below nearest remaining evidence (78.3 %) and B3 (78.7 %).
  - GT accuracy: B4 is higher (64.1 vs 59.9 / 61.8).
  - GT mIoU: B4 is lower (12.1 vs 13.8 / 15.0).
- **Reading:** B4 is more often right on frequent classes and less often right on rarer ones.

**3. Does delayed privileged supervision improve genuinely hidden voxels?**
- **Not measurably in this design.** On later-revealed completed voxels B4 scores 6.92 against B0 7.06 and B3 7.29.
- Training agreement with the delayed class rose from 0.53 to 0.63.
- No B4-without-delayed ablation exists (one method only, by predeclaration), so its separate contribution is not isolated.

**4. Does B4 beat B0 by ≥ 2 mIoU on source completed_unobserved?**
- **No.** −0.57 [−1.37, +0.23], negative on every seed.

**5. Does B4 beat B3?**
- **No.** −1.15 [−2.13, −0.20].

**6. Are gains broad across classes?**
- **No.** 6 classes improve over B0: car +5.6, road +9.9, vegetation +1.9, other ground +1.4, other vehicle +0.4, pole +0.1.
- Losses: sidewalk −5.8, other structure −5.9, fence −4.4, truck −5.8, motorcycle −3.4, and minority classes to ~0.

**7. Does B4 improve candidate-unsupported voxels specifically?**
- **Yes (+3.6)**, which is the only region where transport is structurally incapable.
- It **loses 13 points on candidate-supported voxels** (1.57 M voxels vs 1.17 M unsupported). The net is negative.
- **By distance:** B4 is better 2–8 m from evidence (2–4 m: 9.57 vs 7.72) and worse within 1 m (10.33 vs 12.39) and beyond 8 m (0.79 vs 3.07).
- So the generative path fills a gap that transport can't, but it gives up the instance-specific identity that nearby evidence carries.

**8–10. SemanticKITTI and Occ3D zero-shot SSC; consistency across seeds.**
- **Not evaluated.** SOURCE FAIL keeps the targets locked (no target predictions exist; verification check 6).
- **Frozen context:** in the final zero-shot evaluation, B0 / B3 reached 4.32 / 4.43 official SSC mIoU on SemanticKITTI and 3.71 / 3.80 on Occ3D.

**11–12. Official semantic mIoU reached.**
- Unchanged from the frozen final evaluation: SemanticKITTI 4.43 and Occ3D 3.80 (B3).
- B4 was not promoted.

**13. Numerical comparison with OccAny.**
- OccAny's published semantics (verified in `artifacts/final_zero_shot/occany_comparison_audit.md`): SemanticKITTI 7.28 mIoU / 13.53 superclass (5-frame forward); Occ3D 6.66 / 10.32 (six surround cameras).
- Ours stays below both.

**14. Differences that make the comparison non-identical.**
- **Cameras:** OccAny uses six surround cameras with the full camera mask for its Occ3D semantics; ours uses one front camera.
- **Temporal context:** OccAny's SemanticKITTI setting uses future frames; ours uses four past frames.
- **Training:** OccAny trains on Waymo, DDAD, PandaSet, VKITTI2 and ONCE with LiDAR supervision; our learned heads use KITTI-360 only.
- **Semantics:** OccAny uses Grounded-SAM2 with many prompts per class plus novel-view rendering; ours uses Trident-H with one frozen phrase per class.

**15. Credible practical advantage despite these differences?**
- **For semantics, no.**
- **For geometry, yes.** Under the same evaluator, the causal single-camera geometry beats OccAny's released geometry (final evaluation: +1.4 IoU on SemanticKITTI and +6.8 on Occ3D in Track P).

**16. Is occupancy unchanged?**
- **Yes.** It is bit-identical to Gate 15 seed 0 on all 590 anchors, and no gradient reached the geometry network (tests and digests).

**17. Strong enough semantic SSC result to close the paper?**
- **No.** The last materially different semantic idea fails the frozen source gate.
- It does show that a generative field can reach classes that transport can't, at the cost of the identity that nearby evidence provides.

**SOURCE FAIL → semantic-completion method development is closed.** Decision: `paper_decision.md`.
