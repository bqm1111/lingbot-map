# Final zero-shot evaluation — causal, single-camera, semantic-annotation-free semantic scene completion (2026-09-15)

## Setup

**Frozen system.** Nothing was retrained or tuned after target access.
- Five causal frames, local-reset LingBot with LB-M depth and MoGe-2 clip scale.
- Hard mapper and CompletionUNet (geometry seed 0).
- Trident-H evidence.
- Semantic heads: **B0** (nearest evidence) and **B3** (Gate 16 candidate-selection head, seeds 0/1/2, trained only on KITTI-360).
- One camera: SemanticKITTI `image_2`, Occ3D `CAM_FRONT`.

**Samples.** All SemanticKITTI 08 / Occ3D-nuScenes val samples that OccAny scored and that have a causal window.
- SemanticKITTI: **801**.
- Occ3D: **4,219**.
- Labels: OccAny's stored labels, verified identical to the official files on every sample.

**Freeze and verification.**
- Protocol, evaluators and analysis were hash-frozen before any prediction or score existed (`practical_evaluation_predeclaration.md`).
- Verification: 9/9 (`verification.md`).
- Numbers: `results_numbers.md`, `tables/`, `figures/`. Deviations are at the end.

## Headline

| dataset | method | cameras | temporal context | evaluation | SC IoU | SSC mIoU | completed no-evidence mIoU |
|---|---|---:|---|---|---:|---:|---:|
| SemanticKITTI | B0 | 1 | 5 causal | Track P | 32.65 | 5.45 | 12.33 |
| SemanticKITTI | B3 (3 seeds) | 1 | 5 causal | Track P | 32.65 | 5.64 | 12.58 |
| SemanticKITTI | OccAny released geometry | 1 | 5 forward (non-causal) | Track P, Level A | 31.28 (raw) / 31.05 (pooled) | — | — |
| Occ3D | B0 | 1 | 5 causal | Track P | 36.87 | 4.36 | 8.18 |
| Occ3D | B3 (3 seeds) | 1 | 5 causal | Track P | 36.87 | 4.54 | 8.72 |
| Occ3D | OccAny released geometry | 1 | 5 forward | Track P, Level A | 30.05 / 30.78 | — | — |

| dataset | method | official SC IoU | official SSC mIoU | superclass mIoU |
|---|---|---:|---:|---:|
| SemanticKITTI | ours B0 / B3 | 26.40 | 4.32 / 4.43 | 8.33 / 8.42 |
| SemanticKITTI | OccAny released geometry | 25.21 raw / 25.93 pooled | — | — |
| SemanticKITTI | OccAny + GSAM2, published (5 forward frames) | 25.91 | **7.28** | **13.53** |
| Occ3D | ours B0 / B3 | 31.94 | 3.71 / 3.80 | 6.63 / 6.76 |
| Occ3D | OccAny released geometry (5 forward frames, front camera) | 20.66 raw / 23.55 pooled | — | — |
| Occ3D | OccAny + GSAM2, published (**6 surround cameras**, full camera mask) | 34.15 | **6.66** | **10.32** |

## Answers

**1. What practical protocol was frozen, and why?**
- **Region (Track P):** the input camera's horizontal field of view, within 40 m, on the official grids and valid masks.
  - A single forward camera is responsible for what is in front of it.
  - 40 m is the common forward extent of both benchmark grids.
- **Observed vs completed space:** scored separately. Observed = evidence present; completed = predicted occupied without evidence; geometrically unobserved is its own population.
- **Central metric:** semantic mIoU on voxels that are GT occupied, predicted occupied and have no semantic evidence.
- **Other breakdowns:** distance to nearest evidence and range.
- **Track O:** the official computation on the same predictions; its confusions are identical to Track P's full-region ones.
- **When frozen:** before any prediction.

**2. B0 on SemanticKITTI (Track P):**
- occupancy IoU 32.65 (precision 43.9, recall 56.0)
- SSC mIoU 5.45
- completed no-evidence mIoU 12.33 (accuracy 42.0 %)
- observed-with-evidence 26.47

**3. B3 on SemanticKITTI:**
- SSC 5.64 (+0.19 [+0.05, +0.34])
- completed no-evidence 12.58 (**+0.25 [−0.08, +0.57]**; seeds +0.32 / +0.16 / +0.26; 25 of 33 blocks)
- no-evidence at GT occupancy 12.85 (+0.46 [+0.03, +0.87])
- geometrically unobserved 11.74 (+0.36 [+0.03, +0.65])
- accuracy on the central population 44.1 % vs 42.0 %

**4. Does B3 improve truly completed, no-evidence voxels?**
- **Occ3D: yes, reliably but modestly:** +0.54 [+0.29, +0.86] (all seeds positive; 104 of 150 scenes).
- **SemanticKITTI:** the point estimate is positive and every seed is positive, but the CI includes zero.
  - The related populations (no evidence, geometrically unobserved, SSC) are significantly positive.
  - Evidence-voxel semantics are identical on SemanticKITTI by construction. On Occ3D they differ slightly (+0.39), because 0.4 m voxels average their subvoxels.

**5. As distance from evidence grows:**
- Everything degrades steeply.
  - SemanticKITTI mIoU / accuracy: 15.3 / 47 % at 0–1 m → 3.8 / 19 % beyond 8 m.
  - Occ3D: 11.9 / 47 % → 2.9 / 23 %.
- B3's accuracy gain over B0 persists at every distance: SemanticKITTI +1.4 to +2.5 points; Occ3D +0.6 to +2.4 points.
- B3's mIoU gain is largest close to evidence. It vanishes or turns slightly negative beyond 4 m on Occ3D.
- Learned completion does not become *more* useful than nearest-evidence transfer far from evidence.

**6. B0 on Occ3D (Track P):**
- occupancy IoU 36.87 (precision 77.0, recall 41.4)
- SSC mIoU 4.36
- completed no-evidence 8.18
- observed-with-evidence 20.16

**7. B3 on Occ3D:**
- SSC 4.54 (+0.18 [+0.09, +0.30])
- completed no-evidence 8.72 (+0.54 [+0.29, +0.86])
- no-evidence at GT occupancy 12.30 (+0.55 [+0.41, +0.75])
- geometrically unobserved 7.73 (+0.50 [+0.20, +0.84])

**8. Does the improvement generalise across both datasets?**
- The *direction* does: B3 ≥ B0 in every semantic population, every seed, on both datasets.
- The *size* does not clear the bars: under 1 mIoU point everywhere.
- Under the predeclared rule it generalises on Occ3D but not on SemanticKITTI (central CI lower bound −0.08).
- It is a small, consistent, KITTI-360-learned effect, of about the same size as on the source dev drive (+0.58).

**9. Classes that generalise (central population, B3 − B0 IoU):**
- SemanticKITTI:
  - vegetation +2.6, car +2.4, truck +3.9, other-vehicle +1.9, bicycle +1.1 and sidewalk +1.0
  - road +0.4
- Occ3D:
  - construction vehicle +3.1, driveable surface +1.6, bus +1.5, car +1.0 and truck +0.8
  - trailer +0.7, vegetation +0.5 and sidewalk +0.5

**10. Classes that fail:**
- **Rider classes on SemanticKITTI (B3 worse):** bicyclist −6.0, motorcyclist −2.0, person −0.6, pole −0.5.
- **Wrong teacher readout (teacher-level failures that no head fixes):**
  - **SemanticKITTI terrain** is 21 % of central voxels but has IoU 0.01. Its voxels are read as *other-ground*, which makes up 15 % of predictions against 0.2 % of GT.
  - **Occ3D:** *other_flat* makes up 17 % of predictions (GT 1.8 %), *others* 6 % (GT 0.07 %); terrain IoU is 2 and manmade 3.
- These are readout failures under the frozen phrases and vocabulary. They were not tuned after the scores were seen.

**11. Is occupancy unchanged?**
- Yes. B0 and B3 share one occupancy volume.
- Per sample it is identical to the frozen causal evaluation of the OccAny comparison study (verification check 4).

**12. Official benchmark numbers:**
- **SemanticKITTI:** SC IoU 26.40; SSC mIoU B0 4.32 / B3 4.43; superclass 8.33 / 8.42.
- **Occ3D:** SC IoU 31.94; SSC mIoU 3.71 / 3.80; superclass 6.63 / 6.76.
- Samples: 801 / 4,219, causal, one camera, raw, front-half mask on Occ3D.

**13. What OccAny uses** (`occany_comparison_audit.md`):
- **Training:** Waymo, DDAD, PandaSet, VKITTI2 and ONCE, from all cameras, with projected LiDAR supervision. No target data.
- **Semantics:** Grounded-SAM2 with several prompt variants per class, plus Segmentation Forcing distillation.
- **SemanticKITTI input:** 5 frames running *forward* from the target, with novel-view rendering.
- **Occ3D semantics:** 6 surround cameras at one timestep.
- **Size:** 651 M parameters, plus SAM2-L and GroundingDINO.
- **Published:** SC IoU 25.91 / 23.55 (5-frame) / 34.15 (surround); mIoU 7.28 / 6.66.

**14. Can OccAny be evaluated under Track P?**
- **Geometry: yes (Level A).** Its released predictions go through the same evaluator.
- **Semantics: no.**
  - OccAny's semantic pipeline needs GroundingDINO's deformable-attention CUDA kernel, which fails on this GPU (sm_120). It failed as released and again after a clean in-place rebuild.
  - The released outputs contain no semantics.
  - The attempt is recorded in `logs/occany_semantic_attempt.json`; OccAny's original build was restored.
  - Semantics are therefore compared at Level B/C.

**15. Under the fairest comparison:**
- **Geometry, same evaluator (Level A):** ours is stronger.
  - SemanticKITTI Track P: +1.37 [+0.39, +2.30] vs raw and +1.60 vs pooled; ours is much better within 20 m (0–10 m: 50.5 vs 23.3) and OccAny is better at 30–40 m (30.1 vs 23.2).
  - SemanticKITTI official: +1.19 [+0.35, +2.09] vs raw; +0.47 [−0.42, +1.36] vs pooled (not distinguishable).
  - Occ3D: +6.8 [+3.9, +9.9] Track P; +8.4 official vs pooled. This holds even though OccAny sees future frames.
- **Semantics (Level B):** ours is weaker. Official SSC mIoU is 4.43 vs 7.28 on SemanticKITTI and 3.80 vs 6.66 on Occ3D.
  - On Occ3D OccAny uses 6 cameras and the full camera mask, so that pair is not like-for-like.
  - On SemanticKITTI both use one camera, but OccAny sees future frames, renders novel views and uses many prompts per class.
  - The superclass gap is similar (8.4 vs 13.5).

**16. Does OccAny receive more input?**
- Yes.
  - Future frames on SemanticKITTI and Occ3D 5-frame; six surround cameras for Occ3D semantics.
  - Test-time novel-view rendering.
  - A 651 M-parameter model trained on five multi-camera datasets with LiDAR supervision.
- Ours: one camera, four past frames, and a 25 k-parameter semantic head on frozen foundation models.

**17. Is the current system practically useful?**
- **Geometry: yes, in its operating region.**
  - Occupancy IoU 50.5 (SemanticKITTI) and 46.2 (Occ3D) within 10 m of the camera; 41.1 / 35.8 at 10–20 m.
  - Causal, 0.2 s of geometry per sample after the foundation models.
- **Semantics: not yet.**
  - SSC mIoU is ~5 in the operational region.
  - Even observed voxels with direct evidence reach only 26 / 20 mIoU, so the teacher's readout on target imagery caps everything downstream.
  - Completed-space semantics are 8–13 mIoU and fall below 4 beyond 4 m from evidence.

**18. Is this enough evidence to close the paper?**
- Enough to close the *semantic-completion* line.
  - The learned head's advantage is real in direction but small, and it does not meet the predeclared SemanticKITTI criterion.
  - The semantic level is far below the published OccAny semantics.
- The geometry result is the strong, verified finding: causal, single-camera, better than OccAny's released geometry under the same evaluator.

## Deviations and disclosures

- **Freeze with a failing test.** The freeze command ran while one protocol test failed. The test's expected value was wrong, not the code. The test was corrected; no frozen file changed.
- **Stuck background job.** A background wait loop stalled because its process check matched its own shell. It was restarted; no computation was affected.
- **OccAny build.** OccAny's GroundingDINO extension was rebuilt in place for one attempt and then restored byte-identical.
- **Occupancy seed.** Geometry seed 0 is used throughout, because the semantic heads were trained on it. Earlier multi-seed occupancy numbers are context only.

**PAPER C** — see `paper_decision.md`.
