# Does the Waymo geometry change semantic occupancy?

**Run 2026-10-20.** Raw: `artifacts/waymo_arm/results/semantic_impact_B0.json`. Code:
`tools/waymo_arm/cli/predict_semantic.py` (predictions),
`tools/waymo_arm/cli/score_semantic.py` (scoring). Predictions:
`/media/minh/MINH1/lingbot_waymo_arm/zero_shot_predictions`, 801 SemanticKITTI and 4,219
Occ3D anchors, both complete.

This report answers **Q2**: *does better geometry translate into better semantic occupancy
mIoU?*

---

## 0. What was substituted, and what was not

One object: the `CompletionUNet` returned by
`semantic_completion.frozen_geometry.load_frozen_geometry`, bound to the Waymo arm's seed-0
checkpoint. `GEOMETRY_SEED` stays 0, so it is a one-for-one replacement of the KITTI-360
seed-0 checkpoint.

Unchanged: Trident-H, the evidence lift, the B0 nearest-evidence rule, the frozen text
readout, the 18-phrase vocabulary and class mappings, the clips, the tokens, the anchors, the
prediction and evaluation grids, and the evaluation protocol.

Two safeguards, both confined to the Phase 4 CLI and neither modifying a frozen file:

* The frozen loader's sha256 assertion still runs. It is given a freeze record generated from
  the Waymo checkpoints' real hashes, so it has something true to check rather than being
  bypassed.
* `PREDICTION_ROOT` is redirected to the Waymo arm's own directory, with an assertion that
  refuses to start if the redirect failed. The frozen KITTI-360 prediction cache cannot be
  written to.

Both arms are scored by **one function in one process**, so no column can differ by scorer.

---

## 1. Measured evidence — the headline

Δ = Waymo − KITTI-360. Paired block bootstrap, 2,000 resamples, seed 0; blocks are 25
contiguous anchors on SemanticKITTI (33 blocks) and one per scene on Occ3D (150 blocks).
Both arms are always resampled on the same anchors, so each interval is an interval on the
**difference**. `*` = 95 % CI excludes zero.

### SemanticKITTI sequence 08, 801 anchors

| population | SC IoU k360 | Waymo | Δ SC IoU | SSC mIoU k360 | Waymo | Δ mIoU |
|---|---:|---:|---:|---:|---:|---:|
| full | 26.40 | 23.55 | **−2.84** [−3.41, −2.26]* | 4.32 | 4.90 | **+0.58** [+0.44, +0.71]* |
| observed | 33.43 | 29.34 | −4.09 [−4.99, −3.19]* | 6.97 | 8.12 | +1.16 [+0.81, +1.46]* |
| completed | 24.92 | 21.77 | −3.15 [−3.83, −2.44]* | 3.46 | 3.59 | +0.14 [+0.04, +0.25]* |

### Occ3D-nuScenes validation, 4,219 anchors

| population | SC IoU k360 | Waymo | Δ SC IoU | SSC mIoU k360 | Waymo | Δ mIoU |
|---|---:|---:|---:|---:|---:|---:|
| full | 31.94 | 22.85 | **−9.09** [−10.19, −8.00]* | 3.71 | 3.74 | **+0.04** [−0.24, +0.28] |
| observed | 41.82 | 33.50 | −8.32 [−9.87, −6.90]* | 5.66 | 6.93 | +1.27 [+0.87, +1.64]* |
| completed | 27.36 | 17.01 | −10.34 [−11.53, −9.25]* | 2.69 | 1.83 | −0.85 [−1.12, −0.63]* |

`observed` = valid ∧ `pred.observed` (the causal hard map resolved something).
`completed` = valid ∧ ¬`pred.observed`. These are not the frustum split; that is
`waymo_vs_kitti360_report.md` §9.

**The geometry got clearly worse and the semantic metric did not follow.** SC IoU falls by
2.8 and 9.1 points with intervals clear of zero, while SSC mIoU moves **+0.58** and **+0.04**.

### Validity control

Full-population Δ SC IoU here (−2.84, −9.09) agrees in sign and roughly in size with the
frozen Phase 3 evaluator's (−3.03, −8.36) on a different anchor population, different masks
and three seeds. And `k360` SemanticKITTI full SC IoU is **26.40**, which is the frozen
historical baseline figure to the digit.

---

## 2. Measured evidence — semantics on a fixed voxel set

The brief asks to separate semantic change caused by a *different occupancy mask* from
semantic change on a *fixed* set. Two populations do that:

* `correct_occupied` — **per arm**: valid ∧ predicted occupied ∧ GT occupied. Answers "when
  the geometry is right, is the label right?", but the voxel *set* differs between arms.
* `shared_correct` — the **intersection** of the two arms' correct-occupied sets. The voxel
  set is identical by construction, so a difference here is semantics alone.

| | k360 | Waymo | Δ |
|---|---:|---:|---:|
| **SemanticKITTI** | | | |
| `correct_occupied`, accuracy | 41.54 % | 47.45 % | **+5.92** |
| `correct_occupied`, voxels | 44,053,033 | 39,683,185 | −9.9 % |
| `shared_correct`, accuracy (33,687,018 voxels) | 42.371 % | 42.371 % | **+0.00** [0.00, 0.00] |
| **Occ3D** | | | |
| `correct_occupied`, accuracy | 31.98 % | 33.63 % | **+1.65** |
| `correct_occupied`, voxels | 16,201,022 | 11,914,775 | −26.5 % |
| `shared_correct`, accuracy (9,602,301 voxels) | 30.528 % | 30.584 % | **+0.06** [+0.03, +0.08]* |

**The +5.92 is a selection effect, not better semantics.** The Waymo arm predicts a 9.9 %
smaller true-positive set, and that smaller set is easier to label. Hold the voxels fixed and
the difference is zero.

The Occ3D +0.06 pp has a CI excluding zero and is still negligible: it is 0.06 of a
percentage point on 9.6 M voxels. Statistical significance at this sample size does not make
it a semantic improvement, and §3 identifies exactly where it comes from.

---

## 3. Measured evidence — why, structurally

Direct voxel-level comparison of the two arms' B0 labels on the intersection of their query
sets (10 anchors per dataset, `label_identity` check):

| dataset | shared query voxels | B0 label mismatches | occupancy mismatches |
|---|---:|---:|---:|
| SemanticKITTI | 1,921,828 | **0** (0.000 %) | 1,522,698 |
| Occ3D | 125,370 | 455 (0.363 %) | 422,987 |

**On SemanticKITTI the B0 label cannot depend on the geometry checkpoint at all.** The
prediction grid *is* the evaluation grid (256×256×32 @ 0.2 m both), so each native voxel has
exactly one subvoxel, and the aggregation selects it identically whether or not it is
occupied — occupied subvoxels if any, otherwise all of them, which for one subvoxel is the
same subvoxel. 1.5 M voxels change occupancy and not one changes label.

On Occ3D the prediction grid is 2× finer (400×400×32 @ 0.2 m against 200×200×16 @ 0.4 m), so
a native voxel aggregates eight subvoxels and the occupied subset does enter the readout.
That is the whole mechanism behind the 0.363 % of labels that move, and behind the +0.06 pp
in §2.

So the SemanticKITTI mIoU change of +0.58 is **entirely** a consequence of which voxels are
predicted occupied, with labels held fixed by construction. On Occ3D it is that plus a
0.363 % label perturbation.

---

## 4. Per-class results

SemanticKITTI, per-class IoU, full population:

| class | k360 | Waymo | Δ |
|---|---:|---:|---:|
| road | 20.15 | 27.09 | **+6.93** |
| sidewalk | 8.75 | 10.29 | +1.55 |
| car | 7.57 | 8.91 | +1.34 |
| other-vehicle | 1.90 | 3.23 | +1.33 |
| motorcycle | 4.58 | 5.37 | +0.80 |
| parking | 1.73 | 2.28 | +0.55 |
| person | 1.43 | 1.93 | +0.50 |
| motorcyclist | 1.90 | 2.22 | +0.33 |
| building | 6.52 | 6.84 | +0.32 |
| traffic-sign | 0.04 | 0.34 | +0.30 |
| fence | 2.67 | 2.90 | +0.23 |
| truck | 3.98 | 4.20 | +0.22 |
| bicycle | 3.68 | 3.89 | +0.21 |
| trunk | 0.83 | 0.99 | +0.16 |
| other-ground | 0.06 | 0.09 | +0.02 |
| terrain | 0.006 | 0.006 | −0.000 |
| pole | 0.46 | 0.46 | −0.004 |
| **bicyclist** | 2.32 | 0.56 | **−1.76** |
| **vegetation** | 13.49 | 11.54 | **−1.95** |

Occ3D, per-class IoU, full population:

| class | k360 | Waymo | Δ |
|---|---:|---:|---:|
| vegetation | 2.11 | 6.58 | **+4.46** |
| truck | 3.64 | 6.95 | +3.31 |
| bus | 3.29 | 6.22 | +2.93 |
| trailer | 1.22 | 2.20 | +0.98 |
| manmade | 0.28 | 1.01 | +0.73 |
| car | 5.24 | 5.82 | +0.59 |
| barrier | 0.89 | 1.33 | +0.44 |
| bicycle | 1.35 | 1.75 | +0.40 |
| motorcycle | 2.52 | 2.87 | +0.36 |
| others | 0.044 | 0.041 | −0.003 |
| terrain | 0.77 | 0.76 | −0.01 |
| pedestrian | 0.83 | 0.68 | −0.14 |
| other_flat | 0.85 | 0.70 | −0.15 |
| traffic_cone | 1.90 | 1.61 | −0.29 |
| construction_vehicle | 3.40 | 2.49 | −0.90 |
| **sidewalk** | 11.11 | 7.92 | **−3.20** |
| **driveable_surface** | 23.58 | 14.74 | **−8.85** |

The per-class pattern is an occupancy pattern. The Waymo arm trades recall for precision
(`waymo_vs_kitti360_report.md` §9), which raises IoU on large classes where false positives
dominated and lowers it where recall did — road +6.93 on SemanticKITTI against
driveable_surface −8.85 on Occ3D, the same kind of surface moving opposite ways on the two
benchmarks. No class gains because its semantics were learned better; on SemanticKITTI no
label changed at all.

---

## 5. Hypotheses

* **H2 (semantic supervision is the bottleneck) is what this result is consistent with.**
  Neither our system nor OccAny trains semantics from labels, so swapping the training corpus
  cannot give the teacher new classes or better 2D predictions. §3 shows this is not merely
  unlikely but structurally impossible on SemanticKITTI.
* **H1 (source data is the bottleneck) fails for semantics**, as it already failed for
  geometry in Phase 3. There is no path from the training corpus to the semantic pathway.
* **The SSC mIoU metric is weakly coupled to occupancy quality in this system.** A 9.1-point
  SC IoU collapse on Occ3D produced a mIoU change indistinguishable from zero. Consistent
  with `reports/semantic_bottleneck_diagnosis.md` §2: perfect semantics on today's geometry
  reaches only 23.88 / 19.87 mIoU, so mIoU is dominated by terms occupancy barely moves.

---

## 6. Statistical uncertainty

All intervals in §1 and §2 are paired block bootstraps at 2,000 resamples, seed 0, under
`practical.protocol.Statistics` — the same statistics as Phase 3, not chosen here.

Two asymmetries to state:

* **§3's structural check is 10 anchors per dataset, not 801 / 4,219.** It is a check on a
  mechanism, and the SemanticKITTI result (exactly 0 of 1,921,828) is a statement about how
  the aggregation is defined, which the 801-anchor `shared_correct` figure independently
  confirms by coming out bit-identical (42.371 % both arms). The Occ3D 0.363 % is the one
  number in this report that rests on a 10-anchor sample.
* **§4's per-class deltas carry no intervals.** Rare classes on these populations are
  volatile — `bicyclist` at 2.32 and `traffic-sign` at 0.04 rest on few voxels. Treat the
  large-class entries as informative and the rare-class ones as noise unless bootstrapped.

Single geometry seed throughout (seed 0), per the frozen canonical pipeline.

---

## 7. Confounding factors

* **Waymo's geometry is worse, not better.** Q2 asks whether *better* geometry improves
  semantics; the available contrast is *different and worse* geometry. The finding — that
  labels are invariant and only the mask moves — is direction-independent, but no measurement
  here shows what *better* geometry would do.
* **One seed per arm**, so a seed-level geometry difference is not separable from the arm
  difference in §1. Phase 3's per-seed spreads were non-overlapping between arms, which makes
  this unlikely to matter at these effect sizes.
* **`shared_correct` is conditioned on both arms being right**, which is a selected
  population — easier voxels than either arm's own true positives. It is the right instrument
  for isolating semantics and the wrong one for estimating absolute accuracy.
* **Occ3D's any-subvoxel reduction** is what makes its labels move at all; its +0.06 pp is an
  artefact of grid ratio, not a property of the semantic model.
* **Occ3D uses CAM_FRONT only**, the frozen single-camera protocol, and is not comparable to
  OccAny's published surround-view numbers.

---

## 8. Unsupported interpretations — explicitly not claimed

* **Not claimed:** that the Waymo arm has better semantics. The `correct_occupied` +5.92 pp
  is a selection effect (§2) and the brief's warning applies to it exactly.
* **Not claimed:** that +0.58 mIoU on SemanticKITTI is a semantic gain. No label changed
  (§3); it is an occupancy-mask gain measured by a semantic metric.
* **Not claimed:** that the Occ3D +0.06 pp on the fixed set means anything, despite its CI
  excluding zero.
* **Not claimed:** that the +0.58 / +0.04 pattern would reverse under *better* geometry. Not
  measured.
* **Not claimed:** that these numbers are comparable to the Phase 3 table. Different
  population, masks and seed count; §1 states the two side by side only to show agreement in
  sign.

---

## 9. Decisions and next actions

**Q2 answered: no.** Changing the geometry checkpoint moves SSC mIoU by +0.58 and +0.04, and
on SemanticKITTI the semantic labels are provably invariant to it. Geometry quality and
semantic label quality are decoupled in this system, because nothing in the semantic pathway
is trained on the geometry's training corpus.

Three consequences:

1. **Retire the "better source data" route to semantics.** H1 is now falsified for both
   geometry (Phase 3) and semantics (here). The Waymo arm is a domain result, not an
   improvement, and should be reported as one.
2. **Any future semantic claim must be scored on a fixed occupancy mask.** `shared_correct`
   is the population that does this; `correct_occupied` and `full` both let occupancy leak
   in, and §2 shows the leak is larger than any semantic effect this project has measured.
3. **The next experiment belongs on the semantic side**, under the constraints in
   `reports/semantic_bottleneck_diagnosis.md` §9 — and whether privileged 3D semantic
   supervision is a credible route is answered in
   `reports/privileged_semantic_supervision_feasibility.md`.
