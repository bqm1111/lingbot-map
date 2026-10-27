# Privileged 3D semantic supervision — feasibility

**Written 2026-10-20.** Analysis only; no model was trained for this report. Sources, all
pre-existing frozen artifacts: `artifacts/gate20_future_revealability/analysis.json`,
`artifacts/gate21_future_semantic_prior/{summary.txt,metrics.json,selection_bias.json}`,
`artifacts/gate18_temporal_semantics/{scaling.json,teacher_quality.json}`,
`artifacts/delayed_semantic_pilot/`, `artifacts/semantic_field/`,
`artifacts/waymo_arm/results/bottleneck_B0.json`.

This report answers **Q4**: *does privileged 3D semantic supervision offer a credible path
beyond B0/B3?*

> ## Verdict: **NO-GO** as formulated. Recommend stopping this approach.
>
> The brief's own gate is that the supervision must provide *meaningful additional **reliable**
> hidden-region semantic coverage beyond the previously tested delayed-target strategy*. It
> does not, and the reason is measured three times over: coverage is thin, the labels are
> noisy enough to destroy three quarters of the headroom in the very voxels they cover, and
> the one experiment that trained on them came out **significantly worse than the baseline it
> was meant to beat**.

---

## 1. What "privileged 3D semantic supervision" means here

Supervision available at training time but not at inference: a 3D semantic label for a voxel
the anchor's own five-frame clip cannot see. Two distinct sources, and they must not be
conflated:

* **(A) Future-revealed teacher labels** — the voxel is observed later in the same drive, and
  the frozen 2D teacher's reading at that later time is carried back. This is what
  `tools/delayed_semantic/targets.py` builds and what Gates 20 and 21 tested. **The brief's
  "previously tested delayed-target strategy" is this.**
* **(B) Real 3D semantic ground truth** from the dataset's own annotations, used as
  source-domain training supervision.

§2–§4 evaluate (A) on measured evidence. §5 treats (B), which is a different proposition with
a different objection.

---

## 2. Measured evidence — (A) coverage

Gate 20, dev drive `2013_05_28_drive_0006_sync`, held out from every threshold. 590 anchors,
P0 = 2,743,212 hidden occupied voxels. Horizon 10 s, `visible` level.

| | train (3 drives, 1,276 anchors) | dev (held out) |
|---|---:|---:|
| P0 voxels | 8,379,519 | 2,743,212 |
| ever revealed within 10 s | 27.21 % | **31.94 %** |
| never revealed | 72.79 % | 68.06 % |
| revealed voxels available as supervision | 2,280,255 | 876,254 |
| revealability, *thing* classes | 30.72 % | 39.44 % |
| revealability, *stuff* classes | 27.14 % | 31.83 % |

**About two thirds of the hidden surface is never revealed at any horizon**, so for two thirds
of the population this supervision does not exist at all. Gate 20's own recorded outcome was
**B**, and it already attributed the ceiling to camera field of view rather than trajectory.

---

## 3. Measured evidence — (A) reliability, and the headroom it destroys

Where the supervision does exist it is genuinely better than the deployed baseline — on the
876,254 revealed dev voxels:

| on revealed voxels (dev) | top-1 | mIoU |
|---|---:|---:|
| B0 (nearest evidence, deployed) | 39.49 % | 6.39 |
| future-revealed privileged target | **53.33 %** | **16.88** |

That is +10.49 mIoU on that subset, which is why the idea looked promising. The decisive
measurement is what it is worth on the **whole** hidden population, and what perfect labels
in the same voxels would have been worth:

| ceiling on all P0 (dev, 2,743,212 voxels) | mIoU | gain over B0 |
|---|---:|---:|
| B0 | 8.796 | — |
| labels from the privileged target where revealed, else B0 | 12.958 | **+4.16** |
| **GT** where revealed, else B0 | 24.751 | **+15.96** |

**The privileged labels' own noise destroys 11.80 of the 15.96 points available in exactly
the voxels they cover — 73.9 % of the headroom, before any learning problem.** Thin coverage
costs the rest: even perfect labels on the revealed third cap out at +15.96, because the
other two thirds stay as B0.

Consistent teacher figures from a separate gate: Gate 18 measures the teacher at 62.57 %
top-1 / 16.91 mIoU / 33.24 % mean-class accuracy on dev, over the 4.12 % of evidence voxels
that have a GT label to score against. Phase 5 measures 56.2 % on SemanticKITTI. Different
populations and datasets; the brief's instruction not to generalise the Stage A audit's
12.3 % / 63.9 % applies to all of these equally, in both directions.

---

## 4. Measured evidence — (A) learnability: already tested, and it failed

This is the part that settles the question. Gate 21 trained on this supervision — three
seeds, four variants, dev drive 0006 held out, 12 leakage checks all passing, geometry
bit-identical across runs.

| dev population | B0 | best learned variant | Δ |
|---|---:|---:|---:|
| **E1** — revealed (where it was trained) | 6.39 | C3 6.44 ± 0.22 | **+0.05** |
| **E2** — never revealed (where it must generalise) | 10.18 | C3 6.19 ± 0.25 | **−3.99**, CI [−5.03, −3.13] |

All four variants on all of P0, against B0's 8.796: C1 prior **1.13 ± 0.72**, C2 geometry
**4.78 ± 0.03**, C3 full **6.28 ± 0.23**, C4 semantics-only **6.37 ± 0.07**. **Every variant
is below the baseline.**

Three details that close off the usual escapes:

* **It is not a near-miss.** The E2 deficit's 95 % CI is [−5.03, −3.13], clear of zero, with
  per-seed values −3.78, −4.34, −3.85.
* **It is worst where the voxels are.** The E2 deficit is largest in the 0–1 m bin
  (−6.29 pp over 851,732 voxels) and smallest at 2–4 m (−0.43 pp) — it fails hardest on the
  near voxels that dominate the population (Phase 5 §3).
* **It is worst on the classes that matter for mIoU.** *Thing* E2 delta −15.68 against *stuff*
  −4.31.

Gate 21's recorded conclusion — written under its own preregistered decision rule, before
this report — was *"future supervision does not generalise from revealed to never-revealed
hidden voxels"*, recommended next step *"close this branch; no automatic rescue by size,
epochs, data or a transformer"*.

### Why it fails: the supervision is a biased slice

From `selection_bias.json`, class composition of the two populations (train):

| | revealed (UR) | never revealed (UN) |
|---|---:|---:|
| vegetation | 41.3 % | 54.7 % |
| road | 23.8 % | 14.2 % |
| terrain | 15.2 % | — |

The revealed set over-represents road and under-represents vegetation relative to the
population the model must serve. A model fitted on it is fitted on the wrong mixture, which is
the mechanism behind "learned-vs-copied" rather than "revealed-vs-never".

---

## 5. (B) Real 3D semantic ground truth — a different objection

Source (A)'s ceiling is set by teacher noise (§3). Real 3D labels would remove that term:
the gap between +4.16 and +15.96 is precisely what correct labels in the revealed region
would recover. So (B) is not refuted by §3. It is refuted on two other grounds.

**The scientific objection, which is decisive.** Neither our system nor OccAny trains
semantics from labels; that is the central claim distinguishing both from supervised SSC
methods. Training on real 3D semantic annotations converts the method from label-free
open-vocabulary completion into supervised semantic completion. That is not an improvement to
the current contribution — it is a different contribution, competing in a different field
against methods built for it, and it forfeits the comparison the project is built on. The
brief forbids introducing *target-domain* annotations; Waymo is the source domain, so that
rule does not bar it, but the claim-level cost is independent of which domain the labels come
from and is larger than the rule.

**The data situation, measured.** Two separate facts, and the second is the stronger one.

*Our cache has no labels.* The per-frame cache under
`/media/minh/MINH1/lingbot_waymo_pilot/waymo_frames/` holds `points_vehicle`, `cp_points`,
per-camera images and poses, `calibration_json`, `vehicle_to_world`, `timing_json`,
`sync_error_json` — and **no semantic field of any kind**. The extractor deleted each 1 GiB
tfrecord after conversion.

*The source does have them.* Three segments from our own corpus were re-downloaded and
counted (`artifacts/waymo_arm/logs/waymo_semseg_probe.json`):

| segment | frames | with 3D semseg | with camera panoptic |
|---|---:|---:|---:|
| `...10017090168044687777_6380_000_6400_000` | 198 | 30 | 20 |
| `...10023947602400723454_1120_000_1140_000` | 199 | 20 | 20 |
| `...1005081002024129653_5313_150_5333_150` | 199 | 30 | 20 |
| **total** | **596** | **80 (13.4 %)** | **60 (10.1 %)** |

Only **laser id 1, TOP**, carries `segmentation_label_compressed`; the four side and rear
lasers carry none. So real 3D point semantics exist for roughly one frame in seven, on the
sensor that supplies 89.7 % of the points. Scaled to the corpus that is order 20,000 labelled
frames across 798 segments — an **extrapolation from three segments**, not a count, and the
per-segment spread (20, 30, 30) already shows it varies.

So (B) is not blocked by data availability, which is why the verdict in §5 rests on the
claim-level objection above rather than on this. Two costs remain, and both are real: a
corpus pass to extract the labels, measured at **~372 GiB downloaded over 638 minutes** by
the sidecar run; and Phase 1's T2 gate below.

A second precondition would also bind: Phase 1's T2 gate. `waymo.project` applies no lens
distortion, giving a p99 residual of 0.237 m against a 0.20 m one-voxel criterion. It was
scoped `semantic_target_only` precisely because it does not touch the occupancy path — but a
Waymo *semantic* target build is exactly the path it does touch. A verified
`_project_distorted` exists (p99 0.193 m, 99.3 % within a voxel) and would have to be adopted
first.

---

## 6. The size of the prize, for scale

Even granting (A) or (B) everything, Phase 5 bounds the reward. On SemanticKITTI, perfect
semantics on all correctly-predicted geometry reaches **23.88** mIoU; on Occ3D, **19.87**.
The remaining ~76 and ~80 points of the metric belong to occupancy. Within the semantic
share, perfecting the teacher where it already speaks is worth **+1.63 / +1.89**.

So the whole of the semantic programme — every variant of privileged supervision, better
teachers and better propagation combined — is competing for a fraction of ~20 mIoU points,
while occupancy holds four times as much.

---

## 7. Hypotheses

* **H-coverage-ceiling.** Hidden-region semantic coverage is bounded by sensor geometry, not
  by algorithm or data volume. Supported from two directions: Gate 20 attributes its ceiling
  to field of view rather than trajectory, and Gate 18's temporal ladder saturates — evidence
  voxels go 17,887 at 5 frames to 18,289 at 20 frames (+2.2 %) while T3 mIoU moves 8.796 to
  8.845 across four times the history. If true, no supervision scheme fixes coverage and only
  a different sensor arrangement would.
* **H-mixture.** The failure in §4 is a population-mixture problem, not a capacity problem.
  Supported by `selection_bias.json` and by C4 (semantics-only, fewest confounds) failing
  identically to C3. Testable by reweighting to the never-revealed mixture — cheap, but see
  §8 for why it is still not recommended.
* **H-label-noise.** With correct labels in the revealed region the ceiling rises from +4.16
  to +15.96 (§3). This is measured, not hypothesised; what is hypothesised is that a learner
  would then generalise, and §4 is direct evidence against that, since Gate 21's C3 failed to
  beat B0 even on E1 where the labels it trained on were available (+0.05).

---

## 8. Statistical uncertainty

* Gate 21's E2 deficit carries a **95 % CI of [−5.03, −3.13]** across three seeds. This is the
  one number in this report with a proper interval, and it is the one the verdict rests on.
* Gate 20's **ceilings carry no intervals** — they are pooled counts on one dev drive. The
  train/dev contrast (+2.89 vs +4.16) is the only uncertainty handle on them, and it is a
  two-point spread, not a confidence statement.
* **Coverage figures come from a single held-out drive** (590 anchors). Gate 20's per-drive
  table shows real between-drive variation in P0 (1.08 M to 4.41 M voxels), so ±several
  points on the 31.94 % is plausible and not quantified.
* Gate 18's saturation is a single ladder, one dev split, no resampling.
* Phase 5's oracles are pooled point estimates (see that report's §6).

**Failure to reject is not evidence of equivalence, and the converse holds too:** §4 does not
merely fail to find a benefit, it finds a **deficit with an interval clear of zero**. That is
a positive finding of harm on the population that matters, not an inconclusive result.

---

## 9. Confounding factors

* **One dev drive, one dataset.** Everything in §2–§4 is KITTI-360. Nothing here establishes
  what revealability or teacher accuracy would be on Waymo, and the brief's warning against
  generalising the Stage A numbers applies.
* **Gate 21 trained at step 3000** with a 257 k-parameter head. A larger model or longer
  schedule was not tried — but Gate 21's preregistered rule anticipated that escape and
  recorded "no automatic rescue by size, epochs, data or a transformer", and the E1 result
  (+0.05 where it *was* trained) is the reason: the model is not underfitting its own
  supervision, it is being asked to transfer across a mixture shift.
* **`visible`-level revealability with a 10 s horizon.** A longer horizon raises coverage
  slightly (the curves are near-flat past 4 s) and worsens registration, which the delayed
  pilot's G1 gate already failed once at the pooled +10-frame p90 of 0.204 m.
* **Comparisons are to B0, not B3.** B3 over B0 is +0.25 on SemanticKITTI with the CI
  containing zero (PAPER C), so the choice of baseline does not change any sign here.

---

## 10. Unsupported interpretations — explicitly not claimed

* **Not claimed:** that pseudo-label coverage equals pseudo-label correctness. §2 and §3 are
  reported separately and the 73.9 % headroom loss is the gap between them.
* **Not claimed:** that privileged supervision is useless in principle. On the revealed
  voxels it beats B0 by +10.49 mIoU (§3). The failure is in coverage, label noise and
  transfer — not in the premise.
* **Not claimed:** that real 3D labels (B) would fail to raise mIoU. They probably would. The
  objection in §5 is about what the method would then be claiming, plus two unmet
  preconditions.
* **Not claimed:** that Waymo lacks 3D semantic labels — it has them, on 13.4 % of frames in
  the three segments probed, TOP laser only. The corpus-wide count is an extrapolation from
  three segments and is labelled as one.
* **Not claimed:** any causal account of *why* the mixture shift breaks transfer beyond the
  correlation in `selection_bias.json`.
* **No new architecture is proposed.** Per the brief, no transformer, foundation-model
  pipeline, clustering method or semantic diffusion network is designed here, and none is
  justified by anything above.

---

## 11. Decision and next actions

**Q4 answered: no.** Privileged 3D semantic supervision of form (A) does not offer a credible
path beyond B0/B3. It reproduces the B3/B4 pattern the brief named as a stop condition: B3
over B0 +0.25 with CI ∋ 0; B4 at 8.22 against B0 8.80 and B3 9.38 (SOURCE FAIL); the delayed
pilot's B2 ≈ B1 at +0.08; Gate 17's oracle instance grouping +0.09 [−0.27, +0.34]; and now
Gate 21 at −3.99 [−5.03, −3.13]. Five attempts, no gain, one significant loss.

**Recommended: close this branch.** Do not spend compute on a learned semantic completion
mechanism trained on future-revealed labels.

What to do instead, in priority order:

1. **Report the negative result properly.** Gates 20, 21 and this analysis form a complete,
   preregistered account of why hidden-region semantic completion is capped in a monocular
   causal system. That is publishable as a finding, and
   `reports/occany_observed_vs_unobserved.md` already shows the geometric counterpart — our
   entire lead over OccAny is out-of-frustum.
2. **Move the effort to occupancy**, which Phase 5 shows owns ~76 of 100 metric points
   against the semantic share's ~24.
3. **If the semantic branch is revisited**, the only untested lever that §7 does not already
   argue against is changing the *observation geometry* (H-coverage-ceiling) — not the
   supervision. Gate 20 measured **−59.96 pp** of its revealability loss at the field-of-view
   step and recorded the conclusion *"Extending the horizon is not the lever; widening the
   field of view would be."* The concrete form, with the data already on disk, is in
   `artifacts/waymo_arm/final_summary.md` under Q5.
4. **Do not run** the mixture-reweighting test of H-mixture as a rescue attempt. It is cheap,
   but Gate 21's E1 result (+0.05 where the labels existed) says the model does not profit
   from this supervision even without a mixture shift, so a reweighted fit has nothing to
   recover.
