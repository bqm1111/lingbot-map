# ONCE-trained model versus OccAny — BLOCKED, not executed

**2026-10-20.** Phase D of the brief. **No comparison was made, because there is no
ONCE-trained model.** See `once_geometry_results.md` for the Phase C block and
`once_feasibility.md` for the audit.

This file exists so the stage is recorded as blocked rather than missing, and so the comparison
that *was* executed is not mistaken for the one that was not.

---

## 1. What Phase D asked for, and its status

| required row | status |
|---|---|
| KITTI-360-trained CompletionUNet | available (A0), measured |
| Waymo-trained CompletionUNet | available (A1 of the earlier brief), measured |
| **ONCE-trained CompletionUNet** | **does not exist** |
| Released OccAny checkpoint, matched five-frame setting | available [reproduced] |
| OccAny one-frame | available on SemanticKITTI [reproduced]; **no one-frame nuScenes run exists** |

The primary question — *"Does our method trained only on ONCE outperform the released OccAny
checkpoint?"* — **cannot be answered.** Nothing in this directory should be read as answering
it.

---

## 2. What was executed instead, and what it does and does not establish

Phase A ran in full: `one_frame_ablation.md`. It is a different experiment answering a
different question — how the *existing* model behaves when temporal input is removed — and it
is informative about the method-versus-data question only indirectly.

The three existing training-source arms, for reference (frozen zero-shot protocol, seed-mean,
five causal frames):

| training source | SemanticKITTI SC IoU | Occ3D SC IoU |
|---|---:|---:|
| KITTI-360 | **26.17** ± 0.32 | **31.22** ± 1.78 |
| Waymo | 23.14 ± 0.55 | 22.86 ± 0.22 |
| ONCE | **not measured** | **not measured** |
| OccAny released [reproduced] | 25.21 raw / 25.93 pooled (5-frame forward) | 20.66 raw / 23.55 pooled (5-frame forward) |

**KITTI-360 is not one of OccAny's five source datasets.** That is why the ONCE arm mattered:
it was the one authorized route to a result of the form "our method trained on a dataset OccAny
also uses". Without it, the subset claim rests on nothing, and the Waymo arm is the only
OccAny-source arm we have — and it **failed** (−3.03 and −8.36, both with CIs clear of zero).

So on present evidence, the honest position on the subset question is: **one OccAny-source
dataset has been tried and it did not transfer; a second could not be tried.** That is weaker
than where the brief hoped to be, and it should be stated that way rather than filled with the
KITTI-360 numbers, which do not bear on it.

---

## 3. What Phase A *does* contribute to the method-versus-data question

Two things, both independent of training source:

**The advantage is not temporal accumulation.** The accumulated map scored alone is 2.90–22.16
across benchmarks and conventions, against OccAny's 20.66–24.06. On Occ3D pooled, our
*one-frame* arm beats OccAny by as much as our five-frame arm does (+18.37 vs +18.01). So
whatever produces our result, it is the completion network, not the memory.

**The reconstruction stack is training-data independent by construction.** `map_only` IoU is
bit-identical between the KITTI-360 and Waymo arms (8.820500920980095 both), since the hard map
does not depend on the completion checkpoint. Every data-sensitive part of the system is the one
967 k-parameter network — which is also where the Waymo failure lives.

Neither establishes architectural causality. Both narrow where it would have to come from.

---

## 4. The comparison that would still be needed

Even with an ONCE-trained model in hand, Phase D would establish *"our method generalizes from
one OccAny source dataset"* — not *"the architecture is why"*. The confounds in
`matched_data_control_feasibility.md` §2 survive a dataset match: different supervision drawn
from the same bytes, different pretrained initialization, and three orders of magnitude of
trained capacity.

That report's recommendation stands unchanged: **do not retrain OccAny now.** It is a
16-GPU-shaped job on our 4 GPUs, its warm-start checkpoint is unpublished and unquantified, and
for DDAD our side of the matched pair cannot be built at all.

---

## 5. Not claimed

* No ONCE performance number, estimate, or projection appears anywhere in this directory.
* No claim that ONCE would have succeeded or failed.
* No substitution of the KITTI-360 result for the missing ONCE result.
* No claim that Phase A answers Phase D's question.

---

# Addendum — 2026-10-27

§1–§5 above are preserved as the October 20 historical record. One correction and one update.

**Correction to §1 and §2.** Those sections attribute the block to ONCE's 2 FPS release making
"the frozen sweep policy impossible to preserve either way". That reasoning was mine and was
wrong: the frozen window is 21 sweeps at 0.523 s spacing over 10.46 s, so ONCE's native 0.500 s
cadence matches the frozen *spacing* exactly. The cadence was never the obstacle.

**Correction to §2's arm table.** It places our arms' **frozen-evaluator** figures (KITTI-360
26.17 / 31.22, Waymo 23.14 / 22.86, on 163 and 1,182 anchors) beside OccAny's
**canonical-protocol** figures (25.21 / 20.66, on 801 and 4,219 anchors). Those columns are not
comparable, which is the rule `geometry_head_to_head.md` §1 states. The head-to-head on one
protocol is now in `waymo_arm_vs_occany.md` §3; its conclusions do not change this file's.

**Update.** Phase B has been executed on real ONCE data under the authorized 2.1 s policy and
returns **CONDITIONAL GO** (`once_supervision_validation.md`). Phase C is therefore still not
executed, now under brief §10 rather than the retracted cadence blocker.

**Phase D remains BLOCKED and unmeasured.** The required row —

| required row | status |
|---|---|
| ONCE-trained CompletionUNet | **still does not exist** |

— is unchanged. Nothing in this directory answers *"does our method trained only on ONCE
outperform the released OccAny checkpoint?"* No ONCE performance number, estimate or projection
appears anywhere, and §5's "Not claimed" list stands in full.

---

# Addendum 2 — 2026-10-27: Phase D EXECUTED on SemanticKITTI

§1–§5 and Addendum 1 above are preserved as historical records; both described Phase D as
blocked for want of an ONCE-trained model. **That model now exists** (`once_training_report.md`)
and the comparison has been made. This addendum reports SemanticKITTI; Occ3D follows in
Addendum 3.

**Status: exploratory.** The predeclared occupied-prevalence gate is failed, not passed
(0.0131 against [0.015, 0.040]). It is recorded as failed everywhere this arm appears.

## 1. Protocol

OccAny's own anchors (**801** SemanticKITTI), their stored labels, their placement and their
scorer — the only protocol on which our arms and OccAny are comparable. Scored by the same code
that produced the existing OccAny contrasts
(`tools/method_vs_data/once_arm/cli/score_vs_occany.py`, logic from `one_frame_score.py`).
Paired block bootstrap: 2,000 resamples, seed 0, 33 blocks of 25 contiguous anchors, all arms
resampled on the same anchors. Raw and pooled reported separately and never mixed.

**Validity check:** our KITTI-360 arm scores **26.40** raw here, which is the stored canonical
figure to the digit, and OccAny's rows reproduce 20.32 / 25.21 raw and 24.06 / 25.93 pooled.
The harness is the established one.

## 2. The headline: ONCE-only does NOT outperform OccAny on SemanticKITTI

| arm | raw SC IoU | pooled SC IoU | raw P / R |
|---|---:|---:|---|
| ours, KITTI-360 | **26.40** | 26.34 | 37.45 / 47.22 |
| **ours, ONCE** | **18.20** | **20.26** | 31.14 / 30.45 |
| OccAny 1-frame causal [reproduced] | 20.32 | 24.06 | 55.64 / 24.25 |
| OccAny 5-frame **forward** [reproduced] | 25.21 | 25.93 | 45.64 / 36.03 |

| contrast | raw | pooled |
|---|---|---|
| ONCE − OccAny 1-frame causal | **−2.12** [−3.06, −1.13] * | **−3.80** [−4.69, −2.98] * |
| ONCE − OccAny 5-frame forward | **−7.01** [−7.97, −6.04] * | **−5.67** [−6.68, −4.67] * |
| ONCE − KITTI-360, same recipe | **−8.20** [−9.02, −7.32] * | **−6.08** [−6.96, −5.25] * |

`*` = 95 % CI excludes zero. **The answer is no, in both conventions, against both of their
modes, with intervals clear of zero.** It loses even to their one-frame causal run, which sees
four fewer frames than we do.

## 3. Where it loses and where it does not — the structure survives

Raw convention; `*` = CI excludes zero.

| region | share of GT occ | k360 | **ONCE** | OccAny 1f | ONCE − OccAny 1f |
|---|---:|---:|---:|---:|---|
| all | 100 % | 26.40 | 18.20 | 20.32 | −2.12 [−3.06, −1.13] * |
| anchor_observed | 4.8 % | 11.47 | 7.51 | 15.76 | **−8.25** [−9.38, −7.19] * |
| earlier_only | 6.2 % | 17.95 | 11.48 | 12.97 | −1.49 [−2.60, −0.40] * |
| **never_observed** | **89.1 %** | 30.07 | 22.17 | 21.53 | **+0.65** [−0.53, +1.89] **ns** |
| in_frustum | 74.5 % | 25.16 | 16.72 | 24.16 | **−7.45** [−8.51, −6.28] * |
| **out_frustum** | 25.5 % | 29.98 | 21.55 | **7.43** | **+14.13** [+13.03, +15.22] * |

Pooled, the same two cells: never_observed **−0.04** [−1.12, +1.08] **ns**; out_frustum
**+8.38** [+7.19, +9.54] *. Against their 5-frame forward run out_frustum is still
**+13.51** raw / **+8.64** pooled, both significant.

**Read it as: ONCE-only training destroys accuracy on surfaces the camera actually sees and
preserves completion into space it never saw.** The out-of-frustum advantage — ours ~21.6
against their ~7.4-8.0, a factor of roughly three — survives the corpus change intact in both
conventions. Never-observed space, 89.1 % of ground-truth occupancy, is a **statistical tie**
with their causal mode in both conventions. The aggregate still goes to OccAny because
in-frustum voxels are 69.3 % of the valid mask, and that is exactly where this arm collapses
(−7.45 raw).

Against KITTI-360 under the identical recipe the ONCE arm loses in **every** region, −3.96 to
−8.45 raw, with no region spared. That is the cleanest available statement that this is a
source-data effect concentrated in one 967 k-parameter network — the hard map is bit-identical
across arms (`map_only` 8.8205).

## 4. What this does and does not license

**Does.** The brief's primary question is answered for SemanticKITTI: **no**, our method trained
exclusively on ONCE does not outperform the released OccAny checkpoint. The claim that our
geometric completion advantage is robust to source-dataset choice is **not supported** by this
arm.

**Does not.** It does not show our architecture is inferior. The recipe is 6,000 steps and
967 k trained parameters from scratch against OccAny's ≥70,300 steps from a pretrained MUSt3R
plus an unrecorded warm start, and the ONCE arm additionally carries a measured ~34 %
supervision-density deficit, a 1.35× monocular-scale bias against KITTI-360's 1.20×, and six
independent scenes. What it shows is that **our present training recipe is sensitive to source
selection** — which is the conclusion the brief anticipated for this outcome.

**Not claimed:** no causal attribution of the deficit to any single one of those differences.
The intervention that would separate them has not been run.

---

# Addendum 3 — 2026-10-27: Phase D on Occ3D-nuScenes

Same protocol as Addendum 2: OccAny's own anchors (**4,219**), their stored labels, placement
and scorer, paired block bootstrap with one block per scene (150 blocks), 2,000 resamples,
seed 0. All 4,219 anchors have both an ONCE prediction and an OccAny output; none was dropped.

**Validity check:** our KITTI-360 arm scores **31.94** raw here, the stored canonical figure to
the digit, and OccAny reproduces 20.66 raw.

Occ3D has no strict-visibility split in this harness (that mask is built for SemanticKITTI's
pose chain), so the frustum and map-evidence splits are reported instead, as brief §8 allows.

## 1. Headline — and on Occ3D the answer depends on the convention

| arm | raw SC IoU | pooled SC IoU | raw P / R |
|---|---:|---:|---|
| ours, KITTI-360 | **31.94** | **41.56** | 77.87 / 35.13 |
| **ours, ONCE** | **18.67** | **30.52** | 52.66 / 22.44 |
| OccAny 5-frame **forward** [reproduced] | 20.66 | **23.55** | 42.96 / 28.47 |

OccAny's pooled 23.55 is their **published** Occ3D figure (0.2355) reproduced to five digits, so
the pooled column is the one their paper reports in.

| contrast | raw | pooled |
|---|---|---|
| ONCE − OccAny 5-frame forward | **−1.99** [−3.94, −0.08] * | **+6.98** [+5.57, +8.42] * |
| ONCE − KITTI-360, same recipe | **−13.27** [−14.61, −11.94] * | **−11.04** [−12.13, −9.90] * |

**The sign flips.** Raw, the ONCE arm loses narrowly — the interval's upper bound is −0.08, only
just significant, against −7.01 for the same contrast on SemanticKITTI. Pooled, it **wins by
+6.98 with the interval clear of zero.**

Why: the 3×3×3 max-pool is worth **+11.85** to the ONCE arm, **+9.62** to the KITTI-360 arm and
only **+2.89** to OccAny. Dilation rewards sparse predictions, and the ONCE arm is the sparsest
of the three (raw recall 22.44 against OccAny's 28.47). This is the same asymmetry recorded in
`geometry_head_to_head.md` §1, acting more strongly here because this arm predicts less.

**Neither column is the "right" one and both are reported.** The honest statement is that on
Occ3D the comparison is **convention-dependent**: ONCE-only loses narrowly under raw scoring and
wins substantially under the convention OccAny publishes in.

## 2. By region — the same structure as SemanticKITTI

| region | share of GT occ | k360 | **ONCE** | OccAny 5f | ONCE − OccAny raw | ONCE − OccAny pooled |
|---|---:|---:|---:|---:|---|---|
| all | 100 % | 31.94 | 18.67 | 20.66 | −1.99 [−3.94, −0.08] * | **+6.98** [+5.57, +8.42] * |
| in_frustum | 45.5 % | 36.18 | 17.64 | 29.98 | **−12.34** [−15.19, −9.42] * | +1.16 [−0.96, +3.24] ns |
| **out_frustum** | **54.5 %** | 28.25 | **19.54** | **8.94** | **+10.60** [+8.81, +12.27] * | **+16.82** [+15.28, +18.36] * |
| map-observed | 32.9 % | 41.82 | 24.93 | 28.05 | −3.12 [−5.67, −0.56] * | +1.24 [−0.65, +3.20] ns |
| map-completed | 67.1 % | 27.36 | 14.54 | 17.75 | −3.21 [−5.26, −1.17] * | **+7.04** [+5.28, +8.75] * |

**The out-of-frustum advantage replicates on the second benchmark: +10.60 against SemanticKITTI's
+14.13.** Ours ~19.5 against their ~8.9 — again roughly a factor of two to three. And again the
collapse is in-frustum, −12.34. Occ3D's frustum split is nearly even (51 % / 49 % of valid), so
the two effects very nearly cancel in the aggregate, which is why the total loss is only −1.99
here against −7.01 on SemanticKITTI, where in-frustum carries 69.3 % of the valid mask.

Against KITTI-360 under the identical recipe the ONCE arm again loses in **every** region,
−8.71 to −18.54, with the largest loss in frustum.

## 3. Answer to the brief's primary question

> *Does our method trained exclusively on ONCE outperform the released OccAny checkpoint under
> matched inference and evaluation conditions?*

**Reported independently per benchmark, as required — and the two benchmarks differ.**

| benchmark | convention | ONCE vs OccAny | verdict |
|---|---|---|---|
| SemanticKITTI | raw, vs their **causal 1-frame** | −2.12 [−3.06, −1.13] * | **loses** |
| SemanticKITTI | raw, vs their forward 5-frame | −7.01 [−7.97, −6.04] * | **loses** |
| SemanticKITTI | pooled, vs their causal 1-frame | −3.80 [−4.69, −2.98] * | **loses** |
| SemanticKITTI | pooled, vs their forward 5-frame | −5.67 [−6.68, −4.67] * | **loses** |
| Occ3D | raw, vs their forward 5-frame | −1.99 [−3.94, −0.08] * | **loses, narrowly** |
| Occ3D | **pooled**, vs their forward 5-frame | **+6.98 [+5.57, +8.42] *** | **WINS** |

So: **no on SemanticKITTI in both conventions; on Occ3D it depends on the convention** — a
narrow loss raw, a clear win under the pooled convention OccAny's own paper reports. Five of
six cells favour OccAny, and the one that favours us is the one scored the way they publish.

**The claim this arm was built to test — that our completion advantage is robust to
source-dataset choice — is not supported.** A result that is negative on one benchmark and
convention-dependent on the other, from an arm that loses 11-13 points to the same recipe
trained on KITTI-360, does not establish robustness.

The brief's instruction that a method-level claim must not be declared when the aggregate
improves while strict never-observed deteriorates does not bind here, because **the aggregate
did not improve**. The converse pattern is what was measured: the aggregate fell while the
unobserved-space advantage held.
