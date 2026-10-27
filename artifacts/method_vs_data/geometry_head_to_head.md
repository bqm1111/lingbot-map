# Geometry head-to-head — our method against OccAny and our own baselines

**Written 2026-10-20.** Sources: `artifacts/occany_comparison/analysis.json` (801 SemanticKITTI
and 4,219 Occ3D anchors, OccAny's own stored labels, placement and scorer),
`artifacts/waymo_arm/results/waymo_vs_kitti360.json`. Protocol fixed in
`fair_evaluation_protocol.md`. No new training; the DDAD arm is blocked
(`ddad_feasibility.md`).

Every row is tagged. Track **O** throughout (every `label != 255` voxel — OccAny's convention).

---

## 1. The convention that must be stated first

**OccAny's published numbers are the pooled (dilated) ones.** Verified: their published
SemanticKITTI SC IoU is 0.2591 and our scorer gives **0.25916 pooled** against 0.25205 raw;
their published Occ3D figure is 0.2355 and our scorer gives **0.23555 pooled** against 0.20666
raw. The match is exact to five digits on both benchmarks.

So a raw number of ours against a published number of theirs is a raw-against-pooled
comparison and is inadmissible. Both conventions are given for every row below.

Pooling is **not** neutral and not symmetric: on Occ3D it adds **+9.25** to ours and **+2.89**
to theirs; on SemanticKITTI it adds **−0.43** to ours and **+0.72** to theirs. Our predictions
are already dense, so dilation barely helps us and helps them substantially.

---

## 2. SemanticKITTI sequence 08 — 801 shared anchors, 33 blocks

| row | window | raw SC IoU | pooled SC IoU | raw P / R | tag |
|---|---|---:|---:|---|---|
| **hard map alone** | causal 5 frames | **8.79** | 15.04 | 32.75 / 10.72 | [ours] ablation |
| **ours** | **causal 5 frames** | **26.32** ± 0.30 | 25.89 ± 0.39 | 36.40 / 48.79 | [ours] primary |
| OccAny 1-frame | causal 1 frame | 20.32 | 24.06 | 55.64 / 24.25 | [reproduced] |
| OccAny 5-frame | **forward**, t…t+40 | 25.21 | 25.93 | 45.64 / 36.03 | [reproduced] |
| ours | forward 5 frames | 29.82 ± 0.40 | 28.66 ± 0.35 | 39.94 / 54.07 | [ours] **not our system** |

Our per-seed causal raw: 26.40 / 26.57 / 25.99.

## 3. Occ3D-nuScenes validation — 4,219 shared anchors, 150 scene blocks

| row | window | raw SC IoU | pooled SC IoU | raw P / R | tag |
|---|---|---:|---:|---|---|
| **hard map alone** | causal 5 frames | **9.82** | 22.19 | 60.15 / 10.51 | [ours] ablation |
| **ours** | **causal 5 frames** | **31.63** ± 1.74 | 40.88 ± 1.00 | 75.65 / 35.20 | [ours] primary |
| OccAny 5-frame | **forward** | 20.66 | 23.55 | 42.96 / 28.47 | [reproduced] |
| ours | forward 5 frames | 24.95 ± 1.45 | 40.05 ± 0.98 | 79.14 / 26.70 | [ours] **not our system** |

Our per-seed causal raw: 31.93 / 33.20 / 29.75. Occ3D is the frozen single-front-camera
protocol, so it is not comparable with OccAny's published surround numbers.

---

## 4. The comparison, stated carefully

Against OccAny's **forward** 5-frame run (their strongest released geometry, consuming future
frames we do not use):

| benchmark | ours causal − OccAny forward, raw | pooled |
|---|---:|---:|
| SemanticKITTI | **+1.11** [+0.30, +1.93] | **−0.04** [−0.77, +0.70] |
| Occ3D | **+10.98** [+9.01, +12.97] | **+17.36** [+16.04, +18.66] |

Paired block bootstrap, 2,000 resamples, seed 0.

**SemanticKITTI is a tie, Occ3D is a clear win.** The pooled SemanticKITTI interval contains
zero, and a tie is what should be reported — not a win, and not equivalence either.

Against OccAny's **1-frame causal** run (the only genuinely causal comparator, and one with
four fewer frames than we get): SemanticKITTI +6.08 raw [+4.82, +7.33], +2.28 pooled
[+1.23, +3.33]. That asymmetry favours us and is stated wherever the number is used; their
forward window is worth only +1.9 SC IoU to them, which bounds what a causal 5-frame OccAny
mode would plausibly add.

**Seed conventions differ between the two delta sets and are not interchangeable.** The
vs-forward deltas are **seed means** over seeds 0–2 (`occany_comparison`). The vs-1-frame
deltas are **seed 0 only**, because they come from the canonical prediction pipeline, which
fixes `GEOMETRY_SEED = 0`; our seed-0 raw SemanticKITTI figure is 26.40 against the 26.32 seed
mean. The gap is 0.08, well inside the ±0.30 seed spread, but the two sets should not be put
in one column.

---

## 5. The ablation that answers the strongest objection

The natural objection to our result is that we win by *accumulating* five frames of map rather
than by completing anything. The hard map scored on its own refutes that on its own terms:

| benchmark | hard map alone (causal) | + CompletionUNet | completion contributes | OccAny best released |
|---|---:|---:|---:|---:|
| SemanticKITTI | **8.79** | **26.32** | **+17.53** (67 % of the total) | 25.21 |
| Occ3D | **9.82** | **31.63** | **+21.81** (69 % of the total) | 20.66 |

Raw convention, identical anchors, identical scorer.

**Accumulated reconstruction alone scores 8.79 on SemanticKITTI — 16.4 points below OccAny's
25.21.** Memory is not what closes the gap; roughly two thirds of our score is produced by the
learned completion network, on both benchmarks.

Validity check on this ablation: `map_only` IoU is **bit-identical** between the KITTI-360 and
Waymo training arms (8.820500920980095 on the 163-anchor frozen evaluator, both arms), exactly
as it must be since the hard map does not depend on the completion checkpoint. A second,
independent evaluator on 801 anchors gives 8.79 — consistent.

**What this does not settle:** where the *marginal advantage over OccAny* sits. "Two thirds of
our absolute score comes from completion" and "our lead over OccAny sits in region X" are
different claims from different measurements. The second is
`observed_vs_completed_analysis.md`.

---

## 6. Where we are behind

Reported because the brief asks for measured evidence, not a favourable selection.

* **Semantics.** Ours 4.45 SSC mIoU against OccAny's 7.28 on SemanticKITTI forward, and 4.09
  against 6.66 on Occ3D surround. The geometry result does not extend to semantics.
* **Surround geometry.** OccAny's 6-camera surround geometry reaches 34.15 against our 32.62.
  Our advantage is **single-camera only**.
* **Precision.** On SemanticKITTI our causal precision is 36.40 against their 45.64; we win on
  recall (48.79 against 36.03). We predict more and are right less often per prediction.
* **Observed surfaces.** On voxels the anchor camera actually sees, OccAny is better — the
  earlier observability study measured −2.70 pooled SC IoU for us on the observed region.

---

## 7. What these numbers do and do not support

**Supported — benchmark superiority, with scope.** Under a matched single-camera causal
protocol, our system beats the released OccAny checkpoint on Occ3D (+10.98 raw, CI clear of
zero) and ties it on SemanticKITTI (−0.04 pooled, CI containing zero), while consuming only
past frames where their strongest released run consumes future ones.

**Supported — task-specific training efficiency.** 6,000 optimizer steps and 24,000 sample
draws with 967 k trained parameters, against ≥70,300 steps and 4,500,000 draws from a
pretrained MUSt3R plus an unrecorded warm start. Both sides verified
(`occany_training_audit.md`). The qualifier "task-specific" is load-bearing: our pipeline
carries frozen LingBotMap and MoGe-2 dependencies that this count excludes.

**Not supported — a methodological advantage.** Every row above comes from systems trained on
different corpora with different supervision and different pretraining. Nothing here
attributes the difference to the reconstruction mechanism. §5 narrows it usefully — the
advantage is not mere accumulation — but narrowing is not attribution.

**Not supported — fewer *datasets* implies more data-efficient.** We verified OccAny's sample
draws per epoch, not its unique scenes or frames; those live in sequence pickles not present
locally. The efficiency claim must rest on steps and draws, which are verified, not on dataset
counts, which are not.

**Not established — that a single OccAny-source dataset suffices for us.** That was A2's job
and A2 is blocked. KITTI-360 is not one of OccAny's five sources, so the strong rows above
cannot settle the subset question the brief poses.
