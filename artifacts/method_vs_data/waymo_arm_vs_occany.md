# Our method trained on an OccAny source dataset, against OccAny

**2026-10-27.** This file answers one question: *when our CompletionUNet is trained on a
dataset OccAny also trains on, how does it compare to the released OccAny checkpoint?*

It exists because that comparison was **not written down anywhere** before today, although
both halves of it had been measured on 2026-10-20. The halves sat in
`geometry_head_to_head.md` (OccAny's rows, and ours trained on KITTI-360) and
`reports/waymo_semantic_impact.md` §1 (ours trained on Waymo, on the same anchors). No new
training, inference or scoring was run to produce this file; every number is read from a
stored JSON and cited to it.

**Supersedes `once_vs_occany.md` §2's arm table**, which mixed protocols — see §5.

---

## 1. Why Waymo is the relevant arm

OccAny's released checkpoint trains on five sources, from its own stored `args.train_dataset`
(`occany_training_audit.md` §1.2): **Waymo, DDAD, PandaSet, ONCE, VKITTI2**. Neither
SemanticKITTI nor KITTI-360 is among them, so both evaluation benchmarks are zero-shot for
them as well as for us.

Of those five, exactly one has been trained on by us:

| OccAny source | our arm | status |
|---|---|---|
| **Waymo** | **A1, 3 seeds, 6,000 steps** | **trained and measured** |
| DDAD | A2 | NO-GO — aggregated 4-sensor cloud, sensor origin unrecoverable (`ddad_feasibility.md`) |
| ONCE | A4 | gates 6/6 pass except occupied prevalence, 0.0133 vs the 0.015 predeclared floor (`once_feasibility.md` §16) |
| PandaSet | — | unaudited beyond its per-point sensor ID |
| VKITTI2 | — | synthetic; not considered |

So this report rests on **one OccAny-source dataset out of five**. That is the limit of what
can be said, and §6 says it as such.

## 2. The protocol all four rows share

Canonical zero-shot prediction pipeline, which fixes `GEOMETRY_SEED = 0`: **801 SemanticKITTI
and 4,219 Occ3D anchors**, OccAny's own stored labels, placement and scorer, OccAny's valid
mask. **Raw** convention (no pooling) throughout this section.

This is the only protocol on which the Waymo arm and OccAny have both been scored. It is *not*
the frozen `baseline_reconstruction` evaluator used in `waymo_vs_kitti360_report.md` §2–§3
(163 / 1,182 anchors, own masks, three seeds); those numbers are not interchangeable with
these. The two protocols agree on sign and roughly on size (Δ −2.84 / −9.09 here against
−3.03 / −8.36 there), which is the consistency check that makes either believable.

## 3. Measured

Sources: `artifacts/waymo_arm/results/semantic_impact_B0.json`
(`/results/*/populations/full/`), `artifacts/occany_comparison/analysis.json` via
`geometry_head_to_head.md` §2–§3.

### SemanticKITTI sequence 08 — 801 anchors, raw SC IoU

| row | window | SC IoU |
|---|---|---:|
| ours, trained on KITTI-360 | causal 5 frames | **26.40** |
| **ours, trained on Waymo** | **causal 5 frames** | **23.55** |
| OccAny released, 1-frame | causal 1 frame | 20.32 |
| OccAny released, 5-frame | **forward**, t…t+40 | 25.21 |

### Occ3D-nuScenes val — 4,219 anchors, raw SC IoU

| row | window | SC IoU |
|---|---|---:|
| ours, trained on KITTI-360 | causal 5 frames | **31.94** |
| **ours, trained on Waymo** | **causal 5 frames** | **22.85** |
| OccAny released, 5-frame | **forward** | 20.66 |
| OccAny released, 1-frame | — | **no nuScenes 1-frame run exists** |

### The one difference with a confidence interval

Paired block bootstrap, 2,000 resamples, seed 0; 33 blocks of 25 contiguous anchors on
SemanticKITTI, one block per scene (150) on Occ3D; both arms resampled on the same anchors.

| benchmark | Δ SC IoU, Waymo-trained − KITTI-360-trained | significant |
|---|---:|---|
| SemanticKITTI | **−2.84** [−3.41, −2.26] | yes |
| Occ3D | **−9.09** [−10.19, −8.00] | yes |

The hard map scored alone is **bit-identical** between the two arms (8.820500920980095), so
the entire difference is the 967 k-parameter completion network and nothing else.

## 4. Derived — the comparison itself

These are **differences of two measured means on identical anchors**, not quantities that were
bootstrapped. No confidence interval was computed for any row in this table, and none is
estimated here.

| benchmark | comparison | Δ |
|---|---|---:|
| SemanticKITTI | ours Waymo-trained − OccAny **1-frame causal** | **+3.23** |
| SemanticKITTI | ours Waymo-trained − OccAny **5-frame forward** | **−1.66** |
| Occ3D | ours Waymo-trained − OccAny **5-frame forward** | **+2.19** |

For scale, the same comparisons for our KITTI-360-trained arm are +6.08, +1.19 and +11.28.

Read plainly: trained on one of OccAny's own source datasets, our completion network **beats
their only genuinely causal mode on SemanticKITTI, loses to their strongest released mode on
SemanticKITTI, and stays ahead on Occ3D by a margin a fifth the size of the KITTI-360 arm's.**

## 5. The two gaps in this answer

**(a) The pooled convention was never computed for the Waymo arm.** This is the more
consequential gap, because **OccAny's published numbers are the pooled ones** — verified to
five digits (published 0.2591 against our 0.25916; published 0.2355 against our 0.23555,
`geometry_head_to_head.md` §1). Pooling is a 3×3×3 max-pool and is not symmetric:

| | raw → pooled |
|---|---|
| OccAny 5-frame forward, SemanticKITTI | +0.72 |
| OccAny 1-frame, SemanticKITTI | +3.74 |
| OccAny 5-frame forward, Occ3D | +2.89 |
| ours KITTI-360-trained, SemanticKITTI | **−0.43** |
| ours KITTI-360-trained, Occ3D | **+9.25** |
| **ours Waymo-trained, either** | **not measured** |

The Waymo arm's gain cannot be assumed equal to the KITTI-360 arm's, and the direction of the
inequality is predictable: the Waymo arm is **sparser** in every cell of
`frustum_split.json` — SemanticKITTI in-frustum recall 39.13 against 44.90, Occ3D in-frustum
26.86 against 40.68 — and dilation rewards sparse predictions (hence OccAny's own 1-frame
+3.74). So the Waymo arm would likely gain *more* from pooling than the KITTI-360 arm did,
which could move §4's SemanticKITTI rows in either direction. **They are left unstated rather
than estimated.**

**(b) No paired bootstrap exists for ours-Waymo against OccAny.** The intervals in §3 are on
the Waymo-minus-KITTI-360 difference. §4's rows are point estimates.

**Both gaps are closable without training.** All 801 and 4,219 Waymo-arm predictions are
intact at `/media/minh/MINH1/lingbot_waymo_arm/zero_shot_predictions`. Closing them needs a
scorer that applies `tools/occany_comparison/scoring.py:pool` to those stored volumes and runs
the existing block bootstrap — the Waymo arm's own scorer
(`tools/waymo_arm/cli/score_semantic.py`) deliberately applies no pooling, by the frozen
protocol's convention, so this is new code rather than a flag. Not written, not run.

## 6. What this does and does not support

**Supported.** Trained on one of OccAny's five source datasets, with 6,000 optimizer steps,
967 k trained parameters and from scratch, our completion network remains ahead of the released
OccAny checkpoint on Occ3D (+2.19 raw) and ahead of its causal mode on SemanticKITTI (+3.23
raw), while consuming only past frames where their 5-frame run consumes future ones.

**Not supported — a methodological advantage.** The direction of this result cuts against it.
Holding architecture, loss, schedule and step budget fixed and changing only the training
corpus moved the metric by **2.8 and 9.1 points with intervals clear of zero**. We therefore
demonstrated our own method's strong data-sensitivity on our own system, and any claim of the
form "the architecture is why we win" has to survive that. It currently does not.

**Not supported — generalization across OccAny's pool.** One source tried, one substantial
degradation. Four untried, two of them blocked. The honest statement remains the one in
`once_vs_occany.md` §2: *one OccAny-source dataset has been tried and it did not transfer at
the KITTI-360 arm's level.*

**Not claimed.** No pooled Waymo-arm number. No confidence interval on any §4 row. No
projection of what ONCE or PandaSet would give. No substitution of the KITTI-360 arm's figures
for a missing one.

## 7. Correction to an existing file

`once_vs_occany.md` §2 tabulates the three training-source arms against OccAny using the
**frozen evaluator's** figures for our arms (KITTI-360 26.17 ± 0.32 / 31.22 ± 1.78, Waymo
23.14 ± 0.55 / 22.86 ± 0.22, on 163 and 1,182 anchors) beside **OccAny's canonical-protocol**
figures (25.21 / 20.66, on 801 and 4,219 anchors). Those columns are not comparable, which is
the rule `geometry_head_to_head.md` §1 states. The discrepancy is small — 26.17 against 26.40,
22.86 against 22.85 — so the file's *conclusions* are unaffected, but its table should not be
read as a head-to-head. §3 above is the head-to-head. That file is left otherwise unmodified.
