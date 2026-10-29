# The Waymo arm under OccAny's canonical protocol — both conventions, with intervals

**2026-10-27.** This closes the two gaps `waymo_arm_vs_occany.md` §5 recorded against itself:
the Waymo arm had never been scored under OccAny's **pooled** convention — the one OccAny
publishes in — and no row of its OccAny comparison carried a **confidence interval**. Both are
now measured. No training was launched, no prediction was regenerated, no dataset was
downloaded, no threshold was changed.

Code: `tools/method_vs_data/canonical_rescore.py` (and `rescore_tables.py`,
`region_crosstab.py`). Raw: `artifacts/method_vs_data/waymo_canonical_rescore/`.

> ## What changed
>
> Three of the four statements `waymo_arm_vs_occany.md` §4 made about the Waymo arm do not
> survive the full protocol. The arm's raw SemanticKITTI win over OccAny's causal mode becomes
> a **tie** under their published convention; its raw Occ3D lead becomes a **tie** once the
> interval is computed; and its pooled Occ3D lead, never previously measured, is a **large
> significant win**.
>
> | comparison | previously | now |
> |---|---|---|
> | SK, vs OccAny causal 1-frame, **raw** | +3.23, no CI | **+3.23 [+2.26, +4.21] \*** win |
> | SK, vs OccAny causal 1-frame, **pooled** | **not measured** | **−0.69 [−1.52, +0.10] ns** — tie |
> | SK, vs OccAny forward 5-frame, raw | −1.66, no CI | **−1.66 [−2.52, −0.79] \*** loss |
> | SK, vs OccAny forward 5-frame, **pooled** | **not measured** | **−2.55 [−3.52, −1.65] \*** loss |
> | Occ3D, vs OccAny forward 5-frame, raw | +2.19 "stays ahead" | **+2.20 [−0.02, +4.35] ns** — **tie** |
> | Occ3D, vs OccAny forward 5-frame, **pooled** | **not measured** | **+14.51 [+12.92, +16.06] \*** win |
>
> And the result that does survive, in both conventions, is the one about unobserved space:
> on the **strict never-observed** mask the Waymo arm beats OccAny's causal mode by
> **+7.02 [+5.66, +8.37]** raw and **+4.04 [+2.77, +5.29]** pooled. That cell is a tie for the
> ONCE arm; it is a significant win here.

---

## 1. What was scored, and the seed limitation

**Population.** OccAny's own anchors: **801** SemanticKITTI sequence-08 and **4,219**
Occ3D-nuScenes validation. Verified on disk: `/media/minh/MINH1/lingbot_waymo_arm/
zero_shot_predictions` holds exactly 801 + 4,219 = **5,020** volumes, as
`waymo_arm_vs_occany.md` §5 claimed. **Zero missing, zero anchors dropped**, for all three of
our arms (`n_missing_predictions` = 0 for `k360`, `waymo`, `once` in both datasets).

**Seed coverage — the one analysis that could not be completed as asked.** Canonical
predictions exist for **seed 0 only**, for every one of our arms. The prediction pipeline fixes
`GEOMETRY_SEED = 0` (`tools/zero_shot_eval/config.py:28`), and both arms' prediction scripts
substitute their seed-0 checkpoint "one for one" for the KITTI-360 seed-0 checkpoint. Seeds 1
and 2 are trained and their checkpoints are on disk, but no canonical prediction volume exists
for them, and regenerating them is outside this task.

So **every number in this report is a single-seed number**, including the KITTI-360 reference
(26.40 / 31.94 raw, which is the seed-0 row, not the three-seed mean of 26.32 / 31.63). No
three-seed canonical interval is reported, estimated or implied. The three-seed spread that
*does* exist was measured by the frozen evaluator on its own smaller populations — SemanticKITTI
±0.55 and Occ3D ±0.22 for the Waymo arm, on 163 and 1,182 anchors — and **those numbers are not
interchangeable with these**, per the rule in `geometry_head_to_head.md` §1. They are quoted in
§7 and never placed in a column with a canonical figure.

**Evaluation configuration.** OccAny's stored `voxel_label` as ground truth, their placement,
their valid mask (`label != 255`), Track O occupancy (`label != empty_class`), grid
(256, 256, 32) at 0.2 m in the anchor's velodyne frame. Paired block bootstrap: **2,000
resamples, seed 0**; 33 blocks of 25 contiguous anchors on SemanticKITTI, one block per scene
(150) on Occ3D. **Every arm is resampled on the same anchors in the same pass**, so all pairs
are paired by construction. Global IoU inside the bootstrap is recomputed from **summed
TP/FP/FN**, never averaged over per-anchor IoUs.

**Provenance.** Code revision `db465c2ce2a692817d9ce4a141df7c24c4fa2d13`, branch
`feature-completion-study`. Working tree carries one unrelated modification,
`M lingbot_map/heads/dpt_head.py`, plus this task's new files; nothing historical was touched.
Checkpoints, seed 0: KITTI-360 `7f5e853dd05b…`, Waymo `e40fb93467e4…`, ONCE `03fc0d26fc83…` —
each matching the sha256 in its own training record.

## 2. The scorer is the existing one, and it reproduces the references

`canonical_rescore.py` is `once_arm/cli/score_vs_occany.py` with one change: the "ours"
prediction root is a parameter instead of a constant, so all arms score in one pass. The
scoring function, region definitions, the `PL.geometry_dilation` call, the block construction
and the bootstrap are the same lines. **The pooling operation and its order are preserved
exactly**: the 3×3×3 max-pool is applied to the **prediction only**, after it is loaded and
before any confusion count; ground truth, the valid mask and every region mask are left
undilated, and no region mask is pooled.

Verification against the stored references, all four required cells plus five more:

| quantity | stored | this run | Δ |
|---|---:|---:|---:|
| OccAny 1-frame causal, SK **raw** | 20.32 | **20.32** | 0 |
| OccAny 1-frame causal, SK **pooled** | 24.06 | **24.06** | 0 |
| OccAny 5-frame forward, SK **raw** | 25.21 | **25.21** | 0 |
| OccAny 5-frame forward, SK **pooled** | 25.93 | **25.93** | 0 |
| OccAny 5-frame forward, Occ3D **raw** | 20.66 | **20.66** | 0 |
| OccAny 5-frame forward, Occ3D **pooled** | 23.55 | **23.55** | 0 |
| ours KITTI-360, SK raw / pooled | 26.40 / 26.34 | **26.40 / 26.34** | 0 |
| ours KITTI-360, Occ3D raw / pooled | 31.94 / 41.56 | **31.94 / 41.56** | 0 |
| ours ONCE, SK raw / pooled | 18.20 / 20.26 | **18.20 / 20.26** | 0 |
| ours ONCE, Occ3D raw / pooled | 18.67 / 30.52 | **18.67 / 30.52** | 0 |
| ours Waymo, SK raw | 23.55 | **23.55** | 0 |
| ours Waymo, Occ3D raw | 22.85 | **22.85** | 0 |

Two further independent checks, neither of which was tuned toward:

* **the whole stored frustum split reproduces.** All twelve cells of
  `waymo_arm/waymo_vs_kitti360_report.md` §9 come back identical — SemanticKITTI in-frustum
  25.16 / 22.38 (Δ −2.78), out-of-frustum 29.98 / 26.58 (Δ −3.40), Occ3D in-frustum
  36.18 / 22.86 (Δ −13.32), out-of-frustum 28.25 / 22.85 (Δ −5.40).
* **every ONCE contrast reproduces** to the digit, including its intervals: Occ3D raw
  −1.99 [−3.94, −0.08], Occ3D pooled +6.98 [+5.57, +8.42], SemanticKITTI raw
  −2.12 [−3.06, −1.13].

**One discrepancy, investigated and explained.** The Occ3D pooled in-frustum KITTI-360 value is
46.4528 here against 46.4558 stored in `one_frame_occ3d_pooled.json`. Cause: the one-frame study
excluded **1** of the 4,219 Occ3D anchors, which has no one-frame metric scale — an exclusion
inherent to the *one-frame* protocol and inapplicable here. That study ran on 4,218 anchors;
this one runs on the full 4,219, which is the same population `once_vs_occany.md` Addendum 3
used, and the ONCE figures agree with it exactly. The gap is one anchor, not a scoring
difference. **It is not rounding and is not described as such.**

**Per-sample integrity gates**, asserted on all 5,020 anchors: every arm's stored
`T_cam_to_grid` agrees with the reference arm's to 1e-6; grid dims and flat shapes agree; the
stored transform and intrinsics are finite; and on SemanticKITTI the independently reconstructed
ground-truth pose chain collapses to the stored anchor transform. Additionally verified
directly: the hard-map `observed` mask is **bit-identical** across the three arms (so the
map-evidence split is a legitimate common denominator), while `occupied` differs (so the arms
really are three different checkpoints).

## 3. Headline — the Waymo arm, both conventions, with intervals

Absolute SC IoU, `all` = the full official valid volume, seed 0:

| arm | input window | SK raw | SK pooled | Occ3D raw | Occ3D pooled |
|---|---|---:|---:|---:|---:|
| ours, KITTI-360 (**not** an OccAny source) | **causal 5** | **26.40** | **26.34** | **31.94** | **41.56** |
| **ours, Waymo** (an OccAny source) | **causal 5** | **23.55** | **23.37** | **22.85** | **38.05** |
| ours, ONCE (an OccAny source) | **causal 5** | 18.20 | 20.26 | 18.67 | 30.52 |
| OccAny released [reproduced] | **causal 1** | 20.32 | 24.06 | — none exists | — |
| OccAny released [reproduced] | **forward**, t…t+40 | 25.21 | 25.93 | 20.66 | 23.55 |

**The windows are not the same and shared scoring does not make them so.** Our three arms
consume five **past** frames. OccAny's 1-frame mode is the only genuinely causal comparator it
has, and it sees four frames fewer than we do. OccAny's 5-frame mode consumes **future** frames
t…t+40 that our system never sees. OccAny publishes no 1-frame nuScenes run, so the Occ3D
columns have no causal comparator at all.

The Waymo arm's verdict, eight cells:

| benchmark | convention | comparator | Δ (Waymo − comparator) | verdict |
|---|---|---|---|---|
| SemanticKITTI | raw | OccAny **causal 1-frame** | **+3.23** [+2.26, +4.21] \* | **win** |
| SemanticKITTI | **pooled** | OccAny **causal 1-frame** | **−0.69** [−1.52, +0.10] **ns** | **tie** |
| SemanticKITTI | raw | OccAny forward 5-frame | **−1.66** [−2.52, −0.79] \* | loss |
| SemanticKITTI | **pooled** | OccAny forward 5-frame | **−2.55** [−3.52, −1.65] \* | loss |
| Occ3D | raw | OccAny forward 5-frame | **+2.20** [−0.02, +4.35] **ns** | **tie** |
| Occ3D | **pooled** | OccAny forward 5-frame | **+14.51** [+12.92, +16.06] \* | **win** |
| SemanticKITTI | raw / pooled | ours KITTI-360, same recipe | −2.84 \* / −2.97 \* | loss |
| Occ3D | raw / pooled | ours KITTI-360, same recipe | −9.09 \* / −3.51 \* | loss |

`*` = 95 % CI excludes zero. **ns** = the interval contains zero, so no significant difference
was detected — which is **not** a demonstration of equivalence.

**Neither convention is selected and both are reported.** Of the six OccAny cells, two favour
the Waymo arm significantly, two favour OccAny significantly, and two are ties. The one cell
that was previously quoted as the arm's headline Occ3D result (+2.19 raw) is a tie once its
interval exists; the cell that most favours us (+14.51) is scored the way OccAny publishes.

### Why pooling moves the arms so unequally

| arm | SK raw→pooled | Occ3D raw→pooled |
|---|---:|---:|
| ours, KITTI-360 | −0.05 | **+9.62** |
| **ours, Waymo** | **−0.18** | **+15.20** |
| ours, ONCE | +2.06 | +11.85 |
| OccAny 1-frame causal | +3.74 | — |
| OccAny 5-frame forward | +0.72 | **+2.89** |

`waymo_arm_vs_occany.md` §5 predicted, from the Waymo arm's lower recall, that it would gain
*more* from dilation than the KITTI-360 arm. **That prediction holds on Occ3D (+15.20 against
+9.62) and fails on SemanticKITTI (−0.18 against −0.05, i.e. it loses slightly more).** It is
recorded as a half-confirmed prediction, not a mechanism.

## 4. Where the arm wins and where it collapses

Full regional tables, every arm and every contrast, with TP/FP/FN per region, are generated in
`waymo_canonical_rescore/tables.md` and `metrics.csv`. The decisive rows:

### SemanticKITTI, strict visibility split (the archived `observed_strict` masks)

| region | % of valid | % of GT occ | KITTI-360 | **Waymo** | ONCE | OccAny 1f | OccAny 5f fwd |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 100 | 100 | 26.40 | **23.55** | 18.20 | 20.32 | 25.21 |
| anchor_observed | 32.69 | **4.76** | 11.47 | 10.29 | 7.51 | **15.76** | 11.72 |
| earlier_only | 24.81 | **6.18** | 17.95 | 15.13 | 11.48 | 12.97 | 11.65 |
| **never_observed** | 42.50 | **89.06** | 30.07 | **28.55** | 22.17 | 21.53 | 29.02 |

Waymo − OccAny, raw / pooled:

| region | vs causal 1-frame, raw | vs causal 1-frame, pooled | vs forward 5-frame, raw | vs forward 5-frame, pooled |
|---|---|---|---|---|
| all | +3.23 [+2.26, +4.21] \* | −0.69 [−1.52, +0.10] ns | −1.66 [−2.52, −0.79] \* | −2.55 [−3.52, −1.65] \* |
| anchor_observed | −5.47 [−6.31, −4.74] \* | −3.92 [−4.48, −3.41] \* | −1.42 [−1.97, −0.90] \* | −0.50 [−0.88, −0.10] \* |
| earlier_only | +2.16 [+1.21, +3.08] \* | −2.02 [−2.84, −1.22] \* | +3.48 [+2.83, +4.09] \* | +0.85 [+0.20, +1.45] \* |
| **never_observed** | **+7.02 [+5.66, +8.37] \*** | **+4.04 [+2.77, +5.29] \*** | −0.48 [−1.69, +0.74] ns | −1.16 [−2.47, +0.18] ns |

**This is the report's most consequential new number.** Against OccAny's only causal mode, on
the mask where no permitted input frame ever observed the surface — 89.06 % of ground-truth
occupancy — the Waymo-trained arm wins in **both** conventions with intervals clear of zero.
That cell was a statistical tie for the ONCE arm (+0.65 raw, −0.04 pooled, both ns), so the two
OccAny-source arms differ qualitatively here and not only in magnitude.

Against OccAny's **forward** mode the same cell is a **tie in both conventions**. That is the
honest ceiling of the claim: the never-observed advantage holds against their causal mode, and
does not hold against the mode that gets to look at the future.

### Frustum split, both benchmarks

| | % valid | % GT occ | KITTI-360 | **Waymo** | ONCE | OccAny |
|---|---:|---:|---:|---:|---:|---:|
| SK in_frustum, raw | 69.34 | 74.53 | 25.16 | 22.38 | 16.72 | 24.16 (1f) / 29.55 (5f) |
| SK out_frustum, raw | 30.66 | 25.47 | 29.98 | **26.58** | 21.55 | **7.43** (1f) / 8.04 (5f) |
| Occ3D in_frustum, raw | 50.95 | 45.51 | 36.18 | 22.86 | 17.64 | 29.98 |
| Occ3D out_frustum, raw | 49.05 | 54.49 | 28.25 | **22.85** | 19.54 | **8.94** |

Waymo − OccAny out-of-frustum: **+19.16** [+18.01, +20.29] and **+12.34** [+11.12, +13.51]
(SK raw / pooled, vs causal 1-frame); **+18.54** / **+12.60** (SK, vs forward 5-frame);
**+13.90** [+11.94, +15.82] and **+23.16** [+21.49, +24.80] (Occ3D raw / pooled). **Eight of
eight out-of-frustum cells across the two OccAny-source arms and two benchmarks are significant
wins.** In-frustum, the Waymo arm loses in every cell (−1.78 to −7.17 on SK, −7.11 raw on
Occ3D).

### Why the aggregate behaves differently on the two benchmarks — confusion counts, not fractions

The brief asks for this explained by confusion counts. SemanticKITTI raw, Waymo against OccAny
1-frame, `all`: **17,059,410** voxels are net converted from their false negatives into our
true positives
(TP 39.68 M against 22.62 M) at the cost of **57,138,787** extra false positives (FP 75.18 M
against 18.04 M). IoU rises because the TP gain outweighs the FP cost *in that denominator*.
Under pooling the same trade reverses: dilation adds 18.5 M FP to their 18.04 M (a 2.0×
increase from a small base) against 38.3 M added to our 75.18 M (a 1.5× increase from a large
base), so their IoU gains 3.74 and ours loses 0.18, and a +3.23 win becomes a −0.69 tie. **The
convention flip is entirely a property of where each method sits on the precision/recall
trade**, visible in the raw numbers: their precision 55.64 against our 34.55, their recall
24.25 against our 42.53.

The benchmark asymmetry has the same origin. SemanticKITTI's valid mask is **69.3 %**
in-frustum; Occ3D's is **51.0 %**. The in-frustum collapse therefore costs ~1.4× more on
SemanticKITTI, which is why the Waymo arm's aggregate is −1.66 against OccAny's forward mode
there and +2.20 on Occ3D, from in-frustum deficits of comparable size (−7.17 and −7.11).

### The three region families are not interchangeable

`region_crosstab.json`, 200 SemanticKITTI anchors, 303,492,979 valid voxels. All three splits
are verified exact partitions of the valid mask (`strict_regions_sum_equals_valid`,
`frustum_split_sum_equals_valid`, `map_evidence_split_sum_equals_valid` all true) — but they are
**three different partitions and they are not additive**. On GT-occupied voxels:

| | GT-occupied voxels |
|---|---:|
| never_observed | 21,645,991 |
| out_frustum | 6,399,945 |
| **both** | **5,280,272** |
| fraction of **out-of-frustum** that is never-observed | **82.5 %** |
| fraction of **never-observed** that is out of frustum | **24.4 %** |

So out-of-frustum is nearly a subset of never-observed, but it covers only **a quarter** of it.
**An out-of-frustum win therefore speaks to ~24 % of never-observed surface and says nothing
about the other 76 %**, and the two must not be renamed into each other. The joint table also
shows why: 58.22 % of all GT-occupied voxels are `never_observed ∩ in_frustum ∩ map_completed`
— never seen, yet squarely inside the anchor camera's view volume.

The three concepts are kept distinct throughout: **frustum membership** (does the view volume
contain the voxel centre), **strict surface visibility** (does a GT raycast from one of the five
permitted frames reach it, stopping at the first GT-occupied voxel), and **hard-map evidence
state** (did our accumulated reconstruction resolve it). Occ3D has no strict-visibility split in
this harness — that mask is built on SemanticKITTI's pose chain — so Occ3D is reported with the
frustum and map-evidence splits only, and no strict never-observed claim is made for it.

## 5. The unified comparison — what may share a table and what may not

All three of our arms and both OccAny modes **can** share one table, because they were scored
in one pass on one population with one scorer: 801 / 4,219 anchors, seed 0, OccAny's labels and
masks. That is §3's table and the per-arm pairs are below.

| contrast, `all` | SK raw | SK pooled | Occ3D raw | Occ3D pooled |
|---|---|---|---|---|
| Waymo − OccAny causal 1f | +3.23 \* | −0.69 ns | — | — |
| Waymo − OccAny forward 5f | −1.66 \* | −2.55 \* | +2.20 ns | +14.51 \* |
| ONCE − OccAny causal 1f | −2.12 \* | −3.80 \* | — | — |
| ONCE − OccAny forward 5f | −7.01 \* | −5.67 \* | −1.99 \* | +6.98 \* |
| KITTI-360 − OccAny causal 1f | +6.08 \* | +2.28 \* | — | — |
| KITTI-360 − OccAny forward 5f | +1.19 \* | +0.41 ns | +11.28 \* | +18.01 \* |
| **Waymo − ONCE**, same recipe | **+5.36 \*** | **+3.12 \*** | **+4.18 \*** | **+7.53 \*** |
| Waymo − KITTI-360, same recipe | −2.84 \* | −2.97 \* | −9.09 \* | −3.51 \* |
| ONCE − KITTI-360, same recipe | −8.20 \* | −6.08 \* | −13.27 \* | −11.04 \* |

The two KITTI-360-against-OccAny rows are **seed-0** rows and therefore differ slightly from
`geometry_head_to_head.md` §4, which reports the same contrasts as **three-seed means**
(+1.11 [+0.30, +1.93] raw and −0.04 [−0.77, +0.70] pooled on SemanticKITTI, against +1.19 and
+0.41 here). Both are correct for what they are; the seed-0 row is the one that pairs with the
Waymo and ONCE rows, since those have no other seed.

**Waymo beats ONCE in all four cells with intervals clear of zero.** The two OccAny-source arms
are not equivalent, and `training_supervision_provenance.md` shows one reason they would not be:
Waymo preserved the frozen 21-frame / 10 s supervision window and ONCE did not.

**What may not share a table.** The frozen `baseline_reconstruction` evaluator's figures — 163
SemanticKITTI and 1,182 Occ3D anchors, its own masks, three seeds — are a different population
and a different protocol. They are reported separately in §7 and no row above mixes them. In
particular, the three-seed frozen figures (KITTI-360 26.17 / 31.22, Waymo 23.14 / 22.86, ONCE
18.50 / 19.29) are **not** placed beside OccAny's canonical numbers; that is the error
`once_vs_occany.md` §2 made and its Addendum 1 corrected.

## 6. Pairing, blocks, and what the intervals do and do not cover

* **Pairing unit:** the anchor. Every arm is scored on the identical anchor list in one pass,
  and the bootstrap resamples the **same** block draw for all arms, so each Δ is a paired
  difference, not a difference of two independent estimates.
* **Block unit:** on SemanticKITTI, a contiguous run of **25** sequence-08 anchors (33 blocks),
  which absorbs the serial correlation of a single driven sequence. On Occ3D, **one block per
  nuScenes scene** (150 blocks).
* **Resamples:** 2,000; **bootstrap seed:** 0; percentile interval at 2.5 / 97.5.
* **Inside each resample, global IoU is recomputed from the summed TP/FP/FN** of the drawn
  blocks. Per-anchor IoUs are never averaged.
* **These intervals cover evaluation-sample uncertainty only.** They do **not** cover
  training-seed variation, because the canonical predictions are seed 0 (§1). OccAny is one
  released checkpoint and has no seed spread at all. Both facts are recorded in the JSON as
  `uncertainty_covered`.

## 7. Analyses that could not be produced, and why

| asked for | status |
|---|---|
| **per-seed canonical metrics for all three Waymo seeds** | **not available.** Canonical predictions exist for seed 0 only; generating seeds 1–2 means regenerating model predictions, which this task excludes. Seed-0 comparison completed in full instead. |
| three-seed canonical confidence intervals | **not produced**, and not estimated from the seed-0 intervals. |
| strict never-observed split on Occ3D | **not available.** The strict mask requires SemanticKITTI's GT pose chain; Occ3D is reported with frustum and map-evidence splits, as the protocol allows. |
| OccAny causal 1-frame on Occ3D | **does not exist.** No such released run; the Occ3D comparison has no causal comparator. |
| conflict fraction for the KITTI-360 supervision cache | **not recorded** in the frozen target files (no `stats_json`). Reported as absent rather than recomputed under a different rule. |

For reference, and **not comparable with anything above**, the frozen-evaluator three-seed
figures: Waymo SemanticKITTI **23.14 ± 0.55** (per-seed 23.73 / 22.64 / 23.05) and Occ3D
**22.86 ± 0.22** (22.84 / 23.08 / 22.65), on 163 and 1,182 anchors with that evaluator's own
masks. The two protocols agree on sign and roughly on size for the Waymo-minus-KITTI-360
contrast (−2.84 / −9.09 canonical against −3.03 / −8.36 frozen), which is the consistency check
that makes either believable. The seed spread is small against the effects measured here, but
it is a different quantity and is not folded into any interval.

## 8. Reproduction

```bash
cd /home/minh/workspace/lingbot-map_fork
for ds in semantickitti occ3d; do for post in raw pooled; do
  PYTHONPATH=$PWD python tools/method_vs_data/canonical_rescore.py \
      --dataset $ds --post $post --device cuda:0 \
      --arms k360 waymo once occany_1f occany_5f_fwd
done; done
PYTHONPATH=$PWD python tools/method_vs_data/rescore_tables.py
PYTHONPATH=$PWD python tools/method_vs_data/region_crosstab.py --limit 200 --device cuda:0
PYTHONPATH=$PWD python tools/method_vs_data/supervision_provenance.py
```

Outputs, all in `artifacts/method_vs_data/waymo_canonical_rescore/`:
`canonical_{semantickitti,occ3d}_{raw,pooled}.json` (per-arm per-region IoU, precision, recall,
TP/FP/FN, valid and GT-occupied voxel counts, every paired contrast with its interval, and the
provenance block), `tables.md`, `metrics.csv` (324 rows), `region_crosstab.json`,
`supervision_provenance.json`, and `logs/`.

## 9. Not claimed

* **No three-seed canonical result for any arm**, and no projection of what seeds 1–2 would give.
* **No architectural claim.** Every OccAny comparison crosses different corpora, supervision
  objectives, pretraining and three orders of magnitude of trained capacity at once.
* **No equivalence from a tie.** The three `ns` cells mean no significant difference was
  detected on 801 or 4,219 anchors; they do not establish that the methods are equal.
* **No never-observed claim for Occ3D**, and no restatement of an out-of-frustum result as a
  never-observed result (§4 quantifies why that substitution is invalid).
* **No change to the ONCE arm's status.** Its occupied-prevalence gate is still **failed**
  (0.0131 against a [0.015, 0.040] floor) and the arm remains exploratory.
* **No selection between conventions.** Raw and pooled are both reported in every cell.
