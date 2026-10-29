# Method versus data — reconciled conclusions

**2026-10-27.** Written after the canonical rescore of the Waymo arm and the
training-supervision provenance audit. Supersedes the *interpretation* in
`method_vs_data_conclusions.md` and `method_vs_data_updated_conclusions.md` (body and both
Addenda) where they differ. **Those files' measurements stand and are preserved**; what changes
here is what may be concluded from them, plus three numbers that now have intervals they did
not have before.

Sources: `waymo_canonical_rescore.md`, `training_supervision_provenance.md`,
`once_vs_occany.md` Addenda 2–3, `once_training_report.md`, `one_frame_ablation.md`,
`observed_vs_completed_analysis.md`, `occany_training_audit.md`,
`fair_evaluation_protocol.md`. Geometry only; this task touched nothing semantic.

---

## Corrections to the existing record

Each of these retracts or qualifies a statement of mine. The superseded text is left in place
in its own file.

1. **`waymo_arm_vs_occany.md` §4 and §6, "stays ahead on Occ3D (+2.19 raw)" — QUALIFIED.** The
   difference is +2.20 with a paired interval of **[−0.02, +4.35]**, which contains zero. The
   correct statement is *no significant difference detected on Occ3D under raw scoring*. §6's
   "remains ahead of the released OccAny checkpoint on Occ3D (+2.19 raw)" overstates a tie.
2. **`waymo_arm_vs_occany.md` §4, "beats their only genuinely causal mode on SemanticKITTI
   (+3.23 raw)" — CONVENTION-DEPENDENT.** True raw (+3.23 [+2.26, +4.21]); under OccAny's
   **published pooled** convention it is **−0.69 [−1.52, +0.10]**, a tie. A raw-only reading of
   that cell is the exact error `geometry_head_to_head.md` §1 exists to prevent.
3. **`method_vs_data_updated_conclusions.md` Addendum 2 Q4, "Data-subset robustness — REFUTED
   … two independent attempts" — OVERSTATED.** The two attempts are not independent in the way
   the claim needs. Measured from the training caches: the Waymo arm **preserved** the frozen
   21-frame / 10.44 s supervision window; the ONCE arm **shortened it to 5 frames / 2.00 s**.
   ONCE is therefore not a second clean source substitution, and "two of OccAny's five sources
   failed" conflates a corpus change with a corpus-plus-supervision change.
4. **`method_vs_data_updated_conclusions.md` Addendum 1 Q4, "No never-observed figure exists for
   the Waymo or ONCE arms" — NOW MEASURED, and the answer is not the same for the two arms.**
   On the strict never-observed mask against OccAny's causal mode: Waymo **+7.02 raw / +4.04
   pooled, both significant**; ONCE **+0.65 raw / −0.04 pooled, both ns**.
5. **`once_supervision_validation.md` §7, "sparsity is falsified" — SCOPE NARROWED.** Correct
   for what it tested; every thinning condition ran at the **21-sweep** window and measured
   **target prevalence**, not any trained model. It does not bound density effects under
   5-sweep supervision, and no thinned KITTI-360 arm was ever trained, so no thinning result in
   this directory speaks to transfer.
6. **`method_vs_data_updated_conclusions.md` Addendum 1 §"On the 2 Hz KITTI-360 control", "its
   diagnostic value has been obtained by measurement" — WITHDRAWN.** That reasoning compared a
   *rebuilt* 5-sweep KITTI-360 target probe with the ONCE arm's targets. The KITTI-360
   *checkpoint* every comparison uses as its reference was trained at 21 sweeps / 10.44 s. A
   statistics probe is not a trained control, and no duration-matched KITTI-360 training
   control exists.

---

## Q1. Does Waymo training beat OccAny — by benchmark, convention and comparator?

**Mixed, and the answer flips on both axes. Two significant wins, two significant losses, two
ties, out of six cells.** Seed 0, 801 / 4,219 anchors, paired block bootstrap.

| benchmark | convention | comparator | Δ | verdict |
|---|---|---|---|---|
| SemanticKITTI | raw | OccAny **causal 1-frame** | +3.23 [+2.26, +4.21] \* | **win** |
| SemanticKITTI | **pooled** | OccAny **causal 1-frame** | −0.69 [−1.52, +0.10] ns | **tie** |
| SemanticKITTI | raw | OccAny forward 5-frame | −1.66 [−2.52, −0.79] \* | loss |
| SemanticKITTI | **pooled** | OccAny forward 5-frame | −2.55 [−3.52, −1.65] \* | loss |
| Occ3D | raw | OccAny forward 5-frame | +2.20 [−0.02, +4.35] ns | **tie** |
| Occ3D | **pooled** | OccAny forward 5-frame | **+14.51 [+12.92, +16.06] \*** | **win** |

Read by comparator: **against OccAny's only causal mode the Waymo arm is a win raw and a tie
pooled; against its forward mode it loses on SemanticKITTI and wins on Occ3D under the
published convention.** The forward mode consumes frames t…t+40 that our system never sees,
and is worth a measured **+1.9** SC IoU to OccAny over its own causal mode — so the comparator
choice matters less than the convention does here.

**And it falls below KITTI-360 in all four cells** (−2.84 / −2.97 SemanticKITTI, −9.09 / −3.51
Occ3D, all significant). Those two findings are different and are kept apart: *falling below
KITTI-360 is a statement about source sensitivity; falling below OccAny is a statement about
competitiveness.* The Waymo arm does the first consistently and the second only in two of six
cells.

One ordering that was not previously on the record: **Waymo beats ONCE in all four cells**
(+5.36, +3.12, +4.18, +7.53, all significant), so the two OccAny-source arms are far from
equivalent.

## Q2. Does any lead persist on the strict never-observed mask?

**Yes — against OccAny's causal mode, in both conventions, for the Waymo arm. No, against its
forward mode.**

| region (SemanticKITTI, 89.06 % of GT occupancy) | Waymo − OccAny causal 1f | Waymo − OccAny forward 5f |
|---|---|---|
| **never_observed**, raw | **+7.02 [+5.66, +8.37] \*** | −0.48 [−1.69, +0.74] ns |
| **never_observed**, pooled | **+4.04 [+2.77, +5.29] \*** | −1.16 [−2.47, +0.18] ns |
| anchor_observed, raw / pooled | −5.47 \* / −3.92 \* | −1.42 \* / −0.50 \* |
| earlier_only, raw / pooled | +2.16 \* / −2.02 \* | +3.48 \* / +0.85 \* |

This is the first never-observed figure for any OccAny-source arm, and it separates the two
such arms qualitatively: **ONCE is a tie in this cell in both conventions (+0.65 ns, −0.04 ns);
Waymo is a significant win in both.**

Three limits on the claim, all load-bearing:

* it holds against their **causal** comparator and is a **tie** against their forward one;
* the Waymo arm is still **below the KITTI-360 arm** in this cell (−1.52 raw, −0.73 pooled,
  both significant), so never-observed completion is itself source-sensitive;
* **Occ3D has no strict never-observed mask** in this harness, so this is a one-benchmark
  result. The frequently quoted out-of-frustum wins are **not** a substitute: measured on
  200 anchors, only **24.4 %** of never-observed GT-occupied voxels are out of frustum (while
  82.5 % of out-of-frustum voxels are never-observed). An out-of-frustum win addresses a
  quarter of the never-observed surface and is silent on the rest.

## Q3. Which regions explain the aggregate — by confusion counts

**Our arms trade precision for recall; the aggregate is decided by whether that trade pays in
the region that dominates the valid mask, and the pooling convention changes whether it does.**

SemanticKITTI raw, `all`, Waymo against OccAny causal 1-frame: TP 39.68 M against 22.62 M
(**+17.06 M** net, since TP + FN is the same 93.30 M for every arm), FP 75.18 M against
18.04 M (**+57.14 M**). Precision 34.55 against 55.64; recall 42.53 against 24.25.

Pooling adds 18.5 M false positives to their 18.04 M (**2.03×** from a small base) and 38.3 M
to our 75.18 M (**1.51×** from a large base). Their IoU gains **+3.74**, ours loses **−0.18**,
and the +3.23 win becomes a −0.69 tie. **The convention flip is a property of where each method
sits on the precision/recall trade, not of the region composition.**

The benchmark asymmetry *is* region composition: SemanticKITTI's valid mask is **69.3 %**
in-frustum against Occ3D's **51.0 %**, so near-identical in-frustum deficits (−7.17 and −7.11
raw against OccAny's forward mode) cost ~1.4× more on SemanticKITTI. That is why the aggregate
is −1.66 there and +2.20 on Occ3D.

And the regional structure is stable across both OccAny-source arms: **eight of eight
out-of-frustum cells** (two arms × two benchmarks × two conventions, where defined) are
significant wins, ours roughly two to three times OccAny's, while **in-frustum accuracy is lost
in every cell**. Whatever transfers across the corpus change, it is completion into space
outside the camera's view volume; what does not transfer is accuracy on surfaces the camera
sees.

## Q4. What source sensitivity has been demonstrated?

**Strong source sensitivity, on our own system, with intervals clear of zero — and now with one
arm's confound identified.**

Architecture, loss, optimizer, schedule, step budget, batch, crop, grid, mapper, evaluator and
per-seed initialization digest all held fixed; `map_only` is bit-identical across arms
(8.820500920980095), and the hard-map `observed` mask is bit-identical across all three
prediction sets, so every difference lives in one 967,322-parameter network.

| arm | changed vs KITTI-360 | SK raw / pooled | Occ3D raw / pooled |
|---|---|---|---|
| KITTI-360 | — | 26.40 / 26.34 | 31.94 / 41.56 |
| **Waymo** | corpus, 5 lasers, 50° camera; **supervision window preserved** | 23.55 / 23.37 | 22.85 / 38.05 |
| **ONCE** | corpus, 90° camera, **window 21→5 frames, 10.44→2.00 s**, prevalence 0.0227→0.0131 | 18.20 / 20.26 | 18.67 / 30.52 |

**What is demonstrated:** replacing the training corpus while holding the recipe and the
supervision window fixed costs **2.84 / 2.97** SC IoU on SemanticKITTI and **9.09 / 3.51** on
Occ3D, all significant. That is the Waymo arm, and it is a clean source-only result.

**What is not demonstrated:** that two OccAny sources failed independently. The ONCE arm's
larger deficit (−8.20 / −13.27 against KITTI-360) is confounded by the **largest single
departure from the frozen recipe anywhere in this directory** — a 5.2× shorter supervision
window — alongside a 34 % occupied-voxel deficit, a 1.35× monocular scale bias against
KITTI-360's 1.20×, and six independent scenes. **No causal attribution is made**, and the
previously claimed refutation of source robustness rests on one clean arm, not two.

**Input-distribution similarity still fails as a transfer predictor**, now on three points:
ONCE's 90.1° camera is the closest of the three to KITTI-360's 103.9° and is wider than both
test cameras (82° and 64.5°), yet ONCE performs worst. Waymo's 50° is the narrowest and it
performs better. The FOV ordering does not predict the result ordering.

## Q5. Which supervision differences remain uncontrolled?

| difference | Waymo | ONCE |
|---|---|---|
| training corpus | changed | changed |
| **supervision window** (frames / span) | **preserved: 21 / 10.00 s** | **changed: 5 / 2.00 s** |
| sweeps per window | 105 (5 lasers) vs 21 | 5 vs 21 |
| camera HFOV | 50.0° vs 103.9° | 90.1° vs 103.9° |
| independent scenes | 751 vs 3 | 6 vs 3 |
| target occupied prevalence | 0.0200 vs 0.0227 | **0.0131** vs 0.0227 |
| monocular metric-scale bias | not measured | 1.35× vs 1.20× |
| input frames, direction, span | identical (5, causal, ~2 s) | identical (5, causal, 2.0 s) |
| target temporal direction | identical (**future only**) | identical (**future only**) |
| carve rule, conflict rule, grid, recipe, init | identical | identical |

**Uncontrolled for both arms:** corpus identity, scene content, camera FOV, sensor geometry and
independent scene count move together. **Uncontrolled for ONCE additionally:** supervision
window length and target density.

**The missing control, specified but not launched.** Retrain the frozen recipe on KITTI-360 with
the supervision window truncated to ~2.1 s and everything else fixed — drives, anchors, input
clips, grid, loss, schedule, seeds. That isolates **window length** from corpus identity and is
the only experiment that would make the ONCE arm interpretable. It does not exist: all three
KITTI-360 5-sweep artefacts (`k360_duration_matched` span2.1, `k360_sweep_control` dec5, the
beam and point-budget controls) are **target-statistics probes with no training**, and the
thinning probes ran at the 21-sweep window throughout. Note also that truncating KITTI-360 to
2.1 s drops it from 32 occupied z-levels to **21**, so such a control would change the *shape*
of supervision, not only its amount.

Separately: the proposed alternative of **selecting more open KITTI-360 anchors until prevalence
reaches 0.0131** is **not a pure density intervention** — selecting by openness varies scene
content and density together. It is a different experiment and neither substitutes for the
other.

## Q6. What can honestly be claimed in the paper now?

**The strongest supported claim.**

> Under matched single-camera causal evaluation on OccAny's own anchors, labels, placement and
> scorer, our 967 k-parameter completion network trained from scratch for 6,000 optimizer steps
> **wins decisively on Occ3D-nuScenes against the released OccAny checkpoint's strongest
> released mode, in the convention OccAny publishes in** — +18.01 [+16.67, +19.34] trained on
> KITTI-360 and +14.51 [+12.92, +16.06] trained on Waymo, one of OccAny's own five source
> datasets — while consuming only past frames where their mode consumes future ones. On
> SemanticKITTI the comparison is a tie-to-narrow-win depending on convention and comparator.
> The advantage is **localized**: it is completion rather than accumulation (the accumulated
> map alone scores 8.79 / 9.82 against their 25.21 / 20.66, and the completion contribution is
> +15 to +32 across all four cells with every interval clear of zero), it survives dropping
> four of five input frames, and regionally it lives outside the camera's view volume — eight
> of eight out-of-frustum cells across two training corpora and two benchmarks are significant
> wins — while in-frustum accuracy on directly visible surfaces is consistently worse than
> theirs.

**The secondary claim that is now supportable, and was not before.**

> The never-observed completion advantage survives a change of training corpus to one of
> OccAny's own sources: on the strict mask where no permitted input frame observed the surface
> (89.06 % of SemanticKITTI ground-truth occupancy), the Waymo-trained arm beats OccAny's only
> causal mode by +7.02 raw and +4.04 pooled, both significant. Against their future-frame mode
> the same cell is a tie.

**What must be stated alongside it.**

* **Single-camera only.** We are behind on surround geometry (32.62 against 34.15) and on
  semantics (4.45 against 7.28 SSC mIoU).
* **The comparators are not window-matched.** OccAny has no causal five-frame mode; its 1-frame
  causal run sees four frames fewer than we do, and its 5-frame run sees the future. Both
  asymmetries must be named every time.
* **Strong source sensitivity, on our own system.** Changing only the corpus moved SC IoU by
  −2.84 / −9.09 with intervals clear of zero.
* **Task-specific optimization budget, stated accurately.** 6,000 optimizer steps, 24,000
  sample draws, 967,322 trained parameters, from scratch, against OccAny's **≥**70,300 steps and
  4.5 M draws from a pretrained MUSt3R plus an unrecorded warm start. The "≥" is load-bearing:
  their total is unbounded above by our audit. "Task-specific" is also load-bearing: **the count
  excludes our frozen LingBotMap and MoGe-2 backbones entirely**, and an optimizer-step count is
  **not** total compute and **not** unique-data efficiency — OccAny's unique scene and frame
  counts were never verified. Also: OccAny's own model selection used 10 memory views, so a
  5-frame causal evaluation is not the condition its checkpoint was selected under.

**What must NOT be claimed.**

* **Not architectural causality.** Every OccAny comparison changes pretraining, supervision
  objective, training corpus and ~3 orders of magnitude of trained capacity at once. Nothing
  measured here isolates architecture, and our own source sensitivity cuts against the claim.
* **Not source-dataset robustness — but not its refutation either.** The honest position:
  **one clean source substitution (Waymo) has been tried; it cost 2.8–9.1 SC IoU against
  KITTI-360 yet remains competitive with OccAny in two of six cells and keeps the
  never-observed advantage against their causal mode. A second attempt (ONCE) performed worse
  but changed the supervision window as well as the corpus, so it does not constitute a second
  independent source failure.** Failure of two replacement arms to preserve KITTI-360
  performance does not establish failure across every source subset — two of OccAny's five
  sources remain untried (PandaSet unaudited, VKITTI2 synthetic) and one is a verified NO-GO
  (DDAD).
* **Not equivalence from a tie.** Three cells here have intervals containing zero. That is no
  significant difference detected on 801 or 4,219 anchors — not a demonstration of parity.
* **Not a never-observed claim from an out-of-frustum result**, and not a never-observed claim
  on Occ3D, where the strict mask does not exist.
* **Not a three-seed canonical result.** Every canonical number is seed 0.
* **Not a convention.** Raw and pooled are both reported everywhere; neither is selected.

---

## Final decision table

| question | answer | strength |
|---|---|---|
| Does Waymo-trained geometry beat OccAny? | **2 wins, 2 losses, 2 ties of 6 cells.** Occ3D pooled +14.51 \*; SK raw vs causal +3.23 \*; SK pooled vs causal a tie; Occ3D raw a tie; both SK cells vs their forward mode a loss | measured, seed 0, paired CIs |
| Does the lead persist on strict never-observed space? | **Yes vs their causal mode, both conventions** (+7.02 / +4.04 \*). **Tie vs their forward mode.** Still below KITTI-360 there | measured, SemanticKITTI only |
| Which regions explain the aggregate? | out-of-frustum wins (8/8 significant) against in-frustum losses (all cells); the balance flips with the 69.3 % / 51.0 % in-frustum share, and the convention flips with the precision/recall trade | confusion counts |
| What source sensitivity is demonstrated? | **−2.84 / −2.97 SK and −9.09 / −3.51 Occ3D** for a clean corpus-only change, all significant | one clean arm (Waymo) |
| Which supervision differences remain uncontrolled? | window length and target density for ONCE; corpus, scene content, FOV, sensor geometry and scene count for both | provenance audit |
| Is the architecture why we win? | **Not established, and not establishable by any comparison run so far** | unchanged |
| Next action | **the duration-matched KITTI-360 training control** (§Q5). Not launched | — |

**The single next action justified by these results:** train the frozen recipe on KITTI-360 with
the supervision window truncated to ~2.1 s, three seeds, everything else fixed. It is the only
experiment that converts the ONCE arm from an uninterpretable double-change into evidence, it
costs ~8 minutes per seed plus one target rebuild, and it needs no new data. **Not launched.**

**Not recommended:** OccAny retraining (16-GPU-shaped job, unreproducible warm start), any
architecture change, a further dataset arm, or any scaling of the existing arms — two of the
three prior "next experiments" in this directory turned out to rest on comparisons that this
audit has had to correct.
