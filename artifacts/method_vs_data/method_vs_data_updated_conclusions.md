# Method versus data — updated conclusions

**2026-10-20.** Supersedes the interpretation in `method_vs_data_conclusions.md` where they
differ; that file's measurements stand. Phase A executed in full; Phases C and D are blocked.

| stage | outcome |
|---|---|
| **Phase A** one-frame vs five-frame ablation | **EXECUTED** — `one_frame_ablation.md` |
| **Phase B** ONCE feasibility audit | **EXECUTED** — `once_feasibility.md`, verdict CONDITIONAL GO |
| **Phase C** ONCE-only training | **BLOCKED** — `once_geometry_results.md` |
| **Phase D** ONCE vs OccAny | **BLOCKED** — `once_vs_occany.md` |
| **Phase E** this analysis | **EXECUTED** |

---

## Decision table

| question | result |
|---|---|
| **Does our single-frame completion outperform OccAny?** | **Measured, and it depends.** SemanticKITTI, matched one-frame vs one-frame: **+2.76** [+1.60, +3.94] raw but **+0.47** [−0.45, +1.43] pooled — a **tie** under OccAny's own convention. Occ3D, our one frame vs their five *forward* frames: **+6.81** [+4.75, +8.94] raw and **+18.37** [+16.93, +19.79] pooled — **yes, decisively.** |
| **Does ONCE pass supervision correctness gates?** | **CONDITIONAL GO.** The origin gate DDAD failed, ONCE passes on authoritative documentation (LiDAR frame = sensor centre, single `lidar_roof`). The brief's **hard temporal gate fails as specified**: ONCE releases at 2 FPS, so ~21 sweeps span 10.5 s while a 2.1 s window gives 5. |
| **Does ONCE-only training outperform OccAny?** | **BLOCKED.** No ONCE model was trained. No number exists and none is projected. |
| **Does the advantage persist in never-observed space?** | **Measured — yes, and that is where it lives.** Out of frustum at one frame: ours 25.57 / 26.56 (SK) and 21.75 / 37.15 (Occ3D) against OccAny's 7.43 / 11.85 and 8.94 / 12.55, while the one-frame *map* scores 0.75 and 0.54 there. Conversely we **lose** in-frustum on SemanticKITTI (22.34 / 23.93 vs 24.16 / 27.33). |
| **Can the performance lead be attributed to the method?** | **Weak-to-moderate, and not causal.** Strengthened by Phase A: the lead survives removing 80 % of input frames, and accumulation alone is far below OccAny. Still confounded by different pretrained weights, corpora, supervision objectives and ~3 orders of magnitude of trained capacity. **Our own Waymo arm shows the method is strongly data-sensitive** (−3.03 / −8.36), which cuts against a pure architecture claim. |
| **What is the next minimum experiment?** | **Decide the ONCE sweep policy, then pull `train_infos.tar` (small) and run the existing probe on one `train_lidar.tar` sequence.** That is the only cheap step that could unblock the one claim currently resting on nothing — that our method generalizes from an OccAny source dataset. |

---

## Q1. Is the performance advantage robust to source dataset choice?

**Unresolved, and currently unsupported.** This is the question the ONCE arm existed to answer
and it could not be run.

What is known: of OccAny's five source datasets, exactly one has been tried as a training
source for our CompletionUNet — **Waymo — and it failed**, −3.03 SemanticKITTI and −8.36 Occ3D,
both with CIs clear of zero. DDAD is a verified NO-GO on supervision grounds. ONCE is blocked
on the temporal gate. PandaSet is unaudited beyond its per-point sensor ID.

**KITTI-360, which produces our strong numbers, is not one of OccAny's five sources**, so it
cannot support a claim of the form "our method works when trained on what OccAny trains on".
Substituting it there would be the central error this report exists to avoid.

Honest statement: **one OccAny-source dataset tried, one failure; the robustness question is
open.**

## Q2. Is the advantage genuinely about completion?

**Yes, on measured evidence, and Phase A strengthens this considerably.**

| | SemanticKITTI raw / pooled | Occ3D raw / pooled |
|---|---|---|
| hard map alone, five frames | 8.79 / 15.04 | 9.81 / 22.16 |
| hard map alone, **one** frame | **3.65 / 9.15** | **2.90 / 9.90** |
| + CompletionUNet, one frame | 23.08 / 24.53 | 27.47 / 41.92 |
| **completion contributes (C2)** | **+19.43 / +15.39** | **+24.58 / +32.01** |
| OccAny | 20.32 / 24.06 (1f) | 20.66 / 23.55 (5f fwd) |

All four C2 intervals exclude zero. Accumulated reconstruction alone is below OccAny in every
cell except the pooled five-frame Occ3D map. **Memory is not what beats them.**

The strict split (`observed_vs_completed_analysis.md`) agrees: voxels only an earlier frame saw
are 6.18 % of the GT surface, and under the pooled convention our advantage there is not
significant (+0.65, CI ∋ 0), while the never-observed 89.06 % carries +4.77.

**The withdrawn claim is not repeated.** "We beat it by remembering, not by completing" was
wrong and remains withdrawn.

## Q3. How dependent is the method on temporal evidence?

**Modestly, and not uniformly.** C1 (one frame − five frames):

| cell | Δ | significant? |
|---|---|---|
| SemanticKITTI raw | −3.31 [−3.89, −2.74] | yes |
| SemanticKITTI pooled | −1.81 [−2.35, −1.29] | yes |
| Occ3D raw | −4.47 [−5.13, −3.89] | yes |
| Occ3D pooled | **+0.35** [−0.18, +0.89] | **no** |

Three of four cells show a real cost of 1.8–4.5 points; the fourth shows none. Removing 80 % of
the input frames and ~2.5× of observed surface is survivable, not catastrophic.

**Where the cost falls is the informative part.** On `anchor_observed` — what the single
remaining frame still sees — the two arms are statistically indistinguishable in both
conventions (−0.09 and +0.12, neither significant). The entire loss is in `never_observed`
(−4.61 / −3.34). Temporal input does not improve what the model does with what it sees; it
improves what it extrapolates.

**Accumulation vs distribution shift is not separated**, and cannot be by this experiment: the
model was trained only on five-frame maps, so every one-frame input is simultaneously sparser
*and* off-distribution. That needs a one-frame-trained CompletionUNet — new training, not
authorized.

**Metric scale is a minor term, measured across four cells:** forcing the five-frame scale into
the one-frame arm gives −0.60 / +0.86 / −4.77 / −1.96. Three of four negative, largest
magnitudes negative, so a mismatched scale generally hurts. (Two earlier readings of this in
session were over-confident in both directions; `one_frame_ablation.md` §7 records the
correction.)

**One unexplained counter-current.** The one-frame arm is robustly *better* on `earlier_only`
(+2.06 raw, +2.96 pooled, both significant) — where an earlier frame saw and the anchor did
not. The five-frame arm has map evidence there and loses. The plausible cause is
misregistration of accumulated evidence; **this is unverified** and it questions whether
accumulation is net-positive at all. It is the single most important loose end in Phase A.

## Q4. Can we attribute the advantage to the method rather than the data?

**Strengthened but still not causal.** Three claims, kept distinct:

**Performance — supported, with scope.** Under matched single-camera causal evaluation we beat
the released OccAny checkpoint on Occ3D (+11.28 raw / +18.01 pooled at five frames, +6.81 /
+18.37 at one frame) and tie-to-win on SemanticKITTI (+6.08 raw / +2.28 pooled at five frames;
+2.76 raw / +0.47 pooled at one frame). We are behind on semantics (4.45 vs 7.28 SSC mIoU) and
on surround geometry (32.62 vs 34.15); the advantage is **single-camera only**.

**Data efficiency — supported.** ≥70,300 optimizer steps and 4,500,000 sample draws from a
pretrained MUSt3R plus an unrecorded warm start, against **6,000 steps, 24,000 draws, 967 k
trained parameters, from scratch**. "≥" is load-bearing: OccAny's total is unbounded above by
our audit. The qualifier "task-specific" is also load-bearing — our pipeline carries frozen
LingBotMap and MoGe-2 dependencies this count excludes, and OccAny's unique scene and frame
counts were **not** verified, so the claim rests on steps and draws, not dataset names.

**Methodological advantage — not supported.** What Phase A adds: the lead survives at one
frame, so it is not temporal context; and the lead is localized to never-observed space, so it
is not visible-surface accuracy. What still blocks causality:

* different pretrained initialization (MUSt3R + a prior OccAny run vs LingBotMap + MoGe-2);
* different supervision drawn from the same data (pointmap regression + SAM2 distillation vs
  occupancy ray-carving) — the difference that blocked DDAD entirely;
* ~3 orders of magnitude of trained capacity;
* and **our own Waymo result**: same architecture, same recipe, only the corpus changed, and
  performance moved −3.03 / −8.36. We have demonstrated that this method is strongly
  data-sensitive, which is direct evidence against "the architecture is what wins".

The one thing that *is* localized: `map_only` is bit-identical across the KITTI-360 and Waymo
arms (8.820500920980095 both), so the reconstruction and mapping stack is training-data
independent by construction, and **all** the data sensitivity — and all the performance — lives
in the single 967 k network.

## Q5. Is another training experiment justified?

**Not yet, and not the ones on the list.**

* **Waymo + ONCE mixture** — no. One component is blocked and the other failed alone. Mixing an
  unavailable arm with a negative one has no predictable value.
* **PandaSet-only** — not now. It publishes a per-point sensor ID `d` (the field DDAD lacks), so
  the validated multisensor carve would apply, but its release cadence is unverified and the
  2 FPS problem may recur. Audit before committing.
* **Matched-data OccAny retraining** — no. 16-GPU-shaped job on 4 GPUs, unreproducible warm
  start, and for DDAD our side of the pair cannot be built. `matched_data_control_feasibility.md`
  §5 stands.
* **ONCE-only** — justified in principle and blocked in practice; it remains the right next
  training arm *if* the sweep-policy question is answered.

**The cheapest informative work is not training at all**, and two candidates stand out:

1. **Resolve the ONCE sweep policy and run the existing probe** on `train_infos.tar` plus one
   `train_lidar.tar` sequence. Small download, hours of work, and it unblocks the only claim
   currently resting on nothing (Q1).
2. **Test the `earlier_only` anomaly** (Q3). If accumulated evidence really is misregistered,
   that is a finding about our own pipeline worth more than another dataset arm — and it is
   inference-only.

---

## What changed today

1. **Phase A executed.** The completion advantage survives removing four of five input frames.
   Per-benchmark: decisive on Occ3D, a tie under OccAny's convention on SemanticKITTI.
2. **The "single-frame beats OccAny" claim was corrected mid-flight** — raw-only reading said
   +2.76 significant; the pooled convention says +0.47 with the CI containing zero. Both are
   reported and neither is presented as the answer alone.
3. **A real bug was found in my own leakage audit.** L3 used a bare inequality and `NaN != NaN`
   passes it. Fixing it surfaced 1 Occ3D anchor with no one-frame metric scale — an inherent
   limit of the one-frame protocol, since one MoGe-B candidate has no median to fall back on.
   That anchor is excluded from every arm and disclosed.
4. **ONCE audited: CONDITIONAL GO, blocked on 2 FPS.** It passes the origin gate DDAD failed.
5. **A second scorer validation.** My Phase A scorer reproduces `observability_split` and
   `occany_comparison` exactly on the shared populations, including OccAny's Occ3D 20.66 raw and
   23.55 pooled — the latter matching OccAny's published 0.2355 to five digits.

---

# Addendum — 2026-10-27: Phase E after the authorized ONCE audit

The body above is preserved as the October 20 record. It is amended in three places and
extended with answers to the current brief's Q1–Q6. Supporting documents:
`once_metadata_audit.md`, `once_supervision_validation.md`, `waymo_arm_vs_occany.md`,
`once_feasibility.md` Addendum 3.

## Amendments to the body

1. **The decision table's row *"Does ONCE pass supervision correctness gates?"*** said the hard
   temporal gate "fails as specified" because ONCE releases at 2 FPS. **Retracted.** The frozen
   window is 21 sweeps at 0.523 s spacing over **10.46 s**, so ONCE's native 0.500 s cadence
   matches the frozen spacing. The correct statement is that every correctness gate passes and
   the open item is target density.
2. **Q1's *"ONCE is blocked on the temporal gate"*** becomes: ONCE is CONDITIONAL GO; the open
   item is a ~34 % occupied-voxel deficit of verified scene-content origin.
3. **The *"next minimum experiment"* row** is complete: `train_infos.tar` was pulled, the probe
   was run on real LiDAR, and the gates were measured. Its successor is in §9 of
   `once_supervision_validation.md`.

## Q1. Did real ONCE data pass the sensor-origin, temporal-pose and occupancy-supervision gates?

**Sensor-origin: yes, empirically.** Road plane at −1.730 to −1.762 m with elevation p1
**−24.97°**, so the ray origin is on the roof and rays descend onto the road. This is the gate
DDAD failed at −1.35°. No mounting height was invented.

**Temporal-pose: yes.** Poses for **100 %** of window frames, round-trip **3.7e-13**,
timestamps strictly increasing, windows of **5 real native sweeps spanning 2.000 s** with gap
rejection above 0.75 s. Nothing interpolated or duplicated.

**Occupancy-supervision: three of four bands yes, prevalence no.** Valid 0.569 ✓, conflict
0.020 ✓ (ceiling 0.12), z-levels **32 of 32** ✓, prevalence **0.0130** against a 0.015 floor ✗
(2 of 6 sequences inside).

**Limitation that changes how the failure reads.** Under the same authorized 2.1 s window,
**KITTI-360 also falls out of bands** — vertical coverage 21 of 32 against a floor of 30. The
bands were calibrated under the frozen 21-sweep / 10.46 s recipe and are inapplicable to the
authorized recipe; each corpus fails a different one. The bands were preserved unchanged.

## Q2. Was ONCE-only training executed successfully?

**No. It was not executed at all.** Phase B returned CONDITIONAL GO and brief §10 permits
Phase C only on GO. No ONCE model, checkpoint, prediction or metric exists;
`artifacts/method_vs_data/once_arm/` was deliberately not created. The remaining port and its
cost are specified in `once_supervision_validation.md` §9.

## Q3. Did the ONCE-trained model outperform OccAny on SemanticKITTI and Occ3D?

**Unanswerable — no ONCE model exists.** No number, estimate or projection is offered.

What *is* now on the record for an OccAny source dataset is the **Waymo** arm, written up for
the first time in `waymo_arm_vs_occany.md` (the comparison had never been put in one table).
On 801 / 4,219 anchors, raw, seed 0: ours-Waymo **23.55** SemanticKITTI and **22.85** Occ3D
against OccAny's 20.32 (1-frame causal), 25.21 (5-frame forward) and 20.66 (Occ3D forward) —
**+3.23, −1.66, +2.19**. Point estimates only; the pooled convention, which is the one OccAny
publishes, was never computed for that arm.

## Q4. Does the advantage persist in genuinely never-observed geometry?

**Measured for the KITTI-360 arm; unmeasured for any OccAny-source arm.** On the strict split
(801 anchors) the lead sits in never-observed space — 89.06 % of GT surface, **+8.54** raw and
**+4.77** pooled — while `earlier_only` (6.18 %) is **+0.65 pooled with a CI containing zero**
and `anchor_observed` (4.76 %) is a **loss** (−2.70 pooled). The one-frame ablation agrees: the
completion contribution C2 is +15 to +32 across all four cells, all CIs clear of zero.

**No never-observed figure exists for the Waymo or ONCE arms**, so this does not yet answer the
brief's question for an OccAny-source arm — and the brief's rule that an aggregate gain must not
be claimed while the strict never-observed region deteriorates therefore cannot be checked for
such an arm.

## Q5. Does the experiment support a method-driven advantage, a source-data effect, or an unresolved interaction?

**A source-data effect is demonstrated; a method-driven advantage is not; the interaction is
unresolved.**

* **Source data matters, shown on our own system.** Holding architecture, loss, schedule and
  step budget fixed and changing only the corpus moved SC IoU by **−2.84 and −9.09** with CIs
  clear of zero. The reconstruction stack is data-independent by construction (`map_only` is
  bit-identical across arms at 8.820500920980095), so all of that sensitivity lives in one
  967 k-parameter network.
* **The advantage is completion, not accumulation.** Map-only scores 8.79 / 9.82 against
  OccAny's 25.21 / 20.66; completion supplies ~two thirds of our score and the lead survives
  dropping four of five input frames.
* **No architecture claim is supported.** Every OccAny comparison crosses different corpora,
  different supervision objectives, different pretraining and three orders of magnitude of
  trained capacity at once.
* **One OccAny source dataset tried, with a substantial degradation.** Two of the remaining four
  are blocked (DDAD NO-GO, ONCE CONDITIONAL GO), one is unaudited (PandaSet), one is synthetic
  (VKITTI2).

## Q6. What is the minimum next experiment needed for a defensible paper claim?

**Two candidates, in cost order. Neither is launched.**

1. **Rescore the existing Waymo arm under OccAny's pooled convention, with the paired block
   bootstrap.** No training, no new data — all 5,020 prediction volumes are intact on disk. It
   closes the only gap in the one OccAny-source result we already own, and that result is
   currently stated in the raw convention while OccAny publishes the pooled one. This is the
   cheapest experiment that changes what can be written.
2. **The ONCE arm**, if the CONDITIONAL GO is accepted as sufficient. It would establish whether
   our method reaches competitive zero-shot geometry from an OccAny source dataset other than
   Waymo, with the ~34 % supervision-density difference and the 90.1° camera stated as
   confounds.

**On the 2 Hz KITTI-360 control (brief §14).** Its premise no longer holds. The control was to
isolate supervision cadence from dataset identity by giving KITTI-360 ~5 sweeps over ~2.1 s —
but under the authorized policy that is **already the condition both corpora are in** (5 sweeps,
2.0–2.1 s), and it has been measured here without training. Sparsity has additionally been
falsified as an explanation of ONCE's statistics (85 % of KITTI-360's returns can be removed
with it still in band). **A separate 2 Hz KITTI-360 training control is therefore not
recommended**; its diagnostic value has been obtained by measurement.

**Not recommended:** OccAny retraining (16-GPU-shaped, unreproducible warm start), any
architecture change, or a further dataset arm beyond the two above.

---

# Addendum 2 — 2026-10-27: the ONCE arm, trained and compared

Everything above is preserved. Addendum 1 reported Phase C as not executed. **It has now been
executed**: the ONCE-only CompletionUNet was implemented, preprocessed, validated, trained on
three seeds and evaluated against OccAny on both benchmarks and both conventions. Sources:
`once_training_report.md`, `once_vs_occany.md`, `once_supervision_validation.md`,
`once_metadata_audit.md`, `tools/method_vs_data/once_arm/`.

**Status: exploratory.** The predeclared occupied-prevalence gate is **failed, not relabelled**
(0.0131 against [0.015, 0.040]) and is recorded as failed in every manifest the arm produced.

## Amendments to Addendum 1

* Its row *"Was ONCE-only training executed?"* — **now yes.**
* Its Q3 *"Unanswerable — no ONCE model exists"* — **now answered below.**
* Its Q6 recommendation to rescore the Waymo arm pooled still stands and is still not done.
* Two factual corrections this phase forced, both retracting earlier claims of mine:
  the camera-to-grid extrinsic was conjugated when it should have been left-multiplied
  (`A·T`, not `A·T·A⁻¹`), and **ONCE's released images are already rectified** — the
  `once_metadata_audit.md` §5 conclusion that they are raw is withdrawn.

## Q1. Does ONCE-only training outperform OccAny?

**No on SemanticKITTI in both conventions. On Occ3D it depends on the convention.** Canonical
protocol, OccAny's own anchors / labels / placement / scorer, paired block bootstrap.

| benchmark | convention | comparator | Δ | verdict |
|---|---|---|---|---|
| SemanticKITTI (801) | raw | OccAny **causal 1-frame** | −2.12 [−3.06, −1.13] * | loses |
| SemanticKITTI | raw | OccAny forward 5-frame | −7.01 [−7.97, −6.04] * | loses |
| SemanticKITTI | pooled | OccAny causal 1-frame | −3.80 [−4.69, −2.98] * | loses |
| SemanticKITTI | pooled | OccAny forward 5-frame | −5.67 [−6.68, −4.67] * | loses |
| Occ3D (4,219) | raw | OccAny forward 5-frame | −1.99 [−3.94, −0.08] * | loses, narrowly |
| **Occ3D** | **pooled** | OccAny forward 5-frame | **+6.98 [+5.57, +8.42] *** | **wins** |

Absolute: ONCE **18.20** raw / **20.26** pooled on SemanticKITTI; **18.67** raw / **30.52**
pooled on Occ3D. OccAny: 20.32 / 24.06 (1f) and 25.21 / 25.93 (5f fwd) on SemanticKITTI;
20.66 / 23.55 on Occ3D — the pooled 23.55 being their **published** figure reproduced to five
digits.

**Harness validity:** our KITTI-360 arm scores **26.40** and **31.94** raw on these two
populations, both stored canonical figures to the digit.

**Why the Occ3D sign flips:** the 3×3×3 max-pool is worth **+11.85** to the ONCE arm, +9.62 to
the KITTI-360 arm and only **+2.89** to OccAny. Dilation rewards sparse predictions and this arm
is the sparsest (raw recall 22.44 against their 28.47). Five of six cells favour OccAny; the
cell that favours us is the one scored the way they publish. **Both are reported; neither is
selected.**

## Q2. Is the advantage present in never-observed geometry?

**Yes — and this is the result worth keeping.** The mechanism that distinguishes the method
survives the corpus change even where the aggregate does not.

| region | benchmark | ONCE | OccAny | Δ |
|---|---|---:|---:|---|
| **out of frustum** | SemanticKITTI raw | **21.55** | 7.43 | **+14.13** [+13.03, +15.22] * |
| out of frustum | SemanticKITTI pooled | — | — | **+8.38** [+7.19, +9.54] * |
| **out of frustum** | Occ3D raw | **19.54** | 8.94 | **+10.60** [+8.81, +12.27] * |
| out of frustum | Occ3D pooled | — | — | **+16.82** [+15.28, +18.36] * |
| **never observed** (89.1 % of GT occ) | SemanticKITTI raw | 22.17 | 21.53 | +0.65 [−0.53, +1.89] **ns** |
| never observed | SemanticKITTI pooled | — | — | −0.04 [−1.12, +1.08] **ns** |
| anchor observed | SemanticKITTI raw | 7.51 | 15.76 | **−8.25** [−9.38, −7.19] * |
| in frustum | SemanticKITTI raw | 16.72 | 24.16 | **−7.45** [−8.51, −6.28] * |
| in frustum | Occ3D raw | 17.64 | 29.98 | **−12.34** [−15.19, −9.42] * |

**Four of four out-of-frustum cells are significant wins**, ours roughly two to three times
OccAny's. Never-observed space — 89.1 % of ground-truth occupancy — is a **statistical tie** with
their causal mode in both conventions. What collapses is everything the camera actually sees.

That also explains the benchmark asymmetry: SemanticKITTI's valid mask is 69.3 % in-frustum
against Occ3D's 51.0 %, so the in-frustum collapse costs more on SemanticKITTI, which is why the
aggregate loss is −7.01 there and only −1.99 on Occ3D raw.

## Q3. How does source training data affect the same architecture?

**Strongly, and now with three points on the axis.** Architecture, loss, schedule, step budget,
grid, mapper and evaluator held fixed; only the training corpus changed.

| training source | SemanticKITTI | Occ3D | camera HFOV | target prevalence |
|---|---:|---:|---:|---:|
| KITTI-360 (not an OccAny source) | **26.17** | **31.22** | 103.9° | 0.0276 |
| Waymo | 23.14 | 22.86 | 50.0° | 0.0222 |
| **ONCE** | **18.50** ± 0.68 | **19.29** ± 0.47 | **90.1°** | **0.0131** |

Frozen evaluator, raw, three seeds. **`map_only` is bit-identical across all three arms**
(8.8205 / 10.0351), so every difference is the one 967 k-parameter network.

**The two failing arms fail differently, so this is not one mechanism.**

* **Waymo** lost mainly in the **completed** region (Occ3D −11.64 against KITTI-360).
* **ONCE** loses mainly in the **observed** region (SemanticKITTI −14.46), with observed-region
  precision **21.82** against recall **57.41** — it over-predicts occupancy wherever the hard map
  resolved something.

The ONCE pattern is what a rarer positive class would produce through the Dice term (prevalence
0.0131 against KITTI-360's 0.0233). **That is a description confirmed on two benchmarks, not a
demonstrated cause**, and the HFOV ordering rules out field of view as the organising variable:
ONCE's 90.1° is the closest of the three to KITTI-360's 103.9° and is wider than both test
cameras, yet it performs worst. Input-distribution similarity again fails as a transfer
predictor, as it did for Waymo.

Four differences move together and none is isolated: supervision density (~34 % fewer occupied
voxels than duration-matched KITTI-360), the monocular metric-scale bias (**1.35× on ONCE against
1.20× on KITTI-360**, same diagnostic), scene content, and six independent scenes against
KITTI-360's three drives with 1,358 anchors. **No causal claim is made.**

## Q4. Does the experiment support a methodological advantage over OccAny?

Taking the brief's four categories separately:

* **Performance superiority — NOT supported from an OccAny source.** Five of six cells lose.
  It remains supported only from KITTI-360 training, which is not one of their sources.
* **Data-subset robustness — REFUTED, and this is the arm's main contribution.** Two of
  OccAny's five sources have now been tried and both failed, ONCE worse than Waymo. DDAD is a
  verified NO-GO, PandaSet unaudited, VKITTI2 synthetic. The claim that our advantage is not
  specific to KITTI-360 training is **not supported by evidence and is contradicted by two
  independent attempts**.
* **Task-specific training efficiency — still supported, and unaffected.** 6,000 steps, 24,000
  sample draws, 967 k trained parameters from scratch against ≥70,300 steps and 4.5 M draws from
  a pretrained MUSt3R plus an unrecorded warm start. The ONCE arm trains in **7.7 min per seed**.
* **Architectural causality — still NOT established**, and nothing here moves it. Different
  corpora, supervision objectives, pretraining and three orders of magnitude of trained capacity
  all differ at once.

**The one structural finding that does survive:** the out-of-frustum completion advantage is
robust to the training corpus (four of four cells, +8.4 to +16.8), while in-frustum accuracy is
not. That localizes what is and is not transferable, which is more than the aggregate numbers
say, but it is still not architectural causality.

## Q5. What is the minimum remaining experiment?

ONCE failed, so the brief directs this at the supervision-density confound. **Two candidates, in
cost order. Neither is launched.**

1. **Rescore the existing Waymo arm under the pooled convention with the paired bootstrap.**
   No training, no new data — all 5,020 Waymo prediction volumes are intact on disk. It is the
   cheapest remaining experiment and it is now *more* valuable than before: the ONCE result shows
   the pooled convention can flip an Occ3D sign, and the Waymo arm — our only other
   OccAny-source arm — is currently stated in raw only.
2. **A KITTI-360 density-matched control.** The brief asks whether the supervision-density
   confound merits a KITTI-360 two-second-target control. **It does not, in the form proposed:**
   under the authorized policy KITTI-360 at 2.1 s is *already* 5 sweeps over 2.09 s, and that was
   measured without training (`once_supervision_validation.md` §5). Sparsity is additionally
   falsified as the driver — 85 % of KITTI-360's returns can be removed with prevalence still in
   band. A control that actually bites would have to match **scene content**, by selecting
   KITTI-360 anchors by openness until prevalence reaches 0.0131. That is constructible from the
   existing cache but is a new training arm.

**On PandaSet:** it remains the only untried non-synthetic OccAny source with a published
per-point sensor ID, so the already-validated multisensor carve would apply. But two source arms
have now failed, and a third attempt is more likely to confirm the pattern than break it.
**Recommendation: do experiment 1, and treat the source-robustness claim as refuted rather than
unproven, unless a density-and-content-matched control says otherwise.**

**Not recommended:** OccAny retraining, any architecture change, or a further dataset arm.

---

## Final decision table

| Question | Result | Evidence |
|---|---|---|
| Was ONCE training completed? | **Yes** | 3 seeds × 6,000 steps, 7.7 min each, 3 distinct checkpoints `03fc0d26`/`f469f80d`/`2057b05e`, logs in `once_arm/{logs,runs}`, recipe sha256 `4ef021b6…` |
| SemanticKITTI SC IoU | **18.50 ± 0.68** (frozen evaluator); **18.20** raw / **20.26** pooled (canonical) | seeds 18.37 / 19.23 / 17.89, 163 and 801 anchors |
| Occ3D SC IoU | **19.29 ± 0.47** (frozen evaluator); **18.67** raw / **30.52** pooled (canonical) | seeds 18.78 / 19.70 / 19.38, 1,182 and 4,219 anchors |
| Advantage over OccAny | **Mixed, mostly no** — loses 5 of 6 cells; wins Occ3D pooled +6.98 | paired block bootstrap, 2,000 resamples, seed 0 |
| Never-observed completion advantage | **Yes, preserved** — out of frustum +14.13 / +8.38 (SK), +10.60 / +16.82 (Occ3D); never-observed a tie | strict visibility split (SK) and frustum split (both) |
| Method-versus-data conclusion | **Source-data effect demonstrated; source robustness refuted; architectural causality unresolved** | 3 arms, one recipe, `map_only` bit-identical |
| Next experiment | **Rescore the Waymo arm pooled** — no training, predictions on disk | `once_vs_occany.md` Addendum 3, Q5 above |
