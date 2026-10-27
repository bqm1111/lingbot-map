# One frame versus five frames — does the completion advantage need temporal input?

**Run 2026-10-20.** Raw: `artifacts/method_vs_data/results/one_frame_{semantickitti,occ3d}_{raw,pooled}.json`,
`one_frame_leakage_audit_{semantickitti,occ3d}.json`. Code: `tools/method_vs_data/one_frame.py`,
`one_frame_predict.py`, `one_frame_score.py`, `one_frame_leakage_audit.py`.

SemanticKITTI-08: 801 anchors, 33 blocks. Occ3D-nuScenes: 4,218 anchors (1 excluded, §8),
150 scene blocks. Track O throughout. No retraining; the frozen seed-0 CompletionUNet
(sha256 `7f5e853dd05b…`, verified against `code_freeze.json`) is used in every arm.

**Both postprocessing conventions are reported for every number**, because they disagree on
two of the five contrasts. OccAny's published figures are the pooled ones.

> ## Headline
>
> **Learned completion over a one-frame map is worth +15 to +32 SC IoU — significant in all
> four benchmark × convention cells.** That is the one unambiguous result here.
>
> **Does our single-frame completion outperform OccAny? It depends on the benchmark:**
> * **SemanticKITTI, matched one-frame vs one-frame:** +2.76 [+1.60, +3.94] raw but
>   **+0.47 [−0.45, +1.43] pooled — a tie** under OccAny's own convention.
> * **Occ3D, our one frame vs their five forward frames** (off-setting in *their* favour):
>   **+6.81 [+4.75, +8.94] raw and +18.37 [+16.93, +19.79] pooled — yes, decisively.**
>
> **Removing four of five frames costs 1.8–4.5 SC IoU in three of four cells, and nothing
> measurable in the fourth** (Occ3D pooled, +0.35 [−0.18, +0.89]).
>
> **Our advantage over OccAny is not temporal accumulation.** The map alone scores 2.90–22.16
> against OccAny's 20.66–24.06, and on Occ3D our *one-frame* arm matches our five-frame arm
> against OccAny under pooling (+18.37 vs +18.01).

---

## 1. Arms, and what is frozen

| arm | input | completion | note |
|---|---|---|---|
| **A0** `five_frame_unet` | five causal frames | frozen CompletionUNet | the deployed system; read from the existing canonical predictions, anchoring it to the published 26.396948 |
| **A1** `one_frame_unet` | **one** causal frame | same frozen CompletionUNet | a five-frame-trained model given one frame — **not** a model optimized for single-frame inference |
| **A2** `one_frame_map` | one causal frame | none (hard map `S` only) | |
| — `five_frame_map` | five causal frames | none | the published map-only floor |
| **A3** `occany_1f` | one frame | OccAny | [reproduced] 1-frame causal preset. **SemanticKITTI only** — no one-frame nuScenes run exists |
| — `occany_5f_fwd` | five **forward** frames | OccAny | [reproduced] not causal. The Occ3D comparator `occany_comparison` uses for its headline number |
| **D** `one_frame_unet_scale5` | one causal frame | frozen CompletionUNet | **labelled diagnostic**, five-frame scale forced in. Deliberate leakage; not a protocol arm |

A1 is a true one-frame arm, not a slice of the five-frame clip. LingBotMap was **re-run** on the
single anchor image (KV cache cleared, RoPE counter asserted at 0; `inference_streaming` clamps
`num_scale_frames` to the sequence length, so one image is a supported configuration). The
metric scale is `exp(log_s)` of that frame's own MoGe-B candidate through the identical
`gate7b.scale.frame_candidate` call — with one candidate the frozen `exp(nanmedian(log_s))`
reduces to this, so the rule is unchanged rather than relaxed.

### Scorer validation

My scorer reproduces the established tools exactly on the shared populations:

| | this tool | established | diff |
|---|---:|---:|---:|
| SemanticKITTI, OccAny 1f, raw | 20.32 | 20.319705 (`observability_split`) | 0 |
| SemanticKITTI, A0, raw | 26.40 | 26.396948 (`observability_split`) | 0 |
| Occ3D, OccAny 5f fwd, raw | **20.66** | 20.66 (`occany_comparison`) | 0 |
| Occ3D, OccAny 5f fwd, pooled | **23.55** | 23.55 (`occany_comparison`), = OccAny's published 0.2355 | 0 |

---

## 2. Input isolation — audited, not asserted

All 801 SemanticKITTI and 4,219 Occ3D samples; all eight checks pass on both.

| check | result |
|---|---|
| L1 one slot, one scale candidate, `anchor_slot=0` | PASS, 0 failures |
| L2 the clip's single key is the anchor's own | PASS, 0 failures |
| L3 one-frame scale finite **and** different from the five-frame scale | PASS, 0 failures |
| L4 one-frame depth differs from the five-frame anchor slot | PASS, 0 failures |
| L5 stored prediction records `n_input_frames=1` and the one-frame scale | PASS, 0 failures |
| L6 the diagnostic arm carries the five-frame scale, as designed | PASS, 0 failures |
| L7 MoGe-B path keyed on `(dataset, key)` only — no window argument | PASS |
| L8 frozen CompletionUNet sha256 matches `code_freeze.json` | PASS |

L3 originally used a bare inequality, which `NaN != NaN` passes silently. It was corrected to
require both scales finite and to count non-finite cases apart — and that correction is what
surfaced §8's excluded anchor. The bug would otherwise have let a scale-less sample through.

**Why slicing the five-frame clip would have been wrong**, quantified:

| | min | median | max |
|---|---:|---:|---:|
| one-frame ÷ five-frame scale, SemanticKITTI | 0.489 | **0.714** | 1.055 |
| one-frame ÷ five-frame scale, Occ3D | 0.322 | **0.792** | 1.687 |
| mean &#124;depth difference&#124;, SemanticKITTI | 0.050 | **0.254** | 0.876 |
| mean &#124;depth difference&#124;, Occ3D | 0.008 | **0.250** | 2.108 |

The five-frame clip's anchor depth comes from a pass that attended across all five images and
its scale is a median over five MoGe-B candidates, so slot 4 of it would have been five-frame
geometry wearing a one-frame label.

**The one admissible shared input** is the per-frame MoGe-B depth cache; L7 establishes it is
keyed only on `(dataset, frame_key)` with no window argument, so it cannot carry temporal
content.

---

## 3. Observation coverage each arm actually has

Share of GT-occupied surface the arm's own map observed:

| arm | SemanticKITTI | Occ3D |
|---|---:|---:|
| A0 / five_frame_map | **15.37 %** | **32.89 %** |
| A1 / A2 (one frame) | **6.49 %** | **12.53 %** |
| D (one frame, five-frame scale) | 13.55 % | 14.15 % |

One frame sees 2.4× less surface than five on SemanticKITTI, 2.6× less on Occ3D.

---

## 4. The five contrasts, all four cells

| contrast | SK raw | SK pooled | Occ3D raw | Occ3D pooled |
|---|---|---|---|---|
| **C1** A1 − A0, remove temporal input | **−3.31** [−3.89, −2.74]* | **−1.81** [−2.35, −1.29]* | **−4.47** [−5.13, −3.89]* | **+0.35** [−0.18, +0.89] |
| **C2** A1 − A2, completion over the 1-frame map | **+19.43** [+18.45, +20.51]* | **+15.39** [+14.36, +16.45]* | **+24.58** [+23.02, +26.16]* | **+32.01** [+30.29, +33.73]* |
| **C3** A1 − A3, ours 1f vs **OccAny 1f** | **+2.76** [+1.60, +3.94]* | **+0.47** [−0.45, +1.43] | n/a | n/a |
| **C3b** A1 − OccAny 5f fwd | n/a | n/a | **+6.81** [+4.75, +8.94]* | **+18.37** [+16.93, +19.79]* |
| C0 A0 − OccAny 1f | +6.08 [+4.82, +7.33]* | +2.28 [+1.23, +3.33]* | n/a | n/a |
| C0b A0 − OccAny 5f fwd | n/a | n/a | +11.28 [+9.25, +13.37]* | +18.01 [+16.67, +19.34]* |
| **D** scale5 − A1, metric scale only | **−0.60** [−1.08, −0.08]* | **+0.86** [+0.48, +1.23]* | **−4.77** [−5.39, −4.20]* | **−1.96** [−2.43, −1.50]* |

Paired block bootstrap, 2,000 resamples, seed 0; 25 contiguous anchors per block on
SemanticKITTI, one block per scene on Occ3D; every arm resampled on the same anchors.
`*` = CI excludes zero.

**C3 and C3b are different comparisons and must not be averaged.** C3 is matched — both
systems get one frame. C3b gives OccAny *five* frames including future ones and still favours
us; it is off-setting in their favour, and there is no one-frame nuScenes run to make it
matched.

---

## 5. SC IoU by region

### SemanticKITTI, raw / pooled

| population | A0 | A1 | A2 1f map | 5f map | D | OccAny 1f | OccAny 5f fwd |
|---|---|---|---|---|---|---|---|
| **all** | 26.40 / 26.34 | 23.08 / 24.53 | 3.65 / 9.15 | 8.79 / 15.04 | 22.48 / 25.39 | 20.32 / 24.06 | 25.21 / 25.93 |
| anchor_observed | 11.47 / 9.02 | 11.38 / 9.14 | 11.93 / 13.09 | 11.16 / 9.57 | 10.84 / 11.46 | **15.76 / 11.72** | 11.72 / 8.30 |
| earlier_only | 17.95 / 14.57 | **20.01 / 17.53** | 3.80 / 8.85 | 12.14 / 11.91 | 19.53 / 18.28 | 12.97 / 13.92 | 11.65 / 11.06 |
| never_observed | **30.07 / 32.37** | 25.46 / 29.03 | 2.84 / 8.43 | 7.86 / 17.67 | 23.93 / 28.18 | 21.53 / 27.60 | 29.02 / 32.80 |
| in_frustum | 25.16 / 25.76 | 22.34 / 23.93 | 4.60 / 11.12 | 8.51 / 15.42 | 22.01 / 25.19 | **24.16 / 27.33** | 29.55 / 29.01 |
| out_frustum | **29.98 / 28.01** | **25.57 / 26.56** | 0.75 / 2.45 | 9.50 / 14.18 | 24.08 / 26.07 | 7.43 / 11.85 | 8.04 / 11.59 |

### Occ3D, raw / pooled

| population | A0 | A1 | A2 1f map | 5f map | D | OccAny 5f fwd |
|---|---|---|---|---|---|---|
| **all** | 31.94 / 41.56 | 27.47 / **41.92** | 2.90 / 9.90 | 9.81 / 22.16 | 22.70 / 39.96 | 20.66 / 23.55 |
| in_frustum | 36.18 / 46.46 | 33.96 / 46.92 | 5.66 / 18.06 | 11.59 / 26.89 | 26.29 / 43.86 | 29.98 / 30.72 |
| out_frustum | 28.25 / 37.09 | 21.75 / 37.15 | **0.54 / 2.59** | 8.26 / 17.87 | 19.58 / 36.37 | 8.94 / 12.55 |
| A0fixed_observed | 41.82 / 54.60 | 36.24 / 55.44 | 7.77 / 24.30 | 26.31 / 44.97 | 24.16 / 49.90 | 28.05 / 34.53 |
| A0fixed_completed | 27.36 / 35.71 | 23.21 / 35.36 | 0.45 / 2.30 | **0.00** / 7.29 | 22.01 / 35.37 | 17.75 / 19.61 |

`five_frame_map` scoring exactly **0.00** on `A0fixed_completed` (raw) is a structural check,
not a result: that stratum is by definition what the five-frame map did not observe. It
confirms the mask semantics.

---

## 6. What the regional pattern shows

**The loss from removing frames is confined to extrapolated space.** On SemanticKITTI, C1 by
region: `anchor_observed` −0.09 [−0.30, +0.13] raw and +0.12 [−0.00, +0.25] pooled — **both
non-significant**; `never_observed` −4.61 [−5.18, −4.05]* raw and −3.34 [−3.97, −2.75]*
pooled. Removing four frames does not degrade what the model does with what it can still see;
it degrades what it invents.

**The one-frame arm is robustly *better* on `earlier_only`** — the voxels an earlier frame saw
and the anchor did not — by **+2.06** [+1.34, +2.66]* raw and **+2.96** [+2.50, +3.42]*
pooled. The five-frame arm has accumulated map evidence there and still loses. The natural
explanation is that accumulated evidence in those regions is partly **misregistered** — pose
drift and scale error placing surfaces slightly wrong — so clean prior-driven completion beats
slightly-wrong evidence. **This is unverified** and it is the most consequential anomaly here,
because it questions whether temporal accumulation is net-positive at all.

**Completion manufactures out-of-frustum geometry from nothing.** The one-frame map is
frustum-limited by construction and scores **0.75** (SK raw) and **0.54** (Occ3D raw) out of
frustum; the one-frame network reaches **25.57** and **21.75** there. OccAny gets 7.43 and
8.94.

**But in-frustum, OccAny is better on SemanticKITTI** — 24.16 / 27.33 against our 22.34 /
23.93. C3's advantage is entirely out-of-frustum, exactly as in the five-frame comparison. On
Occ3D the in-frustum picture reverses under pooling (46.92 vs 30.72 ours).

**A caution on what "completion" means out of frustum.** Much of SemanticKITTI's out-of-frustum
GT surface is road, sidewalk and terrain continuing beside and behind the vehicle.
Extrapolating a ground plane is a real geometric prior but a modest one, and a reviewer will
ask what share of the out-of-frustum advantage is that. This decomposition does not answer it,
and the answer matters for how the result is framed.

---

## 7. The scale diagnostic — four cells, and a correction to my own earlier reading

The brief asks whether a one-frame regression is caused by reduced coverage, one-frame metric
scale, or train/inference distribution shift. Diagnostic D targets the second:

| cell | D = forcing the five-frame scale into A1 |
|---|---|
| SemanticKITTI raw | **−0.60** [−1.08, −0.08]* |
| SemanticKITTI pooled | **+0.86** [+0.48, +1.23]* |
| Occ3D raw | **−4.77** [−5.39, −4.20]* |
| Occ3D pooled | **−1.96** [−2.43, −1.50]* |

**Three of four cells are negative, and the two largest magnitudes are negative.** So forcing a
mismatched scale generally *hurts*, sometimes substantially. The plausible reading is
self-consistency: the one-frame depth and the one-frame scale are estimated from the same
single observation, and pairing a depth field with a scale derived from different evidence
breaks that pairing.

**Two corrections to how I reported this earlier in the session, both recorded here because the
reasoning matters more than the number.** First I read only SemanticKITTI raw (−0.60) and
concluded metric scale was not the cause of the regression. Then, seeing SemanticKITTI pooled
(+0.86), I over-corrected to "the sign flips, direction unresolved" on the strength of two
cells. With all four cells measured, the supported statement is: **a mismatched scale is
generally harmful, with one small positive exception, and the effect is not the dominant term
in C1** (|D| ≤ 4.77 against C1 up to −4.47, and on Occ3D raw the two are comparable).

What remains unseparated is **coverage versus train/inference distribution shift**. The model
was trained only on five-frame maps, so every one-frame input is simultaneously sparser *and*
off-distribution. Separating them needs a one-frame-trained CompletionUNet — new training,
which was not authorized.

---

## 8. One disclosed exclusion

**`scene-1064_9ad02057efee422b9845f563b8565485`** is excluded from the Occ3D comparison — 1 of
4,219 anchors, 0.024 %. Its one-frame MoGe-B candidate is non-finite, so the one-frame protocol
has no metric scale for it: with a single candidate there is no median to fall back on, and the
frozen loader correctly refuses the clip.

This is an inherent property of the one-frame protocol, not a defect in the sample — the frozen
five-frame builder tolerates one bad candidate out of five and a one-frame builder cannot. The
anchor is dropped from **every** arm so the bootstrap stays paired, and the scorer records the
exclusion and its reason in its JSON output. SemanticKITTI had **0** such anchors.

---

## 9. Interpretation

**Supported — the advantage over OccAny is not temporal accumulation.** Map-only floors are
3.65 / 9.15 (SK one frame), 8.79 / 15.04 (SK five frames), 2.90 / 9.90 and 9.81 / 22.16 on
Occ3D — all below OccAny's 20.66–24.06 except the pooled five-frame Occ3D map. Accumulation is
not what closes the gap. And on Occ3D pooled our one-frame arm beats OccAny by as much as our
five-frame arm does (+18.37 vs +18.01).

**Supported — the completion mechanism degrades gracefully.** Removing 80 % of input frames and
~2.5× of observed surface costs 1.8–4.5 SC IoU in three cells and nothing measurable in the
fourth, with no loss at all on what the remaining frame sees.

**Supported, per benchmark — the Phase A question.** On Occ3D our one frame beats OccAny's five
forward frames, significantly, in both conventions. On SemanticKITTI the matched one-frame
comparison is a win under raw and a **tie** under OccAny's own convention.

**Not supported — "our single-frame completion outperforms OccAny", unqualified.** It depends
on benchmark and convention (§4).

**Not supported — architectural superiority.** Every OccAny contrast crosses different
pretrained weights, training corpora, supervision objectives and three orders of magnitude of
trained capacity. The brief says C3 does not isolate architecture; neither does C3b.

**Not supported — that temporal accumulation is useless.** C1 is negative and significant in
three of four cells. Five frames beat one on SemanticKITTI in both conventions. The claim is
only that the *advantage over OccAny* does not require them — with §6's `earlier_only` anomaly
as an unexplained counter-current.

---

## 10. Statistical uncertainty and limits

* All intervals are paired block bootstraps on the **difference**, 2,000 resamples, seed 0,
  every arm resampled on the same anchors.
* **Single geometry seed.** Every arm uses `GEOMETRY_SEED = 0`, the canonical pipeline's fixed
  choice, so no interval here contains model-seed variance. Our three-seed spread is ±0.30 on
  SemanticKITTI and ±1.74 on Occ3D SC IoU. That is comfortably below C1, C2, C3b and C0b, but
  **comparable to or larger than SK-pooled C3 (+0.47), SK-raw D (−0.60) and SK-pooled D
  (+0.86)** — those three should be read as within seed noise, which is part of why §7 does not
  lean on the SemanticKITTI D values.
* **OccAny is a single released checkpoint** with no seed spread, and its one-frame
  configuration is **off-setting** for it: its own model selection used a 10-view KITTI
  validation set (`occany_training_audit.md` §1.5).
* **No matched one-frame comparator on Occ3D.** No one-frame nuScenes OccAny run exists, so
  C3b substitutes their five-frame forward preset — more input than ours, including future
  frames.
* **The strict anchor/earlier/never split is SemanticKITTI-only**, because it needs GT poses for
  all five permitted frames; that chain exists for sequence 08. Occ3D carries `all`, frustum
  and A0-fixed strata only.
* **`never_observed` is never redefined between arms.** The fixed strata come from a GT raycast
  over the five permitted frames' GT poses, identical for every arm; each arm's own visibility
  is in §3.
* **Pooling is strongly asymmetric** — on SemanticKITTI it adds +3.74 to OccAny's sparse output
  against −0.05 to our dense one — and it is what flips C3 and SK-raw D. Neither convention is
  privileged; OccAny's published numbers are pooled.
