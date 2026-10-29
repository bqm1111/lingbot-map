# Streaming-validation phase, part 2: does accuracy survive persistent state with initialization-only scale?

**Short answer.** Yes, at a cost that is small, benchmark-dependent in sign, and much smaller
than the training-seed spread.

* **SemanticKITTI (801 anchors, one physical stream).** Every head of every source and every
  seed loses accuracy: **−0.99 to −2.59 raw** and **−0.24 to −1.18 pooled** SC IoU. The
  KITTI-360 head still beats OccAny's causal one-frame configuration under both conventions;
  what does *not* survive is our win over OccAny's **non-causal** forward configuration, which
  becomes undetectable.
* **Occ3D (4,219 anchors, 150 independent scenes — the better-powered half).** The penalty does
  **not** reproduce and in the raw convention it **reverses**: KITTI-360 gains **+0.86 raw**
  (two of three seeds significantly), while pooled still loses **−0.58**. Our margin over
  OccAny's forward arm survives here (**+12.27 raw / +17.48 pooled**, both significant).

Across both benchmarks the only *consistent* effect is a small pooled loss for KITTI-360
(−0.91 SemanticKITTI, −0.58 Occ3D). On Occ3D the three-seed spread is 3.00–3.44 raw IoU —
several times any mode difference — so retraining moves this number far more than the operating
mode does. Section 9 carries the full Occ3D results.

Two findings carry more weight than the headline numbers:

1. **On SemanticKITTI the upstream map is not what degrades.** Map-only accuracy is
   statistically unchanged between the modes (raw **+0.06 [−0.49, +0.67] ns**, pooled
   **−0.31 [−0.98, +0.40] ns**); the loss appears only once a completion head reads the map — a
   head trained exclusively on clip-reset, per-clip-scaled inputs. **On Occ3D the map does
   degrade** (raw −0.47 ns but pooled **−1.20 [−2.18, −0.17] ***, and −1.04 to −2.06 raw * in
   the older and never-observed regions), and yet the *completed* predictions hold — which is
   why the completion-minus-map-only contribution **grows** under streaming there (§9.4).
2. **Persistent state buys no retrieval of older evidence.** In the region observed earlier in
   the processed stream but not in the recent five frames, the streaming mode is **worse**, not
   better: **−0.41 raw / −0.36 pooled** (both significant) on SemanticKITTI, and −0.42 ns raw
   / **−0.80 pooled *** on Occ3D's 150 independent streams. The explicit occupancy-map window
   was deliberately held at five frames, so the only channel for older evidence is LingBot's
   retained state, and it delivers nothing measurable on either benchmark.

Section 1 states what is and is not being compared; §§2–5 are the four reported comparisons on
SemanticKITTI; §6 is the visibility reinterpretation; §7 the drift diagnostics; §8 the
limitations; **§9 is the full Occ3D arm**, which is the better-powered one.

---

## 1. What the comparison is, and the two things it changes at once

| | mode A — clip-reset reference | mode B — persistent state, startup-only scale |
|---|---|---|
| reconstruction state | reset at every anchor | reset once per physical segment |
| metric scale | re-estimated from that anchor's own five frames | one scalar from the segment's first five frames, frozen |
| MoGe-2 calls | 5 per anchor (4,005 total) | **5 per segment (5 total)** |
| occupancy-map window | the five causal frames *t−4…t* | **the same five frames** |
| mapper, grid, channels, carving, read-out, threshold | frozen | **identical** |
| anchors scored | 801 / 801 canonical | 801 / 801 canonical, **zero exclusions** |

**The A/B contrast changes the reconstruction state and the calibration policy together, by
construction. No table here claims to isolate the effect of calibration frequency.** The
diagnostics in §7 support explanations; they do not establish causality.

A second asymmetry, stated because it cuts against mode B and is easy to miss: **all three
source heads were trained on mode-A inputs** — locally reset, independently scaled five-frame
clips, verified in code for KITTI-360, Waymo *and* ONCE (part 1 §2). Mode B therefore presents
every head with an input distribution it never saw in training. A mode-B deficit is as
consistent with "the head was trained on the wrong distribution" as with "persistent state
produces worse geometry", and nothing here separates those.

Statistics: paired block bootstrap, 2,000 resamples, seed 0, every arm resampled on the same
anchors in one pass. `*` = the interval excludes zero. **`ns` means no difference was
detected, which is not demonstrated equivalence.** Per-seed rows and the three-seed spread are
reported separately from the intervals throughout: the interval describes resampling anchors
at one seed, the spread describes retraining, and they are never combined.

**SemanticKITTI is ONE physical stream.** Its 801 anchors are segments of a single 815-frame,
422.9 s sequence, so the canonical 25-anchor blocks are *within*-stream blocks and the
intervals below do not estimate between-stream variation. §8 gives the block-length
sensitivity, which is the honest bound on how much to read into them.

---

## 2. Streaming minus clip-reset, within each source and seed

Global SC IoU over the whole valid volume, 801 anchors.

| source | seed | raw Δ IoU [95% CI] | pooled Δ IoU [95% CI] |
|---|---|---|---|
| KITTI-360 | 0 | −1.53 [−1.85, −1.17] * | −1.12 [−1.38, −0.84] * |
| KITTI-360 | 1 | −1.24 [−1.51, −0.95] * | −0.93 [−1.16, −0.68] * |
| KITTI-360 | 2 | −0.99 [−1.26, −0.69] * | −0.69 [−0.94, −0.44] * |
| Waymo | 0 | −2.26 [−2.73, −1.74] * | −0.71 [−1.01, −0.40] * |
| Waymo | 1 | −2.16 [−2.59, −1.70] * | −0.87 [−1.17, −0.56] * |
| Waymo | 2 | −2.59 [−3.04, −2.07] * | −1.18 [−1.52, −0.82] * |
| ONCE *(exploratory)* | 0 | −1.69 [−2.06, −1.31] * | −0.42 [−0.71, −0.12] * |
| ONCE *(exploratory)* | 1 | −1.80 [−2.18, −1.41] * | −0.56 [−0.81, −0.28] * |
| ONCE *(exploratory)* | 2 | −1.01 [−1.31, −0.69] * | −0.24 [−0.48, +0.00] ns |
| **map-only (no head)** | — | **+0.06 [−0.49, +0.67] ns** | **−0.31 [−0.98, +0.40] ns** |

Absolute values, and the three-seed spread kept apart from the intervals:

| source | mode | seed 0 | seed 1 | seed 2 | mean | max−min |
|---|---|---|---|---|---|---|
| KITTI-360 | clip-reset | 26.40 | 26.57 | 25.99 | 26.32 | 0.58 |
| KITTI-360 | **streaming** | 24.87 | 25.32 | 25.00 | **25.06** | 0.45 |
| Waymo | clip-reset | 23.55 | 22.54 | 22.84 | 22.98 | 1.01 |
| Waymo | **streaming** | 21.30 | 20.38 | 20.25 | **20.64** | 1.05 |
| ONCE | clip-reset | 18.20 | 19.03 | 17.69 | 18.31 | 1.34 |
| ONCE | **streaming** | 16.51 | 17.23 | 16.68 | **16.81** | 0.72 |

(raw convention; pooled values are in `scores/tables.md`.) The mode-B penalty (−1.25 raw on
the KITTI-360 seed mean) is **larger than the three-seed spread** (0.45–0.58), so it is not a
retraining artefact. It is also smaller than every source-to-source gap, which is why the
rankings in §3 survive.

**Reading the map-only row.** This is the single most informative row in the table. The same
mapper, the same five frames, the same carving — only the depth, the poses and the gauge
change — and the occupied map is as good as before. Whatever mode B costs, it is not paid in
the raw reconstruction's occupied evidence.

---

## 3. Source differences under each fixed inference mode

The main transfer finding is **KITTI-360 > Waymo > ONCE**, and it holds under both modes, at
every seed, under both conventions.

| contrast | mode | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| KITTI-360 − Waymo, raw | clip-reset | +2.84 * | +4.03 * | +3.15 * |
| KITTI-360 − Waymo, raw | **streaming** | +3.57 * | +4.94 * | +4.76 * |
| KITTI-360 − Waymo, pooled | clip-reset | +2.97 * | +3.35 * | +2.70 * |
| KITTI-360 − Waymo, pooled | **streaming** | +2.56 * | +3.30 * | +3.19 * |
| KITTI-360 − ONCE, raw | clip-reset | +8.20 * | — | — |
| KITTI-360 − ONCE, raw | **streaming** | +8.36 * | — | — |

(full grid with intervals in `scores/tables.md`.) Under the streaming mode the KITTI-360
advantage over Waymo **widens** on the raw convention (+2.84 → +3.57 at seed 0) and is
essentially unchanged pooled. So the source-data conclusion is not an artefact of the
clip-reset protocol: **it survives the intended robot operating mode.**

The ONCE comparison remains exploratory and remains confounded: ONCE's targets span ~2 s
against ~10 s for the other two, so a KITTI-360 − ONCE gap is not an isolated
dataset-identity result (part 1 §2).

---

## 4. Completion minus map-only, under each mode

The learned head's contribution is largely preserved, which is the claim the architecture
rests on.

| source | mode | raw Δ | pooled Δ |
|---|---|---|---|
| KITTI-360 | clip-reset | +17.61 * | +11.30 * |
| KITTI-360 | **streaming** | **+16.02 ***| **+10.49 ***|
| Waymo | clip-reset | +14.77 * | +8.33 * |
| Waymo | **streaming** | +12.45 * | +7.93 * |
| ONCE | clip-reset | +9.41 * | +5.21 * |
| ONCE | **streaming** | +7.66 * | +5.11 * |

The head still adds **+16.02 raw / +10.49 pooled** over the map it reads, under the mode a
robot would actually run. 91 % (raw) and 93 % (pooled) of the clip-reset contribution is
retained.

---

## 5. Against the fixed OccAny comparators

OccAny's forward configuration consumes **four future frames (~4.0 s of future)**; both of our
modes consume none. **These rows are not temporally matched and are not described as such.**

| comparator | source | mode | raw Δ [95% CI] | pooled Δ [95% CI] |
|---|---|---|---|---|
| causal 1-frame | KITTI-360 | clip-reset | +6.08 [+4.82, +7.33] * | +2.28 [+1.23, +3.33] * |
| causal 1-frame | KITTI-360 | **streaming** | **+4.55 [+3.35, +5.77] *** | **+1.16 [+0.20, +2.22] *** |
| causal 1-frame | Waymo | clip-reset | +3.23 [+2.26, +4.21] * | −0.69 [−1.52, +0.10] ns |
| causal 1-frame | Waymo | **streaming** | +0.98 [−0.13, +2.06] ns | −1.40 [−2.28, −0.53] * |
| causal 1-frame | ONCE | **streaming** | −3.81 [−4.92, −2.61] * | −4.22 [−5.17, −3.30] * |
| forward 5-frame | KITTI-360 | clip-reset | +1.19 [+0.35, +2.09] * | +0.41 [−0.34, +1.20] ns |
| forward 5-frame | KITTI-360 | **streaming** | **−0.34 [−1.28, +0.63] ns** | **−0.70 [−1.56, +0.15] ns** |
| forward 5-frame | Waymo | **streaming** | −3.91 [−5.00, −2.82] * | −3.26 [−4.32, −2.28] * |

**What changes.** Under the clip-reset protocol the KITTI-360 head beat the non-causal forward
arm on the raw convention (+1.19 *). Under the streaming mode that win is gone: −0.34 ns raw,
−0.70 ns pooled. Against the **causal** comparator — the like-for-like one — the win survives
on both conventions, but the pooled margin narrows from +2.28 to **+1.16**, with the interval
lower bound at **+0.20**: significant, and thin.

Waymo loses its causal-1-frame parity under streaming (pooled −1.40 *, previously −0.69 ns).
ONCE loses on every comparison in every mode.

---

## 6. Observed versus hidden, reinterpreted for a persistent-state system

The five-frame visibility masks every earlier report used **cannot** say what a persistent-state
system has never seen. Both families are therefore reported, and the difference between them is
itself a result.

`visibility/regional_semantickitti.json`, 801 anchors, 39,397 camera ray-casts (mean 49.2 per
anchor, max 134; 285,008 frames provably excluded by the exact 116.3 m prune). Same validated
construction as every earlier report: rays through every second pixel, 0.1 m steps, 80 m range,
every voxel entered up to and including the first GT-occupied voxel counts as observed. The
masks are over **every frame the stream processed**, not over KV-cache residency — frames older
than the 64-frame patch window are evicted from the cache but still shaped the retained state,
and calling a voxel never-observed only when no processed frame could see it is the
conservative choice.

| | recent five-frame window (historical) | **history-aware, whole processed stream** |
|---|---|---|
| observed in the current frame | 32.69 % valid / 4.76 % GT occ. | 32.69 % / 4.76 % |
| observed in the other four | 24.81 % / 6.18 % | 24.81 % / 6.18 % |
| observed earlier in the stream | *not represented* | **17.38 % / 6.39 %** |
| never observed, in frustum | 29.71 % / 68.05 % | **21.16 % / 64.75 %** |
| never observed, out of frustum | 12.79 % / 21.01 % | **3.96 % / 17.92 %** |
| **never observed (union)** | **42.50 % / 89.06 %** | **25.12 % / 82.67 %** |

**This materially revises how the earlier "hidden geometry" tables should be read.** Under a
persistent-state system, 17.4 % of valid voxels — and 6.4 % of ground-truth occupancy — had
been seen earlier in the stream and were previously counted as never observed. The
out-of-frustum hidden region, which the previous phase identified as the sole home of the
pooled advantage, shrinks from 12.79 % to **3.96 %** of the valid volume once history is
accounted for. For mode A the recent-window masks remain the correct description, because a
clip-reset system genuinely never saw anything else; for mode B the history-aware masks are.

### Where the streaming mode gains and loses (seed 0, 95 % CIs)

| region | share of GT occ. | KITTI-360 raw | KITTI-360 pooled | Waymo raw | ONCE raw |
|---|---|---|---|---|---|
| current frame | 4.76 % | −0.56 [−0.80, −0.29] * | −0.39 * | **+1.95 [+1.48, +2.49] *** | **+1.20 *** |
| other four recent | 6.18 % | −1.73 [−2.27, −1.25] * | −1.06 * | +0.46 ns | **+1.09 *** |
| **earlier in stream** | 6.39 % | **−0.41 [−0.67, −0.17] *** | **−0.36 *** | −1.49 * | −1.31 * |
| never obs., in frustum | 64.75 % | −1.47 [−1.97, −0.94] * | −1.08 * | −4.92 * | −3.99 * |
| never obs., out of frustum | 17.92 % | −2.41 [−3.15, −1.70] * | −2.06 * | −6.28 * | −3.59 * |
| never observed (union) | 82.67 % | −1.65 [−2.17, −1.11] * | −1.27 * | −5.20 * | −3.90 * |

Three readings, in order of how much they matter:

**(a) There is no retrieval benefit from persistent state.** In `earlier_in_stream` the
streaming mode is *worse* (−0.41 raw, −0.36 pooled, both significant), and map-only there is
flat (−0.06 ns raw). Geometry the stream saw thirty seconds ago is not recovered. This is the
expected consequence of a design choice, not a surprise: the explicit occupancy-map window was
held at five frames precisely so the comparison would be clean, leaving LingBot's retained
state as the only path for older evidence — and it carries none that reaches the output.
**A system that wants that geometry has to widen the map window, not just keep the state.**

**(b) Waymo and ONCE show a trade; KITTI-360 does not.** Both weaker sources get *better* at
the directly visible surface under streaming (+1.95 and +1.20 raw) and much worse in hidden
space (−4.92 to −6.28). The persistent pass gives the current frame more context and better
local depth, while the frozen gauge and the longer pose chain degrade extrapolation. The
KITTI-360 head loses everywhere, least at the current frame.

**(c) The map-only decomposition localises the upstream change.** Map-only improves at the
current frame (**+1.18 [+0.53, +1.87] * raw**) and degrades in the other four recent frames
(**−1.81 [−2.65, −0.89] * raw**), while being flat overall. That is precisely the signature of
a window whose individual frames are better reconstructed but whose *relative* geometry is less
self-consistent — a longer pose chain with an unrefreshed gauge.

### Mode B against the comparators, by region (KITTI-360 seed 0)

| region | vs OccAny causal 1-frame | vs OccAny forward 5-frame [+4 s] |
|---|---|---|
| current frame | −4.85 * raw / −3.09 * pooled | −0.81 * / +0.33 ns |
| other four recent | +3.26 * / −0.41 ns | +4.57 * / +2.46 * |
| earlier in stream | −0.94 ns / −2.47 * | −1.90 ns / −1.34 ns |
| **never obs., in frustum (64.8 % of GT occ.)** | **+3.48 * / +0.56 ns** | **−6.19 * / −6.58 *** |
| never obs., out of frustum (17.9 %) | +30.89 * / +28.00 * | +29.40 * / +27.30 * |
| never observed (union) | +9.09 * / +6.01 * | +0.76 ns / −0.31 ns |

The shape of the previous phase's finding is confirmed and sharpened. Against the causal
comparator, in the region holding **64.8 % of ground-truth occupancy**, the pooled difference
is **+0.56, interval containing zero — no difference detected**. The global pooled win rests on
the out-of-frustum region, which the history-aware mask shows to be only **3.96 % of the valid
volume and 17.9 % of GT occupancy**. **Out-of-frustum gains do not establish better occlusion
completion**, and with the history correction the region carrying them is smaller than the
earlier analysis implied.

### Completion minus map-only, by region (mode B, KITTI-360 seed 0)

| region | raw Δ | pooled Δ |
|---|---|---|
| current frame | **−1.43 [−2.12, −0.77] *** | −0.26 [−0.80, +0.24] ns |
| other four recent | +5.89 * | +3.03 * |
| earlier in stream | +6.51 * | +1.65 [−0.05, +3.43] ns |
| never obs., in frustum | +21.07 * | +14.52 * |
| never obs., out of frustum | +31.25 * | +23.57 * |
| never observed (union) | +23.12 * | +16.33 * |

The head is **worse than the raw map on directly visible surface** (−1.43 raw) and worth
+23.12 raw / +16.33 pooled in geometry the stream never saw. Its value is completion, not
perception — unchanged from the clip-reset finding, and it holds under the robot mode.

---

## 7. Scale and geometry drift: what actually moves

`scores/drift_semantickitti.json`. Two independent readings, both computed outside the
inference loop; neither can reach a prediction, pose, map or scale. The extra MoGe calls used
here are isolated from the primary loop and from every timing figure.

**(1) Scaled predicted motion against published odometry.** Ratio of (predicted inter-frame
translation × the gauge in force) to the sequence's own odometry translation. 1.0 is correct.

| | early third | middle third | late third | early→late shift |
|---|---|---|---|---|
| clip-reset (re-gauged every anchor) | 0.909 | — | 0.923 | **+0.013** |
| **streaming (startup-only gauge)** | **1.007** | — | **1.114** | **+0.107** |

Two things at once, and both matter. The clip-reset reference is **biased** — it
systematically under-scales motion by ~9 % — but **stable**. The startup-only gauge begins
*markedly more accurate* (1.007 against 0.909) and then **drifts to +11 % by minute seven**, an
8× larger shift. Persistent state buys a better initial metric and spends it on drift.

**(2) MoGe-2 consistency probe** (nine probes per stream, early/middle/late): the maximum
absolute log difference between the frozen gauge and what the same estimator would say later is
**0.062**, i.e. the estimator's own opinion moves by ~6.4 % over the stream. This is a
consistency check between two monocular estimates and **is not a ground-truth scale**.

**(3) Does the drift show up in accuracy?** `figures/performance_over_time.png`. Per-anchor
IoU, first quarter → last quarter of the stream:

| arm | first quarter | last quarter | change |
|---|---|---|---|
| clip-reset, raw | 25.96 | 24.75 | −1.21 |
| **streaming, raw** | 24.65 | 22.64 | **−2.01** |
| clip-reset, pooled | 26.06 | 25.29 | −0.77 |
| **streaming, pooled** | 24.94 | 23.75 | **−1.19** |

Both modes decline, so most of the decline is the sequence getting harder, not the gauge. The
**B−A gap** grows from **−1.31** (first quarter) to **−2.11** (last quarter): about **62 % of
the final deficit is present immediately and 38 % accumulates** over seven minutes. So the
measured 11 % gauge drift costs roughly −0.8 raw IoU across 422.9 s, and the larger part of the
mode-B penalty is *not* drift at all — it is the distribution shift the head sees from the
first anchor onward.

**Attribution, stated as a hypothesis.** Three candidate causes were distinguishable:
upstream scale/pose drift, reconstruction coverage change, and head behaviour on a changed
input distribution. The evidence points at the third for the constant part (map-only unchanged
globally, every head degrades, heads trained only on mode-A inputs) and the first for the
accumulating part (measured +0.107 motion-ratio shift, accuracy gap widening with time).
**Diagnostics support these explanations; they do not establish causality**, and the A/B
contrast cannot separate state from calibration because it changes both.

### 7.1 The same diagnostic on Occ3D's 150 scenes — duration, with a caveat

The drift diagnostic was rerun over all **150** Occ3D scenes (`scores/drift_occ3d.json`).
Ten scenes are too near-stationary to yield any usable motion pair and **126 of 150** have both
an early and a late third; those counts are reported rather than filled in.

| quantity (median over streams) | clip-reset | **streaming** |
|---|---|---|
| whole-stream scaled-motion ratio | — | 1.036 (p05 0.906, p95 1.176) |
| early third → late third | 1.063 → 1.053 | 1.052 → 1.005 |
| aggregate early→late shift | **−0.010** | **−0.047** |
| per-stream shift, median | −0.008 (45.6 % positive) | −0.060 (29.4 % positive) |
| per-stream **\|shift\|**, median | 0.043 | **0.086** |

**What this adds, and what it does not.** The clip-reset reference behaves as designed — its
per-stream shift is centred on zero and almost symmetric (45.6 % positive), which is what a mode
that re-estimates its gauge at every anchor should look like. The persistent mode moves
**2.0× further** over ~20 s. On SemanticKITTI it moved roughly **8× further** over 422.9 s
(+0.107 against +0.013). That the *excess* movement over the reference grows with stream length
is consistent with the frozen gauge being the accumulating term.

**But the sign is not consistent**: Occ3D's streaming ratio drifts **down** (−0.047, only 29 %
of streams positive) where SemanticKITTI's drifted **up** (+0.107). So this is not a universal
monotone inflation of the gauge; the direction is benchmark-specific, and only the *magnitude
relative to clip-reset* behaves as the duration story predicts. Two benchmarks with one duration
each cannot separate "longer stream" from "different scenes", so this remains support for a
hypothesis rather than a demonstration — §7 of `streaming_paper_decision.md` states the gap.

The isolated MoGe consistency probe (never used by inference) disagrees with the frozen gauge
more on Occ3D than on SemanticKITTI: median max \|log difference\| **0.120** (p95 0.236) against
SemanticKITTI's 0.062. Occ3D's scenes span a far wider absolute scale range (9.37–55.28), which
is the obvious candidate explanation and is not tested here.

---

## 8. Limitations

* **SemanticKITTI is one physical stream.** Block-length sensitivity on the pooled
  streaming-minus-clip-reset contrast (KITTI-360 seed 0):

  | block length | 25 | 100 | 400 | 801 |
  |---|---|---|---|---|
  | 95 % CI | [−1.38, −0.84] * | [−1.43, −0.89] * | [−1.22, **+1.81**] | [−1.12, −1.12] |

  At 400-anchor blocks the interval spans zero; at 801 it is a single block and degenerate. **No
  block length turns these into between-stream intervals.** The point estimate is stable; the
  claimed significance is within-stream only.
* **The modes change state and calibration together.** Calibration frequency is not isolated.
* **Every head was trained on mode-A inputs.** The mode-B deficit may be distribution shift
  rather than worse geometry. Resolving it needs a head trained on streaming inputs, and if that
  is ever done, **all three sources must be rebuilt the same way** — comparing a
  streaming-trained KITTI-360 head against clip-trained Waymo/ONCE heads would be worthless.
* **Only ~7 minutes was validated at keyframe interval 1**, with a code ceiling near 8.8 min at
  this cadence (part 1 §3). Nothing longer was run.
* **ONCE is exploratory** throughout: its occupied-prevalence gate still fails and its target
  window differs.
* **An interval containing zero is not equivalence.** The `ns` map-only rows bound the change at
  roughly ±0.7 raw IoU; they do not show the map is unchanged.
* **No training-seed interval exists anywhere here.** The three-seed spreads are three numbers,
  not a distribution.

---

## 9. Occ3D: 4,219 anchors over 150 independent scenes

Occ3D is the better-powered half of this phase — **150 independent validation scenes**, one
bootstrap block each, against SemanticKITTI's single physical stream. All 4,219 canonical
anchors scored with **zero exclusions**, one reset and five MoGe calls per scene, every segment
carrying a valid startup scale.

**It does not reproduce the SemanticKITTI headline, and in the raw convention it reverses it.**

### 9.1 Streaming minus clip-reset (three-seed means)

| source | raw A → B | raw Δ | pooled A → B | pooled Δ |
|---|---|---|---|---|
| KITTI-360 | 31.64 → 32.50 | **+0.86** | 40.91 → 40.33 | **−0.58** |
| Waymo | 22.81 → 21.93 | **−0.89** | 37.69 → 37.12 | **−0.57** |
| ONCE *(exploratory)* | 19.17 → 18.50 | **−0.66** | 30.68 → 30.79 | **+0.11** |
| map-only (no head) | 9.80 → 9.33 | −0.47 | 22.16 → 20.96 | −1.20 |

Per-seed, with 150-scene blocks and 2,000 resamples:

| source | seed | raw Δ IoU [95% CI] | pooled Δ IoU [95% CI] |
|---|---|---|---|
| KITTI-360 | 0 | **+0.99 [+0.18, +1.79]** * | −0.53 [−0.90, −0.15] * |
| KITTI-360 | 1 | +0.58 [−0.23, +1.39] ns | −0.69 [−1.07, −0.33] * |
| KITTI-360 | 2 | **+1.02 [+0.36, +1.67]** * | −0.53 [−0.86, −0.18] * |
| Waymo | 0 | −0.95 [−1.79, −0.10] * | −0.56 [−1.29, +0.19] ns |
| Waymo | 1 | −0.73 [−1.58, +0.10] ns | −0.48 [−1.14, +0.17] ns |
| Waymo | 2 | −0.99 [−1.87, −0.15] * | −0.68 [−1.40, +0.06] ns |
| ONCE | 0 | −0.95 [−1.50, −0.43] * | −0.19 [−0.68, +0.30] ns |
| ONCE | 1 | −0.69 [−1.23, −0.19] * | +0.27 [−0.24, +0.77] ns |
| ONCE | 2 | −0.35 [−0.88, +0.14] ns | +0.25 [−0.21, +0.74] ns |
| map-only | — | −0.47 [−1.19, +0.22] ns | −1.20 [−2.18, −0.17] * |

Two of three KITTI-360 seeds gain raw IoU **significantly**, and the third's interval contains
zero. Taken with SemanticKITTI, where all three lost −0.99 to −1.53 raw, the honest statement
is that **the sign of the raw streaming penalty is benchmark-dependent, and only the pooled
penalty is consistent** — small and negative for KITTI-360 on both benchmarks (−0.58 here,
−0.91 on SemanticKITTI).

Note also that the three-seed *spread* on Occ3D raw is 3.00–3.44 IoU for KITTI-360, far larger
than any mode difference in the table. On this benchmark retraining moves the number several
times more than the operating mode does.

### 9.2 Why raw and pooled disagree (observation, then hypothesis)

**Observed, from the precision/recall columns** (KITTI-360 seed 0): mode B raises recall and
lowers precision in both conventions — raw 77.87/35.13 → 76.84/36.56, pooled 66.49/52.57 →
64.71/52.85. Raw recall gains **+1.43** against a **−1.03** precision loss, so raw IoU rises;
pooled recall gains only **+0.28** against a **−1.78** precision loss, so pooled IoU falls.

**Hypothesis, not tested here:** the 3×3×3 dilation already recovers most of the recall that
raw scoring misses, so pooled scoring cannot be paid for the extra occupied voxels that
persistent state produces, while it still charges for them as false positives. This would make
the convention, not the geometry, the source of the sign flip. Confirming it needs a
threshold/density sweep that this phase did not run.

### 9.3 Source differences: unchanged, and no ranking moves

All **18** source contrasts are significant in raw and pooled, under both modes — same verdict
as SemanticKITTI. KITTI-360 − Waymo raw **widens** under streaming (+9.09/+10.28/+7.12 →
+11.02/+11.60/+9.12 for seeds 0/1/2); KITTI-360 − ONCE widens likewise (+13.27/+13.65/+10.51 →
+15.21/+14.93/+11.87). Waymo − ONCE is essentially unmoved (+4.18/+3.37/+3.39 →
+4.19/+3.33/+2.75).

**The source-data conclusions survive the intended robot operating mode on both benchmarks.**

### 9.4 Completion minus map-only: larger under streaming, not smaller

| source | seed | clip-reset raw | streaming raw | clip-reset pooled | streaming pooled |
|---|---|---|---|---|---|
| KITTI-360 | 0 | +22.14 | **+23.60** | +19.40 | **+20.07** |
| KITTI-360 | 1 | +23.41 | **+24.46** | +19.25 | **+19.76** |
| KITTI-360 | 2 | +19.97 | **+21.46** | +17.61 | **+18.28** |
| Waymo | 0 | +13.05 | +12.58 | +15.90 | +16.53 |
| Waymo | 1 | +13.13 | +12.87 | +14.66 | +15.38 |
| Waymo | 2 | +12.85 | +12.34 | +16.04 | +16.56 |
| ONCE | 0 | +8.87 | +8.39 | +8.37 | +9.37 |
| ONCE | 1 | +9.76 | +9.53 | +8.70 | +10.17 |
| ONCE | 2 | +9.46 | +9.59 | +8.50 | +9.95 |

All 18 intervals exclude zero. For KITTI-360 the completion contribution is **larger** in the
streaming mode in every seed and both conventions — the head is *more* valuable, not less, once
the map degrades. On SemanticKITTI the same quantity retained 91 %/93 % of its clip-reset value;
here it exceeds it.

### 9.5 Against OccAny — and what is unavailable

| comparator | mode | raw Δ IoU [95% CI] | pooled Δ IoU [95% CI] |
|---|---|---|---|
| forward 5-frame | clip-reset | +11.28 [+9.25, +13.37] * | +18.01 [+16.67, +19.34] * |
| forward 5-frame | streaming | **+12.27 [+10.21, +14.36]** * | **+17.48 [+16.20, +18.79]** * |

Unlike SemanticKITTI, where streaming lost our edge over the forward arm, on Occ3D the
KITTI-360 head beats OccAny's **non-causal** five-frame forward configuration under both modes
and both conventions, and by a wider raw margin under streaming. This comparison is still not
temporally matched: that arm consumes ~4 s of future frames and ours consumes none.

**Unavailable and not substituted:** Occ3D has **no causal one-frame OccAny comparator**. The
nuScenes preset was released and reproduced in the five-frame forward configuration only, so the
scorer drops that arm by construction (`score.py` filters `arms` to the OccAny runs that exist)
and no row in this report claims it. The causal head-to-head therefore rests on SemanticKITTI
alone.

### 9.6 History-aware visibility on Occ3D

The five-region partition over all 4,219 anchors, with the same validated ray-casting rules:

| region | share of valid | share of GT occupied |
|---|---|---|
| observed in the current frame | 37.05 % | 21.48 % |
| observed in the other four recent frames | 20.43 % | 16.77 % |
| observed earlier in the stream only | 13.92 % | 13.83 % |
| never observed, inside the anchor frustum | 9.40 % | 18.26 % |
| never observed, outside that frustum | 19.21 % | 29.66 % |
| **preserved union — never observed in the stream** | **28.60 %** | **47.92 %** |

Accounting for the full processed history shrinks the never-observed region from the
recent-window figure of **42.52 % → 28.60 %** of valid voxels (**61.75 % → 47.92 %** of GT
occupied), and the out-of-frustum region from **31.20 % → 19.21 %**. The direction and rough
magnitude match SemanticKITTI (42.50 % → 25.12 % of valid), so the correction is not an
artefact of one benchmark.

### 9.7 Prediction, not retrieval — confirmed on 150 independent streams

KITTI-360 seed 0, by region:

| region | streaming − clip-reset (raw / pooled) | map-only streaming − clip-reset (raw / pooled) |
|---|---|---|
| current frame | **+3.64** * / +0.11 ns | +0.74 ns / +1.36 ns |
| other four recent | **+2.10** * / −1.16 * | +0.65 ns / +0.17 ns |
| **earlier in stream only** | **−0.42 ns / −0.80 *** | **−1.04 * / −2.35 *** |
| never observed, in frustum | −0.51 ns / −0.78 * | −2.06 * / −3.89 * |
| never observed, in stream | −0.32 ns / −0.61 * | −1.29 * / −2.74 * |

The conclusion holds and is now backed by 150 independent streams: **persistent state buys no
retrieval of older evidence.** On geometry observed earlier in the stream but outside the recent
five frames, the streaming mode is flat in raw and **worse** in pooled, exactly as on
SemanticKITTI. The map-only column shows the reconstruction itself degrades there (−1.04 raw *)
and degrades more on never-observed geometry (−2.06 raw * in frustum) — yet the completed
predictions do not, which is why §9.4's completion contribution grows.

What the head contributes on validated full-stream never-observed geometry, mode B, seed 0:

| contrast | raw Δ IoU [95% CI] | pooled Δ IoU [95% CI] |
|---|---|---|
| completion − map-only | **+24.56 [+22.60, +26.47]** * | **+27.28 [+24.81, +29.61]** * |
| vs OccAny forward 5-frame | **+11.20 [+8.76, +13.65]** * | **+14.79 [+12.67, +16.87]** * |

Both exceed their SemanticKITTI counterparts (+23.12 raw / +16.33 pooled, and +9.09 / +6.01
against the *causal* arm). The gain is prediction of geometry unseen anywhere in the processed
stream, on both benchmarks.

### 9.8 What Occ3D cannot test

Occ3D's startup scales span 9.37–55.28 (median 24.48) across the 150 scenes, and each scene is
only ~20 s. The drift exposure is an order of magnitude smaller than SemanticKITTI's, so the
Occ3D result is a well-powered test of **short-horizon** persistent state and **not** a test of
long-horizon gauge stability. The two benchmarks are complementary in exactly this way:
SemanticKITTI has the duration and one stream; Occ3D has 150 streams and little duration.
Neither has both, and no run in this phase provides a long-horizon multi-stream measurement.

---

## 10. Reproduction

```bash
export CUDA_HOME=/usr/local/cuda-12.8 FLASHINFER_CUDA_ARCH_LIST="12.0a"
export PATH=/home/minh/anaconda3/envs/cu128/bin:/usr/local/cuda-12.8/bin:$PATH
cd /home/minh/workspace/lingbot-map_fork

# 22 arms (2 modes x 3 sources x 3 seeds + 2 map-only + 2 OccAny), one pass, both conventions
PYTHONPATH=$PWD python tools/method_vs_data/streaming/score.py \
    --dataset semantickitti --device cuda:3 --sk-stream-sensitivity
PYTHONPATH=$PWD python tools/method_vs_data/streaming/score.py --dataset occ3d --device cuda:1

# history-aware visibility (five mutually exclusive regions + the recent-window family)
PYTHONPATH=$PWD python tools/method_vs_data/streaming/visibility.py \
    --dataset semantickitti --device cuda:2

# drift diagnostics (ground truth read here and nowhere else)
PYTHONPATH=$PWD python tools/method_vs_data/streaming/drift.py \
    --dataset semantickitti --device cuda:2

# tables and figures
PYTHONPATH=$PWD python tools/method_vs_data/streaming/tables.py
PYTHONPATH=$PWD python tools/method_vs_data/streaming/figures.py --which all
```

Artifacts: `scores/streaming_semantickitti.json` (+ `_per_anchor.npz` with every arm's
per-anchor TP/FP/FN), `scores/drift_semantickitti.json`, `scores/tables.md`,
`scores/metrics.csv` (918 rows, both benchmarks), `scores/streaming_occ3d.json`
(+ `_per_anchor.npz`), `visibility/regional_semantickitti.json` and
`visibility/regional_occ3d.json` (each + `_per_anchor.npz`, `_raycast.json`),
`scores/drift_semantickitti.json` and `scores/drift_occ3d.json`,
`manifests/identities.json` (git rev, sha256 of all nine completion
checkpoints, the LingBotMap checkpoint and every file of the phase's code),
`figures/regional_semantickitti.png`, `figures/regional_occ3d.png`,
`figures/performance_over_time.png`,
`figures/accuracy_vs_latency.png`, `figures/qualitative_panel_raw.png` (+ `_raw.json`,
`panel_ranking.json`), `figures/figure_data.json`.

`performance_over_time.png` and `accuracy_vs_latency.png` are SemanticKITTI-only by design: a
drift-over-time curve needs a long stream, and Occ3D's scenes are ~20 s each.

### The qualitative panel

`figures/qualitative_panel_raw.png`, five anchors at fixed quantiles (0.03, 0.27, 0.50, 0.73,
0.97) of the **recorded** per-anchor streaming-minus-clip-reset ranking in `panel_ranking.json`,
so the selection is reproducible and **includes failures**: q=0.03 is `08_003055`, where
streaming loses 6.1 points (24.9 → 18.7); q=0.97 is `08_002375`, where it gains 2.6
(25.3 → 27.8). Six columns: ground truth coloured by **history-aware** visibility region,
map-only, clip-reset completion, persistent-state completion, OccAny's causal 1-frame arm, and
a direct streaming-versus-clip-reset difference.

The first column is the point of the panel. The orange *never in frustum* and purple *never out
of frustum* regions dominate the ground truth, while the blue *earlier in the stream* band is
thin — the visual form of the 6.39 % / 82.67 % split in §6. Read against columns 2–4 it
separates the two mechanisms the brief asked to distinguish: what the completion head adds over
its own map sits on the orange and purple voxels (geometry unseen throughout the processed
stream), not on the blue ones (older evidence the system did see). Per-anchor region counts,
IoUs and the ray-cast tallies are in `figures/qualitative_panel_raw.json`; the
`frames_cast / frames_in_prefix` figures there (e.g. 44 of 612) show the exact-prune at work.

## 11. What this part does NOT claim

* Not that persistent state with a startup-only gauge is equivalent to clip-reset. It costs
  **−1.25 raw / −0.91 pooled** on the KITTI-360 seed mean, consistently and significantly.
* Not that the loss is caused by the calibration policy. The contrast changes two things.
* Not that retraining is necessary. No streaming-trained head was built, so the deficit's
  cause is unresolved and the distribution-shift hypothesis is untested.
* Not that the streaming mode is validated beyond ~7 minutes at this cadence.
* Not that the out-of-frustum advantage over OccAny is occlusion reasoning. It is coverage, in
  a region the history-aware mask shows to be 3.96 % of the valid volume.
