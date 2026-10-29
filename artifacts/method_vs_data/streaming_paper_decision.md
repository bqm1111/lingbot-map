# What the streaming-validation phase does to the paper's claims

**2026-10-28.** Repository `/home/minh/workspace/lingbot-map_fork`, branch
`feature-completion-study`, HEAD `db465c2ce2a692817d9ce4a141df7c24c4fa2d13`. Nothing was
trained, fine-tuned, downloaded or architecturally changed in this phase; nine existing
checkpoints were evaluated under a second inference mode.

Reading order: `streaming_input_provenance.md` (what the modes are and why they are faithful),
`streaming_accuracy_comparison.md` (the numbers), `streaming_runtime_comparison.md` (the cost).
This file is the decision.

---

## 1. The question this phase asked

The paper's results were produced under a **clip-reset** procedure: every evaluation anchor got
its own five-frame clip, its own reconstruction reset, and its own metric-scale estimate from
five fresh MoGe-2 calls. A robot cannot run that way. It carries persistent state and calibrates
metric scale once, at startup.

So: **do the completion models, and the source-data conclusions drawn from them, survive the
intended operating mode?** Not "is startup-only scale good enough" — that was the thing to find
out, and the answer could have been no.

## 2. Answer, in one paragraph

Persistent state with initialization-only scale was implemented and validated, and it **costs
little — less than retraining does, and not even consistently in sign**. On SemanticKITTI's full
canonical 801 anchors the KITTI-360 three-seed mean falls from 26.32 to 25.06 raw SC IoU
(**−1.25**, from unrounded means — the displayed figures subtract to 1.26) and from 25.89 to
24.98 pooled (−0.91); all nine heads lose. On **Occ3D's 4,219 anchors over 150 independent
scenes — the better-powered benchmark — the raw penalty does not reproduce and reverses**:
KITTI-360 rises from 31.64 to 32.50 raw (**+0.86**, two of three seeds significantly) while
pooled falls from 40.91 to 40.33 (−0.58). The only effect consistent across both benchmarks is
a **small pooled loss of 0.58–0.91** for KITTI-360, against an Occ3D three-seed spread of
3.00–3.44 raw IoU. And the mode costs **2.96× less compute per completed occupancy grid**
(368.6 ms → 124.5 ms). **Every conclusion the paper draws about source data survives unchanged
on both benchmarks** — all 18 contrasts significant on each — and so does the completion head's
contribution over its own map, which on Occ3D actually **grows** under streaming. What does not
survive everywhere is the win over OccAny's non-causal forward configuration: it becomes a tie
on SemanticKITTI, though it holds on Occ3D. The paper's claims hold; one of them has to be
restated to name its operating mode, and the penalty must be quoted per benchmark rather than as
one number.

---

## 3. Claim by claim

### 3.1 "The completion head adds large, real geometry over its own reconstruction" — **SURVIVES**

| | clip-reset | persistent state | retained |
|---|---|---|---|
| completion − map-only, raw (K360 s0) | +17.61 | **+16.02** [+15.17, +16.90] * | 91 % |
| completion − map-only, pooled (K360 s0) | +11.30 | **+10.49** [+9.53, +11.47] * | 93 % |

Significant for all three seeds and both conventions. The map itself is **statistically
unchanged** by the mode (raw +0.06 [−0.49, +0.67] ns; pooled −0.31 [−0.98, +0.40] ns), which is
what makes this a clean statement about the head rather than about the upstream.

### 3.2 "Source data, not architecture, drives completion quality" — **SURVIVES, unweakened**

K360 > Waymo > ONCE, under both modes, for all three seeds, in both conventions. **All 18
source contrasts are significant in raw and in pooled.** The KITTI-360 advantage does not shrink
under the robot mode — it grows:

| contrast (seed 0) | clip-reset | persistent state |
|---|---|---|
| K360 − Waymo, raw | +2.84 * | **+3.57** * |
| K360 − Waymo, pooled | +2.97 * | +2.56 * |
| K360 − ONCE, raw | +8.20 * | **+8.36** * |
| Waymo − ONCE, raw | +5.36 * | +4.79 * |

The ranking is the paper's claim and the ranking is identical. This is the single most important
result of the phase, because the source comparison is what the paper is *about*, and it was the
thing most at risk: all nine heads were trained on clip-reset inputs, so a mode change could in
principle have reordered them. It did not.

### 3.3 "We beat a strong published baseline" — **SURVIVES vs causal; vs forward LOST on SemanticKITTI, HELD on Occ3D**

| | clip-reset | persistent state |
|---|---|---|
| vs OccAny, **1 causal frame**, raw | +6.08 * | **+4.55** [+3.35, +5.77] * |
| vs OccAny, **1 causal frame**, pooled | +2.28 * | **+1.16** [+0.20, +2.22] * |
| vs OccAny, 5 **forward** frames, raw | +1.19 * | **−0.34** [−1.28, +0.63] ns |
| vs OccAny, 5 **forward** frames, pooled | +0.41 ns | **−0.70** [−1.56, +0.15] ns |

Against the configuration where both systems see only the past, we win on both conventions and
are **10.8× faster per completed grid** (124.5 ms against 1346.6 ms). That is the comparison a
robotics claim should rest on, and it holds.

Against OccAny's forward configuration — which consumes **four seconds of future frames** — the
clip-reset win of +1.19 raw becomes a −0.34 tie. The honest statement is that our persistent
causal mode matches a non-causal method at 13.5× less compute. **The pooled interval
[−1.56, +0.15] contains zero; that does not establish equivalence**, and I am not claiming it
does. It establishes that the clip-reset advantage over that configuration is not available in
the robot mode.

**On Occ3D the forward comparison does not go away** — the KITTI-360 head beats that same
non-causal configuration under both modes and both conventions, and by a *wider* raw margin in
the persistent mode:

| | clip-reset | persistent state |
|---|---|---|
| vs OccAny, 5 **forward** frames, raw | +11.28 * | **+12.27** [+10.21, +14.36] * |
| vs OccAny, 5 **forward** frames, pooled | +18.01 * | **+17.48** [+16.20, +18.79] * |

So the lost forward-mode win is a SemanticKITTI result, not a general one. **Occ3D has no causal
one-frame OccAny comparator** — the nuScenes preset exists only in the five-frame forward
configuration, the scorer drops that arm by construction, and nothing here substitutes for it.
The causal head-to-head therefore rests on SemanticKITTI alone, which is a real limit on the
strongest robotics claim.

### 3.4 "The advantage is on geometry the cameras never saw" — **SURVIVES, and is now properly validated**

This is where the phase did the most work, because the old masks could not support the claim at
all once state persists. Five-frame visibility masks cannot establish that a persistent-state
system never saw a voxel. I rebuilt the masks over **every processed frame since the sequence
reset**, including the initialization and intermediate context frames, and the region shrank:

| | recent-5-frame masks | history-aware masks |
|---|---|---|
| "never observed" share of valid voxels | 42.50 % | **25.12 %** |
| "never observed" share of GT occupancy | 89.06 % | **82.67 %** |
| out-of-frustum hidden, share of valid | 12.79 % | **3.96 %** |

On the **validated full-stream never-observed region**, in the persistent mode:

| | raw | pooled |
|---|---|---|
| completion − map-only | **+23.12** [+21.59, +24.85] * | **+16.33** [+14.69, +18.11] * |
| vs OccAny, 1 causal frame | **+9.09** [+7.34, +10.85] * | **+6.01** [+4.39, +7.67] * |

So the claim is not only intact, it is now defensible against the obvious objection. The
previously headline out-of-frustum advantage (+30.89 raw / +28.00 pooled against causal OccAny)
is real but sits in a region that is **3.96 % of the valid volume, not 12.79 %** — the paper must
quote the smaller number.

**Occ3D replicates the correction and the result.** Accounting for the full processed history
shrinks never-observed from **42.52 % → 28.60 %** of valid voxels (**61.75 % → 47.92 %** of GT
occupancy) and out-of-frustum from **31.20 % → 19.21 %** — same direction, same rough magnitude,
so the shrinkage is not an artefact of one benchmark. On Occ3D's validated full-stream
never-observed region, in the persistent mode, the head's contribution is **larger** than on
SemanticKITTI: completion − map-only **+24.56 [+22.60, +26.47] raw** / **+27.28 [+24.81, +29.61]
pooled**, and against OccAny's forward arm **+11.20 [+8.76, +13.65] raw** / **+14.79 [+12.67,
+16.87] pooled**. All four intervals exclude zero across 150 independent scenes.

### 3.5 Is the gain retrieval of older evidence, or prediction? — **PREDICTION. Not retrieval.**

The brief asked to separate these, and the answer is unambiguous and slightly unflattering:

| region | streaming − clip-reset, raw | completion − map-only (mode B), raw |
|---|---|---|
| observed earlier in the stream, outside the recent five | **−0.41** [−0.67, −0.17] * | +6.51 * (pooled +1.65 **ns**) |
| never observed in the processed stream | −1.65 * | **+23.12** * (pooled +16.33 *) |

In geometry it had already seen and no longer has in its explicit map window, the persistent-state
system is **worse** than the clip-reset reference, and map-only is flat there (−0.06 ns). The
mapper window was deliberately held to the same five causal frames in both modes, so retained
LingBot state was the *only* channel by which older evidence could help — and it carries nothing
measurable to the output. **The completion head's contribution is prediction of unseen geometry,
not memory of seen geometry.** The paper should say that, and should not claim a memory benefit.

**Confirmed on Occ3D's 150 independent streams.** In the earlier-in-stream region the persistent
mode is again not better: **−0.42 [−1.00, +0.12] raw ns** and **−0.80 [−1.16, −0.48] pooled ***.
There the Occ3D evidence is in fact sharper than SemanticKITTI's, because map-only degrades
measurably in that region (**−1.04 raw ***) and on never-observed geometry (**−2.06 raw *** in
frustum) while the *completed* predictions hold — so the head is absorbing a worse map rather
than exploiting a longer memory. With independent scenes rather than within-stream blocks, the
no-retrieval finding is the best-supported result in this phase.

---

## 4. Cost, paired with the accuracy of its own mode

| operating mode | per completed grid | SC IoU raw | SC IoU pooled |
|---|---|---|---|
| clip-reset | 368.6 ms | 26.40 | 26.34 |
| **persistent state, startup-only scale** | **124.5 ms** (seq. average) | 24.87 | 25.23 |
| OccAny, 1 causal frame | 1346.6 ms | 20.32 | 24.06 |
| OccAny, 5 forward frames (+4 s future) | 1675.9 ms | 25.21 | 25.93 |

Startup 305 ms, ordinary input update 60.5 ms, 8.03 completed grids per second, on 103-second
SemanticKITTI streams; 2.96× cheaper per grid than clip-reset because the five MoGe-2 forwards
and the bidirectional startup block move from per-grid to per-stream. Every pairing above is
measured in the mode it is quoted for; no latency is paired with another mode's accuracy.

---

## 5. Why accuracy drops — observation, and the hypothesis kept separate from it

**Observed.** The frozen gauge drifts. Scaled motion against published odometry, early → late
third of sequence 08: the clip-reset reference is biased ~9 % low but **stable** (0.909 → 0.923,
+0.013); the startup-only gauge starts *more accurate* (1.007) and ends +0.107 away (1.114) — an
8× larger drift. An independent MoGe consistency probe at five points along the stream differs
from the frozen scale by at most 0.062 in log units.

**Observed.** Only part of the deficit accumulates. The B−A gap grows from −1.31 in the first
quarter of the stream to −2.11 in the last, so roughly **38 % accumulates and 62 % is present
immediately**, at the first scored anchor.

**Hypothesis, not established.** The immediate 62 % is consistent with distribution shift — all
nine heads were trained on clip-reset inputs and mode B hands them inputs built differently —
and equally consistent with a persistent-state reconstruction that is simply different. **These
runs do not separate the two.** The main A/B comparison changes reconstruction state and
calibration policy *together*; it does not isolate the effect of calibration frequency, and I
have not claimed it does. No recalibration variant, adaptation or retraining was launched.

**What follows for the paper:** report the loss, report the drift, and do **not** write that
retraining is necessary. There is no evidence for that, and the map-only result argues against
the simplest version of it (the upstream map is statistically unchanged, so a "the geometry got
worse" story has to explain why the map did not).

---

## 6. The strongest claim the evidence supports

> A single completion head of **967,322 trained parameters**, running on a persistent causal
> reconstruction with metric scale calibrated **once at startup and never refreshed**, produces
> a completed occupancy grid every **124.5 ms** on a single GPU — **10.8× faster than the causal
> configuration of the published OccAny baseline, which it beats by +4.55 raw and +1.16 pooled
> SC IoU** — and adds **+16.02 raw / +10.49 pooled** SC IoU over its own reconstruction, with
> **+23.12 raw / +16.33 pooled** of that on geometry validated as never observed at any point in
> the processed stream. Across three training corpora and three seeds each, the ranking
> KITTI-360 > Waymo > ONCE is identical under both the clip-reset and the persistent operating
> mode, in all 18 contrasts and both scoring conventions.

— and the same ranking holds on Occ3D, in all 18 of its contrasts too, giving **36 of 36**
across the two benchmarks.

Stated qualifications, all in the reports: the persistent mode costs 1.25 raw / 0.91 pooled
against clip-reset **on SemanticKITTI**, while on Occ3D it *gains* 0.86 raw and loses 0.58
pooled, so the penalty is quoted per benchmark and the only consistent effect is the small
pooled loss; the win over OccAny's **non-causal forward** configuration is a tie in this mode on
SemanticKITTI, though it holds on Occ3D; **the causal OccAny comparison exists only on
SemanticKITTI**, so the 10.8×-faster-and-better claim rests on that benchmark's one stream;
validated streams are 103 s on SemanticKITTI and ~20 s on Occ3D, so no run in this phase is both
long-horizon and multi-stream (§7).

## 7. The smallest remaining obstacle

**It is no longer the independent-unit count — that one is now closed, and it changed an
answer.** Occ3D's 4,219 anchors over **150 independent scenes** have been scored, and they do
not agree with SemanticKITTI on the sign of the raw penalty: KITTI-360 *gains* +0.86 raw there
(two of three seeds significantly) where it lost −1.25 on SemanticKITTI. The pooled penalty is
the only consistent one (−0.58 / −0.91).

That is the right outcome for the phase even though it complicates the headline. It means the
SemanticKITTI magnitude — whose interval already widened to [−1.22, +1.81] at 400-anchor blocks
with only three independent units — should not be generalised, and the paper should quote the
penalty per benchmark.

**What remains is a genuine gap neither benchmark fills: no measurement is both long-horizon and
multi-stream.** SemanticKITTI has 103 s and one stream; Occ3D has 150 streams and ~20 s each.
The two disagree in exactly the dimension that separates them, so the natural reading — that the
penalty grows with stream duration as the frozen gauge drifts — is **consistent with** both
results but **demonstrated by** neither.

The drift diagnostics, now run on both benchmarks, support it in one respect and undercut it in
another. **Support:** the persistent mode's gauge moves further from the clip-reset reference the
longer the stream runs — median |early→late shift| is **2.0×** the reference's over Occ3D's ~20 s
scenes (0.086 against 0.043) and roughly **8×** over SemanticKITTI's 422.9 s (+0.107 against
+0.013) — and ~38 % of the SemanticKITTI deficit accumulates rather than appearing at the first
anchor. **Undercut:** the *direction* is not consistent. SemanticKITTI's ratio drifts up
(1.007 → 1.114); Occ3D's drifts **down** (1.052 → 1.005, only 29 % of its streams positive). So
there is no universal monotone inflation of the frozen gauge, and with one duration per benchmark
nothing separates "longer stream" from "different scenes". That is evidence, not proof, and §5
and §7.1 of the accuracy report keep it labelled as a hypothesis.

Closing it needs several multi-minute streams with dense ground truth, which this phase's data
does not contain. It is a data-availability limit, not an implementation or compute limit, and no
run in these reports should be read as having settled it.

## 8. One next action

**Quote the streaming penalty per benchmark in the paper, and drop the single-number version.**
That is a writing change, not an experiment, and it is the one action the new evidence forces.

Everything the brief asked for is now complete: both benchmarks scored on their full canonical
populations with zero exclusions, history-aware visibility built and validated on both, runtime
measured on an isolated GPU, and all four reports and five figures delivered. The phase stops
here, as instructed.

**Explicitly not recommended now:** streaming-mode fine-tuning. It is the obvious reflex and the
evidence argues against it more strongly than before — on the better-powered benchmark the
persistent mode is not worse in raw IoU at all, the completion head's contribution over its own
map *grows* under streaming, and every source ranking survives. There is no demonstrated
accuracy problem for retraining to solve. If adaptation is ever justified, the source comparison
must be rebuilt with **consistently constructed inputs for all three corpora**; comparing a
streaming-trained KITTI-360 head against clip-trained Waymo and ONCE heads would silently
reintroduce the confound this whole line of work exists to remove.

A second thing deliberately not launched: the raw/pooled sign flip on Occ3D has a plausible
mechanism (§9.2 of the accuracy report — dilation already saturates the recall that persistent
state adds, while still charging for the false positives), but testing it needs a
threshold/density sweep. It is recorded as a hypothesis, not run.

---

## 9. Machine-readable artifacts

Everything above is derived from files under
`artifacts/method_vs_data/streaming_scale_validation/`, for **both** benchmarks: `manifests/`
(protocol and stream manifests, checkpoint and code identities, anchor and frame lists,
exclusions), `scores/` (per-anchor confusion counts, per-seed summaries, bootstrap outputs,
block-length sensitivity, drift for both benchmarks), `visibility/` (history-aware and
recent-window regional counts and the ray-cast record), `runtime/` (timing traces, isolation
record, discarded contended files), `logs/`, and `manifests/identities.json` (git rev plus
sha256 for all nine completion checkpoints, the LingBotMap checkpoint and every file of the
phase's code).

The three required figures are `figures/accuracy_vs_latency.png` (each point's accuracy and
latency from the same operating mode), `figures/performance_over_time.png`, and
`figures/qualitative_panel_raw.png` (five fixed quantiles of a recorded ranking, failures
included); the regional breakdowns are `figures/regional_semantickitti.png` and
`figures/regional_occ3d.png`. The first two are **SemanticKITTI-only by design** — a
drift-over-time curve and a same-mode latency pairing both need a long stream, and Occ3D's
scenes are ~20 s each.

The one shared map cache per benchmark and mode is **not** under `artifacts/`: it is bulk binary
at `/media/SSD1/MINH_DATASETS/lingbot_streaming_validation/maps/<dataset>/mode{A,B}/`
(801 + 801 and 4,219 + 4,219 `.npz` files, 364 MB in all), and the empty
`streaming_scale_validation/maps/` directory is only a placeholder. Historical artifacts were
not modified.
