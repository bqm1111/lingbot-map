# Predeclaration — A2 and its substitutes

**Written 2026-10-20, before any new training run.** Nothing in this file has been revised
after seeing a result, because **no new training run was executed**: A2 was blocked at the
data audit (`ddad_feasibility.md`). The design and criteria below are recorded so that
whichever arm is later authorized is governed by a hypothesis fixed in advance, and so the
block itself is documented as a decision rather than as silence.

Status of each arm, per brief §5:

| arm | source data | status |
|---|---|---|
| A0 | KITTI-360 | established baseline, not re-run |
| A1 | Waymo | completed negative control, not re-run |
| **A2** | **DDAD only** | **NOT LAUNCHED — blocked, see §4** |
| A3 | Waymo + DDAD | not authorized |
| A4 | Waymo + DDAD + ONCE | not authorized |
| A5 | released OccAny checkpoint | external comparator, already reproduced |

---

## 1. Hypothesis

**H-subset.** Our reconstruction-plus-completion method, trained on a *single* member of
OccAny's five-dataset source pool, reaches zero-shot geometric SC IoU competitive with the
released OccAny checkpoint under matched causal monocular evaluation.

**H-dataset-identity.** The Waymo-only failure (SemanticKITTI −3.03, Occ3D −8.36) is specific
to Waymo rather than general to single-source training from OccAny's pool. If a different
source dataset transfers well, H-subset survives and the Waymo result is a dataset property.
If it fails the same way, the limitation is single-source training or our recipe, not Waymo.

**These are distinct from the method claim.** Neither hypothesis, if confirmed, attributes
anything to architecture. Attribution needs the matched-data control
(`matched_data_control_feasibility.md`), and that is stated here so a positive result is not
later over-read.

---

## 2. Fixed training recipe — no element is free

Per brief §7, unchanged from the frozen baseline and from A1:

| parameter | value |
|---|---|
| model | CompletionUNet |
| input channels | 3 |
| width | 24 |
| parameters | ≈967 k |
| loss | Focal + Dice |
| optimizer | AdamW |
| schedule | cosine |
| steps | 6,000 |
| batch size | 4 |
| crop | 128×128×32 |
| seeds | 0, 1, 2 |
| supervision | the frozen Gate 8C-1 carve rule, unchanged |
| grid | 256×256×32 at 0.2 m |

Also fixed: the five-frame causal input contract (t−4 … t), LingBotMap reconstruction, MoGe-2
metric scaling, hard mapping, and the whole evaluation pipeline.

**Not permitted:** extending steps, changing the loss or its weighting, changing the
occupancy rule, tuning on target labels, or altering the architecture because early results
look unfavourable.

---

## 3. Evaluation and statistics, fixed in advance

Frozen zero-shot protocol: SemanticKITTI sequence 08 (163 anchors on the
`baseline_reconstruction` evaluator) and Occ3D-nuScenes validation (1,182 anchors), identical
masks, grids, label mappings, thresholds and metric code, no target-domain adaptation.

Primary metric: **SC IoU on both benchmarks.**

Secondary diagnostics, all pre-specified: occupancy precision and recall; observed-region and
completed-unobserved SC IoU; inside- and outside-frustum SC IoU; the strict
anchor-observed / earlier-only / never-observed split; range-dependent performance.

Statistics: paired block bootstrap, 25 contiguous anchors per block on SemanticKITTI, one
block per scene on Occ3D, 2,000 resamples, bootstrap seed 0. All three training seeds
reported. Block variation and seed variation reported separately.

---

## 4. Decision rule, fixed in advance

Let Δ be the arm's SC IoU minus A0's, per benchmark, seed-mean, with a paired 95 % CI.

* **ADOPT** — Δ > 0 with the CI excluding zero on at least one benchmark, **and** the other
  benchmark does not degrade beyond the non-inferiority margin below.
* **REJECT** — Δ < 0 with CIs excluding zero on both benchmarks, or a degradation worse than
  the margin on either.
* **INCONCLUSIVE** — anything else.

### Non-inferiority margin: 1.0 SC IoU point

Justified before any result, from two measured quantities:

1. **Historical seed variation.** Our three-seed spread on SemanticKITTI SC IoU is ±0.32
   (per-seed 26.25 / 26.45 / 25.82), and on Occ3D ±1.78. A margin below the seed spread would
   fail arms for noise; a margin far above it would license real regressions.
2. **The research objective.** We are comparing against an external method whose causal
   monocular SemanticKITTI figure is 24.02, about 2.4 points below A0's 26.17. A regression
   larger than 1.0 point would consume nearly half of the margin that makes the comparison
   interesting at all.

1.0 point sits above SemanticKITTI's seed noise and below half the OccAny gap. It is **not**
above Occ3D's ±1.78 seed spread, so on Occ3D the margin is dominated by seed variance and
that is stated rather than hidden: an Occ3D regression between 1.0 and 1.8 points will be
reported as *failing the margin while being within seed noise*, and that combination is an
INCONCLUSIVE outcome, not a REJECT.

**A CI containing zero is not evidence of equivalence** and will not be reported as one.

---

## 5. Why A2 was not executed

Recorded here because the brief requires the predeclaration to precede evaluation, and the
honest sequence is: audit → block → no training.

DDAD fails two of brief §6's correctness gates, for a reason implementation cannot remove.
DDAD publishes one aggregated point cloud per sample, merged from four Luminar-H2 sensors,
in a vehicle frame whose origin sits **on the ground** below the rear axle, with per-point
channels X/Y/Z/INTENSITY only. Our supervision carves free space along rays **from the sensor
origin**, and that origin is unrecoverable: there is no per-point identity, no per-sensor
counts in the release (unlike Waymo, where recovery was possible and verified), and no
azimuthal sector structure from which to infer it.

Measured on real DDAD data with the frozen builder called unchanged: occupied prevalence
0.0042–0.0094 against KITTI-360's 0.0276 and Waymo's 0.0222 — **3–6× out of family** — and the
target moves by ~20 percentage points of valid fraction and roughly 2× in prevalence when the
assumed mount height is varied across plausible values. An arm trained that way would
confound DDAD with a constant we invented, so it could not serve as the control A2 exists to
be.

---

## 6. Predeclared criteria for the substitutes

If ONCE or PandaSet is authorized, §1–§4 apply unchanged, with these additional
pre-specified gates. Recorded now so they cannot be relaxed after seeing data.

**Supervision comparability gate — pass before training.** On ≥3 scenes, the frozen carve must
produce, per anchor:

* occupied prevalence in valid within **[0.015, 0.040]** — a band containing both KITTI-360's
  0.0276 and Waymo's 0.0222, set by the two existing arms, not by the candidate;
* conflict-to-unknown fraction ≤ **0.12** (KITTI-360 0.0865, Waymo 0.0679);
* valid fraction within **[0.45, 0.75]** (KITTI-360 0.5703, Waymo 0.6616);
* occupancy present in ≥30 of 32 z-levels, confirming the grid's vertical extent suits the
  sensor frame.

Failing any of these is a block, reported as DDAD was, not a licence to adjust the recipe.

**Sensor-origin gate.** The arm may use only a **published** sensor origin. ONCE satisfies this
with a single roof LiDAR; PandaSet satisfies it through its per-point sensor ID `d` plus
per-device poses, via the already-validated multisensor carve. No arm may use an assumed mount
position.

**Input-geometry disclosure.** The chosen camera's HFOV must be recorded before training and
reported beside the result. This is required because DDAD's only forward camera is 47.9° and
Waymo's are 50.0°, so a narrow-camera arm would re-run the condition A1 already failed under;
if the substitute is also narrow, the arm's power to discriminate "dataset identity" from
"narrow training FOV" is correspondingly weak and must be stated, not discovered afterwards.

**Predeclared directional expectation.** None. The A1 experiment predeclared a direction from
input-statistic similarity and that prediction was falsified — Waymo lost most on the
benchmark its input statistics most resembled. Having learned that input-distribution
similarity is not a transfer predictor here, declaring a direction from the same kind of
reasoning would be unjustified. The arm is run as an estimate, not a test of a direction.
