# Waymo vs KITTI-360 — controlled geometry comparison

**Run 2026-10-20.** Raw: `artifacts/waymo_arm/results/waymo_vs_kitti360.json`,
`artifacts/waymo_arm/zero_shot_eval/`. Prediction fixed beforehand in
`artifacts/waymo_arm/predeclaration.md`; gates in `phase1_gates.md`; audit in
`stage_a_audit.md`.

## Verdict, against the preregistered decision rule

> **REJECT** — the Waymo arm is worse on **both** benchmarks with confidence intervals
> clear of zero.

The predeclared directional prediction was *SemanticKITTI decreases, Occ3D increases*. It is
**supported on SemanticKITTI and falsified on Occ3D**. The prediction is not revised; the
reading behind it was wrong and §5 says why.

## 1. What changed and what did not

Only the `CompletionUNet` checkpoint. Identical clips, hard maps, target-side caches,
adapters, official masks, range bins, occupancy threshold, metric code and anchors.

The control that proves it: the **raw hard map scored on its own** is bit-identical between
the two arms — SemanticKITTI **8.820501** vs **8.820501**, Occ3D **10.035121** vs
**10.035121** SC IoU. Any difference below is the network and nothing else.

## 2. SemanticKITTI sequence 08 — 163 anchors, 7 blocks

| arm | SC IoU | precision | recall |
|---|---:|---:|---:|
| KITTI-360 (E0) | **26.17** ± 0.32 | 35.83 | 49.31 |
| Waymo | **23.14** ± 0.55 | 33.89 | 42.19 |

Δ SC IoU **−3.03**, paired block bootstrap 95 % CI **[−3.74, −2.48]**, significant.
Per-seed: E0 [26.25, 26.45, 25.82], Waymo [23.73, 22.64, 23.05] — non-overlapping.

## 3. Occ3D-nuScenes val — 1,182 anchors, 150 scene blocks

| arm | SC IoU | precision | recall |
|---|---:|---:|---:|
| KITTI-360 (E0) | **31.22** ± 1.78 | 77.07 | 34.41 |
| Waymo | **22.86** ± 0.22 | 68.44 | 25.55 |

Δ SC IoU **−8.36**, CI **[−9.31, −7.42]**, significant.
Per-seed: E0 [31.56, 32.81, 29.30], Waymo [22.84, 23.08, 22.65] — non-overlapping.

## 4. By region — where the loss happens

`H_evidence` = voxels the causal five-frame hard map resolved (**observed**).
`no_H_evidence` = the rest of the official mask (**must be completed**).
These are **not** the in/out-of-frustum split: a voxel can sit inside the frustum and be
unresolved, and one outside it can be resolved from an earlier clip frame. The frustum split
is a separate measurement and is not produced here.

| benchmark | region | KITTI-360 | Waymo | Δ |
|---|---|---:|---:|---:|
| SemanticKITTI | observed | 33.23 | 28.26 | −4.97 |
| SemanticKITTI | completed | 24.74 | 21.46 | −3.28 |
| Occ3D | observed | 39.03 | 35.05 | −3.98 |
| Occ3D | **completed** | 27.50 | **15.86** | **−11.64** |

Only `full` carries a bootstrap CI: the harness stores per-anchor counts for that region
only, so the sub-regions are across-seed means and spreads. Stated rather than implied.

## 5. The predeclared reading was wrong

The prediction came from two input statistics in which KITTI-360 resembled the SemanticKITTI
test condition and Waymo resembled the Occ3D one (out-of-frustum share 0.422/0.429 vs
0.509/0.522; in-frustum map density 5.9 %/7.4 % vs 16.0 %/19.6 %). The inference was that
input-statistic similarity would predict transfer.

**It does not.** Waymo loses most heavily on exactly the benchmark its input statistics
resemble, and loses most heavily in the region the resemblance was supposed to help: Occ3D
completion, −11.64. Input-distribution similarity is not a transfer predictor here, and no
table in the paper should use it as one.

What is *not* an explanation: undertraining or a broken arm. The Waymo models fit their own
held-out domain **better** than the KITTI-360 models fit theirs — final validation IoU
0.181 / 0.198 / 0.197 against 0.147 / 0.142 / 0.143. The arm learned; it transfers worse.

## 6. Candidate explanations, none yet tested

Measured and available:

* **Narrow camera.** 50° vs 103.9° HFOV. A model trained where the map covers a 50° wedge
  may place occupancy poorly when handed an 82° or 64.5° wedge at test time. §9 now tests
  this directly and the pattern does **not** match: the loss is not confined to one side of
  the frustum boundary, and on SemanticKITTI it is slightly *larger* outside the frustum.
* **Richer supervision.** Waymo targets resolve more of the grid (valid 0.662 vs 0.570) with
  fewer conflicts (0.068 vs 0.087) and five lasers rather than one. A denser, cleaner target
  may teach a less aggressive completion prior — the Occ3D completion collapse and the
  recall drop (34.41 → 25.55) both point that way.
* **Scene composition.** 751 short urban segments vs 3 long drives.

Separating these needs further controlled runs; §7 lists the cheapest.

## 7. Next measurements, cheapest first

1. **Scene-count control inside KITTI-360** — the 8-drive 9,672-anchor cache already exists
   (`tools/kitti360_data_scaling_7drives`). If more KITTI-360 scenes also hurt, the cause is
   scale/diversity, not Waymo.
2. ~~**Frustum-split evaluation** of both arms, distinct from §4~~ — **done**, §9.
3. **Target-density control** — rebuild Waymo targets from the TOP laser only (89.7 % of
   points, one origin) to approximate KITTI-360's single-sensor density.

## 8. Answer to Q1

**Does Waymo-only geometry training outperform KITTI-360 zero-shot, and on which
benchmarks?** No, on neither. SemanticKITTI **−3.03** SC IoU [−3.74, −2.48]; Occ3D **−8.36**
[−9.31, −7.42]. Both significant, three seeds each, identical evaluation. The source-data
bottleneck hypothesis **H1 is not supported for geometry**.

## 9. Addendum — the frustum split (§7.2, measured)

Raw: `artifacts/waymo_arm/results/frustum_split.json`,
`tools/waymo_arm/cli/frustum_split.py`.

**This is a different population and protocol from §2–§4** and the two are not
interchangeable. §2–§4 come from the frozen `baseline_reconstruction` evaluator: its own
anchor sets (163 / 1,182), its own masks, three seeds. §9 reads the canonical zero-shot
predictions: **geometry seed 0 only**, anchors 801 / 4,219, OccAny valid mask. What makes §9
valid is that both arms are scored here by one function on identical anchors, and the two
protocols agree on the sign and roughly on the size of the full-population effect: on this
protocol the full-population Δ SC IoU is **−2.84** and **−9.09**
(`reports/waymo_semantic_impact.md` §1), against §2–§3's −3.03 and −8.36.

`in_frustum` = voxel centre in front of the anchor camera and inside its image rectangle
(1226×370 / 1600×900). This is **orthogonal** to §4's observed/completed axis, and the tables
cross the two.

| benchmark | region | share of valid | share of GT occupied |
|---|---|---:|---:|
| SemanticKITTI | in frustum | 69.3 % | 74.5 % |
| SemanticKITTI | out of frustum | 30.7 % | 25.5 % |
| Occ3D | in frustum | 51.0 % | 45.5 % |
| Occ3D | out of frustum | 49.0 % | 54.5 % |

Occupancy IoU, Δ = Waymo − KITTI-360:

| benchmark | population | KITTI-360 | Waymo | Δ |
|---|---|---:|---:|---:|
| SemanticKITTI | in frustum, all | 25.16 | 22.38 | −2.78 |
| SemanticKITTI | in frustum, observed | 33.74 | 31.32 | −2.42 |
| SemanticKITTI | in frustum, completed | 23.50 | 20.03 | −3.47 |
| SemanticKITTI | out of frustum, all | 29.98 | 26.58 | −3.40 |
| SemanticKITTI | out of frustum, observed | 32.73 | 25.87 | −6.86 |
| SemanticKITTI | out of frustum, completed | 29.26 | 26.90 | −2.36 |
| Occ3D | **in frustum, all** | 36.18 | 22.86 | **−13.32** |
| Occ3D | in frustum, observed | 43.84 | 31.02 | −12.81 |
| Occ3D | in frustum, completed | 31.54 | 16.96 | −14.58 |
| Occ3D | out of frustum, all | 28.25 | 22.85 | −5.40 |
| Occ3D | out of frustum, observed | 39.32 | 36.67 | −2.65 |
| Occ3D | out of frustum, completed | 24.27 | 17.05 | −7.22 |

### What this rules out and what it supports

**The narrow-camera explanation does not survive in the form §6 stated it.** A 50°-trained
model mishandling a wider test wedge predicts a loss concentrated *inside* the frustum, where
the wedge geometry differs. On SemanticKITTI the loss is slightly *larger outside* it
(−3.40 vs −2.78); only on Occ3D is it concentrated inside (−13.32 vs −5.40), and Occ3D's
64.5° camera is the one closest to Waymo's 50°. The two benchmarks point opposite ways, so
frustum width alone does not organise the result.

**The less-aggressive-completion-prior explanation gains direct support.** Read the
precision/recall columns of the raw JSON: in the `observed` population the Waymo arm's recall
*rises* (SemanticKITTI 64.54 → 83.17 out of frustum, Occ3D 41.76 → 44.07) while its precision
falls hard (39.91 → 27.30, 87.08 → 68.59); in the `completed` population its recall *drops*
everywhere (SemanticKITTI in-frustum 40.95 → 32.31, Occ3D in-frustum 36.88 → 19.30). The arm
trained on denser five-laser targets commits more readily where the map resolved something
and extrapolates less where it did not.

That is a **description of the behaviour, confirmed on a new split** — not a demonstration of
its cause. Attributing it to target density still needs §7.1 (the 8-drive KITTI-360 control)
or §7.3 (TOP-laser-only Waymo targets). Neither has been run.
