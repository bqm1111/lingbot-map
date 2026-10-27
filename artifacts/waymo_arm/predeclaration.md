# Waymo training arm — predeclaration

**Written 2026-10-19, before a single training step.** Tooling `tools/waymo_arm/`;
raw records in `artifacts/waymo_arm/logs/`.

## 1. The question

Replace the three KITTI-360 drives with 798 Waymo Open segments and change nothing else.
Does the completion head learn something different, and does it transfer differently to
SemanticKITTI and Occ3D-nuScenes?

## 2. What is held fixed, and the gates that prove it

| held fixed | gate | result |
|---|---|---|
| the occupancy supervision rule | `core/supervision/rawtarget_multisensor.py` must reproduce the frozen single-sensor `rawtarget.build` bit for bit | **PASS** — KITTI-360 anchors: `state`, `n_obs`, `n_frames_occ`, `n_free`, `frames_used` all identical. Waymo TOP-only through both paths: identical |
| the training loop | `tools/waymo_arm/train.py` must be the hash-frozen `current/train.py` | **PASS** — on the same KITTI-360 data: draw digest, target digest, init digest and anchor counts *exact*; loss trajectory inside the frozen loop's own run-to-run envelope |
| the recipe | the same `training_config.yaml` file, by path | CompletionUNet cin 3 / width 24, focal+Dice, AdamW, cosine, 6,000 steps, batch 4, 128×128×32 crops, seeds 0/1/2 |
| the causal contract | five frames `t−4..t` of one camera, LingBot + MoGe-2 clip scale, hard mapper | `waymo_pilot/waymo_geometry.py`, file-audited |

The frozen trainer is **not** bitwise reproducible against itself on this machine — two runs
at one seed differ on 98/100 steps, max |Δloss| 1.5e-4, different final sha256, because bf16
autocast picks non-deterministic cuDNN algorithms. So the trainer gate's criterion is *exact
on the data, within the frozen loop's own envelope on the losses*, and the envelope is
measured in the same session rather than assumed.

## 3. The supervision target matches; the input does not

Target side, same rule, 21 sweep-frames both arms:

| | KITTI-360 drive 0000 (2,094 anchors) | Waymo (smoke, 15 anchors) |
|---|---:|---:|
| valid fraction of grid | 0.5703 | 0.5710 |
| occupied prevalence in valid | 0.0276 | 0.0216 |
| cross-sweep conflict → unknown | 0.0865 | 0.0616 |
| sweep-frames used | 20.94 | 21.0 |

Waymo laser point shares, measured: TOP **0.897**, FRONT 0.023, SIDE_LEFT 0.025,
SIDE_RIGHT 0.027, REAR 0.028. The four short-range lasers sit 1.2–2.6 m from TOP's origin
(1.430, 0.000, 2.184), which is why the per-laser carve was needed rather than one origin.

Input side — 200 anchors per train arm, 200 samples per test arm, all on their own official
grid. "out share" is the fraction of the resolved map lying outside the anchor frustum;
"density" is the fraction of in-frustum voxels the five-frame map resolves.

| arm | HFOV | frustum share | map coverage | out share | in-frustum density |
|---|---:|---:|---:|---:|---:|
| **KITTI-360** (train) | 103.9° | 0.7616 | 0.0779 | **0.4224** | **5.91 %** |
| **Waymo** (train) | 50.0° | 0.4294 | 0.1401 | **0.5093** | **16.00 %** |
| SemanticKITTI (test) | 82.0° | 0.6789 | 0.0874 | **0.4294** | **7.35 %** |
| Occ3D-nuScenes (test) | 64.5° | 0.1442 | 0.0593 | **0.5224** | **19.63 %** |

Surface density of the input is matched (`S` = 0.0166 of the grid on Waymo, 0.0163 on
KITTI-360), so the difference is where the map sits, not how much surface it holds.

## 4. Predeclared prediction

On **both** statistics that describe the network's input, KITTI-360 training resembles the
SemanticKITTI test condition and Waymo training resembles the Occ3D one:

* out-of-frustum share: K360 0.422 vs SK 0.429; Waymo 0.509 vs Occ3D 0.522;
* in-frustum density: K360 5.9 % vs SK 7.4 %; Waymo 16.0 % vs Occ3D 19.6 %.

**Prediction, fixed before training: the Waymo arm will lose SC IoU on SemanticKITTI and
gain SC IoU on Occ3D-nuScenes, relative to the KITTI-360 arm.** The sign is the prediction;
no magnitude is claimed. A result that gains on both, or loses on both, falsifies the
reading above and must be reported as such.

This matters because `reports/occany_observed_vs_unobserved.md` showed the system's entire
advantage over OccAny lives outside the anchor frustum. A training set whose own
out-of-frustum share is 8 points higher is training that regime harder.

## 5. Decision rule

Evaluation is the frozen zero-shot protocol, unchanged, on the same anchors as the KITTI-360
arm, with the paired block bootstrap already used throughout (25-frame blocks on
SemanticKITTI, per-scene on Occ3D, 2,000 resamples, seed 0). Three seeds per arm.

* **ADOPT** the Waymo arm only if it beats the KITTI-360 arm with a CI clear of zero on at
  least one benchmark and is not worse with a CI clear of zero on the other.
* **REPORT AS A DOMAIN RESULT** if the two benchmarks move in opposite directions — which is
  what §4 predicts. That is a finding about train/test camera geometry, not a better system,
  and it must be written that way.
* **REJECT** if the Waymo arm is worse on both with CIs clear of zero.

Anchor policy: 5 anchors per segment, packed at 3 stream frames' separation → ~3,955 anchors
over 798 segments, against the KITTI-360 arm's 1,358 over 3 drives. **Independent anchors
remain 798 (one per 20 s segment); 3,955 is a training volume, never an independent count,
and no table may report it as one.**

## 6. Cost, measured

| step | rate | total |
|---|---|---|
| laser sidecar recovery (re-download) | 3.5 segments/min at 17 MiB/s | ~3.8 h |
| causal clips | 0.5 s/anchor | ~33 min |
| hard-map inputs | 0.08 s/anchor | ~6 min |
| occupancy targets | 0.8 s/anchor | ~53 min |
| training, 3 seeds × 6,000 steps | 8–30 min/seed | ~1.5 h |
| zero-shot evaluation, both benchmarks | existing harness | ~3 h |
