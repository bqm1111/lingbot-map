# Gate 5.2 — Frozen calibrated-gauge transfer to SSCBench-KITTI-360 validation

**Date:** 2026-09-01 · **Status:** complete · **Nothing was trained, fitted, tuned or
selected on this dataset.**

---

## 1. Diagnosis

# `FRESH_CALIBRATED_GAUGE_TRANSFERS`

All five predeclared criteria pass. No leakage and no protocol failure was found. The
overlap discussion is in §7; there is none — the `PARTIAL` and `FAILS` branches each
require at least one criterion to fail, and none does.

---

## 2. Plain-language conclusion

The G51-B gauge — full LingBot image → frozen MoGe-2 given the calibrated horizontal FOV
→ one scalar per five-frame clip → scale all depths and pose translations once → fixed
0.4 m dilation — transfers to SSCBench-KITTI-360 without any change.

On a benchmark, a sequence and a camera never used to design Gates 1–5.1, it lands the
metric scale to **6.0 % median relative error** against a LiDAR oracle, lowers depth
AbsRel from the frozen constant's **0.305 to 0.101**, and raises dilated occupancy IoU
from **0.1077 to 0.1342** — recovering **94 %** of the gain the LiDAR oracle itself
achieves. It is statistically non-inferior to the uncalibrated Gate-5 gauge.

Two findings matter more than the headline:

* **The scale error is stable across three benchmarks.** B's median relative scale error
  is now 6.0 % (KITTI-360), 6.4 % (SemanticKITTI) and 7.9 % (Occ3D-nuScenes). The gauge is
  not tuned to any of them and does not degrade on the fresh one.
* **The learned Gate-2 clip head does not merely fail to transfer — it anti-transfers.**
  C3 is *significantly worse than the constant* `s0` here (dilated ΔIoU −0.0091, CI
  [−0.0131, −0.0052]). It confidently moves the scale in the wrong direction: `s0` is
  already ~25 % too large on KITTI-360, and C3 pushes it further to ~30 % too large, while
  correlating strongly with the oracle (Pearson 0.969). It has learned the *shape* of the
  scale signal and the wrong *gain*. Any future student must be validated
  leave-one-dataset-out for exactly this failure.

An honest limitation, unchanged from Gate 4.1: absolute IoU is capped by **coverage, not
geometry**. The target is a LiDAR-accumulated *completed* scene (78,632 occupied voxels per
clip on average); a five-frame visible-surface reconstruction can only fill part of it, so
recall is 0.18 even at the oracle scale. Nothing in this gate addresses completion.

---

## 3. Validation-data provenance

| Item | Value |
|---|---|
| Benchmark | SSCBench-KITTI-360 |
| Source | `https://huggingface.co/datasets/ai4ce/SSCBench` (`sscbench-kitti/`) |
| Revision | `badde69bbd01552be186dc0ddd0989885b01c532` |
| Distribution | SquashFS 4.0 image, zlib, **206,742,990,564 bytes**, split into 10 parts |
| License | CC BY-NC-SA 3.0 (SSCBench-KITTI-360) |
| Validation sequence | `2013_05_28_drive_0006_sync` — confirmed as the official val split |
| Official anchors | **1,812** (`000000` … `009055`, every 5 native frames) |
| Native RGB | 1408 × 376, `image_00/data_rect` |
| Voxel grid | 256 × 256 × 32 @ 0.2 m, origin (0, −25.6, −2), **velodyne frame of the anchor** |
| Calibration | KITTI-360 `calibration.zip`, sha256 `2691b35c0cb7…f7af`, 2,923 B |
| Poses | KITTI-360 `data_poses.zip`, sha256 `814dcb9e623c…c40b`, 10,683,072 B |
| RGB / timestamps / raw velodyne | the KITTI-360 release already on this machine, at `/media/welf/MINH/datasets/kitti360/KITTI-360` |

### 3.1 Selective extraction — 0.21 % of the archive

The brief asks for a blocker report if validation-only extraction is impossible. It is
possible. SquashFS stores its inode and directory tables **near the end** of the image, so
the whole 323,907-entry tree is enumerable from ~16 MB of metadata and individual files can
be pulled with HTTP byte-range reads across the split parts (part *i* holds bytes
`[i·20 GiB, (i+1)·20 GiB)`).

**Downloaded: 0.441 GB of 206.743 GB (0.21 %), in 815 range requests over 792 s.**
7,230 files: `.bin`, `.label`, `.invalid` and `preprocess/labels/…_1_1.npy` for all 1,812
validation anchors. Training sequences, the test sequence, fisheye and right-camera imagery,
and the nuScenes/Waymo subsets were never transferred. A further 74.4 MB of archive imagery
was fetched once, solely to verify the frame mapping (§3.3), and 10.7 MB of official
calibration/poses came from KITTI-360 directly.

The reader is `sscbench_kitti360/squashfs.py` (+ `remote.py`); the extractor is
`tools/gate5_2/prepare_data.py`.

### 3.2 Three conventions verified against the files, not assumed

SemanticKITTI's conventions do **not** all carry over. Each of these would have silently
corrupted the evaluation:

1. **Grid frame** — the volume is in the **velodyne frame of the anchor**, not the camera
   frame. Verified by voxelising the raw KITTI-360 sweep of the anchor and
   cross-correlating with the official `.bin`: the 3-D correlation peak sits at **exactly
   zero shift** for every anchor tested, and **99.8–99.99 %** of `.bin` voxels are voxels
   the raw sweep also fills. (Plain IoU ranges 0.35–0.92 only because SSCBench filters the
   sweep before voxelising — `.bin` always holds *fewer* voxels than the raw sweep, so IoU
   is not the right statistic here.) The camera frame gives IoU ≈ 0.03.

2. **Target semantics** — `preprocess/labels/<seq>/<anchor>_1_1.npy` (float32, 0/1–18/255)
   is exactly:

   ```
   255   <=>  invalid == 1  AND  label == 0      (unobserved -> excluded from scoring)
   0     <=>  invalid == 0  AND  label == 0      (observed free)
   1..18 <=>  learning_map(label),  label > 0    (occupied, kept even where invalid)
   ```

   Verified **elementwise** on sampled anchors spread across the sequence
   (`test_official_target_equals_label_and_invalid_rule`). Note the asymmetry: an occupied
   voxel survives even when `.invalid` marks it, so **`.invalid` alone is not the
   evaluation mask** — using it as one would discard 62,581 genuinely occupied voxels per
   clip at anchor 0.

3. **Binary scene-completion scoring** reproduces the MonoScene/SSCBench evaluator:
   `mask = (gt != 255)`, `occupied = (gt > 0)` inside that mask. No visibility or validity
   mask is derived from the prediction anywhere.

### 3.3 Frame indexing — SSCBench renumbered KITTI-360

SSCBench index *i* is **not** KITTI-360 frame *i*. KITTI-360 ships poses for only 9,186 of
the 9,699 frames of this drive, and SSCBench indexes into that subset, skipping its first
entry:

```
kitti360_frame  =  pose_frames[sscbench_index + 1]
```

Verified by **exact pixel equality** between the archive's own `data_rect` PNGs and the
local KITTI-360 release at **40/40** sampled indices spanning the sequence
(`artifacts/gate5_2/index_mapping_verification.json`). This is what makes it legitimate to
read RGB, timestamps and velodyne from the local KITTI-360 copy instead of re-downloading
7.9 GB of imagery. Had the mapping been assumed to be the identity, every clip would have
been ~35 frames out of register with its target.

---

## 4. Clip count

| | |
|---|---|
| Official anchors | 1,812 |
| **Eligible clips** | **1,753** |
| Excluded | 59 — 4 without four earlier anchors (sequence start), 55 with non-contiguous native frames |
| Clip form | `[t−4, …, t]` consecutive anchors, anchor last, sliding by one anchor |
| Temporal span | median **2.090 s**, min 2.083 s, max 2.099 s |
| Scale-estimation failures | **0** for C0, C3, A, B and the oracle |
| Clips below the 200-px LiDAR minimum | 0 |

The eligibility rule was declared before any result was seen and is purely temporal: four
earlier anchors must exist **and** consecutive anchors must be exactly 5 native frames
apart. The second condition is what "do not cross sequence discontinuities" means here —
where KITTI-360 lacks poses, consecutive SSCBench anchors can be seconds apart (anchors 0–4
span 8.58 s, not 2 s), and such a clip would silently violate the protocol.

---

## 5. Main result table

1,753 clips, official SSCBench 0.2 m grid, binary scene completion.

| ID | Scale | FOV | Corr. | IoU | Precision | Recall | TP | FP | FN |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| K360-C0-R | constant `s0` | — | raw | 0.0239 | 0.3884 | 0.0250 | 1,938 | 3,005 | 76,694 |
| K360-C0-D | constant `s0` | — | `dilate_r2` | 0.1077 | 0.3628 | 0.1376 | 10,573 | 18,736 | 68,058 |
| K360-C3-R | Gate-2 head | — | raw | 0.0200 | 0.3791 | 0.0208 | 1,617 | 2,566 | 77,015 |
| K360-C3-D | Gate-2 head | — | `dilate_r2` | 0.0985 | 0.3581 | 0.1223 | 9,404 | 17,105 | 69,227 |
| K360-A-R | MoGe | inferred | raw | 0.0381 | 0.3753 | 0.0410 | 3,085 | 5,097 | 75,547 |
| K360-A-D | MoGe | inferred | `dilate_r2` | 0.1351 | 0.3642 | 0.1825 | 13,727 | 24,401 | 64,905 |
| K360-B-R | MoGe | **calibrated** | raw | 0.0376 | 0.3724 | 0.0405 | 3,035 | 5,063 | 75,597 |
| **K360-B-D** | MoGe | **calibrated** | `dilate_r2` | **0.1342** | 0.3619 | 0.1812 | 13,596 | 24,308 | 65,036 |
| *K360-OR-R* | *LiDAR oracle* | — | raw | *0.0364* | *0.3820* | *0.0389* | *2,941* | *4,669* | *75,691* |
| *K360-OR-D* | *LiDAR oracle* | — | `dilate_r2` | *0.1358* | *0.3724* | *0.1811* | *13,630* | *23,259* | *65,001* |

*Italic rows are the non-deployable diagnostic oracle.*

Per-clip means: 313,402 valid evaluation voxels, 78,632 GT-occupied voxels, 302,418 fused
points. Points landing inside the grid: C0 98,475 · C3 94,430 · A 132,933 · B 134,532 ·
oracle 125,521 (of 302,418) — the smaller MoGe scale pulls ~37 % more of the reconstruction
into the 51.2 m volume.

Per-clip IoU quartiles (dilated): C0 0.0791 / 0.1067 / 0.1349 · C3 0.0760 / 0.0968 / 0.1185
· A 0.1121 / 0.1328 / 0.1556 · **B 0.1089 / 0.1309 / 0.1548** · oracle 0.1118 / 0.1324 / 0.1564.

### 5.1 By distance and height (dilated IoU)

| | 0–10 m | 10–20 m | 20–30 m | 30–40 m | 40–51.2 m |
|---|---:|---:|---:|---:|---:|
| C0-D | 0.2282 | 0.1603 | 0.0734 | 0.0269 | 0.0045 |
| C3-D | 0.2048 | 0.1460 | 0.0686 | 0.0270 | 0.0074 |
| A-D | 0.3051 | 0.1868 | 0.0629 | 0.0123 | 0.0016 |
| **B-D** | **0.3020** | **0.1851** | 0.0618 | 0.0125 | 0.0011 |
| OR-D | 0.3022 | 0.1903 | 0.0672 | 0.0165 | 0.0016 |

The MoGe gauge wins decisively inside 20 m and loses beyond 20 m — the direct, expected
consequence of correcting an over-large scale: geometry that was being flung into the far
grid is pulled back to where it belongs, and the residual far-field predictions thin out.
The oracle shows the same profile, so this is a property of the correct scale, not of MoGe.

| | −2…−1 m | −1…0 m | 0…1 m | 1…4.4 m |
|---|---:|---:|---:|---:|
| C0-D | 0.1026 | 0.1025 | 0.1100 | 0.1213 |
| C3-D | 0.0910 | 0.0966 | 0.1042 | 0.1052 |
| A-D | 0.1603 | 0.1194 | 0.1182 | 0.1560 |
| **B-D** | **0.1608** | 0.1183 | 0.1172 | 0.1552 |
| OR-D | 0.1491 | 0.1195 | 0.1249 | 0.1623 |

---

## 6. Decision-rule audit

| # | Criterion for `FRESH_CALIBRATED_GAUGE_TRANSFERS` | Measured | Verdict |
|---|---|---|---|
| 1 | K360-B median relative scale error ≤ 10 % | **6.02 %** | **PASS** |
| 2 | K360-B-D significantly improves on K360-C0-D, CI excludes zero | +0.0266, CI [+0.0215, +0.0319] | **PASS** |
| 3 | K360-B-D non-inferior to K360-A-D at −0.01 | lower bound −0.0021 > −0.01 | **PASS** |
| 4 | K360-B depth AbsRel below C0 depth AbsRel | **0.1013** vs **0.3054** | **PASS** |
| 5 | No leakage or protocol failure | preflight clean, 53/53 tests, 0 forbidden file accesses | **PASS** |

**Branch overlap:** none. `FRESH_CALIBRATED_GAUGE_PARTIAL` requires the occupancy or
non-inferiority criterion to fail; both pass. `FRESH_CALIBRATED_GAUGE_FAILS` requires
> 10 % scale error, serious occupancy degradation or a protocol failure; scale error is
6.02 %, occupancy improves significantly, and no protocol failure occurred. The diagnosis is
uniquely determined.

One point deserves stating plainly rather than being buried: **B is chosen because it was
predeclared, not because it won.** A actually scores marginally *higher* occupancy IoU
(0.1351 vs 0.1342). B is the better *metric gauge* on every metric that measures metric
accuracy — scale error 6.02 % vs 6.38 %, Pearson 0.923 vs 0.854, depth AbsRel 0.1013 vs
0.1085, within-clip dispersion 0.0385 vs 0.0472 — and the two are statistically
indistinguishable on IoU. Selecting on IoU here would have meant discarding calibration on a
0.001 difference inside its own confidence interval.

---

## 7. Statistics — paired contiguous-block bootstrap

**87 blocks** of 20 consecutive anchor clips (the trailing 13 appended to the last block),
10,000 paired resamples, seed 0. These are contiguous blocks of **one KITTI-360 drive**;
they are not independent scenes and are never described as such.

| Contrast | ΔIoU | 95 % CI | Blocks improved |
|---|---:|---|---:|
| C0-R → A-R | +0.0143 | [+0.0125, +0.0161] | 83/87 |
| **C0-D → A-D** | **+0.0275** | **[+0.0228, +0.0325]** | 79/87 |
| C0-R → B-R | +0.0138 | [+0.0119, +0.0158] | 82/87 |
| **C0-D → B-D** | **+0.0266** | **[+0.0215, +0.0319]** | 76/87 |
| C3-R → B-R | +0.0176 | [+0.0155, +0.0196] | 83/87 |
| **C3-D → B-D** | **+0.0356** | **[+0.0306, +0.0407]** | 80/87 |
| A-R → B-R | −0.0005 | [−0.0010, +0.0000] | 34/87 |
| **A-D → B-D** | **−0.0010** | **[−0.0021, +0.0001]** | 38/87 |
| B-R → OR-R | −0.0012 | [−0.0020, −0.0005] | 32/87 |
| B-D → OR-D | +0.0015 | [−0.0001, +0.0032] | 53/87 |
| C0-R → C3-R | −0.0038 | [−0.0053, −0.0023] | 35/87 |
| **C0-D → C3-D** | **−0.0091** | **[−0.0131, −0.0052]** | 35/87 |

**Non-inferiority** (predeclared margin −0.01 IoU):

```
K360-B-D  -  K360-A-D  =  -0.0010,  95% CI [-0.0021, +0.0001]
lower bound -0.0021  >  margin -0.01     ->  NON-INFERIOR
```

The CI also crosses zero, so A and B are statistically indistinguishable on occupancy IoU;
that is *not* claimed as equivalence — the non-inferiority test is what the margin licenses.

**Gain retention** relative to the C0 → oracle gain: dilated **A 97.8 %, B 94.3 %**; raw
A 113.8 %, B 109.9 % (both MoGe variants slightly exceed the oracle in the raw setting,
because the oracle optimises metric depth agreement, not voxel overlap).

---

## 8. Scale diagnostics (vs the LiDAR oracle, 1,753 clips)

| | median | p05 | p25 | p75 | p95 | median abs rel err | median abs log err | median ratio to oracle | Pearson | Spearman | log-ratio MAD | per-frame log dispersion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 27.366 | 27.366 | 27.366 | 27.366 | 27.366 | 0.2452 | 0.2193 | 1.2452 | −0.0000 | 0.2953 | 0.0985 | — |
| C3 | 28.225 | 22.100 | 26.323 | 30.509 | 34.467 | 0.2966 | 0.2597 | 1.2966 | 0.9693 | 0.9798 | 0.0240 | — |
| A | 20.750 | 16.348 | 19.038 | 22.937 | 25.555 | 0.0638 | 0.0651 | 0.9531 | 0.8539 | 0.8326 | 0.0540 | 0.0472 |
| **B** | **20.586** | 15.578 | 18.732 | 22.685 | 25.665 | **0.0602** | **0.0620** | **0.9442** | **0.9233** | **0.9145** | **0.0358** | **0.0385** |
| OR | 21.978 | 16.376 | 19.669 | 24.084 | 27.147 | 0 | 0 | 1 | 1 | 1 | 0 | — |

C0's Pearson is exactly 0 by construction (it is a constant). The striking row is **C3**: it
tracks the oracle almost perfectly in *rank* (Spearman 0.980) yet carries the **largest**
absolute error of any condition (29.7 %) because its gain is wrong. It is a well-shaped,
badly-calibrated estimator — the precise failure mode a distilled student would inherit.

Supplying the calibrated FOV reduces the MoGe gauge's dispersion on every measure
(MAD 0.054 → 0.036, per-frame dispersion 0.047 → 0.038, Pearson 0.854 → 0.923). This
**confirms the open nuance from Gate 5.1** — calibrated FOV buys consistency — and, unlike on
SemanticKITTI, it costs no gain bias here (6.4 % → 6.0 %).

### 8.1 FOV

| | KITTI-360 (this gate) | SemanticKITTI (Gate 5.1) |
|---|---:|---:|
| Calibrated FOV_x | 103.745° | 81.845° |
| MoGe-inferred FOV_x (median) | 102.881° | 86.897° |
| Difference | **−0.864°** | **+5.052°** |
| Processed lattice | 518 × 140, AR 3.700 | 518 × 154, AR 3.364 |

This is why A ≈ B here and A ≪ B on SemanticKITTI: MoGe's self-inferred FOV happens to be
nearly right on KITTI-360's 104° wide-angle camera, and the calibrated FOV has little to
correct. It is a clean confirmation that the mechanism identified in Gate 5.1 is *the
inferred FOV*, not the aspect ratio — the KITTI-360 aspect ratio (3.700) is even further
outside MoGe's documented 2:1 range than SemanticKITTI's (3.364), yet the gauge performs
*better* here. The Gate-5.1 decision to discard the aspect crop is vindicated.

---

## 9. Depth diagnostics at projected LiDAR (167.2 M pixel pairs)

| | AbsRel | median rel | median log | RMSE (m) | δ1 | δ2 | δ3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.3054 | 0.2580 | 0.2311 | 3.624 | 0.4881 | 0.8720 | 0.9747 |
| C3 | 0.3178 | 0.3032 | 0.2653 | 3.531 | 0.2867 | 0.9515 | 0.9888 |
| A | 0.1085 | 0.0822 | 0.0835 | 2.056 | 0.9009 | 0.9765 | 0.9876 |
| **B** | **0.1013** | **0.0765** | **0.0783** | **1.997** | **0.9164** | 0.9769 | 0.9875 |
| OR | 0.0756 | 0.0432 | 0.0432 | 1.790 | 0.9410 | 0.9790 | 0.9885 |

By depth band (AbsRel):

| | 0–10 m | 10–20 m | 20–40 m | 40–80 m |
|---|---:|---:|---:|---:|
| C0 | 0.3243 | 0.2745 | 0.2133 | 0.2148 |
| C3 | 0.3283 | 0.3008 | 0.2675 | 0.2504 |
| A | 0.1024 | 0.1166 | 0.1416 | 0.2829 |
| **B** | **0.0945** | **0.1110** | **0.1365** | 0.2780 |
| OR | 0.0701 | 0.0829 | 0.1053 | 0.2395 |

By temporal offset (AbsRel) — B: t0 0.1031, t1 0.0979, t2 0.0986, t3 0.1011, t4 (anchor)
0.1061. Flat to ±5 %, so the single clip scalar is not being dragged by the oldest frame,
and the 2 s window introduces no drift the gauge cannot absorb.

### 9.1 Oracle cross-validation

Three independent oracle constructions agree, which is why the coordinate chain can be
trusted:

| | vs primary (projected LiDAR, 5 frames) |
|---|---|
| Anchor-frame-only projected LiDAR | median ratio 1.0050, Pearson 0.968 |
| Official `.bin` voxel centres (0.2 m quantised) | median ratio 1.0182, Pearson 0.959 |

The `.bin` oracle is described throughout as a **0.2 m voxel-centre LiDAR scale oracle**,
not an exact projected-point oracle; its +1.8 % offset is the expected quantisation bias.

---

## 10. Leakage, correctness and reproducibility

The label-free preflight ran on 10 clips spread across the drive **inside a file-access
auditor** that intercepts `builtins.open`, `numpy.load` and `numpy.fromfile`: 66 unique
paths opened, **0 forbidden** (`.label`, `.invalid`, `_1_1.npy`, `velodyne`, `voxels/`). All
per-clip checks passed — five chronological frames, anchor last, no future frame, 2.09 s
span, pose-direction cosine > 0.9, transformed intrinsics, exact LingBot/MoGe lattice
correspondence, finite positive teacher depth, `depth == points[...,2]`, RGB in [0,1],
MoGe mask fraction 0.928, minimum 210,398 valid scale pixels, one scalar per clip,
rotation-preserving and translation-scaling similarity fusion.

The diagnostic oracle ran only after the deployable scale tables were finalised and
**sha256-pinned**; the pins are stored in `depth_diagnostics.json` and re-verified by a test,
so a later edit to a scale table would fail the suite rather than pass silently.

The tampering test randomises the `_1_1.npy` target, `.label`, `.invalid`, `.bin` and the
raw velodyne into a temporary tree, first proving the tampering is real (the adapter reads
back a different target), then showing every deployable scale is **bit-identical**. The
released datasets are never written to.

**Tests: 53 new in `tests/gate5_2`; full repository suite 538 passed, 2 skipped, 0 failed**
(485 before this gate + 53). All 20 enumerated requirements are covered; §8's synthetic
coordinate tests are `test_synthetic_points_land_in_expected_voxels`,
`test_camera_frame_points_map_into_the_grid_as_documented` and
`test_lidar_confirms_the_grid_frame`.

---

## 11. Interpretation constraints

Stated explicitly, as the brief requires:

* This is a **frozen inference-only** test. No optimiser was constructed, no backward pass
  ran, no parameter changed. LingBot's checkpoint hash and MoGe's repository revision,
  model revision, weight hash and parameter count are byte-identical to Gate 5.
* **Camera intrinsics are standard inference metadata**, already required to project into
  the voxel grid. The method is therefore **calibration-aware, not RGB-only**, and should
  never be described as the latter.
* **MoGe-2 received large-scale metric supervision during its own training.** The metric
  knowledge is the teacher's; this gate only shows it can be transferred through a single
  scalar per clip.
* **Our downstream method receives no KITTI-360 LiDAR or occupancy supervision.** LiDAR and
  targets are opened only for evaluation and diagnostics, after predictions are pinned.
* **SSCBench-KITTI-360 is related to SemanticKITTI** — same city, same vendor lineage, a
  similar forward camera. This is **not** proof of unrestricted open-domain generalization.
  It is nevertheless a **fresh benchmark, sequence and camera** not used to design Gates
  1–5.1, with a materially different camera formation (104° vs 82° horizontal FOV).
* **Only binary occupancy is evaluated. No semantic capability is claimed.**
* **Five frames are retained for protocol consistency**, not because five is proven optimal.
* **Absolute IoUs must not be compared across SSCBench-KITTI-360, SemanticKITTI and
  Occ3D-nuScenes** — the grids, masks, extents and protocols differ. Only the within-dataset
  contrasts and the scale/depth errors are comparable across gates.
* **G51-D's SemanticKITTI 0.1800 is not evidence of better metric calibration** and is not
  used as such anywhere here; G51-D was not run in this gate.

---

## 12. Recommended next experiment (one)

The diagnosis is `FRESH_CALIBRATED_GAUGE_TRANSFERS`, so the authorised next step is:

> **Lightweight teacher-to-student pseudo-scale distillation on diverse unlabeled RGB clips
> with calibrated camera metadata.** Freeze LingBot and MoGe-2; run the G51-B gauge over
> unlabeled multi-source clips to emit one pseudo-scale per clip; train a small student to
> predict that scalar from LingBot's own features plus the calibrated FOV, so deployment no
> longer needs a 300 M-parameter teacher at inference.

Two conditions this gate makes non-negotiable for that work, both earned from C3's failure
here:

1. **Validate leave-one-dataset-out.** C3 achieves Spearman 0.980 against the oracle while
   being 29.7 % wrong in gain, and is significantly *worse than a constant* on a dataset it
   never saw. Rank correlation on held-in data would have called C3 a success. The student's
   acceptance metric must be median relative scale error on a held-out *dataset*, not
   correlation on a held-out split.
2. **Include the calibrated FOV as a student input.** It is what separates a gauge that
   holds at 6 % across three benchmarks from one that drifts with the camera.

Distillation must not begin until those two conditions are written into its brief. Nothing
in this gate authorises multi-source *training* of the geometry, the corrector or semantics.

---

## 13. Files, commands, cost

**Created — modules:** `sscbench_kitti360/{__init__,squashfs,remote,adapter,audit}.py`
· **tools:** `tools/gate5_2/{prepare_data,build_manifest,cache_lingbot,cache_scales,preflight,lidar_oracle,eval_gate5_2,diagnostics,figures,verify_index_mapping}.py`
· **config:** `configs/gate5_2/kitti360_transfer.yaml`
· **tests:** `tests/gate5_2/test_gate5_2.py` (53)
· **manifest:** `manifests/gate5_2/val.jsonl`
· **artifacts:** `artifacts/gate5_2/` (27 MB)
· **report:** this file.

**No existing module, config, checkpoint, artifact or report was modified.** `git status`
shows only the two files that were already modified before this gate began
(`lingbot_map/models/gct_stream.py`, `research/sem_bypass/model.py`).

```bash
python tools/gate5_2/prepare_data.py --what all
python tools/gate5_2/build_manifest.py
python tools/gate5_2/verify_index_mapping.py --n 40
python tools/gate5_2/cache_lingbot.py
python tools/gate5_2/preflight.py
python tools/gate5_2/cache_scales.py --variant {C0C3,A,B}
python tools/gate5_2/lidar_oracle.py          # only after the three scale tables exist
python tools/gate5_2/eval_gate5_2.py
python tools/gate5_2/diagnostics.py && python tools/gate5_2/figures.py
python -m pytest tests -q
```

| Stage | Time | Peak VRAM |
|---|---:|---:|
| Archive extraction (0.441 GB over 815 range requests) | 792 s | — |
| Index-mapping verification (40 images) | 62 s | — |
| LingBot caching, 1,753 clips × 5 frames | 216 s | 8.99 GiB |
| Preflight, 10 clips × 5 frames × 2 FOV settings | 6 s | 2.21 GiB |
| C0/C3 scales | 69 s | 0.27 GiB |
| MoGe scales, variant A | 545 s | 2.21 GiB |
| MoGe scales, variant B | 545 s | 2.21 GiB |
| LiDAR oracle + depth diagnostics (167 M pixel pairs) | 211 s | CPU |
| Occupancy evaluation, 10 conditions | 302 s | 0.08 GiB |
| Full test suite | 166 s | — |
| **Total** | **≈ 49 min** | **8.99 GiB** |

Hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q, torch 2.7.1+cu128.
Disk: 26 GB extracted targets, 3.1 GB LingBot + MoGe caches, 27 MB artifacts.

**Artifacts:** `scales_{C0C3,A,B}.{csv,json}` · `scale_frames_{A,B}.csv` ·
`oracle_scales.csv` · `gate5_2_per_clip.csv` · `gate5_2_per_{distance,height}_band.csv` ·
`gate5_2_per_block.csv` · `gate5_2_summary.json` · `scale_diagnostics.json` ·
`depth_diagnostics.json` · `preflight.json` · `manifest_summary.json` ·
`index_mapping_verification.json` · `cache_index.json` · `bev_selection.json` ·
`gate5_2_{scales,fov,bev}.png`.

**Figures.** `gate5_2_scales.png` — scale distributions, estimate-vs-oracle scatter (C3
sits visibly *above* the identity line and above `s0`; A and B straddle it) and log-scale
error. `gate5_2_fov.png` — inferred-vs-calibrated FOV and the A→B scale shift.
`gate5_2_bev.png` — bird's-eye panels for the clips at the **10th, 50th and 90th percentile
of K360-C0-D IoU**, a baseline-only selection rule that never sees A, B or the oracle. They
also make the coverage ceiling of §2 immediately visible.

---

## 14. Deviations from the brief

1. **RGB, timestamps and raw velodyne are read from the local KITTI-360 release**, not
   re-downloaded from the SSCBench archive. Justified by §3.3: the archive's own images are
   pixel-identical to the local ones at 40/40 sampled indices under the verified mapping.
   This avoided ~27 GB of redundant transfer. The official *targets* are always the
   archive's own.
2. **The primary oracle is raw projected LiDAR, not the `.bin` voxel centres.** §10 permits
   raw validation LiDAR "if it is already available and clearly aligned"; it is, and its
   alignment is proven in §3.2. Using it keeps the oracle definition identical to Gates
   4/4.1/5, so the numbers are comparable across gates. The `.bin` voxel-centre oracle is
   computed anyway as a cross-check and is labelled as quantised throughout (§9.1).
3. **A temporal eligibility rule was added** beyond "four valid historical samples":
   consecutive anchors must be exactly 5 native frames apart. Without it, 55 clips would
   span up to 8.6 s instead of 2 s, silently violating the protocol the brief specifies.
   Declared before any result was computed; all 59 exclusions are itemised in
   `manifest_summary.json`.
4. **`R_rect_00` is applied** in the velodyne↔camera chain, per the KITTI-360 devkit, even
   though the SSCBench reference loader omits it (it is a ~0.4° rotation). The grid frame
   itself was verified empirically and does not depend on this choice.
5. **The MoGe depth cache is float16.** The authoritative scales in the CSVs are computed
   from float32 outputs in the same pass; recomputing from the cache agrees to ~1e-5
   relative. Tamper-invariance is still asserted bit-exactly, since both sides read the same
   cache. Noted in the test that relies on it.

**No blockers.** The 206.7 GB archive was expected to be one; byte-range extraction of the
SquashFS image reduced it to 0.441 GB.
