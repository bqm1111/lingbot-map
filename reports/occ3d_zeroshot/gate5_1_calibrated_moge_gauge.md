# Gate 5.1 — calibration-aware MoGe-2 gauge factorization

## 1. Diagnosis: `CALIBRATED_GAUGE_PARTIALLY_WORKS`

Two of the five `CALIBRATED_GAUGE_WORKS` criteria pass. `CALIBRATED_GAUGE_FAILS` also
partly fires; §5 documents the overlap and the reason for the chosen diagnosis.

## 2. Plain-language conclusion

Gate 5 left a puzzle: a frozen MoGe-2 reached LiDAR-oracle parity on Occ3D-nuScenes yet
made SemanticKITTI worse than a constant. Gate 5.1 shows that puzzle was **mostly a
camera-formation artefact, and that fixing it trades one dataset against the other**.

Supplying the calibrated horizontal FOV repairs SemanticKITTI decisively. The KITTI scale
bias falls from **14.8 % to 6.4 %** (G51-B) and occupancy rises from 0.1246 to 0.1584; adding
the aspect-safe crop as well (**G51-D**, the predeclared primary) reaches **0.1800**, which
beats the constant (0.1592), the KITTI-learned head (0.1759) and even the LiDAR oracle
(0.1757), recovering **125.6 %** of the oracle gain and testing **non-inferior** to C3
(+0.0047, CI lower bound −0.0003 > the −0.01 margin). The Gate-5 source regression is gone.

But the same change costs Occ3D. There the processed aspect ratio is 1.76:1, so the crop is
the identity and G51-D reduces to G51-B — pure calibrated FOV. That moves the Occ3D scale
from a near-perfect 0.983× the oracle to 1.079×, and occupancy from 0.2282 down to 0.2070
(**−0.0212**, CI [−0.0283, −0.0148], twice the −0.01 margin, so **not** non-inferior).
Oracle-gain retention drops from 98.5 % to 68.0 %.

Two honest observations complicate the simple story. First, **calibrated FOV makes the
gauge more *consistent* but not less *biased***: on Occ3D it raises the log-scale
correlation with the oracle from 0.961 to 0.986 and cuts within-clip dispersion from 0.026
to 0.019, while introducing a systematic +8 % gain error. It fixes the ranking and breaks
the offset. Second, **occupancy IoU is not monotone in scale bias**: on KITTI, G51-B has the
smaller bias (6.4 % vs 9.9 %) yet G51-D scores higher (0.1800 vs 0.1584), because over-scaling
pushes more points into the far half of the grid where the ground truth lives. G51-D is not
better because its scale is better — it is better despite being worse-calibrated.

Nothing was trained, tuned or selected on results. G51-D was named the primary candidate in
the config before any occupancy number existed.

## 3. Does FOV, aspect ratio, both, or neither explain the KITTI bias?

**Predominantly FOV. The aspect-safe crop on its own makes the bias worse, not better.**

| variant | what changed | KITTI scale bias (median \|rel err\|) | KITTI ratio to oracle | KITTI IoU + `dilate_r2` |
|---|---|---:|---:|---:|
| G51-A | nothing (Gate 5) | 14.8 % | 0.852 | 0.1246 |
| **G51-B** | **calibrated FOV only** | **6.4 %** | 0.936 | 0.1584 |
| G51-C | aspect-safe crop only | **18.5 %** ↑ | 0.815 | 0.1074 ↓ |
| G51-D | both | 9.9 % | 1.099 | **0.1800** |

Isolating the factors: FOV alone removes 57 % of the scale bias and adds +0.0334 IoU
(CI [+0.0265, +0.0416], 21/21 blocks). The crop alone *increases* bias by a quarter and costs
−0.0169 IoU (CI [−0.0193, −0.0146], 0/21 blocks improved). MoGe over-estimated KITTI's FOV
by 5.1° on the full lattice (86.9° vs 81.8°) and by **12.4°** on the crop (66.9° vs 54.5°) —
cropping to 2:1 did not make MoGe's own FOV inference more accurate, it made it *less*
accurate, which is why C is the worst variant. The crop is only useful in combination,
because it changes which calibrated FOV is supplied (54.5° instead of 81.8°), and that
larger correction happens to land better in occupancy.

So: the out-of-range aspect ratio is **not** the mechanism. The mechanism is that MoGe's
self-inferred FOV is systematically too wide on KITTI-like imagery, and a too-wide FOV
yields too-small metric depth.

## 4. Main results

### 4.1 Provenance and frozen state

Everything re-verified before any Gate-5.1 number was computed. All eight LingBot-side
hashes match Gate 4.1 (`lingbot ee665103…`, `depth_head 2a91822e…`, `full_s0 05730ac7…`,
`occ_only_s0 b90a6e4e…`, `c0_corrector_s0 65900493…`, plus the three configs). MoGe pin
unchanged: repo `74fbce054ebed49800de42d0ad0e83495065719a`, model `Ruicheng/moge-2-vitl`
@ `39c4d5e957afe587e04eec59dc2bcc3be5ecd968`, class `moge.model.v2.MoGeModel`, weights
`3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5`, MIT, 326 209 221 params.
Frozen scalars: `s0 = 27.3665`, conf ≥ 1.5, teacher depth range (1, 60) m, ≥ 500 valid
pixels, canonical voxel 0.2 m, `dilate_r2` = 0.4 m, V3 not run.

**All 16 Gate-5 baselines reproduced exactly** (|Δ| ≤ 5e-4) before the new variants ran, and
G51-A reproduces the Gate-5 per-clip scales to < 1e-9 on both datasets.

### 4.2 Crop and calibration, as measured

| dataset | processed lattice | AR | crop | teacher lattice | calibrated FOV_x | MoGe self-inferred FOV_x |
|---|---|---:|---|---|---:|---:|
| Occ3D CAM_FRONT | 518 × 294 | 1.762 | **identity** | 518 × 294 | 65.12° | 68.83° |
| SemanticKITTI | 518 × 154 | 3.364 | **x0=100…408, w=308** (22 × 14) | 308 × 154 (AR 2.000) | 54.54° | 66.90° |

KITTI native `fx` 707.091, `cx` 601.887 → processed `fx` 298.755, `cx` 254.305; crop
principal point 154.305, i.e. centred to within 0.3 px. No resize, pad, letterbox or stretch
— the crop is a pure integer slice, applied identically to teacher RGB and to the LingBot
depth/confidence used in the ratio.

Because Occ3D's crop is the identity, **G51-C ≡ G51-A and G51-D ≡ G51-B there, bit-exactly**
(measured ΔIoU = +0.0000 with a zero-width CI). This is asserted as a test.

### 4.3 Occ3D-nuScenes — 1 182 clips, 150 scenes

| id | IoU | P | R | TP | FP | FN | native occ | canon occ | pts in-grid | in mask |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0-R | 0.0647 | 0.519 | 0.070 | 735 | 621 | 9 381 | 7 056 | 25 741 | 0.654 | 0.185 |
| M0-D | 0.1597 | 0.523 | 0.201 | 2 114 | 2 219 | 8 003 | 20 931 | 139 279 | 0.654 | 0.197 |
| M1-R | 0.0411 | 0.547 | 0.044 | 443 | 459 | 9 674 | 7 620 | 27 296 | 0.675 | 0.121 |
| M1-D | 0.1434 | 0.562 | 0.172 | 1 754 | 1 762 | 8 362 | 22 150 | 147 904 | 0.675 | 0.161 |
| **G51-A-R** | **0.1045** | 0.681 | 0.112 | 1 109 | 594 | 9 008 | 6 983 | 25 511 | 0.836 | 0.261 |
| **G51-A-D** | **0.2282** | 0.624 | 0.282 | 2 847 | 2 241 | 7 269 | 20 819 | 138 810 | 0.836 | 0.247 |
| G51-B-R | 0.0799 | 0.648 | 0.085 | 842 | 497 | 9 274 | 7 394 | 26 791 | 0.782 | 0.190 |
| G51-B-D | 0.2070 | 0.620 | 0.250 | 2 528 | 1 954 | 7 589 | 21 784 | 145 328 | 0.782 | 0.210 |
| G51-C-R | 0.1045 | 0.681 | 0.112 | 1 109 | 594 | 9 008 | 6 983 | 25 511 | 0.836 | 0.261 |
| G51-C-D | 0.2282 | 0.624 | 0.282 | 2 847 | 2 241 | 7 269 | 20 819 | 138 810 | 0.836 | 0.247 |
| **G51-D-R** | **0.0799** | 0.648 | 0.085 | 842 | 497 | 9 274 | 7 394 | 26 791 | 0.782 | 0.190 |
| **G51-D-D** | **0.2070** | 0.620 | 0.250 | 2 528 | 1 954 | 7 589 | 21 784 | 145 328 | 0.782 | 0.210 |
| OR-R \* | 0.1027 | 0.692 | 0.109 | 1 093 | 558 | 9 023 | 7 187 | 26 277 | 0.848 | 0.246 |
| OR-D \* | 0.2292 | 0.634 | 0.281 | 2 834 | 2 151 | 7 283 | 21 308 | 142 114 | 0.848 | 0.238 |

Per-clip IoU quartiles (p25 / median / p75), dilated: M0-D 0.054/0.132/0.254 ·
G51-A-D 0.137/0.224/0.312 · **G51-D-D 0.112/0.207/0.286** · OR-D 0.135/0.229/0.307.

By distance (IoU, dilated): G51-A-D 0.443/0.316/0.192/0.071, **G51-D-D
0.393/0.279/0.180/0.077**, OR-D 0.433/0.314/0.200/0.075 for 0–10/10–20/20–30/30–40 m.
G51-D is behind A in the near field and slightly *ahead* beyond 30 m.

Oracle-gain retention (dilated): G51-A 98.5 %, G51-B 68.0 %, G51-C 98.5 %, **G51-D 68.0 %**.

### 4.4 SemanticKITTI — 163 clips, sequence 08, 21 contiguous 8-clip blocks

| id | IoU | P | R | TP | FP | FN | occ | pts in-grid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0-R | 0.0573 | 0.318 | 0.066 | 7 558 | 15 950 | 107 552 | 32 074 | 0.590 |
| M0-D | 0.1592 | 0.263 | 0.301 | 34 420 | 99 847 | 80 690 | 185 696 | 0.590 |
| M1-R | 0.0778 | 0.408 | 0.090 | 9 995 | 14 722 | 105 115 | 31 442 | 0.640 |
| M1-D | 0.1759 | 0.284 | 0.334 | 38 044 | 99 819 | 77 066 | 184 243 | 0.640 |
| G51-A-R | 0.0390 | 0.220 | 0.045 | 5 009 | 18 949 | 110 101 | 25 734 | 0.666 |
| G51-A-D | 0.1246 | 0.217 | 0.241 | 27 062 | 102 797 | 88 048 | 149 526 | 0.666 |
| G51-B-R | 0.0675 | 0.358 | 0.078 | 8 813 | 16 427 | 106 297 | 29 395 | 0.652 |
| G51-B-D | 0.1584 | 0.260 | 0.305 | 34 784 | 102 337 | 80 326 | 173 138 | 0.652 |
| G51-C-R | 0.0252 | 0.150 | 0.029 | 3 270 | 19 506 | 111 840 | 24 032 | 0.672 |
| G51-C-D | 0.1074 | 0.193 | 0.208 | 23 246 | 101 429 | 91 864 | 138 938 | 0.672 |
| **G51-D-R** | **0.0573** | 0.340 | 0.065 | 7 085 | 13 738 | 108 025 | 33 207 | 0.577 |
| **G51-D-D** | **0.1800** | 0.293 | 0.335 | 37 853 | 94 834 | 77 257 | 199 425 | 0.577 |
| OR-R \* | 0.0769 | 0.406 | 0.088 | 9 767 | 14 483 | 105 343 | 31 441 | 0.637 |
| OR-D \* | 0.1757 | 0.284 | 0.334 | 38 062 | 98 841 | 77 048 | 185 295 | 0.637 |

By distance (IoU, dilated): **G51-D-D 0.282/0.216/0.180/0.148** vs M1-D
0.282/0.209/0.177/0.136 and G51-A-D 0.216/0.151/0.121/0.073. G51-D matches or beats the
learned head in every band, with its largest margin at 30–40 m.

Oracle-gain retention (dilated): G51-A −209.3 %, G51-B −4.9 %, G51-C −313.7 %,
**G51-D +125.6 %**.

### 4.5 Scale diagnostics (LiDAR, computed only after all predictions were written)

| dataset | variant | median | p05 | p95 | median \|log err\| | median \|rel err\| | ratio to oracle | Pearson | Spearman | per-frame disp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Occ3D | oracle | 23.550 | 13.850 | 38.938 | — | — | — | — | — | — |
| Occ3D | G51-A/C | 23.378 | 13.908 | 37.812 | 0.0564 | **5.63 %** | **0.983** | 0.961 | 0.942 | 0.0256 |
| Occ3D | **G51-B/D** | 25.595 | 14.699 | 41.726 | 0.0764 | 7.93 % | 1.079 | **0.986** | **0.979** | **0.0188** |
| KITTI | oracle | 26.274 | 18.088 | 35.397 | — | — | — | — | — | — |
| KITTI | G51-A | 22.444 | 16.826 | 27.606 | 0.1598 | 14.76 % | 0.852 | 0.925 | 0.900 | 0.0301 |
| KITTI | **G51-B** | 24.638 | 17.693 | 32.575 | 0.0661 | **6.41 %** | **0.936** | **0.973** | 0.967 | 0.0273 |
| KITTI | G51-C | 21.084 | 15.867 | 26.287 | 0.2043 | 18.48 % | 0.815 | 0.936 | 0.916 | 0.0285 |
| KITTI | **G51-D** | 28.653 | 20.361 | 38.917 | 0.0945 | 9.91 % | 1.099 | 0.969 | **0.971** | 0.0285 |

Depth against projected LiDAR, all five frames (median over frames):

| dataset | variant | AbsRel | med rel | med log | RMSE (m) | δ1 | δ2 | δ3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Occ3D | G51-A/C | **0.0884** | 0.0618 | 0.0625 | 3.769 | 0.9412 | 0.9780 | 0.9873 |
| Occ3D | G51-B/D | 0.1023 | 0.0863 | 0.0833 | **3.480** | **0.9556** | 0.9816 | 0.9895 |
| Occ3D | oracle \* | 0.0611 | 0.0305 | 0.0306 | 3.544 | 0.9565 | 0.9795 | 0.9879 |
| KITTI | G51-A | 0.1502 | 0.1376 | 0.1478 | 2.772 | 0.8826 | 0.9809 | 0.9924 |
| KITTI | **G51-B** | **0.0839** | 0.0601 | 0.0614 | **2.129** | **0.9514** | 0.9859 | 0.9934 |
| KITTI | G51-C | 0.1889 | 0.1791 | 0.1973 | 3.223 | 0.7069 | 0.9757 | 0.9912 |
| KITTI | G51-D | 0.1326 | 0.1136 | 0.1077 | 2.488 | 0.9346 | 0.9849 | 0.9934 |
| KITTI | oracle \* | 0.0655 | 0.0367 | 0.0365 | 2.009 | 0.9587 | 0.9866 | 0.9935 |

Depth error is flat across temporal offsets for every variant (e.g. KITTI G51-D
0.137/0.134/0.128/0.133/0.135 for offsets −4…0), confirming one scalar serves all five
frames. By depth band, G51-D improves most in the far field on both datasets
(KITTI 40–80 m: 0.237 → 0.163; Occ3D 40–80 m: 0.403 → 0.349).

### 4.6 Scene- and block-level paired bootstrap (10 000 resamples, seed 0)

**Occ3D — unit = official nuScenes scene (150), clips averaged within scene first**

| contrast | raw ΔIoU | 95 % CI | scenes | dilated ΔIoU | 95 % CI | scenes |
|---|---:|---|---:|---:|---|---:|
| G51-A → G51-B | −0.0248 | [−0.0290, −0.0206] \* | 23/150 | −0.0212 | [−0.0283, −0.0148] \* | 49/150 |
| G51-A → G51-C | +0.0000 | [+0.0000, +0.0000] | 0/150 | +0.0000 | [+0.0000, +0.0000] | 0/150 |
| G51-A → G51-D | −0.0248 | [−0.0290, −0.0206] \* | 23/150 | **−0.0212** | **[−0.0283, −0.0148] \*** | 49/150 |
| M0 → G51-D | +0.0150 | [+0.0058, +0.0240] \* | 104/150 | **+0.0470** | **[+0.0339, +0.0599] \*** | 117/150 |
| M1 → G51-D | +0.0389 | [+0.0332, +0.0449] \* | 138/150 | +0.0636 | [+0.0556, +0.0715] \* | 139/150 |
| G51-D → oracle | +0.0228 | [+0.0195, +0.0262] \* | 139/150 | +0.0222 | [+0.0183, +0.0264] \* | 128/150 |

**SemanticKITTI — unit = contiguous 8-clip block (21). These are blocks of one sequence,
not independent scenes.**

| contrast | raw ΔIoU | 95 % CI | blocks | dilated ΔIoU | 95 % CI | blocks |
|---|---:|---|---:|---:|---|---:|
| G51-A → G51-B | +0.0283 | [+0.0215, +0.0355] \* | 20/21 | +0.0334 | [+0.0265, +0.0416] \* | 21/21 |
| G51-A → G51-C | −0.0135 | [−0.0172, −0.0100] \* | 0/21 | −0.0169 | [−0.0193, −0.0146] \* | 0/21 |
| G51-A → G51-D | +0.0205 | [+0.0092, +0.0330] \* | 17/21 | **+0.0554** | **[+0.0471, +0.0630] \*** | 21/21 |
| M0 → G51-D | +0.0001 | [−0.0059, +0.0063] | 9/21 | **+0.0205** | **[+0.0140, +0.0273] \*** | 19/21 |
| M1 → G51-D | −0.0190 | [−0.0257, −0.0121] \* | 1/21 | +0.0047 | [−0.0003, +0.0093] | 15/21 |
| G51-D → oracle | +0.0184 | [+0.0128, +0.0236] \* | 20/21 | −0.0048 | [−0.0109, +0.0018] | 7/21 |

**Non-inferiority (predeclared margin −0.01 IoU, dilated):**

| test | Δ | 95 % CI | lower bound vs margin | verdict |
|---|---:|---|---|---|
| Occ3D: G51-D-D − G51-A-D | −0.0212 | [−0.0283, −0.0148] | −0.0283 < −0.01 | **NOT non-inferior** |
| KITTI: G51-D-D − M1-D (C3) | +0.0047 | [−0.0003, +0.0093] | −0.0003 > −0.01 | **NON-INFERIOR** |

Where a CI crosses zero we report *no statistically significant difference was detected*;
we do not claim equivalence. Non-inferiority is claimed only where the lower bound exceeds
the −0.01 margin.

## 5. Decision-rule audit

| # | `CALIBRATED_GAUGE_WORKS` criterion | measured | verdict |
|---|---|---|:--:|
| 1 | G51-D median absolute scale bias ≤ 5 % on **both** datasets | Occ3D 7.93 %, KITTI 9.91 % (median \|rel err\|); 7.90 % / 9.91 % as \|median ratio − 1\| | ✘ |
| 2 | Occ3D: G51-D retains ≥ 90 % of the C0→oracle dilation gain | **68.0 %** | ✘ |
| 3 | Occ3D: G51-D non-inferior to G51-A at −0.01 | −0.0212 [−0.0283, −0.0148] | ✘ |
| 4 | KITTI: G51-D non-inferior to C3-D at −0.01 | +0.0047 [−0.0003, +0.0093] | ✔ |
| 5 | no leakage or protocol failure | 32 Gate-5.1 tests + 485 repo-wide | ✔ |

Two of five pass, so `CALIBRATED_GAUGE_WORKS` cannot be declared.

**Overlap with `CALIBRATED_GAUGE_FAILS`.** That branch fires on any one of three clauses,
and its second — "causes an important Occ3D regression" — does fire: −0.0212 is twice the
gate's own −0.01 materiality margin, with a CI excluding zero. Its first clause is
contradicted decisively (G51-D *does* materially repair SemanticKITTI: +0.0554 over G51-A on
21/21 blocks, from below the constant to above the learned head and the LiDAR oracle), and
its third does not apply.

**Why `PARTIALLY_WORKS` is the chosen description.** Its conditions are met literally:
calibration handling significantly reduces the KITTI bias (14.8 % → 9.9 %, and 6.4 % for the
FOV-only variant), significantly improves KITTI occupancy, preserves most of the Occ3D gain
(68 % of the oracle gain; still +0.0470 over the constant and +0.0636 over the learned head,
both with CIs excluding zero), and one or more full criteria fail. Calling this a failure
would misdescribe a variant that turns an inconsistent gauge into a consistent one: the
worst-case dilated IoU across the two datasets rises from **0.1246** (G51-A) to **0.1800**
(G51-D). The overlap is recorded here rather than resolved silently.

## 6. Leakage audit

| # | guarantee | test |
|---|---|---|
| 1 | LingBot and MoGe pins and weights unchanged | `test_lingbot_and_moge_pins_are_unchanged` |
| 2 | no optimizer, backward call or trainable parameter | `test_no_optimizer_backward_or_trainable_parameter_in_gate5_1` |
| 3 | only RGB + (where declared) one scalar FOV reach MoGe — AST check of the single call site: one positional arg, keywords exactly `{apply_mask, fov_x}` | `test_only_rgb_and_scalar_fov_reach_moge` |
| 4 | labels, masks, LiDAR, extrinsics, pose, camera height never reach the estimator or cache tool | `test_estimator_and_cache_tool_never_open_a_label_lidar_or_pose`, `test_cached_scale_tables_carry_no_target_column` |
| 5 | randomising labels/masks/LiDAR leaves all variant scales bit-identical (full and cropped paths) | `test_randomising_targets_leaves_every_variant_scale_bit_identical` |
| 6 | G51-A reproduces Gate 5 to < 1e-9 per clip, both datasets | `test_g51a_reproduces_gate5_scales_exactly` |
| 7 | crop is identity when AR ≤ 2 | `test_crop_is_identity_when_aspect_ratio_at_most_two` |
| 8 | crop deterministic, patch-aligned, ≤ 2:1, full height, clipped | `test_crop_is_deterministic_patch_aligned_and_within_two_to_one` |
| 9 | intrinsics transformed correctly through preprocessing and crop | `test_intrinsics_are_transformed_through_preprocessing_and_crop` |
| 10 | crop slices teacher and student identically | `test_crop_slices_teacher_and_student_identically` |
| 11 | no interpolation, padding, letterboxing or resize in the crop path | `test_crop_is_a_pure_slice_with_no_interpolation_or_padding` |
| 12 | calibrated FOV correct on synthetic cameras (30/60/90/120°) | `test_calibrated_fov_on_synthetic_cameras_with_known_answers` |
| 13–14 | A/C pass no FOV; B/D pass only the intended calibrated FOV | `test_only_rgb_and_scalar_fov_reach_moge`, preflight record |
| 15 | optical-axis z, not Euclidean ray | `test_optical_axis_z_not_euclidean_ray` + runtime assertion |
| 16 | invalid teacher pixels cannot affect the estimate | `test_invalid_teacher_pixels_cannot_affect_the_estimate` |
| 17 | exactly one scalar per clip | `test_exactly_one_scalar_per_clip` |
| 18–19 | one scalar to all five depths and translations, once; rotations and anchor unchanged; pure similarity | `test_one_scalar_applied_once_to_all_depths_and_translations` |
| 20 | MoGe outputs cannot enter fusion (signature + AST scan) | `test_moge_outputs_cannot_enter_fusion` |
| 21 | `dilate_r2` remains the frozen 0.4 m operation, distinct from the 0.6 m band | `test_dilate_r2_remains_the_frozen_0_4_m_operation` |
| 22 | Gates 4, 4.1 and 5 tests still pass | full-suite run, §8 |

Additional: `test_identity_crop_makes_c_equal_a_and_d_equal_b_on_occ3d` (bit-exact),
`test_moge_expects_degrees` (unit assumption checked against the pinned source),
`test_coverage_and_bootstrap_units` (1 182/150 and 163/21, margin −0.01, equivalence
disclaimer present).

**Coverage:** all 1 182 Occ3D clips and all 163 KITTI clips produced a scale for every
variant. **Zero failures, zero exclusions, zero silent fallbacks to C0 or C3.**

## 7. Preflight (label-free, 10 clips per dataset, all four variants)

Every check passed: shapes match LingBot, `depth == points[...,2]`, all masked depths
finite and positive, FOV finite, RGB in [0,1], exactly one scalar per clip (40/40), and the
crop path contains no interpolation, padding or resize. Recorded per variant: crop
coordinates, teacher lattice, supplied and inferred FOV, MoGe mask fraction (0.90 Occ3D /
0.96–0.97 KITTI), valid scale pixels (68 k Occ3D, 35–60 k KITTI), per-frame implied scale
and within-clip dispersion. `artifacts/gate5_1/preflight.json`.

## 8. Tests, runtime and memory

**485 passed, 2 skipped** repo-wide (both skips pre-existing and unrelated). New:
`tests/gate5_1` — 32 tests.

| suite | result |
|---|---|
| `tests/scale_gate` | 49 |
| `tests/depth_gate` | 35 |
| `tests/voxel_gate` | 29 |
| `tests/voxel_gate_validation` | 24 |
| `tests/occ3d_zeroshot` (Gates 4, 4.1) | 43 |
| `tests/gate5` | 23 |
| `tests/gate5_1` (new) | 32 |
| remainder | 250 passed, 2 skipped |

| stage | wall time | peak VRAM |
|---|---:|---:|
| MoGe scales, 4 variants × KITTI (163 clips) | 43–47 s each | 2.21–2.23 GiB |
| MoGe scales, 4 variants × Occ3D (1 182 clips) | 389–393 s each | 2.22 GiB |
| occupancy evaluation, KITTI (7 conditions) | 49 s | 0.34 GiB |
| occupancy evaluation, Occ3D (7 conditions) | 601 s | 0.64 GiB |
| LiDAR diagnostics, KITTI / Occ3D | 21 s / 189 s | — |
| preflight, figures | ~90 s | 2.2 GiB |

Total ≈ 46 min GPU. Hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q (97.9 GB),
torch 2.7.1+cu128.

## 9. Limitations

1. **MoGe-2 acquired its metric knowledge from its own large-scale metric training.** This
   gate reuses that knowledge; it does not create metric information from nothing.
2. **Our downstream pipeline uses no LiDAR supervision for G51 predictions, but the teacher
   is not supervision-free.** "LiDAR-free" describes our pipeline, not the world.
3. **Supplying calibrated FOV changes the method** from RGB-only scale transfer to
   **calibration-aware** scale transfer. G51-B and G51-D are no longer RGB-only.
4. Camera calibration is standard inference metadata, already required for voxel
   projection — not occupancy supervision.
5. **Occ3D and SemanticKITTI are now development datasets for this method.** Both have
   informed the design across Gates 4.1, 5 and 5.1.
6. **A fresh dataset, unused in all previous gates, is required for the final frozen-transfer
   claim.** Nothing here is a held-out result any more.
7. Not directly comparable to OccAny's six-camera surround-view Occ3D number.
8. **A confidence interval crossing zero does not prove equivalence.** The KITTI
   M1 → G51-D dilated contrast (+0.0047, CI [−0.0003, +0.0093]) shows no significant
   difference *and* passes the −0.01 non-inferiority test; those are different claims.
9. **G51-D was predeclared as primary.** G51-A scores higher on Occ3D and G51-B has the
   lowest KITTI scale bias, but neither was selected post hoc, and this report does not
   promote them.
10. The KITTI bootstrap uses 21 contiguous blocks of a single sequence; its intervals are
    optimistic relative to genuinely independent scenes.
11. The processed lattices have mildly non-square pixels (KITTI `sx`/`sy` = 0.4225/0.4162,
    1.5 % anisotropy; Occ3D 0.9 %), while MoGe's `fov_x` path assumes square pixels. This
    affects all variants identically and is not isolated here.
12. **Occupancy IoU is not monotone in scale bias** (§2). Reading G51-D's occupancy win on
    KITTI as evidence of better calibration would be wrong.

## 10. Recommended next experiment (exactly one)

Per `CALIBRATED_GAUGE_PARTIALLY_WORKS`, one narrowly defined remaining gauge test, **no
student training**:

> **Measure whether the calibrated-FOV gain error is a fixed multiplicative offset that a
> single global constant removes — using only the two development datasets already
> prepared, and validating the constant by leave-one-dataset-out.**

The evidence points there and nowhere else. Supplying calibrated FOV *improved the
gauge's consistency on both datasets* — Occ3D Pearson 0.961 → 0.986 with within-clip
dispersion 0.026 → 0.019, KITTI 0.925 → 0.973 — while introducing a systematic gain error
in the same direction on both (Occ3D ratio 1.079, KITTI G51-B ratio 0.936 rising to 1.099
once the crop narrows the supplied FOV). Better ranking with a shifted gain is the exact
signature of a correctable multiplicative offset, and it is cheap to test: the per-clip
scales, oracle scales and FOV provenance for 1 345 clips are already written to
`artifacts/gate5_1/`. Fit one constant on Occ3D, apply it unchanged to KITTI, and then the
reverse; if a single constant brings both datasets' median bias under 5 % it converts
criterion 1 from a fail to a pass without any learning, and criterion 3 becomes testable
again. If no single constant works, the offset is dataset-specific and the distillation
plan needs a different teacher or an ensemble.

This is deliberately **not** teacher-to-student distillation. Criterion 3 failed: G51-D
regresses on Occ3D beyond the gate's own margin, and distilling now would bake that
regression into the student as a ceiling.

**Not executed in this gate:** student distillation, multi-source pseudo-scale generation,
fresh-dataset evaluation, V3.
