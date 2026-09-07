# Gate 5 — frozen MoGe-2 metric-gauge transfer

## Diagnosis: `MOGE_SCALE_PARTIALLY_WORKS`

The rule that fires is *"MoGe significantly improves over C0/C3 but … **behaves
inconsistently across datasets**"*. On the target it is close to perfect; on the source it
is a significant regression. §16 records that `MOGE_SCALE_FAILS` also partly fires and why
it is the worse description.

## Plain-language conclusion

On Occ3D-nuScenes a frozen MoGe-2, given nothing but RGB and contributing **one scalar per
five-frame clip**, essentially solves the metric-scale problem Gate 4.1 identified. Its
median relative scale error against the LiDAR oracle is **5.6 %**, against 22.9 % for the
frozen constant and 25.5 % for the KITTI-learned head — a 4× improvement over both — and it
correlates with the oracle at Pearson 0.961. Fed into the unchanged fusion, it lifts
occupancy from 0.1597 to **0.2282** with `dilate_r2` (+0.0682 [+0.0579, +0.0788], 129/150
scenes), which is **98.5 % of the entire LiDAR-oracle gain**; M2 and the oracle are
statistically indistinguishable (M2-D→OR-D +0.0010, CI spans zero). Depth error at
projected-LiDAR pixels collapses from AbsRel 0.236 (δ1 0.555) to 0.088 (δ1 0.941), against
the oracle's 0.061.

But it does not transfer *back*. On SemanticKITTI — the source domain, where the existing
learned head is already excellent — MoGe systematically **under**-estimates scale by 15 %
(ratio to oracle 0.852), and occupancy falls from 0.1592 to **0.1246** with dilation
(−0.0349 [−0.0437, −0.0260], improving in only 1 of 21 blocks). It is worse than the plain
constant there. So MoGe is not a universally better scale source; it is a much better one
on nuScenes-like imagery and a worse one on KITTI's cropped, wide-baseline stereo rig.

Three things must be said plainly. **MoGe-2 already received large-scale metric supervision
during its own training**; this experiment *reuses* that metric knowledge and does not
create metric information from nothing. A student distilled from it would remove the need
for LiDAR supervision *in our pipeline*, not in the world. And **Occ3D is now a development
benchmark**, because it informed this method design — it is no longer an untouched target.

Nothing was trained, distilled or tuned. No optimiser exists; MoGe and LingBot parameters
are all `requires_grad=False`. MoGe's depth, point map and predicted intrinsics never enter
the reconstruction — only the scalar does.

---

## 1. Provenance and exact revisions

| item | value |
|---|---|
| MoGe repository | `https://github.com/microsoft/MoGe` @ **`74fbce054ebed49800de42d0ad0e83495065719a`** (2026-08-19) |
| model | **`Ruicheng/moge-2-vitl`** @ revision **`39c4d5e957afe587e04eec59dc2bcc3be5ecd968`** |
| model class | `moge.model.v2.MoGeModel` — **v2**, the metric model (has `scale_head`); not v1, not v3, not `-normal` |
| weight file | `model.pt`, SHA-256 **`3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5`** |
| parameters | 326 209 221 |
| license | **MIT** |
| added dependency | `utils3d_moge @ git+…@62f09d58509485564e24d5d9f6aac9ee9ebc0c37`, installed `--no-deps` |
| environment | unchanged: torch 2.7.1+cu128, numpy 2.2.6, opencv 4.13.0.92, timm 1.0.25, huggingface_hub 1.28.0 — verified byte-identical before/after install |

The repository HEAD now contains MoGe-**3**; the pinned class and revision are asserted by
`test_moge_revision_is_pinned_to_the_metric_v2_model`.

### Frozen LingBot-side components, re-verified against Gate 4.1

`lingbot ee665103348e07e6` · `depth_head 2a91822ea5d1182a` · `full_s0 05730ac7addb101d` ·
`occ_only_s0 b90a6e4ee9cdeccd` · `c0_corrector_s0 6590049302ef2044` ·
`gate31_config a548ed08fcbe1ea6` · `gate2_config b7692022114830f9` ·
`gate0_config 472b45311a5e03ba`. Frozen scalars unchanged: `s0 = 27.3665`, conf ≥ 1.5,
depth range (1, 60) m, canonical voxel 0.2 m, `dilate_r2` = 0.4 m, V3 band 0.6 m.

**Step-1 baseline reproduction, before any MoGe number was computed** — all exact:

| | required | reproduced |
|---|---:|---:|
| Occ3D C0 raw | 0.0647 | 0.0647 |
| Occ3D C0 + `dilate_r2` | 0.1597 | 0.1597 |
| Occ3D C3 raw | 0.0411 | 0.0411 |
| Occ3D C3 + `dilate_r2` | 0.1434 | 0.1434 |
| Occ3D LiDAR-oracle raw | 0.1027 | 0.1027 |
| Occ3D LiDAR-oracle + `dilate_r2` | 0.2292 | 0.2292 |
| KITTI C0 raw / C3 raw / C3 + `dilate_r2` | 0.0573 / 0.0778 / 0.1759 | 0.0573 / 0.0778 / 0.1759 |

---

## 2. Depth and coordinate conventions

MoGe returns points in OpenCV camera coordinates (+x right, +y down, +z forward).
The gauge compares **optical-axis z-depth with optical-axis z-depth**:

```
D_moge := points_moge[..., 2]          asserted equal to out["depth"] at every call
                                       (measured max |diff| = 0.0)
```

Euclidean ray distance `norm(points, axis=-1)` is a different quantity — on a real frame it
differs from z by up to **20.8 m** — and is never used. A unit test pins the 3-4-5 case.

MoGe is called as `model.infer(image, apply_mask=False)`: **no `fov_x`, no intrinsics, no
extrinsics, no focal length, no camera height, no ego pose, no LiDAR.** It infers its own
field of view from RGB. An AST test asserts the single call site passes exactly one
positional argument and only the `apply_mask` keyword.

Predicted FOV_x (MoGe's own, from RGB alone): Occ3D median **68.8°** (true CAM_FRONT
64.6°); KITTI median **86.9°** (true ≈ 81°). Both are over-estimates, and the KITTI
over-estimate is the larger one.

---

## 3. Pixel-alignment verification

MoGe is fed the **identical RGB LingBot consumed** — the same `mode="crop"` geometric
resize at `image_size=518, patch_size=14` — converted back to `[0, 1]` without LingBot's
normalisation. MoGe returns depth at the input resolution, so teacher and student pixels
correspond exactly and **no interpolation is performed at all**:

| dataset | native | LingBot / MoGe lattice |
|---|---|---|
| Occ3D-nuScenes CAM_FRONT | 1600 × 900 | **518 × 294** |
| SemanticKITTI seq 08 | 1226 × 370 | **518 × 154** |

A per-frame runtime assertion fails with *"pixel correspondence broken"* on any mismatch,
and a test asserts the caching tool contains no `interpolate`, `grid_sample` or
`cv2.resize`.

### Label-free preflight (10 deterministic clips per dataset)

| check | Occ3D | KITTI |
|---|---|---|
| shapes match LingBot | ✔ | ✔ |
| `depth == points[...,2]` | ✔ | ✔ |
| all depths finite and positive inside MoGe's mask | ✔ | ✔ |
| predicted FOV finite | ✔ | ✔ |
| RGB in [0, 1] | ✔ | ✔ |
| MoGe depth median (m) | 18.24 [12.88, 24.19] | 12.52 [8.05, 20.40] |
| MoGe mask fraction | 0.899 | 0.971 |
| valid-pixel fraction | 0.446 | 0.758 |
| implied per-frame scale | 18.84 | 25.34 |

No occupancy label, camera mask or LiDAR file was opened. No threshold was tuned on these
clips.

---

## 4. The scale estimator (predeclared)

```
valid_i(p) = MoGe mask
           ∧ finite, positive D_moge   ∧ 1 m < D_moge < 60 m      (already metric)
           ∧ finite, positive D_lingbot
           ∧ LingBot confidence ≥ 1.5

r_i(p)     = log D_moge_i(p) − log D_lingbot_i(p)
log_s_moge = weighted_median over ALL valid pixels of ALL FIVE frames, weight = confidence
s_moge     = exp(log_s_moge)                                    ONE scalar per clip
```

`weighted_median` is the same function object the Gate-4/4.1 LiDAR diagnostic uses
(asserted by identity test). The frozen (1, 60) m range is applied to the **MoGe** depth,
which is already metric; applying it to LingBot depth would require the unknown scale and
would be circular. That choice, the 500-valid-pixel minimum and the
`exclude_and_report` failure policy were all fixed in the config before any occupancy
number existed. No semantic, ground, dynamic-object or target-derived mask is used.

### Coverage and failure policy

| dataset | clips | scale estimated | failed | valid pixels/clip (median / min) |
|---|---:|---:|---:|---|
| Occ3D-nuScenes val | **1 182** | **1 182 (100 %)** | 0 | 416 644 / 109 845 |
| SemanticKITTI seq 08 | **163** | **163 (100 %)** | 0 | 320 697 / 231 295 |

No clip was excluded and no clip silently fell back to C0 or C3. Within-clip agreement is
tight: median per-frame log-scale std **0.026** (Occ3D) and **0.030** (KITTI); median
log-ratio MAD 0.037 and 0.049.

---

## 5. Applying the scale

```
depth_i        = s_moge · D_lingbot_i          all five frames, the same scalar
t_rel_i        = s_moge · t_lingbot_i          translation scaled ONCE
R_rel_i        = R_lingbot_i                   rotation never scaled
```

Asserted (synthetic and runtime): `t(s) == s·t(1)` and `≠ s²·t(1)`; rotation blocks
bit-identical across `s`; the anchor relative transform is exactly the identity;
`fuse(s·D, s·t) == s·fuse(D, t)` to 1e-9, i.e. a pure similarity. A signature test proves
`fuse` accepts only LingBot tensors, and a source scan proves the fusion module contains no
reference to MoGe — **MoGe depth, points and intrinsics have no path into the
reconstruction.**

---

## 6. Occ3D-nuScenes results — development benchmark

1 182 clips, 150 official val scenes, five CAM_FRONT keyframes at 2 Hz, last-frame anchor,
canonical 0.2 m grid → native 0.4 m by the frozen any-subvoxel rule, official camera mask
and single-camera x-cut. `*` = non-deployable diagnostic oracle. **V3 was not run.**

| id | scale | corr. | IoU | P | R | TP | FP | FN | native occ | canon occ | pts in-grid | in mask |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0-R | C0 const | raw | 0.0647 | 0.519 | 0.070 | 735 | 621 | 9 381 | 7 056 | 25 741 | 0.654 | 0.185 |
| M0-D | C0 const | dilate | 0.1597 | 0.523 | 0.201 | 2 114 | 2 219 | 8 003 | 20 931 | 139 279 | 0.654 | 0.197 |
| M1-R | C3 learned | raw | 0.0411 | 0.547 | 0.044 | 443 | 459 | 9 674 | 7 620 | 27 296 | 0.675 | 0.121 |
| M1-D | C3 learned | dilate | 0.1434 | 0.562 | 0.172 | 1 754 | 1 762 | 8 362 | 22 150 | 147 904 | 0.675 | 0.161 |
| **M2-R** | **MoGe-2** | raw | **0.1045** | 0.681 | 0.112 | 1 109 | 594 | 9 008 | 6 983 | 25 511 | **0.836** | 0.261 |
| **M2-D** | **MoGe-2** | dilate | **0.2282** | 0.624 | 0.282 | 2 847 | 2 241 | 7 269 | 20 819 | 138 810 | **0.836** | 0.247 |
| OR-R \* | LiDAR oracle | raw | 0.1027 | 0.692 | 0.109 | 1 093 | 558 | 9 023 | 7 187 | 26 277 | 0.848 | 0.246 |
| OR-D \* | LiDAR oracle | dilate | 0.2292 | 0.634 | 0.281 | 2 834 | 2 151 | 7 283 | 21 308 | 142 114 | 0.848 | 0.238 |

Per-clip IoU quartiles (p25 / median / p75): M0-D 0.054 / 0.132 / 0.254 · M1-D 0.067 /
0.122 / 0.198 · **M2-D 0.137 / 0.224 / 0.312** · OR-D 0.135 / 0.229 / 0.307.

### By distance (IoU)

| id | 0–10 m | 10–20 m | 20–30 m | 30–40 m |
|---|---:|---:|---:|---:|
| M0-D | 0.3122 | 0.2173 | 0.1416 | 0.0695 |
| M1-D | 0.2761 | 0.1922 | 0.1253 | 0.0621 |
| **M2-D** | **0.4427** | **0.3157** | 0.1916 | 0.0714 |
| OR-D \* | 0.4334 | 0.3136 | 0.1998 | 0.0751 |

M2 matches or slightly exceeds the oracle in the near field and falls a little behind
beyond 20 m, where the oracle's per-clip scale is better matched to distant surfaces.

**Occupancy-gain recovery (unclipped):**

```
recovery_raw    = (0.1045 − 0.0647) / (0.1027 − 0.0647) = +104.8 %
recovery_dilate = (0.2282 − 0.1597) / (0.2292 − 0.1597) =  +98.5 %
```

---

## 7. SemanticKITTI results — source protocol

Gate-3.1 protocol: 163 clips, sequence 08, 0.2 m grid, same fusion, same `dilate_r2`.

| id | IoU | P | R | native occ | median scale |
|---|---:|---:|---:|---:|---:|
| M0-R | 0.0573 | 0.318 | 0.066 | 32 074 | 27.366 |
| M0-D | 0.1592 | 0.263 | 0.301 | 185 696 | 27.366 |
| M1-R | 0.0778 | 0.408 | 0.090 | 31 442 | 26.245 |
| M1-D | 0.1759 | 0.284 | 0.334 | 184 243 | 26.245 |
| **M2-R** | **0.0390** | 0.220 | 0.045 | 25 734 | 22.348 |
| **M2-D** | **0.1246** | 0.217 | 0.241 | 149 526 | 22.348 |
| OR-R \* | 0.0769 | 0.406 | 0.088 | 31 441 | 26.438 |
| OR-D \* | 0.1757 | 0.284 | 0.334 | 185 295 | 26.438 |

```
recovery_raw    = (0.0390 − 0.0573) / (0.0769 − 0.0573) =  −92.7 %
recovery_dilate = (0.1246 − 0.1592) / (0.1757 − 0.1592) = −209.3 %
```

MoGe moves *away* from the oracle on the source domain. This is a **major regression** and
is the reason `MOGE_SCALE_WORKS` cannot be declared.

---

## 8. LiDAR-oracle scale diagnostics (diagnostic only)

Computed **after** every deployable prediction was written to disk; the diagnostics tool
refuses to run before `eval_gate5.py` has produced its summary. M2 never touched LiDAR.

### Occ3D-nuScenes (1 182 clips, 5 910 frames)

| scale | median | p05 | p25 | p75 | p95 | median \|log err\| | median \|rel err\| | ratio to oracle | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LiDAR oracle | 23.550 | 13.850 | — | — | 38.938 | — | — | — | — | — |
| C0 constant | 27.366 | 27.366 | 27.366 | 27.366 | 27.366 | 0.2198 | **22.95 %** | 1.162 | 0.000 | −0.239 |
| C3 learned | 29.261 | 18.439 | 23.6 | 35.0 | 44.805 | 0.2273 | **25.48 %** | 1.253 | 0.930 | 0.952 |
| **MoGe-2** | **23.378** | 13.908 | 19.0 | 28.8 | 37.812 | **0.0564** | **5.63 %** | **0.983** | **0.961** | 0.942 |

### SemanticKITTI (163 clips, 815 frames)

| scale | median | median \|rel err\| | ratio to oracle | Pearson |
|---|---:|---:|---:|---:|
| LiDAR oracle | 26.274 | — | — | — |
| C0 constant | 27.366 | 12.68 % | 1.042 | 0.000 |
| C3 learned | 26.599 | **2.90 %** | 0.995 | 0.982 |
| MoGe-2 | 22.444 | 14.76 % | **0.852** | 0.925 |

MoGe's *ranking* of clips is good on both datasets (Pearson 0.96 / 0.93); what differs is
the **bias**. On nuScenes it is essentially unbiased (0.983); on KITTI it is 15 % low.

### Depth against projected LiDAR (all five frames; M2 uses no LiDAR, so this is leak-free)

| dataset | scale | AbsRel | med rel | med log | RMSE (m) | δ1 | δ2 | δ3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Occ3D | C0 | 0.2364 | 0.2282 | 0.2183 | 4.848 | 0.5553 | 0.9661 | 0.9878 |
| Occ3D | C3 | 0.2568 | 0.2580 | 0.2299 | 4.438 | 0.4317 | 0.9810 | 0.9917 |
| Occ3D | **MoGe-2** | **0.0884** | 0.0618 | 0.0625 | 3.769 | **0.9412** | 0.9780 | 0.9873 |
| Occ3D | oracle \* | 0.0611 | 0.0305 | 0.0306 | 3.544 | 0.9565 | 0.9795 | 0.9879 |
| KITTI | C0 | 0.1450 | 0.1288 | 0.1310 | 2.901 | 0.9091 | 0.9817 | 0.9921 |
| KITTI | C3 | 0.0703 | 0.0426 | 0.0426 | 2.015 | 0.9580 | 0.9866 | 0.9936 |
| KITTI | **MoGe-2** | 0.1502 | 0.1376 | 0.1478 | 2.772 | 0.8826 | 0.9809 | 0.9924 |
| KITTI | oracle \* | 0.0655 | 0.0367 | 0.0365 | 2.009 | 0.9587 | 0.9866 | 0.9935 |

By depth band (AbsRel, Occ3D): MoGe 0.063 / 0.094 / 0.159 / 0.403 for 0–10 / 10–20 / 20–40
/ 40–80 m, tracking the oracle's 0.030 / 0.065 / 0.132 / 0.389. By frame offset the MoGe
error is flat (0.089 / 0.087 / 0.088 / 0.088 / 0.090 for offsets −4…0), confirming one
scalar serves all five frames equally well.

---

## 9. Statistical analysis

**Occ3D** — 10 000 resamples, seed 0, resampling unit = **nuScenes scene** (150), clips
averaged within a scene first.

| contrast | ΔIoU | 95 % CI | scenes improved |
|---|---:|---|---:|
| **M0-R → M2-R** | **+0.0398** | [+0.0308, +0.0486] \* | 120 / 150 |
| **M1-R → M2-R** | **+0.0637** | [+0.0566, +0.0709] \* | 143 / 150 |
| **M0-D → M2-D** | **+0.0682** | [+0.0579, +0.0788] \* | 129 / 150 |
| **M1-D → M2-D** | **+0.0847** | [+0.0752, +0.0945] \* | 141 / 150 |
| M2-R → OR-R | −0.0020 | [−0.0060, +0.0019] | 79 / 150 |
| M2-D → OR-D | +0.0010 | [−0.0047, +0.0058] | 91 / 150 |

The last two straddle zero: **MoGe is statistically indistinguishable from the LiDAR
oracle on this benchmark.**

**SemanticKITTI** — sequence 08 is a *single* sequence, so its clips are not independent
scenes and are not presented as such. A clearly labelled **block bootstrap** over 21
contiguous 8-clip blocks preserves temporal neighbourhoods:

| contrast | ΔIoU | 95 % CI | blocks improved |
|---|---:|---|---:|
| **M0-R → M2-R** | **−0.0204** | [−0.0322, −0.0098] \* | 5 / 21 |
| **M1-R → M2-R** | **−0.0395** | [−0.0488, −0.0303] \* | 0 / 21 |
| **M0-D → M2-D** | **−0.0349** | [−0.0437, −0.0260] \* | 1 / 21 |
| **M1-D → M2-D** | **−0.0507** | [−0.0580, −0.0436] \* | 0 / 21 |
| M2-R → OR-R | +0.0390 | [+0.0289, +0.0492] \* | 20 / 21 |
| M2-D → OR-D | +0.0506 | [+0.0422, +0.0594] \* | 21 / 21 |

Block resampling understates uncertainty relative to genuinely independent scenes; the
direction and magnitude, not the interval width, carry the conclusion here.

---

## 10. Leakage audit

| # | guarantee | test |
|---|---|---|
| 1 | MoGe receives RGB only — AST check of the single call site (one positional arg, only `apply_mask`) | `test_moge_infer_receives_rgb_only_and_infers_its_own_fov` |
| 2 | no GT FOV/intrinsics/extrinsics/pose/camera height/LiDAR reaches MoGe | `test_no_ground_truth_camera_or_lidar_reaches_moge` |
| 3 | scale estimation opens no occupancy label or mask | `test_estimator_never_opens_a_label_or_mask` |
| 4 | randomising labels/masks leaves M2 bit-identical | `test_randomising_targets_leaves_the_estimator_bit_identical` |
| 5 | randomising/removing LiDAR leaves M2 bit-identical (LiDAR is not an input at all) | same + `test_m2_scale_is_independent_of_labels_and_lidar` |
| 6 | LiDAR oracle computed only after M0/M1/M2 are finalised | `test_lidar_diagnostics_run_after_predictions_are_finalised` |
| 7 | MoGe and LingBot parameters all `requires_grad=False` | adapter raises on any trainable parameter; `test_no_optimizer_…` |
| 8 | no optimiser, backward call or trainable parameter in Gate 5 | `test_no_optimizer_backward_or_trainable_parameter_in_gate5` |
| 9 | optical-axis z used, not Euclidean ray | `test_optical_axis_z_is_used_not_euclidean_ray` + runtime assertion |
| 10 | pixel correspondence exact, never interpolated | `test_pixel_correspondence_is_asserted_not_interpolated` |
| 11 | invalid teacher pixels cannot contaminate the estimate | `test_invalid_teacher_pixels_cannot_contaminate_the_estimate` |
| 12 | exactly one scalar per clip | `test_estimator_returns_exactly_one_scalar_per_clip` |
| 13–15 | one scalar to all five depths; translation scaled once; rotations unchanged; anchor fixed; pure similarity | `test_one_scalar_is_applied_to_all_five_depths_and_translations_once` |
| 16 | MoGe outputs cannot enter the fusion (signature + source scan) | `test_moge_outputs_cannot_enter_the_fused_reconstruction` |
| 17 | `dilate_r2` remains the frozen 0.4 m expansion, distinct from the 0.6 m band | `test_dilate_r2_remains_the_frozen_0_4_m_expansion` |
| 18–19 | Gate 4.1 and all repository tests still pass | full-suite run, §13 |

Source scans strip docstrings and comments via AST, so they check executable code rather
than the prose that documents the prohibition.

---

## 11. Representative visualisations

* `artifacts/gate5/gate5_scale_plots.png` — scale distributions, oracle correlation and
  error boxplots for both datasets. The KITTI scatter shows MoGe sitting systematically
  below the identity line; the Occ3D scatter hugs it.
* `artifacts/gate5/gate5_scenes_bev.png` — bird's-eye comparisons for the p95 / p50 / p05
  Occ3D scenes, ranked by the **deployable M0-D** scene IoU (fixed rule, applied before any
  MoGe or oracle row was inspected): `scene-0915`, `scene-0558`, `scene-1059`.

---

## 12. Runtime and memory

| stage | dataset | wall time | peak VRAM |
|---|---|---:|---:|
| MoGe inference + scale | KITTI (163 clips, 815 frames) | 50 s (309 ms/clip) | 2.23 GiB |
| MoGe inference + scale | Occ3D (1 182 clips, 5 910 frames) | 449 s (379 ms/clip) | 2.22 GiB |
| occupancy evaluation | KITTI | 30 s | 0.33 GiB |
| occupancy evaluation | Occ3D | 327 s | 0.61 GiB |
| LiDAR diagnostics | KITTI / Occ3D | 18 s / 176 s | — |

Hardware: NVIDIA RTX PRO 6000 Blackwell Max-Q (97.9 GB), torch 2.7.1+cu128.
MoGe prediction cache 961 MiB, outside the repository.

---

## 13. Tests

`tests/gate5/test_gate5.py` — **23 tests**. Whole repository: **453 passed, 2 skipped**
(both pre-existing and unrelated).

| suite | result |
|---|---|
| `tests/scale_gate` | 49 passed |
| `tests/depth_gate` | 35 passed |
| `tests/voxel_gate` | 29 passed |
| `tests/voxel_gate_validation` | 24 passed |
| `tests/occ3d_zeroshot` (Gate 4 + 4.1) | 43 passed |
| `tests/gate5` (new) | 23 passed |
| remainder | 250 passed, 2 skipped |

---

## 14. Limitations

1. **MoGe-2 was trained with large-scale metric supervision.** This experiment reuses that
   metric knowledge; it does not manufacture metric information from monocular RGB. Any
   claim of "LiDAR-free metric scale" must credit MoGe's own training data.
2. **Occ3D-nuScenes is now a development benchmark.** It informed Gate 4.1's diagnosis and
   this gate's design. It is no longer an untouched target for a future paper, and a fresh
   held-out dataset will be needed for the final claim.
3. **Not comparable to OccAny's published Occ3D result**, which is a six-camera surround
   protocol; ours is five temporal frames from CAM_FRONT.
4. **The source regression is real and unexplained.** MoGe under-estimates KITTI scale by
   15 %. Plausible contributors — KITTI's cropped 81° frame reconstructed by MoGe as 87°,
   its rectified stereo-rig imagery, its different aspect ratio after resize — are not
   isolated here. This gate measures the effect; it does not diagnose the cause.
5. **One teacher, one estimator.** No teacher ensemble, no uncertainty estimate, no
   variant sweep — deliberately, to avoid selecting on Occ3D.
6. **The KITTI bootstrap uses 21 contiguous blocks of one sequence**, not independent
   scenes; its intervals are optimistic.
7. **Absolute Occ3D IoUs remain protocol-bound** (6.9 % of the grid evaluable, one forward
   camera, 2 s window). Comparisons within the table are meaningful; comparisons across
   protocols are not.
8. **Occ3D depth diagnostics use the anchor-frame LiDAR sweep per frame**; ego-motion
   between camera exposure and LiDAR keyframe adds a small amount of noise to all four
   scale rows equally.
9. **V3 was deliberately not run**, per the brief. Gate 4.1 already established it loses to
   `dilate_r2` under every geometry, and Gate 5's better geometry does not change that
   question.

---

## 15. Exactly one recommended next experiment

**Measure whether MoGe's scale bias is predictable from image-formation quantities that are
available at inference — before distilling anything.**

Concretely: on both datasets already prepared, regress `log(s_moge / s_oracle)` on
inference-available covariates — MoGe's own predicted FOV, the true intrinsics-derived FOV,
image aspect ratio after resize, MoGe mask fraction, the per-frame log-scale dispersion
already recorded, and the median MoGe depth. Nothing is trained on occupancy; this is a
one-afternoon analysis of the 1 345 clips of diagnostics this gate has already written to
disk.

Why this and not distillation: the gate's own numbers say MoGe is nearly oracle-quality on
one domain and 15 % biased on another, with **good rank correlation in both** (0.96 / 0.93).
That signature — right ordering, wrong gain — is exactly what a correctable systematic bias
looks like, and it is exactly what would be *baked in* if we distilled a student now. The
teacher's residual bias would become the student's ceiling, and we would have spent a
multi-source pseudo-label pipeline to inherit it. If the bias is predictable, a corrected
gauge is a small change to this gate; if it is not, the honest conclusion is that MoGe is a
target-domain-specific gauge and the distillation plan needs a different teacher or a
teacher ensemble.

Explicitly **not** recommended yet: offline multi-source pseudo-scale generation and student
training. The `MOGE_SCALE_WORKS` branch — which is what would have authorised that — did not
fire, because of the SemanticKITTI regression.

---

## 16. Decision-rule audit

| criterion for `MOGE_SCALE_WORKS` | result |
|---|---|
| MoGe scale error lower than both C0 and C3 on Occ3D | ✔ 5.63 % vs 22.95 % / 25.48 % |
| M2-D beats M0-D, scene-bootstrap CI excludes zero | ✔ +0.0682 [+0.0579, +0.0788] |
| M2-D recovers ≥ 50 % of the OR-D oracle gain | ✔ **98.5 %** |
| no major regression on the SemanticKITTI protocol | ✘ **−0.0349 [−0.0437, −0.0260], −21.9 % relative** |
| no leakage or protocol failure | ✔ |

Four of five. The fourth fails, so `MOGE_SCALE_WORKS` cannot be declared.

`MOGE_SCALE_FAILS` also partly fires on its third clause ("causes a serious source-domain
regression"), but its first two clauses are contradicted decisively — MoGe **does**
significantly improve M0-D on Occ3D (+0.0682) and **does** reduce scale error there (4×).
Calling this a failure would misdescribe a result that reaches oracle parity on the target.
`MOGE_SCALE_PARTIALLY_WORKS` names the actual pattern — "behaves inconsistently across
datasets" — and is the assigned diagnosis. The rule overlap is recorded here rather than
resolved silently.
