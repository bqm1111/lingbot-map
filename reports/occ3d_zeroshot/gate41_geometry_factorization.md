# Gate 4.1 — Occ3D-nuScenes geometry error factorization

## Diagnosis

```
scale        SCALE_IS_THE_DOMINANT_BOTTLENECK
pose         POSE_IS_NOT_A_BOTTLENECK
depth shape  DEPTH_SHAPE_IS_NOT_A_BOTTLENECK (once scale is correct)
residual     COVERAGE_LIMITED
prior        V3_LOSES_TO_DILATION_UNDER_EVERY_GEOMETRY, INCLUDING G11
```

## Plain-language conclusion

Gate 4 showed the frozen stack fails on nuScenes; this factorization shows the failure is
**one scalar per clip and essentially nothing else**. Substituting a LiDAR-measured scale
while keeping LingBot's own poses lifts raw fused occupancy from 0.0647 to **0.1027**
(+0.0378 [+0.0285, +0.0468], 127/150 scenes), moves the reconstructed ground plane from
−1.15 m to −0.68 m in the ego frame, raises the in-grid point fraction from 0.654 to 0.848,
and collapses depth error at LiDAR pixels from AbsRel 0.235 (δ1 0.543) to **AbsRel 0.056,
median 2.3 %, δ1 0.956**. Substituting ground-truth ego poses does the opposite of what one
might expect: it makes things slightly **worse** (−0.0035 at constant scale, −0.0101 at
oracle scale), because LingBot's relative poses are already accurate — 0.17–0.36° rotation,
2.0–2.2° translation direction, magnitude ratio 1.02–1.04 over 2.8–11.2 m of true motion —
and swapping them in breaks the internal consistency between its depth and its poses. So
neither pose nor depth *shape* transfers badly; the single non-transferring quantity is
metric scale, and both the frozen constant (16 % high) and the learned estimator (25 % high)
get it wrong in the same direction.

Two things this does **not** rescue. First, even the best geometry cell G11 reaches only
0.0926 raw against its own matched in-band ceiling of **0.3254** — with depth now accurate
to 2.3 %, the remaining gap is **coverage**, not geometry: one forward camera over a 2 s
window cannot populate an 80 × 80 m ego-centric grid. Second, the frozen V3 prior loses to
plain `dilate_r2` in **every** geometry cell, and the gap *widens* as the geometry improves
(−0.0080 at G00, −0.0161 at G11, −0.0188 at G01). Fixing the input geometry does not
rehabilitate the learned prior.

> Branches marked `*` use LiDAR-measured scale or nuScenes ground-truth poses. They are
> **non-deployable diagnostic oracles** and exist only to attribute error.

Nothing was trained, tuned or selected here. No optimizer exists; every parameter has
`requires_grad=False`; all eight frozen hashes and seven frozen scalars were re-verified
before any evaluation ran. Gate 4 is reproduced **exactly** as a special case (§4).

---

## 1. Provenance

| item | value |
|---|---|
| reference code | `valeoai/OccAny` @ `13d79e51dd42111171c10fd8d9bc4b9de296d06b`; `Tsinghua-MARS-Lab/Occ3D` @ `6e8563ac4a8b8c9a7352092118663a914f12bec9` |
| paper | arXiv 2603.23502v2 |
| data | nuScenes v1.0-trainval + Occ3D-nuScenes trainval at `/media/SSD1/MINH_DATASETS/nuscenes` (unchanged from Gate 4; nothing downloaded) |
| manifest | `manifests/occ3d_zeroshot/val.jsonl` — `60a74604…f18e1463`, reused verbatim |
| config | `configs/occ3d_zeroshot/gate41_factorization.yaml` — `7c8e91d1231971c5…` |
| per-clip geometry | `artifacts/occ3d_zeroshot/gate41/geometry_per_clip.csv` — `c2410fc82e7717e2…` |
| per-clip results | `artifacts/occ3d_zeroshot/gate41/factorization_per_clip.csv` — `996518440244d249…` |

### Frozen components — re-verified, all match Gate 3.1 / Gate 4

| component | SHA-256 (first 16) |
|---|---|
| LingBot | `ee665103348e07e6` |
| Gate-2 depth head (`use_rgb=False`, `in_ch=5`) | `2a91822ea5d1182a` |
| Gate-3.1 `full_s0` (V3) | `05730ac7addb101d` |
| Gate-3.1 `occ_only_s0` | `b90a6e4ee9cdeccd` |
| Gate-3.1 `c0_corrector_s0` | `6590049302ef2044` |
| Gate-3.1 / Gate-2 / Gate-0 configs | `a548ed08fcbe1ea6` / `b7692022114830f9` / `472b45311a5e03ba` |

Frozen scalars, unchanged: `s0 = 27.3665`, `τ = 0.45`, correction band `0.6 m`
(3 voxels @ 0.2 m), confidence `≥ 1.5`, depth range `(1, 60) m`, canonical voxel `0.2 m`.

---

## 2. Protocol and coordinate conventions

Unchanged from Gate 4: 1 182 clips, all 150 official val scenes, five CAM_FRONT keyframes
at 2 Hz, anchor = last frame, canonical 0.2 m grid (400 × 400 × 32) over the exact official
extent, any-subvoxel conversion to the native 0.4 m grid (200 × 200 × 16), official camera
mask, single-camera x-cut at index 100, OccAny binary reduction (`occupied := label != 17`,
`255` ignored).

```
p_camera[t]                = unproject(depth[t] * s, K_pred[t])
T_rel[t] (pred poses)      = inv(P[anchor]) @ P[t], translation scaled by s   ← ONCE
T_rel[t] (GT poses)        = inv(C[anchor]) @ C[t],  C[t] = T_ego_cam_t_to_world
                                                          @ T_camera_to_ego_cam[t]
                             metric already — NEVER multiplied by any scale
T_anchor_camera_to_ego     = inv(T_ego_keyframe_to_world) @ T_ego_cam_to_world
                             @ T_camera_to_ego_cam      (anchor frame only)
T_ego_to_occ_grid          = floor((p_ego - (-40,-40,-1)) / 0.4)
```

Both pose branches share one fusion function, so masking, unroll order and intrinsics
cannot diverge between them. Asserted in tests: LingBot translation scaled exactly once
(`t(s) == s·t(1)` and `≠ s²·t(1)`); the GT branch is bit-identical for `s ∈ {0.5, 27.3665,
100, NaN}`; rotations orthonormal, right-handed, never scaled; anchor is a fixed point in
both modes; `gt_relative` maps frame-*f* camera points into the anchor camera (checked
against an independent world round-trip); coupled scaling is a similarity in the predicted
branch and is **not** one in the GT branch.

**Two physical quantities kept distinct throughout** — code, config and report:

| quantity | voxels @ 0.2 m | metres | operation |
|---|---:|---:|---|
| `dilate_r2` expansion | 2 | **0.4 m** | 5 × 5 × 5 max-pool |
| V3 correction band `R_infer` | 3 | **0.6 m** | 7 × 7 × 7 neighbourhood |

---

## 3. Oracle scale — construction and coverage

Reused verbatim from the Gate-4 LiDAR diagnostic: project the anchor `LIDAR_TOP` keyframe
into the anchor CAM_FRONT image through official calibration, then take the
confidence-weighted median of `log(d_lidar) − log(D_lingbot)` at valid pixels. One scalar
per clip, applied to all five depth maps. No Occ3D label, mask or occupancy array is opened
— asserted by a source scan.

**Failure policy, fixed before any occupancy result was inspected:** a clip with fewer than
200 valid correspondences is reported by id and **excluded** from every oracle condition;
it is never silently replaced by `s0`.

| quantity | value |
|---|---|
| clips attempted | **1 182** (full protocol, not the 150-clip Gate-4 subsample) |
| oracle scale valid | **1 182 (100 %)** |
| excluded | **0** |
| LiDAR correspondences per clip | median 2 245, p05 1 346, **min 314** |

Every clip cleared the threshold, so the factorial table covers the complete protocol.

---

## 4. Gate-4 reproduction

The factorization code computes Gate 4 as a special case. Enforced before any new number
was read (tolerance 5e-4):

| Gate 4.1 row | Gate 4 row | Gate 4.1 | Gate 4 | \|Δ\| |
|---|---|---:|---:|---:|
| `G00 \| raw` | T0 | 0.06467 | 0.06467 | **0.00e+00** |
| `C3 \| raw` | T1 | 0.04106 | 0.04106 | **0.00e+00** |
| `C3 \| dilate_r2` | T2 | 0.14342 | 0.14342 | **0.00e+00** |
| `C3 \| occ_only` | T4 | 0.13389 | 0.13389 | **0.00e+00** |
| `C3 \| v3` | T6 | 0.12696 | 0.12696 | **0.00e+00** |

Exact to floating point.

---

## 5. Factorial results — 1 182 clips, 150 scenes

`*` = non-deployable diagnostic oracle. `REF` = matched in-band oracle
`GT_occupied ∩ R_infer(cell)`, a coverage ceiling, not a method.

| cell | scale | poses | corrector | IoU | P | R | native occ | pts in-grid | ground z p5 (m) | pred in mask |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **G00** | const `s0` | pred | raw | **0.0647** | 0.519 | 0.070 | 7 056 | 0.654 | −1.153 | 0.185 |
| G00 | | | dilate_r2 | **0.1597** | 0.523 | 0.201 | 20 931 | 0.654 | −1.153 | 0.197 |
| G00 | | | occ_only | 0.1554 | 0.517 | 0.194 | 21 559 | 0.654 | −1.153 | 0.195 |
| G00 | | | v3 | 0.1516 | 0.550 | 0.179 | 19 463 | 0.654 | −1.153 | 0.186 |
| G00 | | | *REF* | 0.2506 | 0.988 | 0.251 | 2 631 | | | |
| **G01** \* | LiDAR | pred | raw | **0.1027** | 0.692 | 0.109 | 7 187 | **0.848** | **−0.675** | 0.246 |
| G01 \* | | | dilate_r2 | **0.2292** | 0.634 | 0.281 | 21 308 | 0.848 | −0.675 | 0.238 |
| G01 \* | | | occ_only | 0.2189 | 0.611 | 0.267 | 21 503 | 0.848 | −0.675 | 0.232 |
| G01 \* | | | v3 | 0.2104 | 0.649 | 0.243 | 19 009 | 0.848 | −0.675 | 0.221 |
| G01 \* | | | *REF* | 0.3357 | 1.000 | 0.336 | 3 392 | | | |
| **G10** \* | const `s0` | GT | raw | **0.0611** | 0.498 | 0.067 | 7 557 | 0.618 | −1.257 | 0.179 |
| G10 \* | | | dilate_r2 | 0.1515 | 0.508 | 0.193 | 21 047 | 0.618 | −1.257 | 0.196 |
| G10 \* | | | occ_only | 0.1488 | 0.497 | 0.189 | 21 921 | 0.618 | −1.257 | 0.197 |
| G10 \* | | | v3 | 0.1472 | 0.522 | 0.178 | 20 350 | 0.618 | −1.257 | 0.191 |
| G10 \* | | | *REF* | 0.2416 | 0.986 | 0.242 | 2 527 | | | |
| **G11** \* | LiDAR | GT | raw | **0.0926** | 0.666 | 0.099 | 7 408 | 0.816 | −0.772 | 0.225 |
| G11 \* | | | dilate_r2 | **0.2194** | 0.624 | 0.269 | 20 655 | 0.816 | −0.772 | 0.237 |
| G11 \* | | | occ_only | 0.2099 | 0.595 | 0.258 | 21 277 | 0.816 | −0.772 | 0.234 |
| G11 \* | | | v3 | 0.2033 | 0.621 | 0.239 | 19 432 | 0.816 | −0.772 | 0.226 |
| G11 \* | | | *REF* | 0.3254 | 1.000 | 0.325 | 3 281 | | | |
| C3 | learned | pred | raw | 0.0411 | 0.547 | 0.044 | 7 620 | 0.675 | −1.188 | 0.121 |
| C3 | | | dilate_r2 | 0.1434 | 0.562 | 0.172 | 22 150 | 0.675 | −1.188 | 0.161 |
| C3 | | | occ_only | 0.1339 | 0.542 | 0.160 | 22 570 | 0.675 | −1.188 | 0.160 |
| C3 | | | v3 | 0.1270 | 0.555 | 0.147 | 21 072 | 0.675 | −1.188 | 0.155 |
| C3 | | | *REF* | 0.2387 | 0.997 | 0.239 | 2 423 | | | |

`C3` is a reference row (frozen learned scale + predicted poses), not a factorial cell.
Ground truth in the evaluation region: 10 116 occupied voxels of 44 074 evaluable
(6.9 % of the 640 000-voxel native grid).

### Per-clip IoU quartiles

| cell | raw p25/med/p75 | dilate_r2 | v3 | REF |
|---|---|---|---|---|
| G00 | 0.013 / 0.033 / 0.100 | 0.054 / 0.132 / 0.254 | 0.063 / 0.126 / 0.226 | 0.094 / 0.221 / 0.386 |
| G01 \* | 0.041 / 0.089 / 0.148 | 0.135 / 0.229 / 0.307 | 0.130 / 0.201 / 0.277 | 0.196 / 0.333 / 0.463 |
| G10 \* | 0.012 / 0.033 / 0.087 | 0.053 / 0.119 / 0.240 | 0.063 / 0.122 / 0.216 | 0.088 / 0.208 / 0.376 |
| G11 \* | 0.038 / 0.075 / 0.133 | 0.128 / 0.219 / 0.295 | 0.128 / 0.194 / 0.266 | 0.184 / 0.323 / 0.454 |
| C3 | 0.013 / 0.027 / 0.051 | 0.067 / 0.122 / 0.198 | 0.072 / 0.110 / 0.161 | 0.119 / 0.213 / 0.335 |

### By distance (IoU)

| cell \| corrector | 0–10 m | 10–20 m | 20–30 m | 30–40 m |
|---|---:|---:|---:|---:|
| G00 \| raw | 0.158 | 0.097 | 0.052 | 0.019 |
| G00 \| dilate_r2 | 0.312 | 0.217 | 0.142 | 0.070 |
| G00 \| v3 | 0.301 | 0.209 | 0.133 | 0.066 |
| G01 \* \| raw | 0.254 | 0.150 | 0.075 | 0.022 |
| G01 \* \| dilate_r2 | **0.433** | **0.314** | **0.200** | 0.075 |
| G01 \* \| v3 | 0.425 | 0.291 | 0.176 | 0.066 |
| G11 \* \| raw | 0.205 | 0.135 | 0.071 | 0.022 |
| G11 \* \| dilate_r2 | 0.415 | 0.298 | 0.192 | 0.073 |
| G11 \* \| v3 | 0.387 | 0.275 | 0.176 | 0.068 |
| G01 \* \| REF | 0.655 | 0.479 | 0.304 | 0.116 |

The scale effect is strongest in the near field (0–10 m: 0.158 → 0.254 raw) and vanishes
beyond 30 m, where coverage rather than scale is binding.

---

## 6. Geometry diagnostics, independent of occupancy

### 6.1 Scale (1 182 clips)

| quantity | median | p05 | p95 |
|---|---:|---:|---:|
| LiDAR-measured `s_oracle` | **23.550** | 13.850 | 38.938 |
| frozen constant `s0` | 27.3665 | — | — |
| frozen learned `s_c3` | 29.261 | 18.439 | 44.805 |

| error vs LiDAR | constant `s0` | learned `s_c3` |
|---|---:|---:|
| median \|log scale error\| | **0.2198** | 0.2273 |
| median relative scale error | **22.9 %** | 25.5 % |
| median ratio (estimate / oracle) | **1.162** | **1.253** |

Both over-estimate; the learned head over-estimates more. For contrast, on SemanticKITTI
seq 08 the same head achieved 2.9 % median relative error against the constant's 12.7 %.
Residual saturation is absent at the median, so the head is confidently wrong rather than
clipping.

### 6.2 Pose — scaled LingBot relative pose vs nuScenes ground truth

At **oracle** scale, by temporal offset from the anchor:

| offset | rotation (deg) | translation error (m) | GT ‖t‖ (m) | magnitude ratio | direction (deg) |
|---:|---:|---:|---:|---:|---:|
| −1 | 0.172 | 0.156 | 2.788 | 1.022 | 2.17 |
| −2 | 0.233 | 0.263 | 5.602 | 1.026 | 2.03 |
| −3 | 0.288 | 0.398 | 8.364 | 1.032 | 2.04 |
| −4 | 0.356 | 0.546 | 11.186 | 1.035 | 2.09 |

Translation error is 5.0 % of true displacement at every offset and grows linearly, i.e. it
is a residual scale effect, not drift. At the **constant** scale the magnitude ratio median
is 1.200 — exactly the scale bias, since rotation and direction are unchanged (they do not
depend on scale). **LingBot's pose direction and rotation are excellent on nuScenes.**

### 6.3 Depth shape at projected-LiDAR pixels (median over clips)

| scale applied | AbsRel | median abs rel | median abs log | RMSE (m) | δ1 |
|---|---:|---:|---:|---:|---:|
| constant `s0` | 0.2354 | 0.2267 | 0.2180 | 4.674 | 0.5434 |
| learned `s_c3` | 0.2480 | 0.2484 | 0.2219 | 4.215 | 0.5100 |
| **LiDAR oracle** | **0.0564** | **0.0234** | **0.0233** | **3.428** | **0.9564** |

With the correct scalar, LingBot's depth on nuScenes is **better than its scale-corrected
depth on the KITTI validation sequence** (Gate 2: AbsRel 0.0898, δ1 0.9158). Depth *shape*
transfers well; only the scalar does not. LiDAR is used here purely as diagnostic ground
truth and influenced no threshold, model or selection.

---

## 7. Scene-level paired bootstrap

10 000 resamples, seed 0, resampling unit = **nuScenes scene**; clips are averaged within a
scene first, so same-scene clips stay grouped. 150 scenes throughout.

### 7.1 Pose effect at fixed scale

| contrast | ΔIoU | 95 % CI | scenes improved |
|---|---:|---|---:|
| G00 → G10, raw | **−0.0035** | [−0.0052, −0.0019] \* | 57 / 150 |
| G01 → G11, raw | **−0.0101** | [−0.0121, −0.0081] \* | 20 / 150 |
| G00 → G10, dilate_r2 | −0.0082 | [−0.0104, −0.0060] \* | 34 / 150 |
| G01 → G11, dilate_r2 | −0.0098 | [−0.0116, −0.0082] \* | 17 / 150 |
| G00 → G10, occ_only | −0.0066 | [−0.0086, −0.0047] \* | 30 / 150 |
| G01 → G11, occ_only | −0.0091 | [−0.0107, −0.0075] \* | 17 / 150 |
| G00 → G10, v3 | −0.0043 | [−0.0063, −0.0025] \* | 46 / 150 |
| G01 → G11, v3 | −0.0072 | [−0.0090, −0.0054] \* | 28 / 150 |

Every pose effect is **negative** and small (≤ 0.010 IoU). Ground-truth poses do not help.

### 7.2 Scale effect at fixed pose

| contrast | ΔIoU | 95 % CI | scenes improved |
|---|---:|---|---:|
| G00 → G01, raw | **+0.0378** | [+0.0285, +0.0468] \* | 127 / 150 |
| G10 → G11, raw | **+0.0312** | [+0.0222, +0.0399] \* | 118 / 150 |
| G00 → G01, dilate_r2 | **+0.0692** | [+0.0576, +0.0809] \* | 132 / 150 |
| G10 → G11, dilate_r2 | **+0.0676** | [+0.0559, +0.0792] \* | 133 / 150 |
| G00 → G01, occ_only | +0.0632 | [+0.0511, +0.0753] \* | 131 / 150 |
| G10 → G11, occ_only | +0.0607 | [+0.0488, +0.0726] \* | 127 / 150 |
| G00 → G01, v3 | +0.0585 | [+0.0463, +0.0705] \* | 124 / 150 |
| G10 → G11, v3 | +0.0557 | [+0.0438, +0.0672] \* | 123 / 150 |

**Effect sizes, not just significance:** the scale effect is 6–17× the magnitude of the
pose effect on the same rows, and both are estimated on the same 150 scenes.

### 7.3 Interaction — `(G11 − G10) − (G01 − G00)`

| corrector | DiD | 95 % CI |
|---|---:|---|
| raw | −0.0066 | [−0.0083, −0.0049] \* |
| dilate_r2 | −0.0016 | [−0.0035, +0.0002] |
| occ_only | −0.0025 | [−0.0042, −0.0006] \* |
| v3 | −0.0028 | [−0.0046, −0.0010] \* |

Small and negative: the scale and pose effects are close to additive, with a slight penalty
for combining them — consistent with the two error sources partially cancelling in the
deployable configuration.

### 7.4 Corrector comparisons within each geometry cell

| cell | raw→dilate_r2 | raw→occ_only | raw→v3 | **dilate_r2→v3** | occ_only→v3 |
|---|---:|---:|---:|---:|---:|
| G00 | +0.0953 \* | +0.0911 \* | +0.0873 \* | **−0.0080 \*** (75/150) | −0.0038 \* |
| G01 \* | +0.1267 \* | +0.1165 \* | +0.1079 \* | **−0.0188 \*** (36/150) | −0.0085 \* |
| G10 \* | +0.0906 \* | +0.0880 \* | +0.0864 \* | **−0.0042 \*** (85/150) | −0.0015 \* |
| G11 \* | +0.1270 \* | +0.1175 \* | +0.1109 \* | **−0.0161 \*** (41/150) | −0.0066 \* |
| C3 | +0.1027 \* | +0.0931 \* | +0.0862 \* | **−0.0165 \*** (49/150) | −0.0069 \* |

All CIs exclude zero. **V3 loses to `dilate_r2` in all five conditions**, and the deficit
grows when the geometry is repaired (−0.0080 → −0.0161 going G00 → G11; −0.0042 → −0.0188
going G10 → G01). The learned prior is not merely sensitive to upstream shift — better
inputs make deterministic dilation pull further ahead.

---

## 8. Leakage audit

| # | guarantee | test |
|---|---|---|
| 1 | randomising `semantics`, `mask_camera`, `mask_lidar` leaves every prediction bit-identical, in all four pose × scale combinations | `test_predictions_bit_identical_under_full_target_randomisation` |
| 2 | labels opened only after predictions are finalised (source-order assertion) | `test_labels_are_opened_only_after_predictions_are_finalised` |
| 3 | oracle scale reads LiDAR + calibration only | `test_oracle_scale_and_gt_pose_sources_read_no_occupancy_label` |
| 4 | GT-pose branch reads pose/calibration metadata only | same, plus `test_factorization_module_never_reads_a_label_or_mask` |
| 5 | no optimizer, `.backward()`, or trainable parameter anywhere in Gate 4.1 | `test_no_optimizer_backward_or_trainable_parameter_in_gate41` |
| 6 | all frozen hashes and scalars match Gate 3.1 / Gate 4 | `test_frozen_hashes_and_scalars_match_gate31_and_gate4` |
| 7 | LingBot translation scaled exactly once | `test_lingbot_translation_is_scaled_exactly_once` |
| 8 | GT translation never scaled (invariant for `s ∈ {0.5, 27.3665, 100, NaN}`) | `test_gt_translation_is_never_scaled` |
| 9 | rotations never scaled, in either branch | `test_rotations_are_never_scaled_in_either_branch` |
| 10 | identical voxelisation and native conversion in all conditions | `test_all_conditions_use_identical_voxelisation_and_conversion` |
| 11 | correction region derived only from its own raw occupancy | `test_correction_region_comes_only_from_its_own_raw_occupancy` |
| 12 | 0.4 m dilation and 0.6 m band never conflated | `test_dilate_r2_expansion_and_v3_band_are_distinct_quantities` |
| 13 | scene-level bootstrap deterministic | `test_scene_level_bootstrap_is_deterministic` |
| 14 | all 1 182 clips and 150 scenes evaluated; failure policy declared and every failure reported by id | `test_oracle_scale_covers_every_clip_with_a_declared_policy`, `test_all_clips_and_scenes_are_evaluated_and_gate4_is_reproduced` |
| 15 | existing repository tests still pass | full-suite run, §11 |

Masks restrict metrics only (`test_masks_restrict_metrics_only`); anchor invariance and
transform direction are covered in §2's assertions.

---

## 9. Representative scenes

`artifacts/occ3d_zeroshot/gate41/scenes_factorization.png`.

**Selection rule, fixed and applied to the deployable row only** so no oracle result could
influence it: rank the 150 official val scenes by `G00|v3` scene IoU and take the 95th,
50th and 5th percentile (ranks 143, 75, 8).

| scene | rank | G00 raw | G10 \* raw | G01 \* raw | G11 \* raw | G01 \* dilate_r2 | G01 \* v3 |
|---|---|---:|---:|---:|---:|---:|---:|
| `scene-0915` | 143/150 | 0.195 | 0.192 | 0.248 | 0.240 | 0.454 | 0.458 |
| `scene-0274` | 75/150 | 0.027 | 0.032 | 0.074 | 0.052 | 0.197 | 0.167 |
| `scene-1062` | 8/150 | 0.012 | 0.021 | 0.022 | 0.032 | 0.099 | 0.125 |

The panels show the mechanism directly: correcting scale (column 1 → 3) thickens and
re-registers the road and façade surfaces, while switching to GT poses (column 1 → 2)
changes almost nothing. Grey dominates every panel — the coverage limit.

---

## 10. Limitations

1. **Diagnostic only.** G01, G10 and G11 consume LiDAR-measured scale or ground-truth ego
   poses and are not deployable. Only G00 (and the C3 reference) describe a shippable system.
2. **The oracle scale is itself estimated**, from a median over ~2 245 LiDAR pixels at the
   anchor frame only. A per-frame or per-region scale could differ; a single clip-level
   scalar is assumed, matching the frozen pipeline's own assumption.
3. **Single source, single target, single camera.** SemanticKITTI → Occ3D-nuScenes CAM_FRONT.
   Not OccAny's five-source protocol, and not comparable to its published Occ3D numbers
   (OccAny evaluates Occ3D in *surround* mode; see the Gate-4 report).
4. **Absolute IoUs are protocol-bound.** Only 6.9 % of the native grid is evaluable and one
   front camera over 2 s cannot cover 80 × 80 m; the matched in-band ceilings (0.25–0.34)
   are the relevant scale, not 1.0.
5. **GT poses are not error-free** — they carry nuScenes ego-pose and timestamp
   uncertainty — so "GT poses do not help" is bounded by that reference quality. The
   measured LingBot pose errors (0.17–0.36°, 2 % direction) are near that noise floor.
6. **The negative pose effect is small** (≤ 0.010 IoU) and, while its CI excludes zero, it
   should be read as "no useful pose headroom", not as a claim that GT poses are harmful in
   general.
7. **τ = 0.45 and the 0.6 m band were fitted to KITTI** and deliberately not re-tuned; a
   re-tuned V3 might close some of its gap to dilation, but that would no longer be a
   transfer measurement.
8. **`s_c3` over-estimates more than `s0`**, but this establishes only that the head learned
   a source-specific mapping that does not transfer. Camera geometry is one plausible
   explanation; this experiment does not isolate the cause.

---

## 11. Recommended next experiment — exactly one

**Replace the learned clip-scale head with a calibration-conditioned metric-scale estimator,
and validate it with this same frozen harness before anything else changes.**

The evidence points here and nowhere else:

* Scale explains the whole cross-dataset gap: +0.0378 raw / +0.069 with dilation from the
  scale substitution alone, against ≤ 0.010 (negative) for poses.
* Once scale is right, LingBot's depth is accurate to 2.3 % median relative error with
  δ1 = 0.956 — better than its scale-corrected KITTI depth. There is nothing to fix in depth
  shape or in pose.
* The current head is not merely untransferable, it is *worse than a constant* on the target
  (25.5 % vs 22.9 % median relative error), so replacing it is justified without any further
  ablation.
* The smallest change that could work is a head that sees quantities which actually
  determine metric scale and are available at inference — camera intrinsics and the
  camera-to-ego extrinsic (height, pitch) — rather than depth statistics alone. Both are
  present in every dataset considered, including all five OccAny sources.

Explicitly **not** recommended yet: full five-source OccAny training. The factorization does
not implicate depth shape, pose or the fusion geometry, so retraining the reconstruction
stack is not the smallest justified redesign. Multi-source data is worth acquiring for the
*scale* estimator specifically, and the decision on whether the reconstruction stage also
needs it should wait until a calibration-conditioned scale head has been measured on this
harness.

Regarding the V3 prior: it loses to `dilate_r2` under every geometry, including the best
one, and the deficit widens as geometry improves. The evidence supports **dropping or
retraining** it — but that decision should follow the scale fix, since every corrector's
absolute numbers move substantially with scale (dilation gains +0.069 from the scale
substitution alone).

No model was changed and nothing was tuned after these numbers were seen.
