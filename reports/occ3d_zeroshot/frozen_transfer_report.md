# Gate 4 — frozen SemanticKITTI → Occ3D-nuScenes transfer

## 1. Scale diagnosis: `SCALE_DOES_NOT_TRANSFER`
## 2. Local-prior diagnosis: `LOCAL_PRIOR_IS_KITTI_SPECIFIC`
## 3. Combined diagnosis: `NEITHER_TRANSFERS`

## 4. Plain-language conclusion

Neither learned component survives the move to nuScenes. The clip-scale estimator, which on
SemanticKITTI cut the median scale error from 12.7 % to 2.9 %, makes things **worse** here:
C3 scores 0.0411 against the constant-scale C0's 0.0647, a scene-paired **−0.0239
[−0.0325, −0.0158]**, and an independent LiDAR measurement confirms why — the true median
scale on nuScenes is 23.4, the frozen constant gives 27.4 (17 % high) and the learned
estimator pushes it further to 28.6 (27 % high). The estimator learned KITTI's camera
height and intrinsics, not a transferable cue. The local occupancy prior does lift C3
substantially (+0.0862, all 150 scenes), but it **loses to plain deterministic dilation**
(T2 0.1434 vs T6 0.1270, −0.0165 [−0.0214, −0.0118]) and even to its own occupancy-only
ablation (T4 0.1339, −0.0069). On SemanticKITTI the same model beat dilation by +0.0395;
here the ordering reverses. What transfers is the *shape* of the operation — expanding
occupancy into a 0.6 m band around observed surfaces helps on both datasets — not the
learned weights, which are worth less than a 5×5×5 max-pool once the domain changes.

> **The model was developed and trained on SemanticKITTI and evaluated without adaptation
> on Occ3D-NuScenes. This is single-source cross-dataset transfer, not OccAny's five-source
> generalization protocol.**

No component was trained, fine-tuned, recalibrated or selected on nuScenes. No optimizer
was constructed; every parameter has `requires_grad=False`; all eight frozen hashes were
re-verified before evaluation. Earlier gate artifacts and reports are untouched.

---

## 5. Provenance

### Reference implementations, pinned

| repository | commit | used for |
|---|---|---|
| `valeoai/OccAny` (CVPR 2026) | **`13d79e51dd42111171c10fd8d9bc4b9de296d06b`** (2026-06-28, `main`) | sequence protocol, camera choice, mask handling, binary label reduction |
| `Tsinghua-MARS-Lab/Occ3D` | **`6e8563ac4a8b8c9a7352092118663a914f12bec9`** (2024-07-29) | Occ3D data layout and label format |
| paper | arXiv **2603.23502v2**, *OccAny: Generalized Unconstrained Urban 3D Occupancy* | protocol description |

Files read at the pinned OccAny commit: `dataset_setup/base_make_seq.py`,
`dataset_setup/occ3d_nuscenes.md`, `sh/make_seqs.sh`, `occany/datasets/nuscenes.py`,
`occany/datasets/base_seq_dataset.py`, `occany/metrics/ssc.py`,
`occany/datasets/class_mapping.py`, `compute_metrics_from_saved_voxels.py`.

### Data

nuScenes v1.0-trainval and Occ3D-nuScenes trainval, already installed; see
`reports/occ3d_zeroshot/storage_and_access.md`. 150 official val scenes, 34 149 CAM_FRONT
keyframe images (4.7 GiB), 850 Occ3D scene label directories (2.71 GiB), `annotations.json`
150 463 571 bytes. Nothing downloaded.

| artifact | SHA-256 |
|---|---|
| `configs/occ3d_zeroshot/frozen_transfer.yaml` | `d82a1350…873f953f8` |
| `manifests/occ3d_zeroshot/val.jsonl` | `60a74604…f18e1463` |

Frozen checkpoint hashes are listed in the preflight report; all match Gate 3.1.

---

## 6. Match and differences against OccAny's protocol

**The single most important difference: OccAny has no five-frame temporal protocol for
Occ3D-nuScenes.** `sh/make_seqs.sh` at the pinned commit reads:

```sh
elif [ "$DATASET" = "occ3d_nuscenes" ]; then
    # occ3d_nuscenes only supports surround mode (no temporal)
    if [ "$SEQ_MODE" = "temporal" ]; then
        echo "Skipping occ3d_nuscenes - temporal mode not applicable"
```

OccAny evaluates Occ3D-nuScenes in **surround** mode — six cameras at one timestamp — and
reserves the temporal five-frame setting for KITTI, Waymo, DDAD, PandaSet, VKITTI and ONCE.
The five-frame single-camera protocol used here is therefore **ours, not OccAny's**.

> **A direct numerical comparison with OccAny's published Occ3D-nuScenes result is not
> valid**, and none is made anywhere in this report.

What we did adopt verbatim from the pinned OccAny source:

| element | OccAny source | adopted |
|---|---|---|
| camera | `NuScenesDataset` default `camera_names = ['CAM_FRONT']` | yes |
| grid | `voxel_size 0.4`, `occ_size [200,200,16]`, `voxel_origin = pc_range[:3] = (-40,-40,-1)` | yes |
| free class | `empty_class = 17`, 18 classes | yes |
| camera mask | `voxel_label[voxel_mask_camera == 0] = 255` (nuscenes.py:894) | yes |
| lidar mask | `apply_lidar_mask` off by default | yes (off) |
| single-camera cut | `voxel_label[:100,:,:] = 255` when `not use_surround_label` (nuscenes.py:902) | yes |
| binary reduction | `ssc.py:get_score_completion` — ignore 255, occupied := `!= empty_class` | yes |
| temporal cadence | temporal mode uses `--subsampling_rate 5` on 10 Hz = 2 Hz | yes (official 2 Hz keyframes) |

Documented differences:

1. **Temporal, not surround.** Ours is five frames from CAM_FRONT; OccAny's Occ3D setting is
   six cameras at one timestamp.
2. **Anchor is the last frame.** OccAny's Occ3D loader labels the `begin_frame_token`
   (first frame). Our Gate 0–3.1 fusion is frozen to anchor on the **last** frame, and
   changing it would alter the frozen geometry. The Occ3D label is taken at that same anchor
   sample.
3. **OccAny's `pc_range` z-max field reads 3.0** in `nuscenes.py:293, while
   `voxel_origin + occ_size * voxel_size` gives 5.4. We use the latter (the official extent,
   and what `voxel_origin`/`occ_size` actually imply); the repository's pre-existing
   `OCC3D_NUSCENES_GRID` already encodes it.
4. **No sequence manifest is released** for this setting, so ours is reproduced
   deterministically and published as `manifests/occ3d_zeroshot/val.jsonl`.

---

## 7. Clip construction

Five consecutive official 2 Hz keyframes, one camera, one scene, non-overlapping
(clip stride 5), anchored on the last frame. Labels are checked for **file presence only**;
`occ3d_zeroshot/nuscenes_adapter.py` contains no `np.load`, `semantics`, `mask_camera` or
`labels.npz` reference, asserted by test.

| quantity | value |
|---|---|
| official val scenes | 150 (all present) |
| scenes contributing clips | **150** |
| **clips** | **1 182** |
| clips per scene | min 7, median 8, max 8 |
| keyframes per scene | 39–41 |
| rejected clips | **0** (`short_scene` 0, `missing_image` 0, `missing_gt` 0, `nonmonotonic_time` 0) |
| inter-frame spacing (s) | median **0.500**, p5–p95 within [0.399, 0.651] |
| clip duration (s) | median **2.000**, range [1.799, 2.251] |

### Temporal comparability with SemanticKITTI

Measured from each dataset's own timestamps rather than assumed from frame counts:

| protocol | frames | median gap | median span |
|---|---:|---:|---:|
| SemanticKITTI (`times.txt`, 163 clips) | 5 | 0.520 s | **2.078 s** |
| Occ3D-nuScenes (keyframe timestamps, 1 182 clips) | 5 | 0.500 s | **2.000 s** |

|difference| = **0.078 s** → the two protocols carry comparable temporal evidence, not just
equal frame counts.

---

## 8. Coordinate conventions

```
p_camera[t]                = unproject(depth[t] * s, K_pred[t])          # optical-axis z
T_camera_to_lingbot_world  = LingBot pred_pose_c2w[t]  (canonical translation units)
T_frame_to_anchor[t]       = inv(P[anchor]) @ P[t], translation scaled by s
T_anchor_camera_to_ego     = inv(T_ego_keyframe_to_world)
                             @ T_ego_cam_to_world @ T_camera_to_ego_cam
T_ego_to_occ_grid          = floor((p_ego - (-40,-40,-1)) / voxel_size)
```

LingBot supplies **every** relative pose between the five frames. The only ground-truth
calibration used is `T_anchor_camera_to_ego`, evaluated at the anchor frame alone: the
static camera extrinsic plus a single-timestamp correction from the camera's own exposure
ego pose to the LiDAR keyframe ego pose (the frame Occ3D labels live in). No other frame's
ego pose is read, no LiDAR depth reaches inference, no oracle scale is used.

### Verification

Synthetic assertions (all tested): quaternions give orthonormal, right-handed, invertible
rotations; scaling depth and translation together is a pure similarity
(`fuse(s·D, s·t) == s·fuse(D, t)` to 1e-9, i.e. scale applied exactly once); rotations are
never scaled; the anchor is a fixed point; a CAM_FRONT-like extrinsic maps camera +z→ego +x
and camera +y→ego −z; floor-binning matches the grid rule at both boundaries.

Real-data verification, before any label was read — **this is what separates a transfer
failure from a coordinate bug**:

| check | result |
|---|---|
| LIDAR_TOP height in ego | 1.840 m; CAM_FRONT 1.511 m → ego origin is at ground level |
| official LiDAR pushed through the identical chain | ground plane at ego z = **−0.05 to −0.44 m** (expected ≈ 0) ✔ |
| C3 fusion with the frozen scale | ground plane at ego z ≈ **−1.07 m** |
| C3 fusion with the LiDAR-measured scale | ground plane at ego z ≈ **−0.54 m** |
| fused points in front of ego | 100 % |

The chain is correct: LiDAR lands where it should. The ~1 m sag in the LingBot fusion is
depth error, and it halves when the correct scale is substituted — the residual is LingBot's
depth *shape* error, the same factor Gate 1 measured at 33.1 % and Gate 2 failed to fix.

---

## 9. Target-leakage audit

| guarantee | mechanism | test |
|---|---|---|
| predictions independent of every label array | labels opened only after all seven predictions for a clip are fixed | `test_predictions_are_bit_identical_under_full_target_randomisation` — randomises `semantics`, `mask_camera`, `mask_lidar` three times, asserts `torch.equal` on C3, the deterministic control and the learned output |
| pipeline never reads a label | source scan | `test_pipeline_module_never_reads_a_label_or_mask` |
| clip builder never reads a label | source scan | `test_clip_builder_opens_no_label` |
| masks restrict the metric only | scorer leaves predictions untouched; voxels outside `keep` cannot contribute | `test_masks_restrict_the_metric_but_never_the_prediction` |
| `R_infer` is a pure dilation | `region_from(occ,3) == dilate(occ,3)` | `test_r_infer_is_a_pure_dilation_of_the_frozen_geometry` |
| gating depends only on `R_infer` | `apply_region` | `test_output_gating_depends_only_on_r_infer` |
| no optimizer or gradient anywhere | source scan for `torch.optim`, `Adam`, `SGD`, `.backward()`, `requires_grad_(True)` | `test_no_optimizer_is_constructed_anywhere_in_gate4` |
| only official val scenes | manifest ∩ train_split = ∅ | `test_all_clips_are_official_val_scenes_only` |
| frozen scalars unchanged | s0, τ, radius, conf, depth range checked against Gate-2/3.1 configs | `test_frozen_scalars_match_gate31` |

`R_infer = dilate(C3_occupied_0.2m, 3)`; for T5, `R_infer_c0 = dilate(C0_occupied_0.2m, 3)`.

---

## 10. Canonical ↔ native grid

Gate 3.1 was trained at 0.2 m with a 3-voxel = **0.6 m** radius and a 7-voxel = 1.4 m
receptive field. Occ3D's native voxel is 0.4 m, where "radius 3" would mean **1.2 m** — a
doubling of the physical operation. The whole frozen stack therefore runs on a canonical
0.2 m grid over the identical official extent, and predictions are converted afterwards.

| grid | dims | voxel | origin | upper |
|---|---|---|---|---|
| native (official) | 200 × 200 × 16 | 0.4 m | (−40, −40, −1) | (40, 40, 5.4) |
| canonical (ours) | 400 × 400 × 32 | 0.2 m | (−40, −40, −1) | (40, 40, 5.4) |

Conversion rule, frozen before any label was read:

```
a native voxel is occupied iff ANY of its eight 0.2 m subvoxels is occupied
```

Applied **identically** to all seven configurations and to the reference row. Tested on
synthetic patterns: empty→empty, full→full, a single subvoxel at each corner maps to the
correct native voxel, and the rule is monotone (more canonical occupancy can never yield
less native occupancy), so it cannot advantage any particular method. Canonical and native
occupied counts are reported side by side in §11.

Runtime assertion: `radius_voxels == 3` and `3 × 0.2 == 0.6 m` exactly, checked at start-up
and against every checkpoint's stored `radius`.

---

## 11. Results — Occ3D-nuScenes val, 1 182 clips, 150 scenes

Official binary occupancy on the native 0.4 m grid, camera mask applied, single-camera
x-cut applied. Mean over clips.

| id | configuration | IoU | P | R | native occ | canonical occ | TP | FP | FN | in eval mask |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **T0** | C0 constant scale | **0.0647** | 0.519 | 0.070 | 7 056 | 25 741 | 735 | 621 | 9 381 | 0.185 |
| **T1** | C3 learned clip scale | **0.0411** | 0.547 | 0.044 | 7 620 | 27 296 | 443 | 459 | 9 674 | 0.121 |
| **T2** | C3 + V1 `dilate_r2` | **0.1434** | 0.562 | 0.172 | 22 150 | 147 904 | 1 754 | 1 762 | 8 362 | 0.161 |
| **T3** | C3 + V2 `fill_r1_k3` | **0.0771** | 0.562 | 0.085 | 12 731 | 74 386 | 860 | 796 | 9 257 | 0.134 |
| **T4** | C3 + A_occ_only | **0.1339** | 0.542 | 0.160 | 22 570 | 144 321 | 1 612 | 1 830 | 8 505 | 0.160 |
| **T5** | C0 + A_c0_corrector | **0.1280** | 0.569 | 0.152 | 16 518 | 94 164 | 1 608 | 1 392 | 8 508 | 0.164 |
| **T6** | C3 + full V3 seed 0 | **0.1270** | 0.555 | 0.147 | 21 072 | 128 738 | 1 485 | 1 552 | 8 631 | 0.155 |
| REF | in-band oracle (GT ∩ `R_infer`) — *diagnostic, not a method* | 0.2387 | 0.997 | 0.239 | 2 423 | 16 731 | 2 423 | 0 | 7 693 | 0.997 |

Additions / removals relative to T1 (native): T2 +2 614/−0, T3 +754/−0, T4 +2 576/−37,
T6 +2 278/−144, T0 +1 089/−635, T5 +2 543/−445.

Fused points per clip: 438 524 (C3) / 440 419 (C0); **67.5 %** fall inside the grid.
Evaluable voxels: **44 074 of 640 000 (6.9 %)** per clip, containing 10 116 GT-occupied.

| id | per-clip IoU p25 | median | p75 |
|---|---:|---:|---:|
| T0 | 0.0127 | 0.0328 | 0.1003 |
| T1 | 0.0134 | 0.0268 | 0.0511 |
| T2 | 0.0669 | 0.1219 | 0.1978 |
| T4 | 0.0753 | 0.1160 | 0.1745 |
| T6 | 0.0723 | 0.1101 | 0.1610 |
| REF | 0.1186 | 0.2128 | 0.3347 |

### By distance — IoU / precision / recall

| id | 0–10 m | 10–20 m | 20–30 m | 30–40 m |
|---|---|---|---|---|
| T0 | 0.158 / 0.51 / 0.17 | 0.097 / 0.52 / 0.11 | 0.052 / 0.50 / 0.06 | 0.019 / 0.42 / 0.02 |
| T1 | 0.074 / 0.48 / 0.08 | 0.058 / 0.53 / 0.06 | 0.035 / 0.53 / 0.04 | 0.017 / 0.43 / 0.02 |
| **T2** | **0.276** / 0.63 / 0.34 | **0.192** / 0.56 / 0.24 | **0.125** / 0.56 / 0.15 | **0.062** / 0.49 / 0.07 |
| T3 | 0.157 / 0.57 / 0.18 | 0.107 / 0.55 / 0.12 | 0.064 / 0.54 / 0.07 | 0.031 / 0.45 / 0.03 |
| T4 | 0.222 / 0.61 / 0.28 | 0.180 / 0.54 / 0.23 | 0.124 / 0.56 / 0.15 | 0.062 / 0.50 / 0.07 |
| T5 | 0.263 / 0.59 / 0.32 | 0.177 / 0.57 / 0.22 | 0.112 / 0.55 / 0.14 | 0.058 / 0.49 / 0.07 |
| T6 | 0.211 / 0.62 / 0.25 | 0.173 / 0.56 / 0.21 | 0.116 / 0.57 / 0.14 | 0.060 / 0.50 / 0.07 |
| REF | 0.474 / 0.96 / 0.47 | 0.334 / 0.99 / 0.33 | 0.210 / 0.95 / 0.21 | 0.103 / 0.80 / 0.10 |

T2 leads every band; the gap to T6 is widest in the near field (0.276 vs 0.211 at 0–10 m).

Compute: **1.8 ms/clip** inference, peak **1.61 GiB**, 383 s for the full evaluation.

---

## 12. Scene-level paired bootstrap

10 000 resamples, seed 0, **resampling unit = nuScenes scene** (150 scenes); clips are
averaged within a scene first, so same-scene clips stay grouped.

| contrast | ΔIoU | 95 % CI | scenes improved |
|---|---:|---|---:|
| **T0 → T1** (scale) | **−0.0239** | **[−0.0325, −0.0158] \*** | **53 / 150** |
| T1 → T2 | +0.1027 | [+0.0950, +0.1103] \* | 150 / 150 |
| T1 → T3 | +0.0360 | [+0.0324, +0.0398] \* | 150 / 150 |
| T1 → T4 | +0.0931 | [+0.0874, +0.0989] \* | 150 / 150 |
| T1 → T6 | +0.0862 | [+0.0808, +0.0916] \* | 150 / 150 |
| **T2 → T6** | **−0.0165** | **[−0.0214, −0.0118] \*** | **49 / 150** |
| T3 → T6 | +0.0502 | [+0.0469, +0.0535] \* | 150 / 150 |
| **T4 → T6** | **−0.0069** | **[−0.0086, −0.0052] \*** | **44 / 150** |
| T5 → T6 | −0.0013 | [−0.0137, +0.0105] | 92 / 150 |

\* CI excludes zero.

Per-scene T1→T6 deltas: min +0.014, median +0.082, max +0.224 — the learned prior helps
every scene relative to its own input. Per-scene T2→T6: min −0.131, median −0.010, max
+0.042 — dilation wins in 101 of 150 scenes.

---

## 13. Scale statistics

| statistic | value |
|---|---|
| frozen constant `s0` | 27.3665 |
| learned `s_learned` median | **29.26**, p5–p95 [18.44, 44.80] |
| `a_clip` median | **+0.067**, p5–p95 [−0.395, +0.493], range [−0.514, +0.684] |
| residual saturation fraction | median **0.0000**, max 0.957 |

Saturation (|r| within 5 % of the head's `0.7·tanh` bound) is essentially absent at the
median, so the estimator is not clipping — it is confidently wrong.

### Independent LiDAR verification (diagnostic only, 150 clips)

Projected LIDAR_TOP keyframes into the anchor camera and computed the Gate-0 robust
log-ratio scale. **Nothing from this was fed back into the frozen stack.**

| | median | p5 | p95 |
|---|---:|---:|---:|
| `s_oracle` (LiDAR-measured, nuScenes) | **23.40** | 11.40 | 35.04 |
| `s_learned` (frozen estimator) | 28.55 | 17.55 | 43.41 |
| `s0` (KITTI constant) | 27.37 | — | — |
| `s_learned / s_oracle` | **1.265** | 1.07 | 1.54 |
| `s0 / s_oracle` | **1.170** | — | — |

| error metric | learned | constant |
|---|---:|---:|
| median \|log scale error\| | **0.2348** | 0.2053 |
| median relative scale error | **26.5 %** | 17.0 % |

For contrast, the same estimator on SemanticKITTI seq 08 achieved 2.9 % median relative
error against the constant's 12.7 %. On nuScenes the ordering reverses and both are badly
biased high. This is the mechanism behind `SCALE_DOES_NOT_TRANSFER`, and it independently
corroborates the occupancy result rather than resting on it.

---

## 14. Deterministic controls

Both are the Gate-3.1 source-selected controls, applied unchanged: **V1 = `dilate_r2`**,
**V2 = `fill_r1_k3`**. Neither was reselected on Occ3D — the candidate grid was not even
enumerated here.

The headline control finding: on SemanticKITTI, V3 beat `dilate_r2` by **+0.0395
[+0.0350, +0.0441]**. On Occ3D-nuScenes, `dilate_r2` beats V3 by **−0.0165
[−0.0214, −0.0118]**. Same frozen weights, same radius, same threshold; the sign flips.

T2 also achieves this with a *smaller* canonical footprint than T4 (147 904 vs 144 321 is
comparable) and a similar native count (22 150 vs 22 570), so the reversal is not a volume
artifact — dilation simply places its voxels better once the depth is mis-scaled.

---

## 15. Camera-visible versus non-visible

Under the official setting (`apply_camera_mask=True`), every evaluable voxel is
camera-visible **by construction** — the mask sets non-visible voxels to 255 — so the split
is degenerate there. A second, clearly separate diagnostic scoring applies only the
single-camera x-cut, leaving the non-visible half evaluable:

| id | official IoU | visible IoU / P / R | non-visible IoU / P / R | predictions camera-visible |
|---|---:|---|---|---:|
| T0 | 0.0647 | 0.065 / 0.52 / 0.07 | 0.024 / 0.06 / 0.05 | 0.209 |
| T1 | 0.0411 | 0.041 / 0.55 / 0.04 | 0.021 / 0.05 / 0.05 | 0.140 |
| T2 | 0.1434 | 0.143 / 0.56 / 0.17 | 0.036 / 0.05 / 0.15 | 0.187 |
| T4 | 0.1339 | 0.134 / 0.54 / 0.16 | 0.036 / 0.05 / 0.15 | 0.182 |
| T6 | 0.1270 | 0.127 / 0.56 / 0.15 | 0.034 / 0.05 / 0.13 | 0.177 |
| REF | 0.2387 | 0.239 / 1.00 / 0.24 | 0.000 | 0.997 |

Only **14–21 %** of predicted occupied voxels are camera-visible; the rest land in space
Occ3D marks unobserved, i.e. behind the true surfaces — the direct spatial signature of a
17–27 % depth over-scale. Precision in the non-visible partition is ~0.05 for every
configuration, so nothing is gaining there.

---

## 16. Representative scenes

`artifacts/occ3d_zeroshot/eval/scenes_zeroshot.png` — bird's-eye views of the official grid
for the best, median and worst scene by T6 scene IoU.

| scene | role | T6 scene IoU | T1 | T2 | T4 | T6 | REF |
|---|---|---:|---:|---:|---:|---:|---:|
| `scene-0102` | best | 0.413 | 0.205 | 0.391 | 0.417 | **0.427** | 0.623 |
| `scene-0268` | median | 0.119 | 0.030 | **0.135** | 0.116 | 0.112 | 0.281 |
| `scene-0344` | worst | 0.016 | 0.001 | 0.006 | **0.008** | 0.008 | 0.008 |

In the worst scene even the in-band oracle reaches only 0.008: the C3 reconstruction covered
essentially none of the ground truth, so no correction could recover it. That is a coverage
failure upstream of the corrector, not a corrector failure.

---

## 17. Limitations

1. **Single-source transfer.** One source (SemanticKITTI) → one target. This is not OccAny's
   five-source protocol and says nothing about how a multi-source-trained model would behave.
2. **Not comparable to OccAny's published numbers.** OccAny's Occ3D-nuScenes setting is
   surround (6 cameras, one timestamp); ours is temporal (5 frames, CAM_FRONT). Different
   input evidence, different task.
3. **Absolute IoUs are protocol-bound, not method quality.** Only 6.9 % of the native grid is
   evaluable, and a single front camera over 2 s cannot cover an 80 × 80 m ego-centric grid.
   The in-band oracle caps any band-restricted method at 0.2387. Comparisons *within* this
   table are meaningful; comparisons to surround-view or LiDAR methods are not.
4. **τ = 0.45 and radius 0.6 m were fitted to KITTI** and deliberately not re-tuned. A
   re-tuned model would very likely score higher; that would no longer be a transfer test.
5. **The anchor convention differs from OccAny's** (last vs first frame), frozen to match
   Gates 0–3.1.
6. **Occ3D GT is aggregated over the full sequence**, so it marks occupied much that is
   never observed in a 2 s front-camera window — part of the low recall ceiling.
7. **The LiDAR scale diagnostic covers 150 of 1 182 clips** (deterministic seed-0 subsample)
   and uses the anchor frame only.
8. **Sequence 08 of SemanticKITTI was seen during development**; nuScenes was not, so this
   evaluation is genuinely held out — but it is one target dataset and one camera.

---

## 18. Exact next decision

The evidence points to one decision, and it is not a tuning decision.

**Both learned components are KITTI-specific and must not be shipped or extended as they
stand.** Concretely:

1. **Do not carry the clip-scale estimator forward.** It is worse than a constant on
   nuScenes (26.5 % vs 17.0 % median relative error) and worse than nothing in occupancy
   (T0 → T1 = −0.0239). Gate 2 already showed it is *only* a scale estimator; Gate 4 shows
   the scale it estimates is a memorised KITTI camera geometry. Any future scale component
   must be trained multi-source or conditioned on calibration that is actually available at
   inference (camera height, intrinsics), and must be validated cross-dataset before use.
2. **Keep the operation, drop the weights.** Expanding occupancy into a 0.6 m band helps on
   both datasets; the learned weights add +0.0395 on KITTI and −0.0165 on nuScenes. Until a
   learned prior beats `dilate_r2` on a dataset it was not trained on, deterministic dilation
   is the honest baseline for this stage of the pipeline.
3. **The binding constraint is upstream.** T1's recall is 0.044 and 67.5 % of fused points
   land in-grid, while the in-band oracle sits at 0.2387. Depth *shape* — the 33.1 % factor
   Gate 1 isolated and Gate 2 failed to correct — plus scale bias is what caps everything
   downstream. Voxel-space post-processing cannot fix it.

**Recommended next experiment:** OccAny-aligned multi-source preparation and training of the
geometry stage (Phase M0 and its successor), with the frozen-transfer harness built here
reused unchanged as the held-out evaluation. This gate's tooling — manifest, adapter,
canonical/native conversion, leakage tests and controls — is directly reusable and should be
run against any future model *before* it is believed.

This report makes no model change. Nothing was tuned after seeing these numbers.

---

## 19. Tests

New: `tests/occ3d_zeroshot/test_transfer.py` — **24 tests** covering frozen scalars and
checkpoint settings, the absence of any optimizer, the 0.6 m radius guard, canonical/native
extent and conversion (including monotonicity and method-agnosticism), quaternion and
transform validity, similarity/anchor/scale-once assertions, CAM_FRONT axis directions,
floor-binning, full target randomisation leaving predictions bit-identical, source scans
proving the pipeline and adapter never read a label, the OccAny binary reduction, masks
restricting metrics only, region purity and gating, control region parity, manifest protocol
(5 frames, one camera, one scene, ordered, ~2 Hz, anchor last), val-only scenes, and
scene-level bootstrap determinism.

Whole repository:

```
411 passed, 2 skipped, 1 warning
```

| suite | result |
|---|---|
| `tests/scale_gate` | 49 passed |
| `tests/depth_gate` | 35 passed |
| `tests/voxel_gate` | 29 passed |
| `tests/voxel_gate_validation` | 24 passed |
| `tests/occ3d_zeroshot` (new) | 24 passed |
| remainder | 250 passed, 2 skipped |

The 2 skips are pre-existing and unrelated (`tests/prompted_lingbot/test_anchors.py:100`).

