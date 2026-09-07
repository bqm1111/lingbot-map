# Architecture ledger — Gates 0–3.1

Every component this project has trained or frozen, with its exact structure extracted
from source and checkpoints (never inferred from parameter counts), and an explicit
statement of what each experiment can and cannot support.

Grid and protocol shared by every occupancy row: `SEMANTICKITTI_GRID` = **256×256×32 at
0.2 m**, origin (0, −25.6, −2.0), frame = velodyne of the clip's anchor frame, `empty=0`,
`ignore=255`, floor-binning voxeliser (`min_points_per_voxel=1`), evaluation mask
`invalid==0`. Clips are 5 frames at native stride 5. Sequence 08 (163 clips) is the
held-out evaluation set throughout; 00–07 are source training and 09–10 source selection.

> **All results everywhere in this ledger are in-domain SemanticKITTI.** One dataset, one
> sensor rig, one city, fixed camera height and intrinsics. No cross-dataset evidence exists.

---

## 1. Frozen LingBot-Map interface

| field | value |
|---|---|
| purpose | streaming 3D reconstruction; the frozen upstream this project builds on |
| status | **frozen**, always. `requires_grad_(False)` on every parameter, and `cache_lingbot.py` raises if any parameter is trainable |
| checkpoint | `checkpoints/lingbot-map/204754b/lingbot-map.pt`, SHA-256 `ee665103…6d8f2e1cd72` |
| parameters | **1 157 943 540** (1 342 tensors, summed from the checkpoint state dict) |
| class / mode | `lingbot_map.models.gct_stream.GCTStream`, direct mode — one independent forward per clip, state reset between clips |
| construction | `img_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024, kv_cache_sliding_window=64, kv_cache_scale_frames=5, kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True, use_sdpa=False, camera_num_iterations=4` |
| input | 5 RGB frames, `load_and_preprocess_images(mode="crop", image_size=518, patch_size=14)`; KITTI 1226×370 → **[5, 3, 154, 518]** (pure resize, no crop); patch grid 11×37 = 407 |
| autocast | bfloat16 for the aggregator; heads stay fp32 |
| outputs consumed | `pred_depth` **[5, 154, 518]** fp16 — optical-axis **z-depth in canonical (non-metric) units**; `pred_depth_conf` **[5, 154, 518]** fp16 — `expp1` precision, `1+exp(x) > 1`, higher is better; `pred_K` **[5, 3, 3]** fp32 for the processed lattice; `pred_pose_c2w` **[5, 4, 4]** fp32, **camera-to-world**, translation in canonical units; `pose_enc` **[5, 9]** |
| outputs *not* used | `pooled_pre_gct` **[5, 1024]** is cached but never consumed by Gates 0–3.1 |
| cache | `artifacts/scale_gate/cache/lingbot/*.npz`, 926 clips |
| supports | that these outputs are internally consistent but non-metric, and that a single positive scalar per clip is the correct metric correction |
| does not support | anything about LingBot's internal features; no Gate reads its token activations |

**Pose-convention correction (Gate 0/1).** `pose_encoding_to_extri_intri` returns
**camera-to-world**, despite the `demo.py` comment saying otherwise. Verified empirically:
raw-as-c2w gives 0.57° rotation error, +1.000 translation-direction cosine and 0.31 m
error against ground truth; inverting gives 2.67°, −0.999 and 16.26 m. All caches were
repaired by `tools/scale_gate/repair_poses.py` from the stored `pose_enc`, without
re-running inference.

---

## 2. Constant-scale C0 baseline

| field | value |
|---|---|
| purpose | deployable geometry with no per-clip scale estimate — the honest floor |
| status | **no trainable parameters** |
| definition | `depth = s0 · D_lingbot`, `translation = s0 · t_lingbot`, rotations unchanged |
| `s0` | **27.3665**, `configs/depth_gate/refine.yaml: scale.constant`; fitted on source sequences only. (Gate 1's `F_deployable` row used the marginally different 27.3498 fitted on a slightly different source set.) |
| inputs | `pred_depth`, `pred_depth_conf`, `pred_K`, `pred_pose_c2w` |
| fusion support | `conf ≥ 1.5` ∧ finite ∧ `1 m < s0·D < 60 m` |
| output | fused point cloud → occupancy **[256, 256, 32]** bool |
| sequence-08 IoU | **0.0573** (Gate 2), reproduced **0.05725** in Gate 3.1 |
| config | `configs/depth_gate/refine.yaml`, `configs/scale_gate/semantickitti.yaml` |
| supports | that a single fitted constant recovers 32.6 % of the gap from zero scale |
| does not support | any claim about per-clip scale variation, which it cannot represent |

---

## 3. Clip-scale extraction from the frozen Gate-2 depth head

| field | value |
|---|---|
| purpose | one positive scalar per clip, converting canonical geometry to metric |
| status | **frozen** — reuses the Gate-2 checkpoint; nothing is trained in Gates 3/3.1 |
| definition | `a_clip = median of r over the C0 fusion support`; `s_learned = s0 · exp(a_clip)` |
| support for the median | `conf ≥ 1.5` ∧ finite ∧ `1 m < s0·D < 60 m` — LingBot only; **no LiDAR, no GT pose, no oracle scale** |
| aggregation | one scalar over all 5 frames jointly; no per-frame scale is fitted |
| `r_shape = r − a_clip` | **discarded**. Never formed in the Gate-3/3.1 code path; a test greps the modules and fails on any occurrence |
| accuracy (seq 08) | median relative error vs the Gate-0 oracle **2.9 %**, against the constant's 12.7 %; Pearson/Spearman 0.98; ratio 5–95 % 0.93–1.06 |
| sequence-08 IoU (C3) | **0.0778**, reproduced **0.07784** in Gate 3.1 |
| implementation | `voxel_gate/c3.py:clip_scale`, importing `tools/depth_gate/decompose_residual.py` |
| supports | that a clip-level metric scale is recoverable in-domain to ~3 % |
| does not support | generalisable scale recovery — sequence 08 shares camera height, intrinsics, sensor and city with 00–07 |

---

## 4. Gate-2 depth residual CNN (`depth_cnn`) — exact architecture

| field | value |
|---|---|
| purpose | tested whether a small head can correct LingBot depth **shape**; the answer was no |
| status | trained in Gate 2; **frozen** in Gates 3 and 3.1 |
| checkpoint | `artifacts/depth_gate/runs/depth_cnn/best.pt`, SHA-256 `2a91822e…19faafbe1` |
| parameters | **48 129** (summed from the state dict) |
| input tensor | **[B, 5, 154, 518]** |

**Input channels — confirmed from `depth_gate/models.py:build_inputs` and the
checkpoint's `use_rgb: False`, `in_ch: 5`:**

| # | channel | definition |
|---|---|---|
| 0 | normalised log depth × valid | `(log(s0·D) − 2.6606) / 0.7538`, masked by the LingBot valid mask |
| 1 | normalised confidence | `(conf − 8.2533) / 7.6816` |
| 2 | valid mask | binary |
| 3 | `x` coordinate | `linspace(−1, 1, W)` |
| 4 | `y` coordinate | `linspace(−1, 1, H)` |

> **The head receives depth, confidence, validity and normalised image coordinates only.
> It receives no RGB and no LingBot visual or token features.** `use_rgb` is `False` in the
> checkpoint and the `rgb` channel is never appended. Its result therefore supports only
> the narrow claim that *this particular geometry-only head learned scale rather than
> useful shape correction*. It is not evidence that shape correction is impossible from
> richer inputs; the `rgbd_unet` variant that did receive RGB was also tested and was
> **worse** (474 273 params, AbsRel 0.0950 vs 0.0898), but neither was given LingBot
> features, so the question of feature-conditioned shape correction remains open.

**Exact layer sequence** (`ConvBlock(c_in, c_out)` = `Conv2d(3×3, pad 1) → GroupNorm(8) →
GELU → Conv2d(3×3, pad 1) → GroupNorm(8) → GELU`; every conv has stride 1):

```
stem  = ConvBlock(5, 32)                          # 1 440 + 32 + 32 + 32 | 9 216 + 32 + 32 + 32
body  = Sequential(ConvBlock(32,32), ConvBlock(32,32))
h     = stem(x); h = h + body(h)                  # residual skip
head  = Conv2d(32, 1, kernel 1)                   # 32 + 1, ZERO-initialised
out   = 0.7 * tanh(head(h))                       # = r, the log residual
D_ref = D_base * exp(r)
```

| field | value |
|---|---|
| receptive field | stem 5×5; body adds 8 → 13×13 px; the residual skip means the effective field spans 5×5 to **13×13 processed pixels** (≈ 31×31 native KITTI pixels) |
| initialisation | default PyTorch everywhere; **head weight and bias zeroed**, so training starts exactly at the unrefined depth |
| loss | `SmoothL1(log D_refined, log D_lidar, beta=0.1) + 0.05·mean|r|`, on valid projected-LiDAR pixels only |
| supervision | projected metric LiDAR depth (z-buffered), sparse |
| splits | train 00–07; early stopping 09–10; evaluation 08 |
| inference dependencies | LingBot depth + confidence, `s0`; **no LiDAR, no GT pose** |
| supports | that the head is a **clip-level scale estimator**: C3 (its scalar alone, correctly coupled) reproduces 101.5 % of the reported occupancy gain, while the median-zero spatial residual gives +0.0005 (n.s.) with oracle scale and −0.0005 (n.s.) with GT poses |
| does not support | that LingBot depth *shape* cannot be corrected; only that this geometry-only head did not do it |

---

## 5. Coupled depth / pose-translation scaling

| field | value |
|---|---|
| purpose | apply one metric scalar consistently, so the fused cloud is a pure similarity |
| status | **no parameters** |
| definition | `c_scaled[t] = c_anchor + s·(c[t] − c_anchor)`; rotations untouched; applied **before** the `cam_to_velo` world alignment |
| equivalent form | `rel = inv(P_anchor) @ P_f; rel[:3,3] *= s` — proven identical to 1e-10 |
| implementation | `tools/depth_gate/decompose_residual.py:scaled_relative_pose`, imported by Gates 3 and 3.1 |
| asserted invariants | anchor is a fixed point; rotation block bit-identical across `s`; `‖t(s)‖/‖t(1)‖ == s` to 1e-12; scale never applied twice; `fuse(s·D, s·t) == s·fuse(D, t)` to 1e-12 |
| why it matters | Gate 2's C1 scaled depth by `s0·exp(r)` while leaving translation at `s0`. Correcting that coupling is worth **+0.0015 IoU [+0.0008, +0.0023]** and lifts the in-grid fraction from 0.623 to 0.639 |
| supports | that depth and translation must be scaled together |
| does not support | any claim about pose *rotation* accuracy, which is never modified |

---

## 6. Multi-frame fusion and voxelisation

| field | value |
|---|---|
| purpose | turn 5 scaled depth maps into one occupancy volume in the benchmark's frame |
| status | **no parameters**; frozen since Gate 1 |
| unprojection | `x = (u − cx)·d/fx`, `y = (v − cy)·d/fy`, `z = d` (verified optical-axis z-depth) |
| fusion | every frame transformed into the **anchor camera** (last frame of the clip) by the scaled relative pose, then `cam_to_velo` from `calib.txt` |
| admission | `conf ≥ 1.5` ∧ finite ∧ `1 m < depth < 60 m`, `pixel_stride = 1` |
| voxelisation | `idx = floor((p − origin)/0.2)`, `min_points_per_voxel = 1` — `prompted_lingbot.occupancy.occupancy_from_points`, reproducing OccAny `ray.py:217` |
| typical output | ~318 000 fused points → ~24 700 occupied voxels per clip (C3, seq 08) |
| in-grid fraction | 0.640 for C3; ~36 % of fused points fall outside the 51.2 × 51.2 × 6.4 m grid |
| supports | comparability with the OccAny/SemanticKITTI SSC binary-occupancy protocol |
| does not support | sub-voxel accuracy claims; floor-binning discards all sub-0.2 m structure |

---

## 7. Gate-3 3D voxel CNN (`VoxelCorrector3D`) — exact architecture

| field | value |
|---|---|
| purpose | correct and locally infill occupancy inside a bounded band around observed surfaces |
| status | **trained** (the only trained component in Gates 3 / 3.1) |
| parameters | **16 577** (summed from the state dict) |
| input tensor | **[B, 6, 256, 256, 32]** |
| output tensor | **[B, 1, 256, 256, 32]** logits |
| implementation | `voxel_gate/models.py` |

**Input channels** (order is part of the checkpoint; channels 1–4 masked by channel 0,
standardised with source-training statistics only):

| # | channel | definition |
|---|---|---|
| 0 | `occupied` | frozen-geometry binary occupancy |
| 1 | `log1p_count` | `log1p` of fused points per voxel |
| 2 | `n_frames` | number of **distinct** contributing clip frames (1–5) |
| 3 | `mean_confidence` | mean LingBot depth confidence over contributing points |
| 4 | `mean_point_depth_m` | mean metric depth of contributing points at their source pixel |
| 5 | `in_correction_region` | the band indicator — **see §7.1, this is where Gate 3 leaked** |

No positional encoding: the model receives no voxel coordinate, height, range, sequence
id or frame id, so it cannot key on absolute scene position.

**Exact layer sequence** (every conv stride 1, padding 1, `bias=True`):

```
body = Sequential(
    Conv3d( 6, 16, kernel 3, pad 1), GroupNorm(4, 16), GELU,      #  2 592 + 16 | 16 + 16
    Conv3d(16, 16, kernel 3, pad 1), GroupNorm(4, 16), GELU,      #  6 912 + 16 | 16 + 16
    Conv3d(16, 16, kernel 3, pad 1), GroupNorm(4, 16), GELU)      #  6 912 + 16 | 16 + 16
head = Conv3d(16, 1, kernel 1)                                     #     16 +  1, ZERO-init
prior = (2*x[:,0:1] - 1) * 4.0                                     # +/-4 logit from C3 occupancy
z     = prior + head(body(x))
pred  = (sigmoid(z) >= tau) inside R_infer, frozen geometry preserved outside
```

| field | value |
|---|---|
| receptive field | three sequential 3×3×3 convs → **7×7×7 voxels = 1.4 × 1.4 × 1.4 m**; the head is 1×1×1 and the prior is pointwise |
| initialisation | default PyTorch; **head weight and bias zeroed**, so the untrained model reproduces the input occupancy exactly at any threshold ≤ 0.5 (tested) |
| residual sign | unbounded, so voxels can be **deleted** (large negative) or **added** (large positive); both directions tested by forcing the head bias to ±12 |
| loss | `weighted BCE(pos_weight resolved from source-train balance) + 0.5 · soft Dice`, restricted to `S_train` |
| supervision | binary LiDAR occupancy, `(label != empty) ∧ valid`. **No semantic class labels are ever loaded** |
| splits | train 00–07 (652 clips); selection 09–10 (111 clips); evaluation 08 (163 clips) |
| optimiser | AdamW, lr 1e-3, weight decay 1e-4, grad-norm clip 1.0, batch 1 full grid, bf16 autocast, ≤20 epochs, patience 4 |
| threshold | τ = 0.45, selected on 09–10 in Gate 3 and **frozen** in Gate 3.1 |
| memory / time | 1.24 GiB peak training, ~31 s/epoch, 1.6 ms inference per clip. The full grid fits, so **no tiling, cropping or downsampling is used** |
| inference dependencies | frozen-geometry voxel evidence and `R_infer` only |

### 7.1 The Gate-3 correction-region defect

| | Gate 3 (`voxel_gate/data.py:81`) | Gate 3.1 (`voxel_gate_validation/data.py`) |
|---|---|---|
| region | `dilate(occ, 3) & keep` | `dilate(occ, 3)` |
| `keep` | `unpack(valid_bits)` — the SemanticKITTI per-sample `.invalid` mask | not used in the region at all |
| used as | input channel 5 **and** the output gate | — |
| effect | removed ~75 700 voxels/clip (25 % of the band) from the inference region | region is 33 % larger and target-independent |

`keep` is label-side: it encodes which voxels the SSC ground truth considers observable.
Feeding it to the model and gating the output with it is a target-derived inference
dependency. Gate 3's 0.2903 is therefore **non-authoritative**; Gate 3.1 reruns it clean.
Gate 3.1 fixes this structurally, not by deletion: `data.input_view` copies out only the
five `c3_*` evidence keys, so the inference path physically cannot see `valid_bits`,
`gt_flat` or `vc_flat`.

The label-side mask is still used, but only where it is legitimate:

```
R_infer = dilate(frozen_occupancy, 3)      inference region and output gate — no target
S_train = R_infer AND valid                loss restriction only
S_eval  = valid                            metric restriction only
```

| supports | that a small local 3D prior improves occupancy far beyond deterministic morphology |
| does not support | occluded-scene completion beyond 0.6 m; anything cross-dataset; and — pending Gate 3.1 — nothing at all, since the Gate-3 number is provisional |

---

## 8. Deterministic morphology controls

| field | value |
|---|---|
| purpose | establish whether any learned gain exceeds indiscriminate densification |
| status | **no parameters, no training** |
| implementation | `voxel_gate/controls.py` |
| families | `dilate(r)`; `close(r)` = dilate then erode; `fill(r, k)` = add empty voxels with ≥ k occupied neighbours in the (2r+1)³ window |
| candidate grid | r ∈ {1, 2} × {dilate, close} plus r ∈ {1, 2} × 18 neighbour thresholds = **44 candidates** |
| primitive | `F.max_pool3d(kernel 2r+1, stride 1, padding r)` for dilation; `avg_pool3d(divisor_override=1)` for neighbour counts |
| region | identical `R_infer` to the learned model; each control preserves the frozen geometry outside it |
| selection | best mean IoU on source-selection sequences 09–10; sequence 08 plays no part |
| Gate-3 result | `dilate_r2` reached **0.1759** on sequence 08 from a 0.0778 base — nearly double, purely by inflating occupied volume 5.6× |
| supports | the single most important calibration fact in this project: **any SemanticKITTI occupancy number without a volume-matched morphology control is uninterpretable** |
| does not support | use as a method; precision collapses from 0.408 to 0.284 |

---

## 9. What the project can and cannot claim

**Can claim (in-domain SemanticKITTI, sequence 08):**

- LingBot geometry is internally consistent but non-metric; one positive scalar per clip is the right correction, and it must be applied to depth and pose translation together.
- The Gate-2 depth head is a clip-scale estimator, not a shape corrector (Gate 2.1 decomposition, `SCALE_CORRECTION_ONLY`).
- Deterministic dilation nearly doubles occupancy IoU by volume inflation alone, so morphology controls are mandatory.
- Depth *shape* error remains the dominant unaddressed geometric factor: −0.0416 IoU, 33.1 % of the 0.1255 five-frame visible-reconstruction score, under matched support and matched scale.

**Cannot claim:**

- Any generalisation. Every number comes from one dataset, one sensor, one city, with sequence 08 held out temporally rather than environmentally.
- That shape correction is impossible — only that a geometry-only head with a 13×13-pixel receptive field and no RGB or LingBot features did not achieve it.
- That 0.1255 is an IoU upper bound. It bounds a perfect *five-frame visible reconstruction*; plain dilation exceeds it, because the SSC target marks occluded near-surface voxels occupied.
- Anything from Gate 3's 0.2903 until Gate 3.1's clean rerun replaces it.
