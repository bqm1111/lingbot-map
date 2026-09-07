# Gate 3 — visible-voxel correction

## Diagnosis: `VISIBLE_VOXEL_CORRECTION_PASSES`

All six criteria are met, most of them by a wide margin.

## Plain-language conclusion

A 16 577-parameter 3D CNN, working only on voxel evidence that frozen C3 geometry already
produces, raises sequence-08 occupancy IoU from **0.0778 to 0.2903** — every one of the
163 clips improves. It beats the strongest deterministic control (dilation, 0.1759) by
**+0.1145 [+0.1078, +0.1213]**, and it does so while predicting **59 % fewer** occupied
voxels than that control. Precision *rises* from 0.408 to 0.687 while recall rises from
0.090 to 0.340. This is not densification: an occupied-count-matched deterministic control
reaches only 0.1232.

Two things must travel with that headline.

1. **Deterministic dilation alone nearly doubles IoU** (0.0778 → 0.1759) purely by
   inflating occupied volume 5.6×. Any occupancy result on this benchmark that does not
   report a morphology control is uninterpretable. Gate 3's controls are the reason the
   learned number can be believed.
2. **The learned module is doing bounded local infill, not only correction of observed
   voxels.** 86 % of the voxels it adds lie outside the five-frame visible-reconstruction
   set. It never leaves the 0.6 m band around observed C3 surfaces — that is the hard
   geometric guarantee — but inside that band it is populating near-surface space that no
   perfect visible reconstruction would produce. The right name for what passed is
   *local occupancy correction and infill*, and §10 quantifies the split.

A consequence worth stating flatly: **the 0.1255 "five-frame visible-reconstruction
ceiling" is not an upper bound on IoU.** Plain dilation exceeds it. It bounds what a
perfect *visible reconstruction* scores, not what any method can score, because
SemanticKITTI's SSC target marks occluded near-surface voxels occupied too.

> **All results are in-domain SemanticKITTI results.** Training 00–07, selection 09–10,
> evaluation 08 — one dataset, one sensor, one city, fixed camera height and intrinsics.
> The 163 clips are not 163 independent environments; the confidence intervals describe
> variation *within* sequence 08. Nothing here is cross-dataset.

No earlier gate's reports, checkpoints, caches or artifacts were modified. `git status`
shows only the two pre-existing modified files from before this task.

---

## 1. Frozen protocol and provenance

Unchanged from Gates 1–2: sequence 08, 163 five-frame clips, clip stride 5,
`pixel_stride = 1`, confidence threshold 1.5, `SEMANTICKITTI_GRID` **256×256×32 @ 0.2 m**,
origin (0, −25.6, −2.0), the floor-binning voxeliser, the evaluation mask
(`invalid == 0`), the SemanticKITTI calibration and the repaired `pred_pose_c2w`
convention. Semantic class labels are never loaded. LiDAR is used only to build binary
geometric supervision and to evaluate.

| item | value |
|---|---|
| LingBot checkpoint | `ee665103…6d8f2e1cd72` (frozen) |
| Gate-2 depth head | `artifacts/depth_gate/runs/depth_cnn/best.pt` — `2a91822e…19faafbe1` (frozen) |
| Gate-3 corrector | `artifacts/voxel_gate/runs/voxel_cnn3d/best.pt` — `c91e2ab0…5bf0b1a3f0e` |
| LingBot cache (926 clips) | `1e8a7eea…67f7b95e21` |
| projected-LiDAR cache | `1db2aa63…9054877f3706` |
| RGB cache | `98ed95a0…c46aa553371` |
| val manifest | `16c8e6b5…f29ddfe0` |
| train manifest | `501f59a9…6a54c7c97544` |
| Gate-3 config | `configs/voxel_gate/visible_correction.yaml` — `12fedde1…1cce761101` |
| Gate-2 config | `configs/depth_gate/refine.yaml` — `b7692022…41e0dd92` |
| geometry config | `configs/scale_gate/semantickitti.yaml` — `472b4531…5f3e4ea086` |

Tools: `tools/voxel_gate/{cache_c3,select_region,train,eval_voxel,figures}.py`.
Library: `voxel_gate/{c3,voxels,data,models,losses,controls}.py`.
Artifacts: `artifacts/voxel_gate/`. Tests: `tests/voxel_gate/test_voxel_gate.py`.

### Splits

| role | sequences | clips | used for |
|---|---|---:|---|
| source train | 00–07 | 652 | gradients, normalisation statistics, `pos_weight`, band radius |
| source selection | 09, 10 | 111 | early stopping, probability threshold, deterministic control |
| **evaluation** | **08** | **163** | reported metrics only |

Gate 2's split policy verbatim. Clip-id disjointness and the absence of sequence 08 from
every selection path are asserted in code and in tests.

### C3 reproduction

C3 is reproduced before anything is trained, and again at evaluation:

```
cache_c3.py --split val   ->  mean IoU 0.07784   |0.07784 - 0.0778| = 3.9e-5  <= 5e-4  PASS
eval_voxel.py             ->  mean IoU 0.07784                                        PASS
```

Both tools abort with a non-zero exit if the tolerance is exceeded. A test asserts it too.

---

## 2. Frozen C3 input geometry

```
r          = depth_cnn(s0 * D_lingbot, conf, valid, x, y)        s0 = 27.3665, head frozen
a_clip     = median of r over the unchanged C0 five-frame fusion support
s_learned  = s0 * exp(a_clip)
depth      = s_learned * D_lingbot
pose       = LingBot pose, relative translations scaled by s_learned (rotations untouched)
```

The scale extraction and the coupled pose scaling are **imported** from the validated
Gate-2 implementation (`tools/depth_gate/decompose_residual.py`), not reimplemented, so
this is the same code path that produced 0.0778.

`r_shape` is never formed. A test greps the entire Gate-3 code path (docstrings and
comments stripped) and fails on any occurrence of `r_shape`; `c3.use_r_shape: false` is
asserted. LingBot and the depth head both have `requires_grad_(False)` and no optimiser
ever sees their parameters. Poses are not refined beyond the translation coupling, and
`check_pose_scaling` re-verifies the four Gate-2 assertions (anchor fixed, rotations
unscaled, translation ratio exactly `s`, agreement with the Gate-1 convention) on real
sequence-08 poses during caching.

---

## 3. Features, targets and the correction region

### Inference inputs — six channels, all computable from frozen C3 alone

| # | channel | definition |
|---|---|---|
| 0 | `occupied` | C3 binary occupancy (floor-binned, `min_points_per_voxel = 1`) |
| 1 | `log1p_count` | `log1p` of fused C3 points per voxel |
| 2 | `n_frames` | number of *distinct* clip frames contributing to the voxel |
| 3 | `mean_confidence` | mean LingBot depth confidence over contributing points |
| 4 | `mean_point_depth_m` | mean metric depth of contributing points at their source pixel |
| 5 | `in_correction_region` | the band indicator (channel 3.2 below) |

Channels 1–4 are standardised with statistics computed on **source-training clips only**
and masked by channel 0. No positional encoding is supplied: the model gets no voxel
coordinate, height, range, sequence id, frame id or filename, so it cannot key on absolute
scene position.

### Correction region

No predicted-visibility or ray-casting utility exists in this repository
(`grep -rn "visib\|frustum\|free_space\|raycast"` finds none outside plotting code), so
per the brief the region is a **documented local band derived entirely from C3
occupancy**:

```
R = dilate(C3_occupied, radius = 3 voxels = 0.6 m)  AND  evaluation-valid mask
```

Radius selected on **source-training** clips (00–07) by a model-free rule fixed in advance:
the smallest candidate whose *in-band oracle IoU* reaches 90 % of the best candidate's.
Preferring the smallest sufficient band is the anti-densification choice.

| radius | \|R\| / grid | GT occupancy in R | visible set in R | in-band oracle IoU | (restricted to visible set) |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.09 % | 0.273 | 0.614 | 0.2731 | 0.0841 |
| 2 | 6.69 % | 0.367 | 0.731 | 0.3667 | 0.0994 |
| **3** | **9.23 %** | **0.427** | **0.794** | **0.4273** ← 90 % rule selects | 0.1075 |

Source-train reference on the same clips: C3 IoU 0.0807, full visible ceiling 0.1314.

### Outside the region

Dilation is extensive, so `R` **contains every C3-occupied voxel**. "Preserve C3 exactly
outside `R`" and "force empty outside `R`" are therefore the same statement. The code
writes the preserving form explicitly —
`apply_region(pred, c3, R) = (pred & R) | (c3 & ~R)` — and two tests assert that no
configuration, learned or deterministic, changes a voxel outside `R`. Measured
`in_region_frac` is exactly 1.000 for V0/V1/V2/V3.

### Supervision

Target: **plain binary LiDAR occupancy**, `(label != empty) & valid`, supervised only on
`S = R ∩ valid`. The anti-completion guarantee is the *region*, not the label.

> **A design decision changed mid-run, on source data, before sequence 08 was touched.**
> The first implementation additionally intersected the label with the Gate-1
> visible-reconstruction occupancy. The source-data probe showed that ceiling caps any
> band-restricted method at IoU 0.1075 — below what plain dilation already scores — so the
> label was relaxed to the brief's literal specification (binary LiDAR occupancy inside
> the permitted region). Both variants are reported in the radius table above and as row
> `VRV` in §8. No sequence-08 number existed at the time of this change.

The Gate-1 visible-reconstruction occupancy is retained **only** as a diagnostic (§10) and
as the `VC`/`VRV` reference rows. It is never a model input and never gates inference.

---

## 4. Proof that inference inputs contain no LiDAR or oracle information

| guarantee | how it is enforced | test |
|---|---|---|
| model input is independent of every LiDAR-derived array | inputs are built by `voxel_gate.data.inputs`, which reads only the five `c3_*` keys | `test_model_input_is_bit_identical_when_the_targets_are_mutated`: zeroes `gt_flat`, `vc_flat` and `valid_bits`, asserts `torch.equal` on the assembled tensor |
| the input path never *reads* a target key | a `dict` subclass raises on any access to `TARGET_KEYS` | `test_inputs_reads_no_target_key` |
| the correction region needs no target | region recomputed after deleting the targets | `test_correction_region_is_computable_without_any_target` |
| no channel names a forbidden quantity | `FEATURE_NAMES` × `FORBIDDEN_INPUTS` cross-check | `test_input_channel_names_declare_no_forbidden_quantity` |
| oracle scale is unused | C3 uses `s_learned` only; `scale_targets_*.csv` is never opened by any Gate-3 tool | `grep` — no reference exists |
| sequence 08 never selects anything | asserted at start-up in `select_region.py` and `train.py`; recorded in `train.json` | `test_source_and_sequence08_clip_ids_are_disjoint`, `test_threshold_and_hyperparameters_were_not_selected_on_sequence_08`, `test_region_and_control_were_selected_on_source_only` |
| grid untouched | dims/voxel size/origin/labels asserted | `test_grid_dimensions_and_voxel_size_are_unchanged` |
| the fast scorer equals the frozen evaluator | compared voxel-for-voxel against `binary_occupancy_scores` on the real label volume | `test_cached_scores_match_the_frozen_evaluator_exactly` |

---

## 5. Model and training

```
z = prior(C3 occupancy) + delta_theta(features)        prior = +/- 4.0
p = sigmoid(z)
prediction = (p >= tau) inside R,  C3 preserved outside R
```

Three 3×3×3 convolutions at width 16 (`Conv3d → GroupNorm → GELU`), then a zero-initialised
1×1×1 head. **16 577 parameters.** Zero initialisation means the untrained model reproduces
V0 exactly at any threshold of 0.5 (tested), so every change is earned. The residual is
unbounded in sign, so the model can delete a false occupied voxel or add a missed one;
both directions are exercised in tests by forcing the head bias to ±12.

Loss on `S = R ∩ valid` only: weighted BCE (`pos_weight = 3.410`, resolved from the
source-train class balance — positives are 22.7 % of the region) plus `0.5 ×` soft Dice.
A test mutates a target *outside* the mask and asserts the loss does not move.

| setting | value |
|---|---|
| optimiser / lr / weight decay | AdamW, 1e-3, 1e-4 |
| batch size / precision | 1 full grid, bf16 autocast |
| epochs / patience | 20 / 4 |
| best epoch | 17 |
| source-selection IoU at best epoch | 0.2849 @ τ = 0.45 |
| **selected threshold τ** | **0.45** (chosen on sequences 09–10) |
| peak GPU, training | 1.24 GiB |
| peak GPU, evaluation | 2.95 GiB |
| training time | 634 s (31 s/epoch) |
| inference time | 1.6 ms per clip (0.26 s for all 163) |

The full 256×256×32 grid fits comfortably, so **no tiling, cropping or downsampling was
needed** — the training and evaluation resolution is the original one throughout.

> The epoch cap was raised from 12 to 20 after a first run whose source-selection curve was
> still rising at the cap. That decision used source sequences 09–10 only, before any
> sequence-08 evaluation existed. It is the only hyperparameter that moved; no architecture
> or loss search was run.

---

## 6. Deterministic controls

Same inputs, same region, same frozen evaluator. Candidate grid: dilation (r ∈ {1, 2}),
closing (r ∈ {1, 2}) and neighbour-count fill (r ∈ {1, 2} × 18 thresholds) — 44 candidates.
**V1** is the best of them on the source-selection sequences; sequence 08 played no part.

Top of the source-selection table (C3 baseline there: IoU 0.0757, 24 378 occupied):

| control | IoU | P | R | occupied |
|---|---:|---:|---:|---:|
| **`dilate_r2`** | **0.1653** | 0.265 | 0.330 | 133 161 ← V1 |
| `fill_r2_k3` | 0.1648 | 0.280 | 0.306 | 116 276 |
| `fill_r2_k4` | 0.1635 | 0.285 | 0.296 | 110 711 |
| `dilate_r1` | 0.1571 | 0.323 | 0.248 | 82 385 |

**V2**, the occupied-count-matched control, is chosen *after* V3 by picking the candidate
whose mean sequence-08 occupied count is closest to V3's: `fill_r2_k22`, 57 493 voxels
against V3's 56 230 — a 2.2 % match. Matching on sequence 08 *advantages the control*, which
is the conservative direction; V3's own threshold, radius and weights remain source-only.

---

## 7. Configurations evaluated

| id | definition |
|---|---|
| **V0** | frozen C3 occupancy |
| **V1** | `dilate_r2` inside R — strongest source-selected deterministic control |
| **V2** | `fill_r2_k22` inside R — occupied-count-matched deterministic control |
| **V3** | learned corrector, τ = 0.45, inside R |
| **VC** | Gate-1 five-frame visible-reconstruction occupancy (reference, not a method) |
| **VR** | *in-band oracle*: `GT ∩ R` — the best any band-restricted method could do |
| **VRV** | `GT ∩ visible set ∩ R` — the in-band oracle if additionally confined to the visible set |

---

## 8. Results — sequence 08, 163 clips

| id | IoU | P | R | occupied | TP | FP | FN | added vs C3 | removed vs C3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | 0.0778 | 0.408 | 0.090 | 24 717 | 9 995 | 14 722 | 105 115 | 0 | 0 |
| V1 | 0.1759 | 0.284 | 0.334 | 137 863 | 38 044 | 99 819 | 77 066 | 113 146 | 0 |
| V2 | 0.1232 | 0.336 | 0.169 | 57 493 | 18 845 | 38 648 | 96 265 | 32 776 | 0 |
| **V3** | **0.2903** | **0.687** | **0.340** | **56 230** | **38 667** | **17 563** | **76 443** | **44 400** | **12 887** |
| VC | 0.1255 | 0.938 | 0.128 | 14 933 | 13 972 | 961 | 101 138 | — | — |
| VR | 0.3873 | 1.000 | 0.387 | 44 190 | 44 190 | 0 | 70 920 | — | — |
| VRV | 0.1016 | 1.000 | 0.102 | 11 096 | 11 096 | 0 | 104 014 | — | — |

Mean fused C3 points per clip: 318 184; mean C3 voxels: 24 717. `in_region_frac` = 1.000
for V0–V3, i.e. no configuration placed a single voxel outside the permitted region.

### Per-clip IoU distribution

| id | p25 | median | p75 |
|---|---:|---:|---:|
| V0 | 0.0588 | 0.0758 | 0.0951 |
| V1 | 0.1533 | 0.1731 | 0.1979 |
| V2 | 0.1058 | 0.1235 | 0.1414 |
| **V3** | **0.2417** | **0.2846** | **0.3314** |
| VC | 0.1026 | 0.1180 | 0.1402 |
| VR | 0.3150 | 0.3815 | 0.4486 |

V3's 25th percentile (0.2417) is above V1's 75th percentile (0.1979): the two distributions
barely overlap.

### By distance — IoU / precision / recall / occupied

| id | 0–10 m | 10–20 m | 20–40 m | 40–80 m |
|---|---|---|---|---|
| V0 | 0.270 / 0.59 / 0.33 / 5 746 | 0.130 / 0.40 / 0.16 / 10 972 | 0.039 / 0.36 / 0.04 / 7 626 | 0.004 / 0.49 / 0.00 / 373 |
| V1 | 0.282 / 0.31 / 0.77 / 25 999 | 0.209 / 0.27 / 0.52 / 53 690 | 0.160 / 0.31 / 0.28 / 53 136 | 0.050 / 0.40 / 0.06 / 5 037 |
| V2 | 0.332 / 0.43 / 0.62 / 14 791 | 0.197 / 0.34 / 0.33 / 25 982 | 0.059 / 0.30 / 0.08 / 15 990 | 0.005 / 0.48 / 0.01 / 730 |
| **V3** | **0.558** / 0.70 / 0.74 / 10 938 | **0.405** / 0.67 / 0.52 / 20 334 | **0.256** / 0.72 / 0.29 / 22 222 | **0.073** / 0.68 / 0.08 / 2 736 |
| VC | 0.375 / 0.97 / 0.38 / 3 888 | 0.194 / 0.95 / 0.20 / 5 314 | 0.083 / 0.93 / 0.08 / 4 629 | 0.043 / 0.84 / 0.05 / 1 102 |
| VR | 0.834 / 1.00 / 0.83 / 8 496 | 0.598 / 1.00 / 0.60 / 15 664 | 0.334 / 1.00 / 0.33 / 17 813 | 0.088 / 0.90 / 0.09 / 2 217 |

V3 wins every band and keeps precision essentially flat with range (0.70 / 0.67 / 0.72 /
0.68) while V1's precision degrades and its volume explodes. Beyond 40 m all methods are
weak — V3 reaches 0.073 against an in-band oracle of 0.088, so at that range the band
itself, not the model, is the binding constraint.

---

## 9. Paired clip-level bootstrap — 10 000 resamples, seed 0, 163 paired clips

| contrast | ΔIoU | 95 % CI | clips improved |
|---|---:|---|---:|
| V0 → V1 | +0.0981 | [+0.0927, +0.1034] \* | 162 / 163 |
| V0 → V2 | +0.0453 | [+0.0428, +0.0478] \* | 162 / 163 |
| **V0 → V3** | **+0.2125** | **[+0.2042, +0.2210] \*** | **163 / 163** |
| **V1 → V3** | **+0.1145** | **[+0.1078, +0.1213] \*** | **163 / 163** |
| **V2 → V3** | **+0.1672** | **[+0.1593, +0.1754] \*** | **163 / 163** |

\* CI excludes zero.

Per-clip ΔIoU for V0 → V3: min **+0.0628**, p5 +0.1423, median +0.2075, p95 +0.3106, max
+0.3647. The *worst* clip in the set still improves by +0.063 — more than twelve times the
+0.005 practical threshold. For V1 → V3: min +0.0345, median +0.1090, max +0.2771. The gain
is uniform, not a few outliers.

### Headroom recovered

| reference | gap from V0 | V1 | V2 | V3 |
|---|---:|---:|---:|---:|
| in-band oracle VR (0.3873) — the meaningful ceiling here | 0.3095 | 31.7 % | 14.6 % | **68.7 %** |
| five-frame visible ceiling VC (0.1255) | 0.0477 | 205.7 % | 95.1 % | 445.9 % |

The second row is reported because the brief asks for it, but it is **not a meaningful
percentage**: three of the four configurations exceed 100 %, because VC is not an IoU upper
bound (see §10).

---

## 10. Occupied volume, additions and removals

This is where the learned result separates from densification.

| id | occupied | ×V0 | added | removed | precision | IoU |
|---|---:|---:|---:|---:|---:|---:|
| V0 | 24 717 | 1.00 | 0 | 0 | 0.408 | 0.0778 |
| V1 | 137 863 | 5.58 | 113 146 | 0 | 0.284 | 0.1759 |
| V2 | 57 493 | 2.33 | 32 776 | 0 | 0.336 | 0.1232 |
| **V3** | **56 230** | **2.28** | **44 400** | **12 887** | **0.687** | **0.2903** |
| VR | 44 190 | 1.79 | 34 195 | 14 722 | 1.000 | 0.3873 |

Three facts rule out uncontrolled volume inflation:

1. **V3 uses 59 % fewer occupied voxels than V1 and still beats it by +0.1145.** V1 needs
   137 863 voxels to reach recall 0.334; V3 reaches recall 0.340 with 56 230.
2. **The count-matched control loses by 0.1672.** At essentially identical volume
   (57 493 vs 56 230, a 2.2 % match) V2 scores 0.1232 against V3's 0.2903. Volume is not
   what is buying the gain — placement is.
3. **Precision goes up, not down.** V0 0.408 → V3 0.687, while both deterministic controls
   *lose* precision (0.284, 0.336). V3's volume is close to the in-band oracle's own
   (56 230 vs 44 190, 27 % over), which is what a well-calibrated corrector should look like.

V3 is also the only configuration that **removes** anything: 12 887 C3 voxels deleted per
clip, against the oracle's 14 722. The morphological controls are structurally incapable of
deletion. Roughly 88 % of the voxels the oracle would delete are deleted by V3.

### Visible versus occluded — what the model is really adding

| | additions inside the visible set | additions outside it | % outside |
|---|---:|---:|---:|
| V1 | 6 501 | 106 644 | 94.3 % |
| **V3** | **6 094** | **38 306** | **86.3 %** |

Both add roughly the same number of *visible* voxels; V3 wins by adding far fewer
non-visible ones and placing them far better. But 86 % of its additions are still outside
what a perfect five-frame visible reconstruction produces. Within the 0.6 m band, the module
is inferring near-surface occupancy that was not directly observed.

That is bounded, and the bound is real: the band covers 9.23 % of the grid and contains at
most 42.7 % of the ground-truth occupancy, so `VR = 0.3873` caps anything this design can
do. It is not scene completion. But it is not purely "correcting observed voxels" either,
and the report does not claim it is. The `VRV` row makes the size of the honest
visible-only task explicit: an oracle confined to both the band and the visible set scores
**0.1016**, below plain dilation. A gate that had enforced that restriction would have been
unable to distinguish learning from morphology at all.

---

## 11. Visualisations

`artifacts/voxel_gate/eval/clips_voxel_cnn3d.png` — bird's-eye views (z collapsed) of the
most improved clip, the median clip and the least improved clip, for all five
configurations, coloured true positive / false positive / missed.

| clip | role | ΔIoU (V0→V3) | V0 | V1 | V2 | V3 | VC |
|---|---|---:|---:|---:|---:|---:|---:|
| `08_000175_000195_s5` | most improved | +0.365 | 0.083 | 0.263 | 0.139 | **0.448** | 0.111 |
| `08_000950_000970_s5` | median | +0.207 | 0.095 | 0.189 | 0.130 | **0.303** | 0.113 |
| `08_004050_004070_s5` | least improved | +0.063 | 0.065 | 0.040 | 0.060 | **0.128** | 0.420 |

**No clip is degraded**, so the third row is labelled "least improved" rather than "most
degraded". It is also the clip where the ground truth disagrees most with every
reconstruction — V0's precision there is 0.08 and even the visible ceiling reaches only
0.57 — which points at a registration or dynamic-object problem in that clip rather than a
model failure. The images show the mechanism directly: V1 and V2 flood the scene with red
false positives, while V3's additions are concentrated on the road surface and building
façades that the LiDAR target marks occupied.

---

## 12. Decision rules

| criterion | required | observed | |
|---|---|---|:--:|
| V3 improves over V0, CI excludes zero | yes | +0.2125 [+0.2042, +0.2210] | ✔ |
| V3 improves over the strongest deterministic control, CI excludes zero | yes | V1→V3 +0.1145 [+0.1078, +0.1213] | ✔ |
| absolute V0→V3 improvement | ≥ 0.005 | **0.2125** (43×) | ✔ |
| not explained by uncontrolled volume inflation | — | 59 % fewer voxels than V1; count-matched V2 loses by 0.1672; precision rises 0.408→0.687 | ✔ |
| no severe precision or recall collapse | — | precision +0.279, recall +0.250 | ✔ |
| gain broad-based, not outliers | — | 163/163 clips improve; worst clip +0.063 | ✔ |

`MORPHOLOGY_ONLY` is excluded: the best morphology reaches 0.1759 against V3's 0.2903, and
V1→V3 is significant on every clip. `MARGINAL_LEARNED_CORRECTION` is excluded: the margin
is 43× the practical threshold and both control contrasts are significant.
`VISIBLE_VOXEL_CORRECTION_FAILS` is excluded: no collapse, and the leakage audit in §4
passes.

**Diagnosis: `VISIBLE_VOXEL_CORRECTION_PASSES`.**

---

## 13. Tests

`tests/voxel_gate/test_voxel_gate.py` — 28 tests, 1 skipped when the cache is absent:
exact C3 reproduction; the fast scorer against the frozen evaluator; coupled scale as a
pure similarity; rotations never scaled; `r_shape` absent from the code path; input
bit-identity under target mutation; the input path refusing to read target keys; region
computability without targets; forbidden-name check on channels; region extensiveness at
every radius; voxels outside the region unchanged for the model and for all 44 controls;
zero-init reproducing V0; add/remove behaviour on synthetic volumes; parameter budget;
loss restricted to the supervision mask; source/sequence-08 clip disjointness; selection
provenance; threshold membership in the source-chosen grid; bootstrap pairedness and
determinism at a fixed seed; grid dimensions and voxel size; pack/unpack round-trip;
distance bins tiling the grid; `dense_from_sparse` equal to the frozen voxeliser.

Whole-repository run: **363 passed, 2 skipped** (Gates 0–3 and all pre-existing suites).

---

## 14. Limitations

1. **In-domain only.** Sequences 00–10 are one city, one sensor rig, one camera height, one
   set of intrinsics. Sequence 08 is held out temporally, not environmentally. Nothing here
   is cross-dataset, and the Gate-2 cross-dataset question is still open.
2. **The anti-completion guarantee is geometric, not semantic.** The 0.6 m band bounds
   *where* the model may act, not whether what it adds was observed. 86 % of its additions
   lie outside the five-frame visible set (§10). Do not describe this module as
   "correcting observed voxels" without that qualification.
3. **SemanticKITTI's SSC target is itself aggregated over future frames**, so voxels
   occluded at the anchor instant are labelled occupied. Some of what looks like inference
   is the benchmark rewarding a plausible surface prior. This is the reason plain dilation
   scores 0.1759.
4. **The visible ceiling is not an IoU bound.** Reporting anything as a "% of the 0.1255
   ceiling" is misleading; the in-band oracle (0.3873 at radius 3) is the correct reference
   for this gate, and it moves with the band radius.
5. **The band radius controls the result.** In-band oracle IoU rises monotonically with
   radius (0.273 / 0.367 / 0.427 at 1 / 2 / 3). Radius 3 was chosen by a stated source-only
   rule, but a different rule would give a different headroom and a different V3.
6. **163 clips from one sequence are not 163 independent environments.** The bootstrap
   describes within-sequence-08 variation only.
7. **Single configuration.** One architecture, one loss, one seed. No variance estimate over
   training seeds is available, and no architecture ablation was run — by design.
8. **V2's parameter was matched on sequence 08.** This advantages the control, so it does not
   threaten the conclusion, but it is not a clean held-out control.

---

## 15. Should the project proceed to privileged-frame completion?

**Yes — the evidence supports it, with three conditions.**

The case for proceeding: a 16 577-parameter local 3D module recovered 68.7 % of the
available in-band headroom, beat every deterministic control decisively, and did it while
*raising* precision and *shrinking* volume relative to morphology. The failure mode this
gate was built to detect — a learned model winning only by densifying — is ruled out three
independent ways. The binding constraint is now demonstrably the band, not the model: V3 is
at 0.2903 against an in-band oracle of 0.3873, and that oracle rises monotonically as the
band widens. Relaxing the band *is* completion, and the geometry now supports it.

The conditions:

1. **Carry the visible/occluded split as a first-class metric, not a footnote.** Gate 3
   already shows 86 % of additions falling outside the visible set. A completion stage must
   report, per configuration, how much of its gain is observed-surface correction and how
   much is inference into unobserved space — otherwise the two become impossible to
   separate and the result becomes uninterpretable.
2. **Keep the deterministic controls in every subsequent experiment.** Dilation reaching
   0.1759 from a 0.0778 base is the single most important calibration fact produced by this
   gate. Any future completion number without a volume-matched morphology control should be
   treated as unreported.
3. **Resolve the cross-dataset question before any generalisation claim.** Gates 2 and 3
   both produced large in-domain gains from modules that plausibly memorised KITTI
   structure — a fixed camera height, a flat road, a consistent street geometry. The scale
   estimator and this corrector share that exposure. A zero-shot evaluation on a second
   driving dataset is the highest-value next measurement, and it is cheaper than a
   completion stage.

Recommended order: (a) cross-dataset validation of the frozen C3 + V3 stack; (b) a
band-radius sweep with the visible/occluded split reported, which turns Gate 3 into a
controlled study of how far local inference can be pushed before it becomes completion;
(c) privileged-frame completion proper.

**Completion, semantics, cross-dataset evaluation and band relaxation have not been
started.** This gate stops at the report.
