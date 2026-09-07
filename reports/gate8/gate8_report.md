# Gate 8 — Causal semantic memory with privileged completion

**Date:** 2026-09-02 · **Commit:** `9a648322e6dc` · **Status:** complete

## 0. Decision summary

**D — fix the training targets and the operating point before anything else.**

The module clearly learns: it improves the identical mapper on all three benchmarks with paired intervals excluding zero, and on the two source benchmarks it comfortably clears the trivial baseline (SemanticKITTI 08 0.240 vs 0.078, Occ3D-nuScenes val 0.413 vs 0.230). But on the held-out benchmark it predicts **2.1x more occupied voxels than exist** and lands below the score of declaring everything occupied. The occupancy gain there is inflation, not structure.

**The most likely cause is in my sampling, not in the architecture.** Training crops are centred on unknown ground-truth-occupied voxels 70 % of the time, and the loss adds a positive weight (up to 8x) and a 2x unknown-voxel weight on top. The model therefore sees a world far denser than the real one and learns a permissive operating point. Concretely, before any scaling:

1. **Rebalance the crop sampler** so the occupied prior in training matches the benchmark prior (7.8 % / 23.0 % / 25.1 % of valid voxels), or reweight the loss to compensate.
2. **Calibrate the decision threshold** on source validation instead of using log-odds > 0 — the residual is added to a frozen map whose own threshold was never meant to absorb a learned logit.
3. **Add the trivial all-occupied baseline to the standing evaluation** so this class of failure cannot pass unnoticed again. (Done: it is now in `gate8_results.json` and this report.)
4. Only then re-run the held-out fold, and only then consider scaling or further folds.

| question | answer |
|---|---|
| What was implemented | incremental world-frame mapper (`gate8/mapper.py`), privileged-target builder, 0.99 M-parameter residual completion U-Net, trainer, evaluator, runtime audit, tests |
| Is the mapper genuinely incremental and causal? | **Yes.** One merge per frame into a persistent sorted table; no replay; dense grids are lookups. With completion off it reproduces the Gate-7B all-past baseline (SemanticKITTI binary IoU 0.0964 vs 0.0976; the residual is world→grid nearest-voxel resampling). |
| Did the completion module learn? | **Yes**: source-validation crop IoU 0.2596 → 0.3096 against the frozen map's 0.1175 on the same crops; teacher KL 1.318 → 1.165. |
| Did it transfer to KITTI-360 (held out, untuned)? | **No — the gain does not survive a trivial-baseline check.** Against the identical mapper the completion raises held-out binary IoU by +0.1738 [+0.1672, +0.1805] and SSC mIoU by +0.0109 [+0.0100, +0.0117], both excluding zero — but its absolute binary IoU (0.2098) is **below** the score of declaring every valid voxel occupied (0.2509). It over-predicts occupancy ~2.1x, and on this benchmark that is what the IoU gain is made of. |
| Did coverage improve without destroying precision or naming? | recall 0.0396 → 0.5455, precision 0.2869 → 0.2542, semantic accuracy on TP 0.4023 → 0.3846, coverage miss 0.9604 → 0.4545. **The coverage gain was bought by inflation, not earned.** Recall rises sixteen-fold, but the prediction covers 2.1x the true occupied volume and the resulting IoU sits below the trivial all-occupied baseline (0.2509). Precision and semantic naming both fall. On the two source benchmarks the same model does clear the baseline, so this is an operating-point failure on the held-out domain, not an absence of learning. |

---

## 1. What was built

| property | how it is guaranteed | test |
|---|---|---|
| one frame at a time | `IncrementalMapper.step` merges one frame into a sorted, double-buffered voxel table; no history is replayed | `test_previous_frames_are_not_reintegrated` |
| causal | output at `t` is a deterministic function of frames ≤ `t`; stepping `t+1` cannot mutate a query taken at `t` | `test_future_frame_cannot_influence_output_at_t` |
| scale separate from map | `ScaleState` object; the table has no scale attribute; forcing a gauge touches no voxel | `test_scale_state_is_separate_from_map_state` |
| identical scale on depth and translation | `_integrate`: `d_m = s·depth`, `T[:3,3] *= s`, rotation untouched | `test_identical_scale_on_depth_and_translation` |
| anchor frames integrated once | five buffered frames, scale fixed, then each integrated exactly once | `test_anchor_frames_integrated_exactly_once` |
| dense grid is an export | `query` is a key lookup per cell; it calls no integration code | `test_dense_export_is_a_lookup_not_a_rebuild` |
| free space stops before the surface; behind stays unknown | Gate-7B ray rule | `test_free_space_stops_before_the_surface_and_behind_stays_unknown` |
| cached Trident maps match their frame | file stores `image_path`; stamp carries dataset/vocab/lattice | `test_trident_cache_matches_its_frame` |
| reproduces the frozen streaming baseline with completion off | SemanticKITTI 08: binary IoU 0.0964 vs Gate-7B S1 0.0976; recall 0.1332 vs 0.1348; mIoU 0.0261 vs 0.0268 | `test_incremental_mapper_reproduces_the_frozen_streaming_baseline` |

**Frozen and unchanged:** LingBot-Map (direct streaming mode, native), MoGe-2 with the
calibrated FOV (scale only; depth rescue disabled), Trident-H (cached; never re-run for
the benchmarks), the union vocabulary, the official evaluation protocol and every prior
baseline result.

**Semantic representation — stated plainly.** Trident-H is cached as per-benchmark,
vocabulary-conditioned probability maps, not dense language-aligned features. The first
version therefore fuses and predicts a **declared 25-class union vocabulary**
(`gate8/vocab.py`), fixed before any result and mapped one-to-one into each benchmark's
classes at evaluation only. **This version is not query-time open-vocabulary.** The map
and the completion head must be moved to fixed-dimensional language-aligned features
before an open-vocabulary claim can be made.

---

## 2. The incremental mapper

| property | how it is guaranteed | test |
|---|---|---|
| one frame at a time | `IncrementalMapper.step` merges one frame into a sorted, double-buffered voxel table; no history is replayed | `test_previous_frames_are_not_reintegrated` |
| causal | output at `t` is a deterministic function of frames ≤ `t`; stepping `t+1` cannot mutate a query taken at `t` | `test_future_frame_cannot_influence_output_at_t` |
| scale separate from map | `ScaleState` object; the table has no scale attribute; forcing a gauge touches no voxel | `test_scale_state_is_separate_from_map_state` |
| identical scale on depth and translation | `_integrate`: `d_m = s·depth`, `T[:3,3] *= s`, rotation untouched | `test_identical_scale_on_depth_and_translation` |
| anchor frames integrated once | five buffered frames, scale fixed, then each integrated exactly once | `test_anchor_frames_integrated_exactly_once` |
| dense grid is an export | `query` is a key lookup per cell; it calls no integration code | `test_dense_export_is_a_lookup_not_a_rebuild` |
| free space stops before the surface; behind stays unknown | Gate-7B ray rule | `test_free_space_stops_before_the_surface_and_behind_stays_unknown` |
| cached Trident maps match their frame | file stores `image_path`; stamp carries dataset/vocab/lattice | `test_trident_cache_matches_its_frame` |
| reproduces the frozen streaming baseline with completion off | SemanticKITTI 08: binary IoU 0.0964 vs Gate-7B S1 0.0976; recall 0.1332 vs 0.1348; mIoU 0.0261 vs 0.0268 | `test_incremental_mapper_reproduces_the_frozen_streaming_baseline` |

**Step cost grows with the table.** The merge scatters every attribute into the spare buffer, so a step is O(table); on SemanticKITTI the table reaches ~12 M rows after 300 frames (86 ms/step) and the KITTI-360 stream ends far larger (§7). Most rows are free-space carvings. A production system would prune far-behind free space or shard the table spatially; neither is done here, and the reported latencies are the unpruned ones.

---

## 3. Privileged training targets

| source | samples | future frames | bytes | note |
|---|---:|---:|---:|---|
| sk_train | 1,656 | 20 | 0.4 GiB | training |
| occ3d_train | 1,570 | 20 | 4.4 GiB | training |
| semantickitti | 162 | 20 | 0.1 GiB | source validation (selection only) |
| occ3d | 300 | 20 | 0.4 GiB | source validation (selection only) |
| total | | | 5.3 GiB | on /media/SSD1 |

**Sample audit (first three training samples).** Which frames built the input and which built the target:

| segment | t | input frames | target frames | rows | future rows |
|---|---:|---|---|---:|---:|
| 00 | 563 | 0–563 | 564–583 | 704,753 | 180,853 |

Every sample file stores `input_frames` and `target_frames`; `tests/gate8` asserts `max(input) == t` and `min(target) == t+1` on the cached files themselves.

---

## 4. The completion module

| item | value |
|---|---|
| architecture | dense 3-level 3D U-Net, widths 24/48/96, GroupNorm, GELU |
| parameters | 986,114 |
| input channels | 32: log-odds, free evidence, observed, unknown, n_obs, age, semantic weight, 25-way union evidence |
| outputs | occupancy residual logit; 25-way semantic logits |
| residual rule | voxels with |log-odds| ≥ 2.0 are never changed; semantics with teacher evidence never overwritten (`gate8/net.py:apply_residual`, tested) |
| sparse conv | `spconv` is installed but not used: completion must predict in unknown space, which is dense; the boxes are small enough for a dense net |

---

## 5. Training

| item | value |
|---|---|
| training samples | 3,226 (SemanticKITTI seqs 00/05/07 + 100 Occ3D train scenes) |
| validation samples (selection) | 120 (SemanticKITTI 08 + Occ3D val; KITTI-360 never) |
| steps / batch / crop | 6,000 / 4 / [128, 128, 32] |
| parameters | 986,114 |
| losses | focal BCE (γ=2.0, pos-weight ≤ 8.0) + soft Dice + 0.5 × teacher KL |
| selection | lowest validation (focal+dice+teacher-KL) loss; no semantic ground truth, no KITTI-360 |
| best step | 4500 (val loss 1.4669) |
| wall time / peak GPU | 25.5 min / 6.22 GiB |
| val crop IoU, first → best | 0.2596 → 0.3096 (frozen map on the same crops 0.1090) |
| val teacher KL, first → last | 1.3179 → 1.1653 |
| small-subset run (200 samples, 300 steps) | val IoU 0.2977 vs frozen 0.1090 |

![training](../../artifacts/gate8/fig_training.png)

Validation crop IoU of the completed map against the frozen map on the same crops: 0.2596 vs 0.1090 at step 500, 0.3002 vs 0.1018 at step 6000. Crops are centred on unknown ground-truth-occupied voxels 70 % of the time, so these numbers over-represent frontier regions relative to the full-grid results in §6.

---

## 6. Results

| benchmark | variant | method | binary IoU | precision | recall | SSC mIoU | sem. acc on TP | bal. recall | coverage miss | naming error |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SemanticKITTI 08 (source val) | raw | frozen five-frame G51-B | 0.0670 | 0.3492 | 0.0766 | 0.0281 | 0.6326 | 0.4388 | 0.9234 | 0.0281 |
| SemanticKITTI 08 (source val) | raw | incremental mapper | 0.0964 | 0.2585 | 0.1332 | 0.0261 | 0.5244 | 0.3072 | 0.8668 | 0.0634 |
| SemanticKITTI 08 (source val) | raw | incremental mapper (union vocab) | 0.0964 | 0.2585 | 0.1332 | 0.0261 | 0.5244 | 0.3072 | 0.8668 | 0.0634 |
| SemanticKITTI 08 (source val) | raw | mapper + completion | 0.2400 | 0.2613 | 0.7468 | 0.0315 | 0.4351 | 0.1575 | 0.2532 | 0.4218 |
| SemanticKITTI 08 (source val) | dilate 0.4 m | frozen five-frame G51-B | 0.1600 | 0.2537 | 0.3022 | 0.0511 | 0.5592 | 0.3551 | 0.6978 | 0.1332 |
| SemanticKITTI 08 (source val) | dilate 0.4 m | incremental mapper | 0.1435 | 0.1931 | 0.3584 | 0.0320 | 0.4794 | 0.2595 | 0.6416 | 0.1866 |
| SemanticKITTI 08 (source val) | dilate 0.4 m | incremental mapper (union vocab) | 0.1435 | 0.1931 | 0.3584 | 0.0320 | 0.4793 | 0.2595 | 0.6416 | 0.1866 |
| SemanticKITTI 08 (source val) | dilate 0.4 m | mapper + completion | 0.1931 | 0.1982 | 0.8824 | 0.0279 | 0.4534 | 0.1694 | 0.1176 | 0.4823 |
| SemanticKITTI 08 (source val) | — | _trivial: every valid voxel occupied_ | _0.0782_ | _0.0782_ | _1.0000_ | — | — | — | _0.0000_ | — |
| Occ3D-nuScenes val (source val) | raw | frozen five-frame G51-B | 0.0793 | 0.6289 | 0.0832 | 0.0245 | 0.4483 | 0.4168 | 0.9168 | 0.0459 |
| Occ3D-nuScenes val (source val) | raw | incremental mapper | 0.1170 | 0.6052 | 0.1266 | 0.0354 | 0.4633 | 0.4372 | 0.8734 | 0.0680 |
| Occ3D-nuScenes val (source val) | raw | incremental mapper (union vocab) | 0.1170 | 0.6052 | 0.1266 | 0.0354 | 0.4634 | 0.4372 | 0.8734 | 0.0680 |
| Occ3D-nuScenes val (source val) | raw | mapper + completion | 0.4126 | 0.4874 | 0.7289 | 0.0517 | 0.3158 | 0.1947 | 0.2711 | 0.4987 |
| Occ3D-nuScenes val (source val) | dilate 0.4 m | frozen five-frame G51-B | 0.2094 | 0.5641 | 0.2499 | 0.0480 | 0.4431 | 0.3795 | 0.7501 | 0.1392 |
| Occ3D-nuScenes val (source val) | dilate 0.4 m | incremental mapper | 0.2309 | 0.5316 | 0.2899 | 0.0541 | 0.4643 | 0.4009 | 0.7101 | 0.1553 |
| Occ3D-nuScenes val (source val) | dilate 0.4 m | incremental mapper (union vocab) | 0.2309 | 0.5316 | 0.2899 | 0.0541 | 0.4643 | 0.4009 | 0.7101 | 0.1553 |
| Occ3D-nuScenes val (source val) | dilate 0.4 m | mapper + completion | 0.3564 | 0.3804 | 0.8495 | 0.0522 | 0.3297 | 0.1953 | 0.1505 | 0.5694 |
| Occ3D-nuScenes val (source val) | — | _trivial: every valid voxel occupied_ | _0.2295_ | _0.2295_ | _1.0000_ | — | — | — | _0.0000_ | — |
| **KITTI-360 (held out)** | raw | frozen five-frame G51-B | 0.0363 | 0.3747 | 0.0386 | 0.0088 | 0.6245 | 0.2789 | 0.9614 | 0.0145 |
| **KITTI-360 (held out)** | raw | incremental mapper | 0.0360 | 0.2869 | 0.0396 | 0.0072 | 0.4023 | 0.2008 | 0.9604 | 0.0237 |
| **KITTI-360 (held out)** | raw | incremental mapper (union vocab) | 0.0360 | 0.2869 | 0.0396 | 0.0072 | 0.4023 | 0.2008 | 0.9604 | 0.0237 |
| **KITTI-360 (held out)** | raw | **mapper + completion** ⚠ | **0.2098** | 0.2542 | 0.5455 | **0.0181** | 0.3846 | 0.1153 | 0.4545 | 0.3357 |
| **KITTI-360 (held out)** | dilate 0.4 m | frozen five-frame G51-B | 0.1321 | 0.3587 | 0.1729 | 0.0266 | 0.5959 | 0.2839 | 0.8271 | 0.0699 |
| **KITTI-360 (held out)** | dilate 0.4 m | incremental mapper | 0.1094 | 0.2854 | 0.1508 | 0.0189 | 0.4220 | 0.2223 | 0.8492 | 0.0871 |
| **KITTI-360 (held out)** | dilate 0.4 m | incremental mapper (union vocab) | 0.1094 | 0.2854 | 0.1508 | 0.0189 | 0.4219 | 0.2223 | 0.8492 | 0.0871 |
| **KITTI-360 (held out)** | dilate 0.4 m | **mapper + completion** ⚠ | **0.2207** | 0.2436 | 0.7011 | **0.0210** | 0.4186 | 0.1252 | 0.2989 | 0.4076 |
| **KITTI-360 (held out)** | — | _trivial: every valid voxel occupied_ | _0.2509_ | _0.2509_ | _1.0000_ | — | — | — | _0.0000_ | — |

![raw results](../../artifacts/gate8/fig_results_raw.png)

![dilated results](../../artifacts/gate8/fig_results_dil.png)

| benchmark | variant | method | occupied TP | FP | FN |
|---|---|---|---:|---:|---:|
| SemanticKITTI 08 (source val) | raw | frozen five-frame G51-B | 1,436,514 | 2,677,557 | 17,326,416 |
| SemanticKITTI 08 (source val) | raw | incremental mapper | 2,499,866 | 7,171,669 | 16,263,064 |
| SemanticKITTI 08 (source val) | raw | mapper + completion | 14,011,988 | 39,612,248 | 4,750,942 |
| SemanticKITTI 08 (source val) | dil | frozen five-frame G51-B | 5,669,753 | 16,680,858 | 13,093,177 |
| SemanticKITTI 08 (source val) | dil | incremental mapper | 6,724,668 | 28,101,786 | 12,038,262 |
| SemanticKITTI 08 (source val) | dil | mapper + completion | 16,556,962 | 66,964,138 | 2,205,968 |
| Occ3D-nuScenes val (source val) | raw | frozen five-frame G51-B | 995,434 | 587,357 | 10,962,156 |
| Occ3D-nuScenes val (source val) | raw | incremental mapper | 1,514,215 | 987,773 | 10,443,375 |
| Occ3D-nuScenes val (source val) | raw | mapper + completion | 8,715,814 | 9,166,618 | 3,241,776 |
| Occ3D-nuScenes val (source val) | dil | frozen five-frame G51-B | 2,987,894 | 2,309,269 | 8,969,696 |
| Occ3D-nuScenes val (source val) | dil | incremental mapper | 3,466,036 | 3,054,500 | 8,491,554 |
| Occ3D-nuScenes val (source val) | dil | mapper + completion | 10,157,415 | 16,541,609 | 1,800,175 |
| **KITTI-360 (held out)** | raw | frozen five-frame G51-B | 5,319,813 | 8,876,088 | 132,521,658 |
| **KITTI-360 (held out)** | raw | incremental mapper | 5,455,113 | 13,559,733 | 132,386,358 |
| **KITTI-360 (held out)** | raw | mapper + completion | 75,196,615 | 220,598,339 | 62,644,856 |
| **KITTI-360 (held out)** | dil | frozen five-frame G51-B | 23,833,788 | 42,612,530 | 114,007,683 |
| **KITTI-360 (held out)** | dil | incremental mapper | 20,780,799 | 52,024,560 | 117,060,672 |
| **KITTI-360 (held out)** | dil | mapper + completion | 96,639,348 | 300,052,420 | 41,202,123 |

| benchmark | comparison | unit (n) | Δ binary IoU | 95 % CI | Δ SSC mIoU | 95 % CI |
|---|---|---|---:|---|---:|---|
| SemanticKITTI 08 (source val) | mapper_raw vs frozen_5frame_raw | contiguous block of 20 clips (8) | +0.0294 | [+0.0206, +0.0389]* | -0.0020 | [-0.0057, +0.0027] |
| SemanticKITTI 08 (source val) | mapper_dil vs frozen_5frame_dil | contiguous block of 20 clips (8) | -0.0165 | [-0.0281, -0.0040]* | -0.0191 | [-0.0235, -0.0114]* |
| SemanticKITTI 08 (source val) | completion_raw vs mapper_raw | contiguous block of 20 clips (8) | +0.1436 | [+0.1254, +0.1679]* | +0.0054 | [+0.0023, +0.0086]* |
| SemanticKITTI 08 (source val) | completion_dil vs mapper_dil | contiguous block of 20 clips (8) | +0.0496 | [+0.0367, +0.0652]* | -0.0041 | [-0.0066, -0.0017]* |
| SemanticKITTI 08 (source val) | completion_raw vs frozen_5frame_raw | contiguous block of 20 clips (8) | +0.1730 | [+0.1535, +0.1975]* | +0.0034 | [-0.0006, +0.0080] |
| SemanticKITTI 08 (source val) | completion_dil vs frozen_5frame_dil | contiguous block of 20 clips (8) | +0.0332 | [+0.0153, +0.0580]* | -0.0232 | [-0.0257, -0.0171]* |
| Occ3D-nuScenes val (source val) | mapper_raw vs frozen_5frame_raw | scene (150) | +0.0376 | [+0.0291, +0.0460]* | +0.0109 | [+0.0075, +0.0143]* |
| Occ3D-nuScenes val (source val) | mapper_dil vs frozen_5frame_dil | scene (150) | +0.0215 | [+0.0105, +0.0333]* | +0.0061 | [+0.0027, +0.0095]* |
| Occ3D-nuScenes val (source val) | completion_raw vs mapper_raw | scene (150) | +0.2956 | [+0.2776, +0.3139]* | +0.0163 | [+0.0144, +0.0183]* |
| Occ3D-nuScenes val (source val) | completion_dil vs mapper_dil | scene (150) | +0.1255 | [+0.1080, +0.1435]* | -0.0020 | [-0.0043, +0.0003] |
| Occ3D-nuScenes val (source val) | completion_raw vs frozen_5frame_raw | scene (150) | +0.3332 | [+0.3161, +0.3505]* | +0.0272 | [+0.0234, +0.0309]* |
| Occ3D-nuScenes val (source val) | completion_dil vs frozen_5frame_dil | scene (150) | +0.1470 | [+0.1296, +0.1649]* | +0.0042 | [+0.0007, +0.0075]* |
| **KITTI-360 (held out)** | mapper_raw vs frozen_5frame_raw | contiguous block of 20 clips (87) | -0.0002 | [-0.0033, +0.0029] | -0.0016 | [-0.0022, -0.0010]* |
| **KITTI-360 (held out)** | mapper_dil vs frozen_5frame_dil | contiguous block of 20 clips (87) | -0.0226 | [-0.0286, -0.0169]* | -0.0077 | [-0.0087, -0.0067]* |
| **KITTI-360 (held out)** | completion_raw vs mapper_raw | contiguous block of 20 clips (87) | +0.1738 | [+0.1672, +0.1805]* | +0.0109 | [+0.0100, +0.0117]* |
| **KITTI-360 (held out)** | completion_dil vs mapper_dil | contiguous block of 20 clips (87) | +0.1112 | [+0.1023, +0.1206]* | +0.0021 | [+0.0012, +0.0032]* |
| **KITTI-360 (held out)** | completion_raw vs frozen_5frame_raw | contiguous block of 20 clips (87) | +0.1735 | [+0.1683, +0.1792]* | +0.0093 | [+0.0083, +0.0103]* |
| **KITTI-360 (held out)** | completion_dil vs frozen_5frame_dil | contiguous block of 20 clips (87) | +0.0886 | [+0.0813, +0.0965]* | -0.0056 | [-0.0068, -0.0042]* |

_Paired bootstrap, 10,000 resamples, seed 0, over Gate 6's units (nuScenes scenes; contiguous 20-clip blocks from one drive on the KITTI family — not independent scenes). * = interval excludes zero._

**SemanticKITTI 08 (source val)**: completion moves raw binary IoU 0.0964 → 0.2400 (precision 0.2585 → 0.2613, recall 0.1332 → 0.7468), SSC mIoU 0.0261 → 0.0315, semantic accuracy on TP 0.5244 → 0.4351; paired Δ IoU +0.1436 [+0.1254, +0.1679]. The frozen five-frame **dilated** map scores 0.1600 / 0.0511.

**Occ3D-nuScenes val (source val)**: completion moves raw binary IoU 0.1170 → 0.4126 (precision 0.6052 → 0.4874, recall 0.1266 → 0.7289), SSC mIoU 0.0354 → 0.0517, semantic accuracy on TP 0.4634 → 0.3158; paired Δ IoU +0.2956 [+0.2776, +0.3139]. The frozen five-frame **dilated** map scores 0.2094 / 0.0480.

****KITTI-360 (held out)****: completion moves raw binary IoU 0.0360 → 0.2098 (precision 0.2869 → 0.2542, recall 0.0396 → 0.5455), SSC mIoU 0.0072 → 0.0181, semantic accuracy on TP 0.4023 → 0.3846; paired Δ IoU +0.1738 [+0.1672, +0.1805]. The frozen five-frame **dilated** map scores 0.1321 / 0.0266.

**Vocabulary note.** `mapper_union` is the incremental mapper with semantics fused in the union vocabulary and mapped back at evaluation — the apples-to-apples baseline for the completion, which lives in that vocabulary. `mapper` (native) is the same geometry with the benchmark's own vocabulary; the small semantic gap between the two is the cost of the union mapping, not of the mapper.

### Classwise IoU (KITTI-360, held out)

| class | frozen 5-frame (dil) | mapper (raw) | mapper + completion (raw) | Δ |
|---|---:|---:|---:|---:|
| car | 0.0553 | 0.0160 | 0.0208 | +0.0048 |
| bicycle | 0.0099 | 0.0022 | 0.0023 | +0.0001 |
| motorcycle | 0.0067 | 0.0008 | 0.0009 | +0.0001 |
| truck | 0.0421 | 0.0200 | 0.0086 | -0.0114 |
| other-vehicle | 0.0123 | 0.0048 | 0.0049 | +0.0001 |
| person | 0.0139 | 0.0060 | 0.0058 | -0.0003 |
| road | 0.0494 | 0.0184 | 0.1023 | +0.0839 |
| parking | 0.0098 | 0.0010 | 0.0011 | +0.0001 |
| sidewalk | 0.0262 | 0.0056 | 0.0135 | +0.0078 |
| other-ground | 0.0072 | 0.0037 | 0.0095 | +0.0058 |
| building | 0.0761 | 0.0124 | 0.0202 | +0.0078 |
| fence | 0.0191 | 0.0059 | 0.0098 | +0.0039 |
| vegetation | 0.1096 | 0.0180 | 0.1066 | +0.0886 |
| terrain | 0.0006 | 0.0003 | 0.0003 | +0.0000 |
| pole | 0.0070 | 0.0027 | 0.0026 | -0.0001 |
| traffic-sign | 0.0044 | 0.0011 | 0.0013 | +0.0001 |
| other-structure | 0.0236 | 0.0089 | 0.0139 | +0.0049 |
| other-object | 0.0052 | 0.0022 | 0.0015 | -0.0007 |

---

## 7. Runtime and memory

| component | median (ms) | p95 (ms) | note |
|---|---:|---:|---|
| image_load | 15.1 | 19.4 | PNG decode + frozen preprocessing |
| lingbot | 44.2 | 47.8 | frozen direct-mode forward, one frame |
| moge | 59.0 | 200.0 | first five frames only (gauge) |
| scale | 1.6 | 2.0 | per-frame candidate, first five frames |
| map_step | 3.8 | 4.4 | incremental table merge (occupied band + decimated free carve) |
| sem_load | 3.5 | 4.1 | cached Trident map read |
| query | 3.2 | 3.5 | dense export onto the benchmark grid |
| net | 45.1 | 55.9 | completion U-Net on the full grid |
| **Trident-H online** | 1596.2 | 1629.3 | frozen teacher, Trident environment, same frames |

| quantity | value |
|---|---|
| per-frame, cached teacher | 66.6 ms |
| per-frame, end-to-end with online Trident | 1659.3 ms |
| map rows after 100 frames | 1,144,444 (0.56 GiB, double-buffered) |
| peak GPU memory (mapper + net + frozen models) | 12.27 GiB |
| live vs cached depth, mean abs error | 3.90e-02 (canonical units) |
| evaluator step (KITTI-360, 1,753 exports) | median 8.0 ms, p95 12.2 ms |
| evaluator query (KITTI-360, 1,753 exports) | median 3.6 ms, p95 3.9 ms |
| evaluator net (KITTI-360, 1,753 exports) | median 45.3 ms, p95 45.4 ms |
| evaluator dilate (KITTI-360, 1,753 exports) | median 31.4 ms, p95 35.9 ms |
| map table at the end of the KITTI-360 stream | 9.00 GiB |

**The system is online, not real-time.** LingBot direct mode, the incremental merge and the dense export are each well under a second per frame; Trident-H online costs 1.6 s per frame on the same GPU and dominates the end-to-end budget. The teacher is asynchronous in any deployment; the cached-teacher number is the mapper's own cost, the end-to-end number is what a single-GPU system would actually pay.

---

## 8. Training-integrity audit

| rule | how it is enforced |
|---|---|
| no semantic ground truth in any loss, selection, threshold or tuning | targets read the GT file and return occupancy only (`binary_occupancy`); the semantic target is the frozen teacher's future evidence; checkpoint selection = geometry loss + teacher KL on source validation (`train_completion.json:selection_rule`) |
| KITTI-360 never used to choose anything | not in `train_sources` or `val_sources`; evaluated once, after selection (`test_no_training_code_touches_kitti360`) |
| future observations only in targets | `future_volume` is a separate mapper instance; sample files record both frame ranges; asserted on the cached files |
| frozen baselines untouched | Gate-6/7A/7B artifacts hashed in `stage0_audit.json` and re-verified (`test_gate7b_artifacts_untouched`) |
| no target on the prediction path | `mapper`, `feed`, `net`, `losses`, `vocab`, `sources` contain no target import (AST-checked) |

---

## 9. Recommendation (one action; awaiting approval)

**D — fix the training targets and the operating point before anything else.**

The module clearly learns: it improves the identical mapper on all three benchmarks with paired intervals excluding zero, and on the two source benchmarks it comfortably clears the trivial baseline (SemanticKITTI 08 0.240 vs 0.078, Occ3D-nuScenes val 0.413 vs 0.230). But on the held-out benchmark it predicts **2.1x more occupied voxels than exist** and lands below the score of declaring everything occupied. The occupancy gain there is inflation, not structure.

**The most likely cause is in my sampling, not in the architecture.** Training crops are centred on unknown ground-truth-occupied voxels 70 % of the time, and the loss adds a positive weight (up to 8x) and a 2x unknown-voxel weight on top. The model therefore sees a world far denser than the real one and learns a permissive operating point. Concretely, before any scaling:

1. **Rebalance the crop sampler** so the occupied prior in training matches the benchmark prior (7.8 % / 23.0 % / 25.1 % of valid voxels), or reweight the loss to compensate.
2. **Calibrate the decision threshold** on source validation instead of using log-odds > 0 — the residual is added to a frozen map whose own threshold was never meant to absorb a learned logit.
3. **Add the trivial all-occupied baseline to the standing evaluation** so this class of failure cannot pass unnoticed again. (Done: it is now in `gate8_results.json` and this report.)
4. Only then re-run the held-out fold, and only then consider scaling or further folds.

---

## 10. Files, commands and manifest

| path | role |
|---|---|
| `gate8/mapper.py` | incremental mapper: ScaleState, double-buffered VoxelTable, ray integration, dense export |
| `gate8/feed.py` | per-frame feed from the cached frozen-model outputs |
| `gate8/sources.py` | train/val stream index (genuine train splits) |
| `gate8/vocab.py` | declared 25-class union vocabulary and fixed maps |
| `gate8/targets.py` | privileged targets: binary occupancy + future-teacher semantics + masks |
| `gate8/net.py` | completion U-Net and the residual rule |
| `gate8/losses.py` | focal BCE, soft Dice, teacher KL |
| `tools/gate8/stream_sources.py` | LingBot native streaming + MoGe-B + gauge candidates for the train sources |
| `tools/gate8/cache_trident.py` | Trident-H cache for the train sources (Trident env) |
| `tools/gate8/build_samples.py` | cached causal inputs + privileged targets with audit |
| `tools/gate8/train.py` | training with source-validation selection |
| `tools/gate8/evaluate.py` | mapper / mapper+completion evaluation with Gate-6 metrics; raw and dilated |
| `tools/gate8/runtime_audit.py` | live one-frame-at-a-time timing |
| `tools/gate8/time_trident.py` | online Trident timing |
| `tools/gate8/aggregate.py` | paired bootstrap and gate8_results.json |
| `tools/gate8/report.py` | this report |
| `configs/gate8/completion.yaml` | training configuration |
| `tests/gate8/test_gate8.py` | correctness and leakage tests |
| `artifacts/gate8/` | results, checkpoints, figures, manifest |
| `reports/gate8/gate8_report.md` | this report |
| `gate7a/frustum.py` | unchanged in Gate 8 |

```bash
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python
$PY tools/gate8/stage0.py                                   # provenance hashes
tools/gate8/run_caches.sh                                   # LingBot + MoGe + Trident on the train sources
for S in semantickitti occ3d kitti360; do $PY tools/gate8/evaluate.py --source $S --tag mapper_native; done
$PY tools/gate8/build_samples.py --source sk_train  --shard 0 --shards 2 &  # (x2)
$PY tools/gate8/build_samples.py --source occ3d_train --anchor-stride 2 --shard 0 --shards 2 &   # (x2)
$PY tools/gate8/build_samples.py --source semantickitti; $PY tools/gate8/build_samples.py --source occ3d --anchor-stride 4
tools/gate8/run_train_eval.sh                               # subset run, primary run, evaluation
$PY tools/gate8/runtime_audit.py --source kitti360 --checkpoint artifacts/gate8/checkpoints/completion_best.pt
(cd /home/minh/workspace/third_party/Trident && PYTHONPATH=$PWD:/home/minh/workspace/lingbot-map_fork \
   /home/minh/workspace/third_party/trident_env/bin/python /home/minh/workspace/lingbot-map_fork/tools/gate8/time_trident.py)
$PY tools/gate8/aggregate.py && $PY tools/gate8/figures.py && $PY tools/gate8/report.py --write
$PY -m pytest tests/gate6 tests/gate7a tests/gate7b tests/gate8 -q
```

**Manifest** `artifacts/gate8/gate8_manifest.json`: commit `9a648322e6dc4e34995f0530650c2fa469779676`, config SHA-256 `a4a365847dd4ad74…`, checkpoint `completion_best.pt` SHA-256 `de9fbc0cf5d15bc1…`, 35 artifact files hashed; frozen LingBot `ee665103348e07e6…`, MoGe-2 `39c4d5e9…`, Trident-H at the Gate-6 provenance.

---

## 11. Deviations and limitations

**D1 — union vocabulary, not language-aligned features.** Trident is cached as probability maps; the semantic head is a fixed 25-way union. Stated in §1; the open-vocabulary claim is deferred.

**D2 — dense U-Net rather than sparse conv.** `spconv` is available, but completion must predict in unknown (dense) space and the grids are small; a dense net is the simpler correct choice. Reported as a deviation from the brief's preference.

**D3 — Occ3D training anchors subsampled 1:2** (every second keyframe of 100 train scenes) to bound disk and build time; SemanticKITTI train anchors are every fifth image frame, matching the frozen stream stride.

**D4 — a memory defect in the first voxel table was found and fixed.** The first implementation re-concatenated every attribute per frame; allocator fragmentation exhausted a 95 GB GPU on one 815-frame sequence. The table is now capacity-doubling and double-buffered; the fix is described in the class docstring and the 300-frame measurement (12.4 M rows, 4.5 GiB, 86 ms/step, 6 GiB peak) is in §2.

**D5 — world→grid export resamples.** The persistent map lives in the scaled LingBot world frame; exporting to a benchmark grid is a nearest-voxel lookup, so the mapper-only result differs from Gate 7B's per-timestamp rebuild by up to one voxel of resampling (SemanticKITTI binary IoU 0.0964 vs 0.0976). This is the price of a persistent map and is not tuned away.

**D6 — MoGe rescue disabled by default** (Gate 7B transfer inconsistent); the mapper keeps `moge_rescue` as an optional channel, unused here.

**D7 — a source-validation/train split within the same benchmarks.** SemanticKITTI train sequences and Occ3D train scenes are disjoint from the official val sets used for selection, so the reported source-validation numbers are clean; KITTI-360 was never read before its single evaluation.
