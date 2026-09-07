# Gate 8C-0 — KITTI-360 alignment and learnability audit

**Verdict.** The KITTI-360 completion target is not geometrically consistent with its own LiDAR — a ground-truth sweep lands on a voxel the official label calls *free* 71% of the time, against 1.7% on SemanticKITTI under the identical code path — so Gate 8B's KITTI-360 numbers were measuring a broken target, not a model failure.

**Recommended next action.** Stop using SSCBench-KITTI-360's `_1_1.npy` label as a completion target, and withdraw every KITTI-360 conclusion from Gates 5.2–8B rather than repairing them. Bring the KITTI-360 question back only if the target is rebuilt from raw aggregated LiDAR under our own verified chain, as a separate approved gate. Do not implement that here, and do not touch the completion model: Gate 8B's *SemanticKITTI* failure is real, unexplained by this defect, and is the right next target.

**Is the KITTI-360 pipeline correctly aligned?** **Yes — ours is.** All 9 anchors round-trip every transform pair to 1.4e-12 m (tolerance 0.001 m); the SSCBench index map, grid, floor-binning, rig geometry and anchor timing all check out; Stage 5 passes every causal and cache invariant on 40 samples. **The published target is not.**

**Does fixed-batch memorization pass?** **Yes, decisively.** Editable-region occupancy AP reaches 0.9823 and 0.9972 on the two fixed batches (best-threshold IoU 0.8519 / 0.9577), gradients stay healthy throughout, and there is no information collision (minimum pairwise input distance 147.7).

**Are the KITTI-360 training drives learnable?** **Undetermined, and untestable as the data stands.** Fixed crops are memorized to AP ≈ 0.99, so nothing about the model, the optimizer, the masks or the inputs blocks learning. But generalisation cannot be measured against a target that contradicts its own sensor, so the Stage-7 experiment was not run.

**Which failure is it?** **A data defect**, in the evaluation/training target itself. It is not an optimization failure (gradients healthy, memorization passes), not cross-drive shift (the defect is present identically on training, source-validation and held-out drives), not mixed-source interference (it reproduces with KITTI-360 alone), and not merely insufficient causal information — although the target is *also* unreachable, which is a consequence of the same defect.

**Decisive evidence.**

1. **Our chain reproduces SSCBench's own voxel input exactly.** Voxelizing the raw velodyne sweep through our adapter and comparing with SSCBench's published `.bin` for the same anchor gives recall 0.9990 at **zero shift**, and both ±1 z shifts destroy it (IoU 0.6001 → 0.1669). The velodyne frame also beats every alternative grid-frame hypothesis, and anchor offset 0 beats ±1 and ±2.

2. **SSCBench's own input disagrees with SSCBench's own label, by the same one voxel.** `.bin` versus `_1_1.npy`, with our code nowhere in the path: precision 0.2983 at zero shift, 0.6187 at +1 z. The offset is internal to the published release.

3. **SemanticKITTI, under the identical code path, is clean.** Raw sweep versus its official label: precision 0.9832 at zero shift, falling to 0.7142 at +1 z. So this is not a general property of SSC completion targets, and not a bug in our voxelizer — KITTI-360 is the outlier.

4. **The disagreement is far larger than a shift.** Even at the best continuous offset (+0.26 m) KITTI-360 precision only reaches 0.54. The gain is concentrated in ground classes (road 1,771 → 9,077 voxels) while vertical structure barely moves, so no single rigid correction fixes it.

5. **A ground-truth oracle therefore cannot reach the target.** GT sweeps, GT poses, all available causal past, integrated once: recall 0.0866 / 0.0733 / 0.0688 and precision 0.2983 / 0.2516 / 0.2735 on train / source-validation / held-out. The four-way depth×pose table confirms the ceiling is the target, not the components: GT depth × GT pose peaks at IoU 0.097, and swapping GT pose for LingBot's changes almost nothing.

6. **The model is not the problem.** Fixed-batch memorization reaches editable-region AP 0.9972 on the same data, so the inputs carry enough signal to fit these targets exactly when generalisation is not required.

---

## Outcome: A — pipeline defect

Per the brief's decision rules this is **A**, because the oracle-LiDAR sanity check fails.
The important qualification is *where* the defect sits: every transform, index, mask and
cache we built is verified correct here, and our voxelization reproduces SSCBench's **own**
published voxel input at 99.90 % recall with zero shift. The inconsistency is between two
official SSCBench-KITTI-360 products — its voxel input and its `_1_1.npy` completion label.

### The exact defect

`preprocess/labels/<drive>/<anchor>_1_1.npy` in the SSCBench-KITTI-360 release marks voxels as **observed free** at locations where the same release's own LiDAR measured a return. Measured on 8 anchors of the held-out drive:

* our voxelized sweep vs the label: precision **0.2882** — 71% of measured surface voxels fall on label-*free* (not invalid, not unknown) voxels;
* SSCBench's own `.bin` vs the same label: precision **0.2983**, rising to **0.6187** under a +1 voxel z shift;
* SemanticKITTI, same code: precision **0.9832**, and +1 z makes it worse.

The dominant component is vertical and concentrated on the ground plane, consistent with the label's ground surface sitting roughly one voxel above the measured return; but a rigid correction recovers precision only to ~0.6, so the label is not simply displaced. The consequence for Gates 6–8B is that the KITTI-360 geometry target was largely unreachable: a perfect-geometry oracle attains ≈7 % recall and ≈27 % precision on the held-out drive.

### Minimal proposed repair

The minimal repair is **not** a code change — our adapter is verified correct, and shifting it to chase the label would break its confirmed agreement with the official voxel input. The options, cheapest first:

1. **Withdraw KITTI-360 as a benchmark and as a training source** (recommended). No cache regeneration; the affected results are marked invalid and the conclusions that rested on them are re-derived from SemanticKITTI and Occ3D alone. This is a documentation and aggregation change only.
2. **Rebuild the KITTI-360 target ourselves** from raw aggregated LiDAR under the verified chain, SemanticKITTI-style. This restores geometric consistency but breaks comparability with every published SSCBench-KITTI-360 number, so it must be reported as a different benchmark.
3. **Adopt the +1 z shift** — rejected. It is not confirmed by official metadata, it recovers precision only to ~0.6, and it would put us out of agreement with SSCBench's own voxel input, which we currently match at 99.9 %.

### Affected Gate 8–8B results and what would need regenerating

| status | artifacts |
|---|---|
| **invalidated** | artifacts/gate5_2/* (all KITTI-360 occupancy scores)<br>artifacts/gate6/counts_kitti360_*.npz and summary_kitti360.json<br>artifacts/gate7a/* KITTI-360 reachability and envelopes<br>artifacts/gate7b/eval_kitti360_*.json<br>artifacts/gate8/eval_kitti360_*.json (Gate 8 held-out fold)<br>artifacts/gate8a/eval_kitti360_locked.json and its decision<br>artifacts/gate8b KITTI-360 fold (target), and the KITTI-360 source-validation half of both new folds' checkpoint/threshold selection |
| unaffected | every SemanticKITTI result (sweep-vs-label precision 0.983 at zero shift)<br>every Occ3D result<br>the transform chain, mapper, caches and causal separation (all verified here) |

Regeneration required **only under repair option 2**: the KITTI-360 targets, then the Gate 8B `k360_train` samples (1 276), the KITTI-360 source-validation samples (175), both new folds' training runs and every KITTI-360 evaluation. Under the recommended option 1, nothing is regenerated.

**Gate 8B's SemanticKITTI failure is NOT explained by this defect and remains open.**

Per the brief, the repair is **not implemented** and Gate 8B is **not re-run**.

## Stage 0 — provenance

| drive | partition | labelled anchors | stream frames | cached samples | keyframe k | anchor stride | native span | missing img/velo/target | missing Trident | missing sample |
|---|---|---|---|---|---|---|---|---|---|---|
| 2013_05_28_drive_0003_sync | train | 198 | 202 | 193 | 1 | 5 | [4, 1007] | 0 | 0 | 5 |
| 2013_05_28_drive_0007_sync | train | 533 | 578 | 529 | 1 | 5 | [2, 2899] | 0 | 0 | 4 |
| 2013_05_28_drive_0010_sync | source_validation | 558 | 605 | 554 | 1 | 5 | [10, 3310] | 0 | 0 | 4 |
| 2013_05_28_drive_0006_sync | heldout | 1812 | 1777 | 175 | 2 | 5 | [35, 9569] | 0 | 35 | 1602 |

3101 anchor rows joined across 4 drives. No missing image, velodyne, target or pose row anywhere; no duplicate stream key; no non-monotonic sequence; no image↔velodyne timestamp gap above 50 ms; 0 frame-key collisions across drives. The missing counts that are non-zero are expected and explained: the last few anchors of each drive have fewer than five future frames so no sample is built, drive 0006 was sampled at anchor-stride 10 by Gate 8B, and its 35 anchors without a Trident cache are the ones Gate 5.2's clip-eligibility rule excluded from the official evaluation manifest.

## Stage 1 — transform chain and round trips

The full chain, with every convention stated, is the module docstring of
`gate8c0/transforms.py` and is reproduced in `stage1_transforms.json`. Round trips over
4 096 sampled points per pair, on all nine audit anchors:

| transform pair (round trip) | max abs error (m) | within 0.001 m |
|---|---|---|
| velo_to_world | 1.38e-12 | yes |
| rect_cam_to_velo | 1.42e-14 | yes |
| rect_cam_to_world | 9.02e-13 | yes |
| velo_src_to_velo_anchor | 1.78e-14 | yes |
| cam0_unrect_to_velo | 2.13e-14 | yes |
| composition_matches_direct | 0.00e+00 | yes |
| grid index → centre → index | exact | yes |

The shipped KITTI-360 matrices are only approximately rigid, so the report states which
inverse is used and what it costs:

| shipped matrix | det | ‖RᵀR−I‖∞ | round trip with Rᵀ (m) | round trip with true inverse (m) |
|---|---|---|---|---|
| R_rect_00 | 0.999999548 | 9.93e-07 | 5.79e-05 | 1.42e-14 |
| cam0_to_velo | 1.000000000 | 5.99e-11 | 2.69e-09 | 2.13e-14 |
| velo_to_world | 1.000001095 | 1.66e-06 | 1.11e-04 | 1.38e-12 |

Nine overlays (three per partition) are in `artifacts/gate8c0/fig_chain_*.png`, all on
identical axes and bounds, carrying the official target, the current sweep, the future
observations, the frustum, the camera origin and forward direction, and the trajectory.
The camera lands at [0.8044, 0.2993, -0.177] m in the grid frame looking along [0.9949, 0.0438, -0.0911] — the
KITTI-360 rig geometry, which is what rules out a frame or sign error by inspection.

## Stage 2 — oracle current-frame alignment

The anchor's own ground-truth sweep is already *in* the grid frame, so this measures the
adapter's origin, resolution, axis order and binning with no pose, depth model or learned
component in the path. Low recall is expected; **precision and surface distance are the
test**.

| partition | n | precision | recall | IoU | frustum precision | median dist (m) | p95 dist (m) | frac of in-grid LiDAR voxels on official occupied |
|---|---|---|---|---|---|---|---|---|
| train (0003, 0007) | 17 | 0.3111 | 0.0607 | 0.0535 | 0.2695 | 0.200 | 0.490 | 0.3091 |
| source-val (0010) | 10 | 0.2106 | 0.0371 | 0.0326 | 0.1788 | 0.200 | 0.616 | 0.2083 |
| held out (0006) | 10 | 0.3134 | 0.0394 | 0.0363 | 0.2908 | 0.200 | 0.600 | 0.3128 |

Diagnostic shifts, reported and **not adopted**:

| diagnostic shift | precision | recall | IoU | note |
|---|---|---|---|---|
| (0, 0, 0) | 0.2848 | 0.0486 | 0.0433 | **identity — the confirmed one** |
| (-1, 0, 0) | 0.3121 | 0.0506 | 0.0455 |  |
| (1, 0, 0) | 0.2857 | 0.0464 | 0.0416 |  |
| (0, -1, 0) | 0.3431 | 0.0555 | 0.0502 |  |
| (0, 1, 0) | 0.2619 | 0.0420 | 0.0376 |  |
| (0, 0, -1) | 0.2931 | 0.0333 | 0.0309 |  |
| (0, 0, 1) | 0.5897 | 0.0918 | 0.0863 |  |
| anchor offset -2 (×5 native frames) | 0.2698 | — | 0.0290 |  |
| anchor offset -1 (×5 native frames) | 0.2840 | — | 0.0320 |  |
| anchor offset 0 (×5 native frames) | 0.2889 | — | 0.0424 | **best** |
| anchor offset 1 (×5 native frames) | 0.2716 | — | 0.0311 |  |
| anchor offset 2 (×5 native frames) | 0.2679 | — | 0.0301 |  |

### The controls that decide where the offset lives

| comparison | shift | precision | recall | IoU |
|---|---|---|---|---|
| KITTI-360: **ours** vs SSCBench's own `.bin` voxel input | (0, 0, 0) | 0.6004 | 0.9990 | 0.6001 |
|  | (0, 0, 1) | 0.2290 | 0.3810 | 0.1669 |
| KITTI-360: SSCBench's `.bin` vs SSCBench's own label *(we are nowhere in this path)* | (0, 0, 0) | 0.2983 | 0.0207 | 0.0197 |
|  | (0, 0, 1) | 0.6187 | 0.0412 | 0.0402 |
| KITTI-360: ours vs the official label | (0, 0, 0) | 0.2882 | 0.0326 | 0.0302 |
|  | (0, 0, 1) | 0.5141 | 0.0545 | 0.0518 |
| SemanticKITTI: raw sweep vs its official label *(reference implementation)* | (0, 0, 0) | 0.9832 | 0.1114 | 0.1112 |
|  | (0, 0, 1) | 0.7142 | 0.0769 | 0.0746 |

Read the second block first: it contains no code of ours. The adapter's elementwise 255/0/occupied rule was also re-verified against the raw `.label`/`.invalid` files and holds exactly on every anchor (`adapter_rule_holds_everywhere` = True), so `_1_1.npy` occupied is precisely `.label > 0`.

## Stage 3 — oracle temporal reconstruction

Ground-truth sweeps of the causal history, ground-truth poses, each integrated once. This
bounds what *any* method reading this map could achieve. `future` is the non-causal
t+1…t+20 diagnostic and is **never a deployable baseline**.

| partition | history | frames used | precision | recall (target coverage) | IoU | valid-grid coverage | median dist (m) |
|---|---|---|---|---|---|---|---|
| train (0003, 0007) | 1 | 1.0 | 0.3215 | 0.0616 | 0.0545 | 0.0414 | 0.200 |
|  | 5 | 4.8 | 0.2989 | 0.0830 | 0.0695 | 0.0618 | 0.200 |
|  | 20 | 18.5 | 0.2983 | 0.0865 | 0.0719 | 0.0659 | 0.200 |
|  | all_past | 10.9 | 0.2983 | 0.0866 | 0.0719 | 0.0659 | 0.200 |
|  | future | 19.0 | 0.3335 | 0.3497 | 0.2058 | 0.2189 | 0.200 |
| source-val (0010) | 1 | 1.0 | 0.2677 | 0.0473 | 0.0419 | 0.0360 | 0.200 |
|  | 5 | 5.0 | 0.2473 | 0.0686 | 0.0567 | 0.0564 | 0.200 |
|  | 20 | 20.0 | 0.2516 | 0.0734 | 0.0602 | 0.0594 | 0.200 |
|  | all_past | 12.3 | 0.2516 | 0.0733 | 0.0602 | 0.0594 | 0.200 |
|  | future | 20.0 | 0.2720 | 0.2665 | 0.1555 | 0.1971 | 0.200 |
| held out (0006) | 1 | 1.0 | 0.2541 | 0.0373 | 0.0336 | 0.0373 | 0.200 |
|  | 5 | 5.0 | 0.2569 | 0.0591 | 0.0505 | 0.0580 | 0.200 |
|  | 20 | 20.0 | 0.2735 | 0.0688 | 0.0581 | 0.0632 | 0.200 |
|  | all_past | 19.8 | 0.2735 | 0.0688 | 0.0582 | 0.0632 | 0.200 |
|  | future | 20.0 | 0.2684 | 0.1734 | 0.1178 | 0.1569 | 0.200 |

Recall rises with history and then saturates — correctly, since a sweep more than 51.2 m back contributes no voxel to this grid, which is why `all_past` equals `20`. The sanity pattern the brief expects therefore **half** holds: recall does improve with history, but ground-truth observations do **not** align with target surfaces (precision ≈0.25–0.30, median distance exactly one voxel), and the non-causal future oracle covers only 17%–35% of the privileged target. Target prevalence is 0.245 on a valid mask covering only 0.140 of the grid, against SemanticKITTI's ≈0.08 prevalence on ≈0.68 of the grid.

## Stage 4 — geometry factorization

All four cells run through the real incremental mapper. "GT depth" is the anchor sweep
projected to the network's lattice, sparse and never densified with target labels;
predicted cells apply the frozen five-frame G51-B scalar identically to depth and to pose
translation.

| partition | depth × pose | h=1 (P/R/IoU) | h=5 (P/R/IoU) | h=20 (P/R/IoU) |
|---|---|---|---|---|
| train (0003, 0007) | gt depth gt pose | 0.291/0.057/0.050 | 0.253/0.091/0.072 | 0.245/0.095/0.073 |
|  | gt depth lb pose | 0.289/0.057/0.050 | 0.271/0.086/0.070 | 0.273/0.093/0.074 |
|  | lb depth gt pose | 0.217/0.004/0.004 | 0.217/0.008/0.008 | 0.229/0.009/0.009 |
|  | lb depth lb pose | 0.216/0.004/0.004 | 0.189/0.011/0.010 | 0.195/0.012/0.011 |
| source-val (0010) | gt depth gt pose | 0.290/0.052/0.046 | 0.289/0.113/0.088 | 0.287/0.128/0.097 |
|  | gt depth lb pose | 0.294/0.053/0.047 | 0.199/0.082/0.061 | 0.199/0.083/0.062 |
|  | lb depth gt pose | 0.346/0.007/0.007 | 0.383/0.022/0.022 | 0.377/0.024/0.023 |
|  | lb depth lb pose | 0.345/0.007/0.007 | 0.280/0.046/0.041 | 0.265/0.047/0.042 |
| held out (0006) | gt depth gt pose | 0.365/0.057/0.052 | 0.338/0.101/0.084 | 0.330/0.115/0.093 |
|  | gt depth lb pose | 0.364/0.056/0.051 | 0.288/0.112/0.088 | 0.298/0.132/0.100 |
|  | lb depth gt pose | 0.405/0.021/0.020 | 0.433/0.055/0.051 | 0.415/0.070/0.064 |
|  | lb depth lb pose | 0.405/0.021/0.020 | 0.424/0.059/0.054 | 0.420/0.060/0.056 |

Pose is not the problem: replacing ground-truth pose with LingBot's changes IoU by less than the sample spread. Predicted depth costs recall (the frozen confidence gate keeps fewer rays) while holding or raising precision, i.e. it is sparser, not misaligned. Every cell lives under a ceiling set by the target.

## Stage 5 — cache and causal-separation audit

| check | result |
|---|---|
| grid declaration (dims, voxel, origin, frame) | yes |
| drive partitions disjoint | yes |
| cache keys cannot collide across drives | yes |
| Trident cache path is drive-scoped | yes |
| causal input only (0..t), 40 samples | yes |
| future target window t+1..t+20 | yes |
| no future frame leaks into the input | yes |
| RGB / Trident / pose / target name the same frame | yes |
| invalid voxels are never supervised | yes |
| each frame integrated exactly once (live mapper) | yes |
| scale anchor sees only its first five frames | yes |

Every invariant is a predicate in `gate8c0/checks.py`, and `tests/gate8c0` injects the five
defects the brief names — a one-frame offset, a reversed pose transform, a drive-ID
collision, a future-frame leak and a one-voxel origin shift — and asserts each is caught
with a diagnostic that identifies it.

## Stage 6 — fixed-batch memorization

Two deterministic batches of four KITTI-360 training-drive samples from different drives
and locations, fixed crop origins, no stochastic augmentation, unchanged architecture,
inputs, targets and loss.

| batch | step | loss | focal | KL | ‖grad‖ | edit AP | AP/prev | best IoU | IoU@0 | P@0 | R@0 | pred density | supervised prevalence | editable frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| batch A | 1 | 2.7264 | 0.3314 | 3.3596 | 4.659 | 0.1738 | 1.03 | 0.1683 | 0.1605 | 0.1632 | 0.9078 | 0.9365 | 0.1834 | 0.9459 |
| batch A | 200 | 0.5657 | 0.1483 | 0.1993 | 1.398 | 0.9067 | 5.39 | 0.6977 | 0.6855 | 0.7445 | 0.8963 | 0.2026 | 0.1834 | 0.9459 |
| batch A | 500 | 0.3338 | 0.1124 | 0.0587 | 0.943 | 0.9683 | 5.75 | 0.8109 | 0.7979 | 0.8490 | 0.9298 | 0.1843 | 0.1834 | 0.9459 |
| batch A | 1000 | 0.2654 | 0.0968 | 0.0267 | 0.372 | 0.9816 | 5.83 | 0.8511 | 0.8269 | 0.8626 | 0.9524 | 0.1858 | 0.1834 | 0.9459 |
| batch A | 2000 | 0.2592 | 0.0958 | 0.0190 | 0.322 | 0.9823 | 5.84 | 0.8519 | 0.8408 | 0.8876 | 0.9410 | 0.1785 | 0.1834 | 0.9459 |
| batch B | 1 | 2.6512 | 0.3784 | 2.9757 | 3.087 | 0.1406 | 1.04 | 0.1357 | 0.1357 | 0.1407 | 0.7936 | 0.7643 | 0.1373 | 0.9625 |
| batch B | 200 | 1.0473 | 0.1994 | 0.5532 | 1.719 | 0.7948 | 5.87 | 0.5790 | 0.5302 | 0.5808 | 0.8587 | 0.2003 | 0.1373 | 0.9625 |
| batch B | 500 | 0.4416 | 0.1258 | 0.1749 | 1.553 | 0.9706 | 7.16 | 0.8398 | 0.8206 | 0.8566 | 0.9513 | 0.1505 | 0.1373 | 0.9625 |
| batch B | 1000 | 0.2562 | 0.0960 | 0.0536 | 0.500 | 0.9955 | 7.35 | 0.9471 | 0.9398 | 0.9625 | 0.9755 | 0.1373 | 0.1373 | 0.9625 |
| batch B | 2000 | 0.2194 | 0.0888 | 0.0201 | 0.181 | 0.9972 | 7.36 | 0.9577 | 0.9529 | 0.9748 | 0.9769 | 0.1358 | 0.1373 | 0.9625 |

Both batches memorize. Cause discrimination: gradients are finite and non-zero throughout (optimization is healthy); the supervised mask carries a real positive prevalence and covers 94.6% of valid voxels (masking is fine); the minimum pairwise input distance across the fixed samples is 147.7 with up to 36.7% target disagreement (the samples are distinct, so neither information collision nor contradictory identical inputs applies). Capacity is not the limit either. **Outcome B is ruled out.**

## Stage 7 — gated off

Stage 7 requires Stages 1-6 to show correct alignment; the KITTI-360 target is not geometrically consistent with its own LiDAR, so the experiment would measure the defect, not learnability.

The descriptive channel comparison is independent of that gate:

| partition | n | observed frac of grid | editable frac of valid | valid frac of grid | target prevalence | mean log-odds where observed | mean n_obs | mean age |
|---|---|---|---|---|---|---|---|---|
| train (0003, 0007) | 12 | 0.0445 | 0.9547 | 0.1657 | 0.2183 | -0.127 | 0.142 | 0.05 |
| source-val (0010) | 12 | 0.0760 | 0.9698 | 0.2028 | 0.1748 | -0.206 | 0.104 | 0.03 |
| held out (0006) | 12 | 0.0433 | 0.9409 | 0.1414 | 0.2733 | 0.080 | 0.227 | 0.65 |

Descriptive only — Gate 8C-0 normalises nothing. The number worth carrying forward is that the causal map observes 4–8 % of the grid while the target asserts 17–27 % occupancy over the valid region, and 94–97 % of that valid region is *editable*, i.e. the map holds no committed evidence there.

## Runtime, memory and commands

| stage | wall time | notes |
|---|---|---|
| 0 provenance | 1 s | 3101 anchors, 4 drives |
| 1 transforms + 9 overlays | 5 s | CPU |
| 2 alignment | 29 s | 37 anchors, CPU |
| 2b controls | 7 s | 8 K360 + 8 SK |
| 3 oracle temporal | 0.8 min | 24 anchors, CPU |
| 4 factorization | 0.6 min | 8 anchors, peak 0.54 GiB |
| 5 cache audit | 5 s | 40 samples |
| 6 memorization | 4.4 min | 2 × 2000 steps, peak 4.38 GiB |
| 7 channel stats | 7 s | descriptive |

```bash
cd /home/minh/workspace/lingbot-map_fork
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1 GATE8_GPUS="1 2 3"
PY=/home/minh/anaconda3/envs/cu128/bin/python

$PY tools/gate8c0/stage0_provenance.py
$PY tools/gate8c0/stage1_transforms.py
$PY tools/gate8c0/stage2_alignment.py
$PY tools/gate8c0/stage2b_target_consistency.py
$PY tools/gate8c0/stage3_temporal.py
$PY tools/gate8c0/stage4_factorization.py     --device cuda:1
$PY tools/gate8c0/stage5_cache_audit.py       --device cuda:1
$PY tools/gate8c0/stage6_memorize.py          --device cuda:2
$PY tools/gate8c0/stage7_channel_stats.py     --device cuda:3   # descriptive only
$PY tools/gate8c0/stage0b_hashes.py
$PY tools/gate8c0/aggregate.py && $PY tools/gate8c0/figures.py && $PY tools/gate8c0/report.py --write
$PY -m pytest tests/gate6 tests/gate7b tests/gate8 tests/gate8a tests/gate8b tests/gate8c0 -q
```

Seed 0 throughout. Configuration and artifact hashes are in `artifacts/gate8c0/hashes.json`
(29 frozen files, re-run to see drift). Machine-readable results:
`artifacts/gate8c0/gate8c0_results.json`, plus one JSON per stage and
`stage0_provenance.csv` (the joined per-anchor table, 3101 rows).

## Tests

`python -m pytest tests/gate6 tests/gate7b tests/gate8 tests/gate8a tests/gate8b tests/gate8c0 -q`

```
209 passed, 3 warnings in 137.53s
```

The relevant existing suites (Gate 6 dataset adapters and metrics, Gate 7B mapper and
causality, Gate 8/8A/8B) plus 21 new Gate 8C-0 tests. The three warnings are NumPy's
`np.fromstring` deprecation inside the frozen `sscbench_kitti360.adapter.parse_calibration`
and matplotlib/`trapz` notices; none originates in Gate 8C-0 code. The expensive historical
Gate 5.x and Gate 7A suites were not re-run because no code they cover changed — Gate 8C-0
added modules and tools only.

The Gate 8C-0 tests that carry this report's claims:

* `test_inv_is_a_true_inverse_not_a_transpose`,
  `test_apply_uses_column_vector_convention_on_row_stored_points`,
  `test_transform_round_trips_within_tolerance` — the conventions the transform-chain
  specification states are the ones the code implements, to 1 mm.
* `test_sscbench_index_maps_to_pose_frames_plus_one`,
  `test_camera_sits_where_the_kitti360_rig_puts_it`,
  `test_grid_declaration_matches_the_frozen_benchmark`, `test_voxelize_floors_and_matches_gate6`,
  `test_grid_index_centre_round_trip_is_exact` — indexing, rig geometry, grid and binning.
* `test_our_voxelization_reproduces_the_official_sscbench_input` — the control that decides
  where the +1 z offset lives, as a test rather than a one-off script.
* The five deliberate defects the brief requires, each of which must be caught:
  `test_one_frame_offset_is_caught`, `test_reversed_pose_transform_is_caught`,
  `test_drive_id_collision_is_caught`, `test_future_frame_leak_is_caught`,
  `test_one_voxel_origin_shift_is_caught` — plus
  `test_integrated_once_and_scale_window_are_caught`,
  `test_supervising_an_invalid_voxel_is_caught` and `test_partitions_disjoint_is_caught`.
* `test_gate8_8a_8b_artifacts_are_untouched`.

## Deviations and failures

1. **Stage 7 was not run, by the brief's own precondition.** It is authorised "only if
   Stages 1–6 show correct alignment and the model passes fixed-batch memorization".
   Memorization passes, but the KITTI-360 target is not geometrically consistent with its
   own LiDAR, so a KITTI-360-only training run would measure the defect rather than
   learnability. The descriptive map-channel comparison Stage 7 also asked for is
   independent of that gate and is reported (`stage7_channel_stats.json`,
   `fig_channels.png`); no conclusion rests on it.

2. **`gate8c0.transforms.inv` uses `np.linalg.inv`, not `(Rᵀ, −Rᵀt)`.** This matches the
   production path (`tools/gate8/evaluate._grid_to_world`). KITTI-360's shipped matrices are
   only approximately rigid — `R_rect_00` has determinant 0.999999548 and `cam0_to_world`
   1.000000611, because the published files carry six decimals — so a rigid inverse leaves a
   7e-5 m round-trip residual at 40 m. `inv_rigid` is kept and Stage 1 reports the difference
   rather than hiding it. The round-trip tolerance is 1 mm (1/200 voxel), chosen so it cannot
   be met by accident; measured worst error is 1.4e-12 m.

3. **The audit subset is 9 anchors for Stage 1, 37 for Stage 2, 24 for Stage 3, 8 for
   Stage 4, 40 samples for Stage 5 and 8 for Stage 6.** Stages 2–4 are O(minutes) per sample
   because each rebuilds oracle volumes and Euclidean distance transforms over a 2.1 M-voxel
   grid. The effects reported are 2–20× in size and consistent across every sample and every
   partition, so the subset sizes are not the limiting factor; the per-sample tables are in
   the artifacts for inspection.

4. **"All available past" is bounded.** A sweep taken more than 51.2 m back contributes no
   voxel to this grid, so Stage 3 walks back until the sensor leaves the box (cap 200 native
   frames). On these drives it resolves to 11–20 stream frames, which is why `all_past` and
   `20` agree to four decimals — that is the correct answer, not a truncation artefact.

5. **Two additive, behaviour-preserving edits from Gate 8B are still in place** and are
   re-hashed here: the `register_source` hook in `gate8/sources.py` and the `sequence`
   argument of `sscbench_kitti360.adapter.load_target`. Gate 8C-0 added none of its own; the
   `hashes.json` frozen list covers 29 files and reports drift on re-run.

6. **`sscbench_kitti360.adapter.parse_calibration` emits a NumPy deprecation warning**
   (`np.fromstring` on a text buffer). It is pre-existing, frozen by this gate's scope, and
   parses correctly; it surfaces as the single warning in the test run.

7. **The Stage-2 shift scan reports `(0, 0, 1)` as the best-scoring alignment and it was not
   adopted.** The brief forbids adopting a better-scoring alignment without confirmation from
   official metadata, and the confirmation went the other way: our unshifted voxelization
   reproduces SSCBench's own `.bin` at 99.90 % recall, while their `.bin` and their label
   disagree by the same +1 z. Shifting would break agreement with the official input to chase
   agreement with an inconsistent label.

