# Gate 8B — leave-one-dataset-out transfer evaluation

**Verdict.** Transfer is target-dependent, not KITTI-360-specific: the completion collapses to chance on SemanticKITTI 08 and KITTI-360 (drive 0006) but keeps a real ranking lift on Occ3D-nuScenes val (SemanticKITTI 08 ×1.62 (AUROC 0.536), Occ3D-nuScenes val ×2.08 (AUROC 0.705), KITTI-360 (drive 0006) ×1.04 (AUROC 0.493)); geometry passes the declared baselines on Occ3D-nuScenes val.

**Recommended next action.** A source-validated map-state canonicalisation gate (Outcome B's remedy), because the failure reproduces on SemanticKITTI -- a target whose grid, frame and stride the KITTI-360 training source shares exactly -- so it is a property of the map state handed to the head, not of one benchmark's export. Run the cheap KITTI-360 export/label audit (Outcome A's remedy) first, since KITTI-360 also fails as an in-domain source-validation set. Do not scale the network, change the architecture, add datasets or redesign semantics.

**Outcome class.** mixed: semantickitti geo fail/near-chance=True, occ3d geo pass/near-chance=False, kitti360 geo fail/near-chance=True

Every fold is *leave-one-dataset-out transfer with no target-domain training or adaptation of the
completion module*. None is claimed as whole-system zero-shot: LingBot-Map's published training
mixture includes KITTI-360, and the training data of MoGe-2 and Trident-H were not audited here.

## 1. Provenance — three folds

| target (held out) | completion-training sources | source-validation domains | never read before freezing | status | KITTI-360 partition |
|---|---|---|---|---|---|
| SemanticKITTI 08 | occ3d_train, k360_train | occ3d, kitti360 | semantickitti | new | train drives ['2013_05_28_drive_0003_sync', '2013_05_28_drive_0007_sync', '2013_05_28_drive_0010_sync']; source-validation drive 2013_05_28_drive_0006_sync |
| Occ3D-nuScenes val | sk_train, k360_train | semantickitti, kitti360 | occ3d | new | train drives ['2013_05_28_drive_0003_sync', '2013_05_28_drive_0007_sync', '2013_05_28_drive_0010_sync']; source-validation drive 2013_05_28_drive_0006_sync |
| KITTI-360 (drive 0006) | sk_train, occ3d_train | semantickitti, occ3d | kitti360 | existing (Gate 8A), reused unmodified | target only (drive 0006); never trained on |

The two new folds were built with Gate 8A's frozen method (`configs/gate8a/cellB_uniform_focal.yaml`:
uniform crop sampler, focal BCE + Dice, 0.5 × future-teacher KL, 6 000 steps, batch 4, crop 128×128×32,
seed 0, candidates {best, last}, macro-average source AP for the checkpoint, one global final-logit
threshold from source validation). A test asserts the fold configs differ from cell B only in their
source lists and never name the target. The two training sources are drawn with equal probability.

## 2. Frozen configuration per fold

| fold | selected checkpoint | sha256 | macro source AP | global τ (final log-odds) | mapper τ |
|---|---|---|---|---|---|
| SemanticKITTI 08 | fold_semantickitti_best | `708951df7b0e136e…` | 0.4883 | +0.0312 | -16.0000 |
| Occ3D-nuScenes val | fold_occ3d_best | `8728000f23756b73…` | 0.3046 | -0.0312 | -3.9688 |
| KITTI-360 (drive 0006) | cellB_last | `b505fdc4facdd53a…` | 0.5018 | +0.5625 | -16.0000 |

Selection candidates (source-validation domains only; the target of each fold was not read until the
frozen file existed):

| fold | candidate | source A | source B | macro AP | AP > prev on both |
|---|---|---|---|---|---|
| SemanticKITTI 08 | fold_semantickitti_best | Occ3D-nuScenes val: AP 0.6468 (prev 0.2295, ×2.82) | KITTI-360 (drive 0006): AP 0.3298 (prev 0.2509, ×1.31) | 0.4883 | yes |
| SemanticKITTI 08 | fold_semantickitti_last | Occ3D-nuScenes val: AP 0.6468 (prev 0.2295, ×2.82) | KITTI-360 (drive 0006): AP 0.3298 (prev 0.2509, ×1.31) | 0.4883 | yes |
| SemanticKITTI 08 | mapper | Occ3D-nuScenes val: AP 0.2821 (prev 0.2295, ×1.23) | KITTI-360 (drive 0006): AP 0.2539 (prev 0.2509, ×1.01) | 0.2680 | yes |
| Occ3D-nuScenes val | fold_occ3d_best | SemanticKITTI 08: AP 0.3103 (prev 0.0782, ×3.97) | KITTI-360 (drive 0006): AP 0.2988 (prev 0.2509, ×1.19) | 0.3046 | yes |
| Occ3D-nuScenes val | fold_occ3d_last | SemanticKITTI 08: AP 0.3101 (prev 0.0782, ×3.96) | KITTI-360 (drive 0006): AP 0.2928 (prev 0.2509, ×1.17) | 0.3014 | yes |
| Occ3D-nuScenes val | mapper | SemanticKITTI 08: AP 0.1064 (prev 0.0782, ×1.36) | KITTI-360 (drive 0006): AP 0.2539 (prev 0.2509, ×1.01) | 0.1801 | yes |
| KITTI-360 (drive 0006) | cellB_best | SemanticKITTI 08: AP 0.3223 (prev 0.0782, ×4.12) | Occ3D-nuScenes val: AP 0.6803 (prev 0.2295, ×2.96) | 0.5013 | yes |
| KITTI-360 (drive 0006) | cellB_last | SemanticKITTI 08: AP 0.3225 (prev 0.0782, ×4.12) | Occ3D-nuScenes val: AP 0.6810 (prev 0.2295, ×2.97) | 0.5018 | yes |
| KITTI-360 (drive 0006) | mapper | SemanticKITTI 08: AP 0.1064 (prev 0.0782, ×1.36) | Occ3D-nuScenes val: AP 0.2821 (prev 0.2295, ×1.23) | 0.1943 | yes |

### KITTI-360 as an in-domain source-validation set

The two new folds put KITTI-360 on the *training* side. Before either target was opened, their
selected checkpoints were scored on the KITTI-360 val drive (a source-validation domain here, so this
is not a held-out number):

* fold **SemanticKITTI 08** (trained on KITTI-360 drives 0003/0007/0010 + occ3d_train): on the KITTI-360 val drive 0006 the selected checkpoint scores AP 0.3298 against prevalence 0.2509 (×1.31, AUROC 0.6241; the mapper alone ×1.01), while on Occ3D-nuScenes val in the same fold it scores ×2.82 (AUROC 0.8175).
* fold **Occ3D-nuScenes val** (trained on KITTI-360 drives 0003/0007/0010 + sk_train): on the KITTI-360 val drive 0006 the selected checkpoint scores AP 0.2988 against prevalence 0.2509 (×1.19, AUROC 0.5603; the mapper alone ×1.01), while on SemanticKITTI 08 in the same fold it scores ×3.97 (AUROC 0.8712).

## 3. Primary setting — causal all-past streaming

Scale fixed from the first five anchor frames, every frame integrated exactly once, persistent map
state, every anchor scored with all causal history to that time, no future frame on the prediction
path. Threshold-free first:

| target | region | voxels | prevalence | AP | AP/prev | AUROC | Brier | ECE | IoU @ 0 | oracle τ | IoU @ oracle τ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SemanticKITTI 08 | completion_full | 239805138 | 0.0782 | 0.1270 | 1.62 | 0.5362 | 0.3076 | 0.4350 | 0.0722 | 1.625 | 0.1263 |
|  | completion_edit | 225701801 | 0.0731 | 0.1028 | 1.41 | 0.4960 | 0.3044 | 0.4404 | 0.0616 | 1.625 | 0.0840 |
|  | mapper_full | 239805138 | 0.0782 | 0.1064 | 1.36 | 0.5523 | 0.2507 | 0.4130 | 0.0787 | 0.031 | 0.0964 |
| Occ3D-nuScenes val | completion_full | 52095614 | 0.2295 | 0.4783 | 2.08 | 0.7051 | 0.2082 | 0.2162 | 0.2958 | 0.500 | 0.3243 |
|  | completion_edit | 49650301 | 0.2226 | 0.4702 | 2.11 | 0.7163 | 0.2064 | 0.2173 | 0.2922 | 0.500 | 0.3201 |
|  | mapper_full | 52095614 | 0.2295 | 0.2821 | 1.23 | 0.5396 | 0.2512 | 0.2816 | 0.2207 | -16.000 | 0.2295 |
| KITTI-360 (drive 0006) | completion_full | 549394259 | 0.2509 | 0.2600 | 1.04 | 0.4928 | 0.3531 | 0.3674 | 0.2058 | -16.000 | 0.2509 |
|  | completion_edit | 518359270 | 0.2511 | 0.2583 | 1.03 | 0.4863 | 0.3486 | 0.3644 | 0.2042 | -16.000 | 0.2511 |
|  | mapper_full | 549394259 | 0.2509 | 0.2539 | 1.01 | 0.5067 | 0.2587 | 0.2526 | 0.2476 | -3.969 | 0.2513 |

At each fold's locked source-selected threshold (paired scene-aware bootstrap, Gate 6 units, 10 000
draws, seed 0):

### SemanticKITTI 08 (target of the new SK fold)

| method | SC IoU | precision | recall | predicted density | pred/GT volume | Δ vs completion (paired 95% CI) |
|---|---|---|---|---|---|---|
| completion | 0.0722 | 0.0781 | 0.4883 | 0.4889 | 6.25 | — |
| completion_edit_region | 0.0615 | 0.0666 | 0.4458 | 0.4895 | 6.69 | — |
| mapper_native | 0.0964 | 0.2585 | 0.1332 | 0.0403 | 0.52 | -0.0242 [-0.0350, -0.0136] |
| mapper_dilate | 0.1435 | 0.1931 | 0.3584 | 0.1452 | 1.86 | -0.0713 [-0.0916, -0.0514] |
| mapper_calibrated | 0.0782 | 0.0782 | 1.0000 | 1.0000 | 12.78 | -0.0060 [-0.0197, +0.0082] |
| frozen_5frame_raw | 0.0670 | 0.3492 | 0.0766 | 0.0172 | 0.22 | +0.0052 [-0.0054, +0.0168] |
| frozen_5frame_dil | 0.1600 | 0.2537 | 0.3022 | 0.0932 | 1.19 | -0.0877 [-0.0991, -0.0757] |
| all_valid_occupied | 0.0782 | 0.0782 | 1.0000 | 1.0000 | 12.78 | -0.0060 [-0.0197, +0.0082] |
| editable_fill | 0.0786 | 0.0788 | 0.9759 | 0.9694 | 12.39 | -0.0064 [-0.0202, +0.0082] |
| random_editable_mean | 0.0794 | — | — | — | — | sd 0.0000, range [0.0794, 0.0795] |

### Occ3D-nuScenes val (target of the new Occ3D fold)

| method | SC IoU | precision | recall | predicted density | pred/GT volume | Δ vs completion (paired 95% CI) |
|---|---|---|---|---|---|---|
| completion | 0.2955 | 0.3474 | 0.6643 | 0.4389 | 1.91 | — |
| completion_edit_region | 0.2919 | 0.3378 | 0.6823 | 0.4495 | 2.02 | — |
| mapper_native | 0.1170 | 0.6052 | 0.1266 | 0.0480 | 0.21 | +0.1786 [+0.1650, +0.1922] |
| mapper_dilate | 0.2309 | 0.5316 | 0.2899 | 0.1252 | 0.55 | +0.0647 [+0.0499, +0.0794] |
| mapper_calibrated | 0.2278 | 0.2289 | 0.9783 | 0.9809 | 4.27 | +0.0678 [+0.0587, +0.0774] |
| frozen_5frame_raw | 0.0793 | 0.6289 | 0.0832 | 0.0304 | 0.13 | +0.2162 [+0.2030, +0.2299] |
| frozen_5frame_dil | 0.2094 | 0.5641 | 0.2499 | 0.1017 | 0.44 | +0.0861 [+0.0718, +0.1009] |
| all_valid_occupied | 0.2295 | 0.2295 | 1.0000 | 1.0000 | 4.36 | +0.0660 [+0.0568, +0.0758] |
| editable_fill | 0.2259 | 0.2282 | 0.9579 | 0.9635 | 4.20 | +0.0696 [+0.0607, +0.0791] |
| random_editable_mean | 0.1960 | — | — | — | — | sd 0.0001, range [0.1959, 0.1961] |

### KITTI-360 (Gate 8A fold, reused)

| method | SC IoU | precision | recall | predicted density | pred/GT volume | Δ vs completion (paired 95% CI) |
|---|---|---|---|---|---|---|
| completion | 0.1873 | 0.2680 | 0.3835 | 0.3590 | 1.43 | — |
| completion_edit_region | 0.1838 | 0.2664 | 0.3722 | 0.3509 | 1.40 | — |
| mapper_native | 0.0360 | 0.2869 | 0.0396 | 0.0346 | 0.14 | +0.1513 [+0.1459, +0.1566] |
| mapper_dilate | 0.1094 | 0.2854 | 0.1508 | 0.1325 | 0.53 | +0.0779 [+0.0702, +0.0855] |
| mapper_calibrated | 0.2509 | 0.2509 | 1.0000 | 1.0000 | 3.99 | -0.0636 [-0.0714, -0.0561] |
| frozen_5frame_raw | 0.0363 | 0.3747 | 0.0386 | 0.0258 | 0.10 | +0.1511 [+0.1470, +0.1554] |
| frozen_5frame_dil | 0.1321 | 0.3587 | 0.1729 | 0.1209 | 0.48 | +0.0552 [+0.0500, +0.0608] |
| all_valid_occupied | 0.2509 | 0.2509 | 1.0000 | 1.0000 | 3.99 | -0.0636 [-0.0714, -0.0561] |
| editable_fill | 0.2506 | 0.2522 | 0.9762 | 0.9714 | 3.87 | -0.0633 [-0.0713, -0.0558] |
| random_editable_mean | 0.1778 | — | — | — | — | sd 0.0000, range [0.1778, 0.1779] |

## 4. Matched five-frame setting

`ScaleState` and map state reset for every official clip; exactly the clip's five real frames from
one camera, LingBot run on those five alone (per-clip caches), scale fixed from those five (the pinned
G51-B scalar), each frame integrated once, query and completion after frame five, nothing carried
between clips. Reported separately: **OccAny and our streaming method do not share a temporal input
budget, and the numbers below are not to be read against Section 3.**

| target | region | voxels | prevalence | AP | AP/prev | AUROC | Brier | ECE | IoU @ 0 | oracle τ | IoU @ oracle τ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SemanticKITTI 08 | completion_full | 239805138 | 0.0782 | 0.1031 | 1.32 | 0.4735 | 0.3278 | 0.4545 | 0.0641 | 1.781 | 0.0840 |
|  | completion_edit | 229505803 | 0.0742 | 0.0658 | 0.89 | 0.4232 | 0.3295 | 0.4620 | 0.0549 | -1.125 | 0.0773 |
|  | mapper_full | 239805138 | 0.0782 | 0.1122 | 1.43 | 0.5641 | 0.2439 | 0.4088 | 0.0815 | 0.031 | 0.0905 |
| Occ3D-nuScenes val | completion_full | 52095614 | 0.2295 | 0.4716 | 2.05 | 0.6976 | 0.2161 | 0.2291 | 0.2790 | 0.531 | 0.3129 |
|  | completion_edit | 49525754 | 0.2227 | 0.4694 | 2.11 | 0.7167 | 0.2140 | 0.2314 | 0.2784 | 0.531 | 0.3142 |
|  | mapper_full | 52095614 | 0.2295 | 0.2690 | 1.17 | 0.5237 | 0.2487 | 0.2732 | 0.2162 | -16.000 | 0.2295 |
| KITTI-360 (drive 0006) | completion_full | 549394259 | 0.2509 | 0.2696 | 1.07 | 0.5045 | 0.3653 | 0.3913 | 0.2134 | -12.781 | 0.2509 |
|  | completion_edit | 514275952 | 0.2490 | 0.2574 | 1.03 | 0.4900 | 0.3660 | 0.3951 | 0.2086 | -12.781 | 0.2490 |
|  | mapper_full | 549394259 | 0.2509 | 0.2624 | 1.05 | 0.5209 | 0.2531 | 0.2444 | 0.2473 | -3.969 | 0.2518 |

### SemanticKITTI 08

| method | SC IoU | precision | recall | predicted density | pred/GT volume | Δ vs completion (paired 95% CI) |
|---|---|---|---|---|---|---|
| completion | 0.0628 | 0.0680 | 0.4549 | 0.5237 | 6.69 | — |
| completion_edit_region | 0.0534 | 0.0579 | 0.4114 | 0.5277 | 7.11 | — |
| mapper_native | 0.0905 | 0.3296 | 0.1109 | 0.0263 | 0.34 | -0.0277 [-0.0332, -0.0229] |
| mapper_dilate | 0.1608 | 0.2456 | 0.3179 | 0.1013 | 1.29 | -0.0980 [-0.1044, -0.0900] |
| mapper_calibrated | 0.0782 | 0.0782 | 1.0000 | 1.0000 | 12.78 | -0.0154 [-0.0235, -0.0085] |
| frozen_5frame_raw | 0.0670 | 0.3492 | 0.0766 | 0.0172 | 0.22 | -0.0042 [-0.0099, +0.0004] |
| frozen_5frame_dil | 0.1600 | 0.2537 | 0.3022 | 0.0932 | 1.19 | -0.0971 [-0.1035, -0.0892] |
| all_valid_occupied | 0.0782 | 0.0782 | 1.0000 | 1.0000 | 12.78 | -0.0154 [-0.0235, -0.0085] |
| editable_fill | 0.0792 | 0.0793 | 0.9892 | 0.9757 | 12.47 | -0.0164 [-0.0244, -0.0095] |
| random_editable_mean | 0.0794 | — | — | — | — | sd 0.0000, range [0.0793, 0.0794] |
| completion_occany_dilation | 0.0712 | 0.0748 | 0.5954 | 0.6227 | 7.96 | -0.0083 [-0.0095, -0.0073] |
| completion_occany_majority | 0.0632 | 0.0682 | 0.4630 | 0.5310 | 6.79 | -0.0004 [-0.0005, -0.0002] |
| mapper_native_occany_dilation | 0.1498 | 0.2879 | 0.2380 | 0.0647 | 0.83 | -0.0870 [-0.0935, -0.0794] |
| mapper_native_occany_majority | 0.0920 | 0.3220 | 0.1142 | 0.0277 | 0.35 | -0.0292 [-0.0349, -0.0243] |
| mapper_dilate_occany_dilation | 0.1537 | 0.2100 | 0.3646 | 0.1358 | 1.74 | -0.0909 [-0.0972, -0.0834] |

### Occ3D-nuScenes val

| method | SC IoU | precision | recall | predicted density | pred/GT volume | Δ vs completion (paired 95% CI) |
|---|---|---|---|---|---|---|
| completion | 0.2804 | 0.3271 | 0.6624 | 0.4647 | 2.02 | — |
| completion_edit_region | 0.2799 | 0.3199 | 0.6909 | 0.4810 | 2.16 | — |
| mapper_native | 0.0928 | 0.6157 | 0.0985 | 0.0367 | 0.16 | +0.1875 [+0.1755, +0.2004] |
| mapper_dilate | 0.2183 | 0.5557 | 0.2644 | 0.1092 | 0.48 | +0.0621 [+0.0493, +0.0757] |
| mapper_calibrated | 0.2273 | 0.2285 | 0.9774 | 0.9818 | 4.28 | +0.0531 [+0.0449, +0.0617] |
| frozen_5frame_raw | 0.0793 | 0.6289 | 0.0832 | 0.0304 | 0.13 | +0.2010 [+0.1891, +0.2138] |
| frozen_5frame_dil | 0.2094 | 0.5641 | 0.2499 | 0.1017 | 0.44 | +0.0710 [+0.0578, +0.0848] |
| all_valid_occupied | 0.2295 | 0.2295 | 1.0000 | 1.0000 | 4.36 | +0.0509 [+0.0426, +0.0595] |
| editable_fill | 0.2242 | 0.2270 | 0.9475 | 0.9582 | 4.17 | +0.0562 [+0.0480, +0.0649] |
| random_editable_mean | 0.1917 | — | — | — | — | sd 0.0001, range [0.1916, 0.1919] |
| completion_occany_dilation | 0.2676 | 0.2898 | 0.7772 | 0.6156 | 2.68 | +0.0128 [+0.0102, +0.0155] |
| completion_occany_majority | 0.2792 | 0.3253 | 0.6633 | 0.4681 | 2.04 | +0.0012 [+0.0011, +0.0014] |
| mapper_native_occany_dilation | 0.1417 | 0.4696 | 0.1687 | 0.0824 | 0.36 | +0.1387 [+0.1259, +0.1523] |
| mapper_native_occany_majority | 0.0929 | 0.6103 | 0.0988 | 0.0372 | 0.16 | +0.1874 [+0.1754, +0.2003] |
| mapper_dilate_occany_dilation | 0.2326 | 0.4521 | 0.3239 | 0.1644 | 0.72 | +0.0478 [+0.0364, +0.0598] |

### KITTI-360

| method | SC IoU | precision | recall | predicted density | pred/GT volume | Δ vs completion (paired 95% CI) |
|---|---|---|---|---|---|---|
| completion | 0.2034 | 0.2694 | 0.4536 | 0.4224 | 1.68 | — |
| completion_edit_region | 0.1960 | 0.2609 | 0.4407 | 0.4207 | 1.69 | — |
| mapper_native | 0.0533 | 0.3774 | 0.0585 | 0.0389 | 0.16 | +0.1501 [+0.1458, +0.1548] |
| mapper_dilate | 0.1424 | 0.3594 | 0.1908 | 0.1332 | 0.53 | +0.0611 [+0.0560, +0.0666] |
| mapper_calibrated | 0.2509 | 0.2509 | 1.0000 | 1.0000 | 3.99 | -0.0475 [-0.0549, -0.0405] |
| frozen_5frame_raw | 0.0363 | 0.3747 | 0.0386 | 0.0258 | 0.10 | +0.1672 [+0.1631, +0.1716] |
| frozen_5frame_dil | 0.1321 | 0.3587 | 0.1729 | 0.1209 | 0.48 | +0.0713 [+0.0663, +0.0768] |
| all_valid_occupied | 0.2509 | 0.2509 | 1.0000 | 1.0000 | 3.99 | -0.0475 [-0.0549, -0.0405] |
| editable_fill | 0.2514 | 0.2531 | 0.9733 | 0.9647 | 3.85 | -0.0480 [-0.0556, -0.0409] |
| random_editable_mean | 0.1942 | — | — | — | — | sd 0.0000, range [0.1942, 0.1942] |
| completion_occany_dilation | 0.2187 | 0.2592 | 0.5836 | 0.5649 | 2.25 | -0.0153 [-0.0173, -0.0134] |
| completion_occany_majority | 0.2049 | 0.2649 | 0.4752 | 0.4501 | 1.79 | -0.0015 [-0.0019, -0.0011] |
| mapper_native_occany_dilation | 0.1029 | 0.3663 | 0.1252 | 0.0857 | 0.34 | +0.1005 [+0.0958, +0.1058] |
| mapper_native_occany_majority | 0.0557 | 0.3778 | 0.0613 | 0.0407 | 0.16 | +0.1477 [+0.1434, +0.1525] |
| mapper_dilate_occany_dilation | 0.1643 | 0.3478 | 0.2374 | 0.1713 | 0.68 | +0.0391 [+0.0341, +0.0447] |

## 5. OccAny reference comparison (matched setting only)

The published OccAny five-frame single-camera numbers appear only here. The post-processing columns
apply OccAny's own `apply_majority_pooling` (separate mode) to our predictions, untuned; the
reproduction is checked bit-for-bit against the official function in the OccAny environment. Raw
completion remains the primary result. In geometry-only mode OccAny's default is `use_dilation=True`
— a 3×3×3 max-pool, i.e. a one-voxel dilation — so both that and the true vote are shown.

**SemanticKITTI 08** — *reference comparison (protocol differs)*

| method | precision % | recall % | SC IoU % | note |
|---|---|---|---|---|
| **OccAny (published, 5 frames, single camera)** | 36.79 | 46.70 | 25.91 | OccAny's own post-processing as published |
| ours: completion, raw (primary) | 6.80 | 45.49 | 6.28 |  |
| ours: completion + OccAny geometry pooling (3×3×3 max-pool) | 7.48 | 59.54 | 7.12 |  |
| ours: completion + OccAny geometry pooling (3×3×3 vote) | 6.82 | 46.30 | 6.32 |  |
| ours: five-frame map, no completion, raw | 32.96 | 11.09 | 9.05 |  |
| ours: five-frame map + OccAny max-pool | 28.79 | 23.80 | 14.98 |  |
| ours: five-frame map + fixed 0.4 m dilation | 24.56 | 31.79 | 16.08 |  |
| Gate 6 frozen five-frame G51-B, raw | 34.92 | 7.66 | 6.70 |  |
| Gate 6 frozen five-frame G51-B + 0.4 m dilation | 25.37 | 30.22 | 16.00 |  |

Protocol audit:

| item | OccAny | ours | match |
|---|---|---|---|
| target split | SemanticKITTI seq 08 (val) | SemanticKITTI seq 08 (val) | yes |
| camera | image_2 (left colour) | image_2 (left colour) | yes |
| five-frame temporal sampling | target frame FIRST; views at native offsets +0,+10,+20,+30,+40 (10-frame video at stride 5, recon_view_idx [0,2,4,6,8]); anchors every 5 frames while begin+50 <= end | target frame LAST; frames at offsets -20,-15,-10,-5,0 (stride 5); anchors every 5 frames from frame 20 | **no** |
| voxel grid / resolution | 256x256x32 @ 0.2 m, origin (0,-25.6,-2) | 256x256x32 @ 0.2 m, origin (0,-25.6,-2) | yes |
| coordinate frame | velodyne frame of the target frame | velodyne frame of the anchor | yes |
| valid / unknown mask | label 255 excluded (predict[target==255] := empty) | keep = target != 255 | yes |
| occupied / free | any non-empty class = occupied | any non-empty class = occupied | yes |
| majority pooling | apply_majority_pooling, separate mode (official code) | gate8b.pooling, checked bit-for-bit against the official function | yes |
| metric aggregation | pooled TP/FP/FN over all evaluated samples | pooled TP/FP/FN over all official clips (Gate 6 metric code) | yes |
| metric-scale source | OccAny's own (reconstruction network) | frozen G51-B: calibrated-FOV MoGe-2 median gauge from the five frames | **no** |

**Occ3D-nuScenes val** — *reference comparison (protocol differs)*

| method | precision % | recall % | SC IoU % | note |
|---|---|---|---|---|
| **OccAny (published, 5 frames, single camera)** | 36.09 | 40.39 | 23.55 | OccAny's own post-processing as published |
| ours: completion, raw (primary) | 32.71 | 66.24 | 28.04 |  |
| ours: completion + OccAny geometry pooling (3×3×3 max-pool) | 28.98 | 77.72 | 26.76 |  |
| ours: completion + OccAny geometry pooling (3×3×3 vote) | 32.53 | 66.33 | 27.92 |  |
| ours: five-frame map, no completion, raw | 61.57 | 9.85 | 9.28 |  |
| ours: five-frame map + OccAny max-pool | 46.96 | 16.87 | 14.17 |  |
| ours: five-frame map + fixed 0.4 m dilation | 55.57 | 26.44 | 21.83 |  |
| Gate 6 frozen five-frame G51-B, raw | 62.89 | 8.32 | 7.93 |  |
| Gate 6 frozen five-frame G51-B + 0.4 m dilation | 56.41 | 24.99 | 20.94 |  |

Protocol audit:

| item | OccAny | ours | match |
|---|---|---|---|
| target split | nuScenes val (Occ3D-nuScenes), CAM_FRONT | nuScenes val (Occ3D-nuScenes), CAM_FRONT, 150 scenes / 1182 clips | yes |
| camera | CAM_FRONT | CAM_FRONT | yes |
| five-frame temporal sampling | target sample FIRST; 5 samples at frame_interval 2 (~1.0 s apart, ~4 s forward) | target sample LAST; 5 samples at 0.5 s spacing (2 s backward) | **no** |
| voxel grid / resolution | 200x200x16 @ 0.4 m, origin (-40,-40,-1) | 200x200x16 @ 0.4 m (prediction at 0.2 m, any-sub-voxel reduction) | yes |
| coordinate frame | ego frame of the reference sample | ego frame of the anchor | yes |
| valid / unknown mask | mask_camera applied, mask_lidar not; x < 100 := 255 (rear half) | mask_camera applied, mask_lidar not; x < 100 cut (Gate 4 frozen setting) | yes |
| occupied / free | class 17 = free; all else occupied | class 17 = free; all else occupied | yes |
| majority pooling | apply_majority_pooling, separate mode (official code) | gate8b.pooling, checked bit-for-bit against the official function | yes |
| metric aggregation | pooled TP/FP/FN over all evaluated samples | pooled TP/FP/FN over all official clips (Gate 6 metric code) | yes |
| metric-scale source | OccAny's own (reconstruction network) | frozen G51-B: calibrated-FOV MoGe-2 median gauge from the five frames | **no** |

Read as a reference comparison only. On Occ3D our raw five-frame completion (28.04 % SC IoU) sits above OccAny's published 23.55 %, and on SemanticKITTI far below it (6.28 % vs 25.91 %); neither number is exact against OccAny's, because OccAny's five frames run *forward* from the target and ours *backward*, and because the metric scale comes from a different source. On SemanticKITTI our five-frame map with the fixed 0.4 m dilation and **no completion** (16.08 %) beats the completion, which is the transfer failure of Section 3 seen again. No number in this section supports a claim over OccAny.

## 6. Geometry-transfer decision per target

Primary setting:

| target | AP/prev | AUROC | AP > 1.5×prev | beats mapper | beats calibrated mapper | beats strongest dilation | beats editable fill | beats matched random | pred/GT vol | geometry |
|---|---|---|---|---|---|---|---|---|---|---|
| SemanticKITTI 08 | 1.62 | 0.5362 | yes | **no** | **no** | **no** | **no** | **no** | 6.25 | FAIL |
| Occ3D-nuScenes val | 2.08 | 0.7051 | yes | yes | yes | yes | yes | yes | 1.91 | **PASS** |
| KITTI-360 (drive 0006) | 1.04 | 0.4928 | **no** | yes | **no** | yes | **no** | yes | 1.43 | FAIL |

Matched five-frame setting (secondary):

| target | AP/prev | AUROC | AP > 1.5×prev | beats mapper | beats calibrated mapper | beats strongest dilation | beats editable fill | beats matched random | pred/GT vol | geometry |
|---|---|---|---|---|---|---|---|---|---|---|
| SemanticKITTI 08 | 1.32 | 0.4735 | **no** | **no** | **no** | **no** | **no** | **no** | 6.69 | FAIL |
| Occ3D-nuScenes val | 2.05 | 0.6976 | yes | yes | yes | yes | yes | yes | 2.02 | FAIL |
| KITTI-360 (drive 0006) | 1.07 | 0.5045 | **no** | yes | **no** | yes | **no** | yes | 1.68 | FAIL |

On Occ3D in the matched setting the completion clears every declared baseline with paired intervals excluding zero (e.g. vs matched-density random +0.0888 [+0.0795, +0.0986], vs the 0.4 m dilation +0.0621 [+0.0493, +0.0757]) and fails the mechanical rule only on the occupied-volume ratio, 2.02 against the 2.0 line; the streaming setting, at 1.91, passes it. The rule is applied as written rather than bent.

## 7. Semantic-transfer decision per target

Primary setting:

| target | completion SSC mIoU | strongest non-completion baseline | beats it (CI > 0) | new-TP semantic acc. | carries information | semantics |
|---|---|---|---|---|---|---|
| SemanticKITTI 08 | 0.0220 | frozen_5frame_dil (0.0511) | **no** | 0.2918 | yes | FAIL |
| Occ3D-nuScenes val | 0.0505 | mapper_dilate (0.0541) | **no** | 0.3676 | yes | FAIL |
| KITTI-360 (drive 0006) | 0.0173 | frozen_5frame_dil (0.0266) | **no** | 0.2995 | yes | FAIL |

### Semantics, primary setting

**SemanticKITTI 08**

| method | SSC mIoU | SC IoU | TP-cond. sem. acc. | coverage miss | naming error | Δ SSC mIoU (paired 95% CI) |
|---|---|---|---|---|---|---|
| completion | 0.0220 | 0.0722 | 0.3549 | 0.5117 | 0.3150 | — |
| mapper_native | 0.0261 | 0.0964 | 0.5244 | 0.8668 | 0.0634 | -0.0042 [-0.0061, -0.0023] |
| mapper_dilate | 0.0320 | 0.1435 | 0.4793 | 0.6416 | 0.1866 | -0.0101 [-0.0130, -0.0077] |
| mapper_calibrated | 0.0230 | 0.0782 | 0.1071 | 0.0000 | 0.8929 | -0.0010 [-0.0030, +0.0009] |
| frozen_5frame_raw | 0.0281 | 0.0670 | 0.6326 | 0.9234 | 0.0281 | -0.0061 [-0.0097, -0.0011] |
| frozen_5frame_dil | 0.0511 | 0.1600 | 0.5592 | 0.6978 | 0.1332 | -0.0292 [-0.0321, -0.0230] |

TP-conditioned semantic accuracy of the completion: **all** true positives 0.3549; **causally observed** true positives 0.5242 (n = 2,486,678); **newly completed** true positives 0.2918 (n = 6,675,477).

**Occ3D-nuScenes val**

| method | SSC mIoU | SC IoU | TP-cond. sem. acc. | coverage miss | naming error | Δ SSC mIoU (paired 95% CI) |
|---|---|---|---|---|---|---|
| completion | 0.0505 | 0.2955 | 0.3873 | 0.3357 | 0.4070 | — |
| mapper_native | 0.0354 | 0.1170 | 0.4634 | 0.8734 | 0.0680 | +0.0151 [+0.0138, +0.0165] |
| mapper_dilate | 0.0541 | 0.2309 | 0.4643 | 0.7101 | 0.1553 | -0.0036 [-0.0063, -0.0011] |
| mapper_calibrated | 0.0322 | 0.2278 | 0.0859 | 0.0217 | 0.8943 | +0.0183 [+0.0168, +0.0197] |
| frozen_5frame_raw | 0.0245 | 0.0793 | 0.4483 | 0.9168 | 0.0459 | +0.0260 [+0.0225, +0.0295] |
| frozen_5frame_dil | 0.0480 | 0.2094 | 0.4431 | 0.7501 | 0.1392 | +0.0025 [-0.0012, +0.0060] |

TP-conditioned semantic accuracy of the completion: **all** true positives 0.3873; **causally observed** true positives 0.4718 (n = 1,503,564); **newly completed** true positives 0.3676 (n = 6,439,881).

**KITTI-360**

| method | SSC mIoU | SC IoU | TP-cond. sem. acc. | coverage miss | naming error | Δ SSC mIoU (paired 95% CI) |
|---|---|---|---|---|---|---|
| completion | 0.0173 | 0.1873 | 0.3098 | 0.6165 | 0.2647 | — |
| mapper_native | 0.0072 | 0.0360 | 0.4023 | 0.9604 | 0.0237 | +0.0100 [+0.0092, +0.0108] |
| mapper_dilate | 0.0189 | 0.1094 | 0.4219 | 0.8492 | 0.0871 | -0.0016 [-0.0025, -0.0007] |
| mapper_calibrated | 0.0070 | 0.2509 | 0.0377 | 0.0000 | 0.9623 | +0.0103 [+0.0094, +0.0110] |
| frozen_5frame_raw | 0.0088 | 0.0363 | 0.6245 | 0.9614 | 0.0145 | +0.0085 [+0.0074, +0.0094] |
| frozen_5frame_dil | 0.0266 | 0.1321 | 0.5959 | 0.8271 | 0.0699 | -0.0093 [-0.0104, -0.0081] |

TP-conditioned semantic accuracy of the completion: **all** true positives 0.3098; **causally observed** true positives 0.4017 (n = 5,346,911); **newly completed** true positives 0.2995 (n = 47,511,997).

### Classwise IoU, primary setting

SemanticKITTI 08:

| class | completion | mapper_native | mapper_dilate | frozen_5frame_dil |
|---|---|---|---|---|
| bicycle | 0.0188 | 0.0188 | 0.0239 | 0.0402 |
| bicyclist | 0.0043 | 0.0039 | 0.0189 | 0.0308 |
| building | 0.0048 | 0.0248 | 0.0344 | 0.0420 |
| car | 0.0397 | 0.0554 | 0.0679 | 0.0991 |
| fence | 0.0127 | 0.0127 | 0.0202 | 0.0289 |
| motorcycle | 0.0109 | 0.0115 | 0.0131 | 0.0590 |
| motorcyclist | 0.0000 | 0.0000 | 0.0000 | 0.0470 |
| other-ground | 0.0004 | 0.0010 | 0.0005 | 0.0007 |
| other-vehicle | 0.0260 | 0.0258 | 0.0224 | 0.0345 |
| parking | 0.0100 | 0.0101 | 0.0101 | 0.0126 |
| person | 0.0124 | 0.0119 | 0.0180 | 0.0258 |
| pole | 0.0064 | 0.0064 | 0.0058 | 0.0125 |
| road | 0.1450 | 0.1808 | 0.1837 | 0.2727 |
| sidewalk | 0.0495 | 0.0489 | 0.0712 | 0.0931 |
| terrain | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| traffic-sign | 0.0063 | 0.0061 | 0.0045 | 0.0135 |
| truck | 0.0331 | 0.0314 | 0.0372 | 0.0595 |
| trunk | 0.0090 | 0.0089 | 0.0103 | 0.0179 |
| vegetation | 0.0279 | 0.0378 | 0.0666 | 0.0813 |

Occ3D-nuScenes val:

| class | completion | mapper_native | mapper_dilate | frozen_5frame_dil |
|---|---|---|---|---|
| barrier | 0.0287 | 0.0267 | 0.0234 | 0.0188 |
| bicycle | 0.0235 | 0.0228 | 0.0252 | 0.0175 |
| bus | 0.0573 | 0.0572 | 0.0718 | 0.0737 |
| car | 0.0864 | 0.0725 | 0.0940 | 0.0943 |
| construction_vehicle | 0.0233 | 0.0231 | 0.0330 | 0.0287 |
| driveable_surface | 0.3355 | 0.1018 | 0.2409 | 0.1894 |
| manmade | 0.0121 | 0.0070 | 0.0147 | 0.0140 |
| motorcycle | 0.0416 | 0.0472 | 0.0706 | 0.0498 |
| other_flat | 0.0140 | 0.0090 | 0.0078 | 0.0080 |
| others | 0.0006 | 0.0006 | 0.0006 | 0.0005 |
| pedestrian | 0.0137 | 0.0125 | 0.0117 | 0.0113 |
| sidewalk | 0.0491 | 0.0541 | 0.1024 | 0.0808 |
| terrain | 0.0015 | 0.0020 | 0.0043 | 0.0047 |
| traffic_cone | 0.0323 | 0.0315 | 0.0376 | 0.0314 |
| trailer | 0.0173 | 0.0181 | 0.0187 | 0.0232 |
| truck | 0.0596 | 0.0620 | 0.0728 | 0.0819 |
| vegetation | 0.0622 | 0.0536 | 0.0910 | 0.0878 |

KITTI-360:

| class | completion | mapper_native | mapper_dilate | frozen_5frame_dil |
|---|---|---|---|---|
| bicycle | 0.0021 | 0.0022 | 0.0074 | 0.0099 |
| building | 0.0191 | 0.0124 | 0.0396 | 0.0761 |
| car | 0.0139 | 0.0160 | 0.0401 | 0.0553 |
| fence | 0.0090 | 0.0059 | 0.0128 | 0.0191 |
| motorcycle | 0.0009 | 0.0008 | 0.0058 | 0.0067 |
| other-ground | 0.0117 | 0.0037 | 0.0065 | 0.0072 |
| other-object | 0.0019 | 0.0022 | 0.0059 | 0.0052 |
| other-structure | 0.0120 | 0.0089 | 0.0169 | 0.0236 |
| other-vehicle | 0.0049 | 0.0048 | 0.0102 | 0.0123 |
| parking | 0.0011 | 0.0010 | 0.0040 | 0.0098 |
| person | 0.0061 | 0.0060 | 0.0127 | 0.0139 |
| pole | 0.0025 | 0.0027 | 0.0055 | 0.0070 |
| road | 0.1011 | 0.0184 | 0.0592 | 0.0494 |
| sidewalk | 0.0173 | 0.0056 | 0.0158 | 0.0262 |
| terrain | 0.0003 | 0.0003 | 0.0010 | 0.0006 |
| traffic-sign | 0.0012 | 0.0011 | 0.0036 | 0.0044 |
| truck | 0.0304 | 0.0200 | 0.0343 | 0.0421 |
| vegetation | 0.0753 | 0.0180 | 0.0582 | 0.1096 |

### The teacher itself

Future-Trident target accuracy against the benchmark's semantic ground truth on future-visible
GT-occupied voxels (evaluation-only use of semantic labels):

| target | anchors | teacher accuracy, all future-visible GT-occupied voxels | …on the causally observed subset | …on the newly visible subset |
|---|---|---|---|---|
| SemanticKITTI 08 | 162 | 0.4290 (n = 2,781,029) | 0.4651 (n = 815,380) | 0.4140 (n = 1,965,649) |
| Occ3D-nuScenes val | 1032 | 0.4113 (n = 959,573) | 0.4442 (n = 495,862) | 0.3762 (n = 463,711) |
| KITTI-360 (drive 0006) | 1748 | 0.2971 (n = 5,597,229) | 0.3315 (n = 922,433) | 0.2903 (n = 4,674,796) |

## 8. Runtime, memory and training cost

| fold | setting | training wall | training peak GPU | target eval wall | eval peak GPU | anchors / clips |
|---|---|---|---|---|---|---|
| SemanticKITTI 08 | stream | 21.5 min | 6.22 GiB | 1.6 min | 13.2 GiB | 163 |
| SemanticKITTI 08 | clips | 21.5 min | 6.22 GiB | 2.0 min | 2.6 GiB | 163 |
| Occ3D-nuScenes val | stream | 21.1 min | 4.84 GiB | 15.8 min | 7.9 GiB | 1182 |
| Occ3D-nuScenes val | clips | 21.1 min | 4.84 GiB | 16.7 min | 6.5 GiB | 1182 |
| KITTI-360 (drive 0006) | stream | 25.8 min | 6.22 GiB | 9.2 min | 14.9 GiB | 1753 |
| KITTI-360 (drive 0006) | clips | 25.8 min | 6.22 GiB | 24.3 min | 2.6 GiB | 1753 |

Caches built for the new source `k360_train` (drives 0003/0007/0010, 1 385 stream frames, 1 289
labelled anchors): targets 6 min (201 MB by byte-range), LingBot stream 0.9–2.1 min per drive at
10 FPS, MoGe-B and scale candidates on the same pass, Trident-H ≈ 1.6–1.8 s/frame on two GPUs, samples
≈ 0.5 s each.

## 9. Reproduction

```bash
cd /home/minh/workspace/lingbot-map_fork
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1 GATE8_GPUS="1 2 3"
PY=/home/minh/anaconda3/envs/cu128/bin/python
$PY tools/gate8b/stage0.py
$PY tools/gate8b/fetch_k360_labels.py --drives 0003 0007 0010
tools/gate8b/run_caches.sh
$PY tools/gate8b/build_samples.py --source kitti360 --anchor-stride 10
tools/gate8b/run_all.sh                      # remaining samples, then both folds concurrently
$PY tools/gate8b/eval_target.py --fold kitti360 --setting clips
for D in semantickitti occ3d kitti360; do $PY tools/gate8b/teacher_accuracy.py --dataset $D; done
$PY tools/gate8b/aggregate.py && $PY tools/gate8b/figures.py && $PY tools/gate8b/report.py --write
$PY -m pytest tests/gate6 tests/gate7a tests/gate7b tests/gate8 tests/gate8a tests/gate8b -q
```

Seeds: 0 everywhere (training, crop and source sampling, the bootstrap); random editable baselines use
seeds 0–4 with a per-clip offset. Hashes of every selected checkpoint and every Gate 8B config are in
`artifacts/gate8b/gate8b_manifest.json`; the frozen surface is hashed in
`artifacts/gate8b/stage0_audit.json`.

## 10. Tests

`python -m pytest tests/gate6 tests/gate7a tests/gate7b tests/gate8 tests/gate8a tests/gate8b -q`

```
256 passed, 2 warnings in 117.45s (0:01:57)
```

239 prior tests (Gate 6: 63, Gate 7A: 68, Gate 7B: 56, Gate 8: 25, Gate 8A: 27) plus 17 new
Gate 8B tests, run after every artifact was produced; nothing skipped on this machine and nothing regressed after the additive edits to
`gate8/sources.py`, `gate6/targets.py`, `sscbench_kitti360/adapter.py` and
`tools/gate8a/evaluate.py`.

The Gate 8B tests that carry the gate's claims:

* `test_every_fold_excludes_its_target_from_train_and_val`,
  `test_fold_configs_match_gate8a_cell_b_except_sources` — each fold's train and validation
  lists never resolve to the target dataset, and the configs differ from Gate 8A's cell B only
  in those lists.
* `test_kitti360_train_and_val_partitions_are_disjoint_and_official`,
  `test_k360_train_frames_never_come_from_the_val_drive` — drives 0003/0007/0010 are official
  SSCBench train drives, 0006 is the official val drive, and no training frame path or record
  names the val drive.
* `test_cached_samples_record_disjoint_input_and_target_frames` (both `k360_train` and
  `kitti360`) — every cached sample's input frames are 0..t and its target frames t+1..t+20.
* `test_k360_train_sample_targets_are_binary_only` — the KITTI-360 geometry target carries no
  class label.
* `test_fold_sampler_draws_each_source_with_probability_one_half`.
* `test_clip_feed_builds_a_fresh_map_per_clip_with_no_carry_over`,
  `test_clip_scale_is_the_pinned_g51b_scalar` — the matched setting keeps no map or scale state
  between clips and uses the frozen five-frame gauge, not MoGe at clip time.
* `test_pooling_matches_the_official_occany_function` — dilation, vote and semantic modes agree
  bit-for-bit with `apply_majority_pooling` run in the OccAny environment;
  `test_occany_geometry_default_is_a_max_pool_not_a_vote`.
* `test_target_evaluator_takes_no_checkpoint_or_threshold_argument`,
  `test_no_target_data_in_the_selection_path`, `test_registry_does_not_change_existing_source_paths`,
  `test_target_loader_default_is_the_validation_drive`, `test_gate8a_artifacts_are_untouched`.

## 11. Deviations and failures

1. **Four additive edits to files Gate 8A treated as frozen, each behaviour-preserving for
   every existing caller.** (a) `gate8/sources.py` gained a `register_source` hook so a later
   gate can add a source without touching the existing branches; every pre-existing path and
   segment builder is unchanged and a test asserts it. (b) `sscbench_kitti360/adapter.py::
   load_target` gained a `sequence` argument defaulting to the validation drive, and (c)
   `gate6/targets.py` passes `rec.get("sequence", <val drive>)` -- every official validation
   record already carries the val drive, so Gates 5.2-8A resolve byte-identically (Gate 6 and
   Gate 8 tests pass unchanged). (d) `tools/gate8a/evaluate.py::run` gained an `art` output
   directory argument defaulting to the Gate 8A root, so Gate 8B could reuse the same
   streaming evaluator without writing into Gate 8A's artifacts.

2. **KITTI-360 training partition = official SSCBench train drives 0003, 0007, 0010; source-
   validation = the official val drive 0006.** All seven official train drives have raw images
   locally; the three chosen give 1 289 labelled anchors, matching `sk_train` (1 656) and
   `occ3d_train` (1 570) in size and keeping every stream inside the RoPE budget at keyframe
   interval 1. Their occupancy targets (`_1_1.npy`) were fetched from the official archive by
   byte-range reads (201 MB, 364 requests, 6 min); the drive poses came from the local
   `data_poses.zip`. The stream is defined by the pose file alone (every 5th SSCBench index);
   a target only marks an anchor.

3. **Equal per-source sampling probability** (p = 1/2 per training source, uniform within) is
   the brief's rule and differs from Gate 8A's uniform-over-files draw, which weighted a
   source by its sample count. Recorded so the two data draws are not conflated.

4. **Balanced within-fold validation set.** Gate 8A's trainer took the first 120 files of the
   concatenated validation list, which -- by construction of that list -- was SemanticKITTI
   only. With two validation domains per fold the same rule would silently validate on one
   domain, so each fold takes the first 60 files of *each* source-validation domain. This
   affects only which of {best, last} is called `best`; the cross-candidate selection is by
   macro AP on the full grids of both domains, exactly as in Gate 8A.

5. **The matched five-frame setting uses the Gate 5.2/6 per-clip LingBot caches, not the
   causal stream.** That is the point of the setting -- LingBot sees exactly the five frames --
   but it means the depth and poses of a given frame differ from the streaming setting's.
   The scale is the pinned G51-B scalar of every prior gate, forced into `ScaleState` (the
   five per-frame candidates would median to the same value; forcing avoids a float round
   trip). Every clip of all three benchmarks has a valid G51-B scale, so no clip is skipped.

6. **OccAny's "separate" pooling is two different operators depending on a flag.** With
   `is_geometry_only=True` its default is `use_dilation=True`: a 3x3x3 `max_pool3d`, i.e. a
   one-voxel Chebyshev dilation, not a vote. The vote (`use_dilation=False`) never removes
   occupancy and needs > 13.5 of 27 neighbours. In semantic mode the operator relabels
   occupied voxels and leaves occupancy untouched. All three are reproduced and checked
   bit-for-bit against the official function in the OccAny environment; the report gives
   dilation and vote as separate columns and never uses either for the primary result.

7. **The comparison with OccAny is a reference comparison, not an exact one.** OccAny's five
   frames start at the target and run *forward* (SemanticKITTI: offsets 0..+40 at stride 10;
   nuScenes: 5 samples ~1 s apart); ours end at the target and run *backward* (offsets -20..0
   at stride 5; nuScenes 0.5 s apart). The anchor sets, grids, masks, occupancy definition,
   pooling and aggregation match; the temporal sampling and the metric-scale source do not.

8. **Whole-system zero-shot is not claimed for any fold.** LingBot-Map's published training
   mixture includes KITTI-360, so the existing KITTI-360 fold is not whole-system zero-shot;
   MoGe-2's and Trident-H's training data were not audited here. Every fold is described as
   leave-one-dataset-out transfer with no target-domain training or adaptation of the
   completion module.

9. **The teacher-accuracy diagnostic reduces Occ3D through the frozen any-sub-voxel rule**
   (`reduce_occ3d_probs`: mean union probability over evidence-bearing sub-voxels, then
   argmax), exactly as every evaluated prediction is, rather than a majority of sub-voxel
   labels. It is an evaluation-only use of semantic ground truth.

10. **One diagnostic crashed and was re-run.** `teacher_accuracy.py` reduced the causally
    observed mask to the Occ3D evaluation grid only when the anchor had teacher evidence, so an
    anchor with none raised a shape error after the smoke test had passed. The reduction was
    moved outside that condition and the Occ3D and KITTI-360 runs restarted. The diagnostic
    is evaluation-only and touches nothing on the prediction path.

