# Gate 8C-1 — KITTI-360-only training, untouched SemanticKITTI and Occ3D evaluation

> **Addendum 2026-09-04 (post-gate).** The released OccAny checkpoint has since been
> downloaded and run on this machine over all 4 819 official val samples; its published
> number reproduces to 0.01. Every statement below that says it *could not be run* is
> superseded. See **§ Addendum** at the end of this file and
> `occany_reproduction.md`. The gate body is left unedited as the record of what was
> known when the gate closed.


| method | training datasets | target datasets excluded from training | protocol | target | SC IoU % | SSC mIoU % | completion params | latency / frame | training cost |
|---|---|---|---|---|---|---|---|---|---|
| **ours** (completion) | KITTI-360 only (drives 0003/0007/0010) | SemanticKITTI 08, Occ3D-nuScenes val | **causal** | SemanticKITTI 08 | 12.74 ± 1.03 | 2.65 | 0.99 M | 8 + 45 ms | 0.72 GPU-h (3 seeds) |
| OccAny (published) | SemanticKITTI + nuScenes (both targets) | none | non-causal (target-first, forward) | SemanticKITTI 08 | 25.91 | — | ≈1.5 B (MUSt3R + decoder) | — | — |
| **ours** (completion) | KITTI-360 only (drives 0003/0007/0010) | SemanticKITTI 08, Occ3D-nuScenes val | **causal** | Occ3D-nuScenes val | 30.89 ± 0.29 | 4.13 | 0.99 M | 12 + 119 ms | 0.72 GPU-h (3 seeds) |
| OccAny (published) | SemanticKITTI + nuScenes (both targets) | none | non-causal (target-first, forward) | Occ3D-nuScenes val | 23.55 | — | ≈1.5 B (MUSt3R + decoder) | — | — |

**Verdict: FAIL (criteria for MINIMUM CONTINUE not met).** The module does not clear the continue bar on both untouched targets; the table above shows which criterion fails and by how much.

**What this is.** target-domain-free transfer of the completion module: trained and selected on KITTI-360 alone, evaluated on untouched SemanticKITTI and Occ3D without adaptation. NOT whole-system zero-shot: the frozen foundation models (LingBot-Map, MoGe-2, Trident-H) have not had their training provenance audited, and LingBot-Map's published mixture includes KITTI-360.

**What this is not.** Not whole-system zero-shot. The completion module never sees a target
dataset, but the frozen components beneath it have not had their training provenance
audited, and LingBot-Map's published mixture includes KITTI-360. The accurate description
is **target-domain-free transfer of the completion module**.

---

## 1. Decision

| target | our SC IoU % | OccAny published % | fraction of OccAny | strongest non-completion baseline | beats it (CI > 0) | AP/prev | AUROC | pred/GT vol | semantics beat baseline |
|---|---|---|---|---|---|---|---|---|---|
| SemanticKITTI 08 | 12.74 | 25.91 | 49 % | frozen_5frame_dil (16.00) | **no** | 2.54 | 0.6507 | 2.38 | **no** |
| Occ3D-nuScenes val | 30.89 | 23.55 | 131 % | all_valid_occupied (22.95) | yes | 2.31 | 0.7629 | 0.54 | **no** |

| criterion | met |
|---|---|
| match or beat occany on at least one | yes |
| at least 90pct of occany on both | **no** |
| beats strongest baseline on both ci excludes zero | **no** |
| semantic beats baseline on both | **no** |
| ranking ok on both (AP/prev>=1.5, AUROC>=0.65) | yes |
| no volume inflation (pred/GT<=2) | **no** |
| near chance on either | **no** |
| far below occany on both | **no** |

## 2. Source: KITTI-360 with rebuilt supervision

Gate 8C-0 disqualified SSCBench-KITTI-360's `_1_1.npy` completion label — a ground-truth
Velodyne return lands on a voxel it calls *free* 71 % of the time. Gate 8C-1 never opens
it. Occupancy is rebuilt from raw sweeps `t … t+20` with ground-truth poses (target
construction only); endpoints occupied, interiors free, unobserved and cross-sweep-
conflicting voxels unknown and excluded from the loss. Full construction rules, drive and
anchor counts, exclusion reasons and validation are in
[`source_target_audit.md`](source_target_audit.md); all checks pass.

| property | SSCBench `_1_1.npy` (rejected) | rebuilt raw-LiDAR (used) |
|---|---|---|
| supervised endpoints marked OCCUPIED | 0.29 | **1.0000** |
| supervised endpoints marked FREE | 0.62 | **0.0e+00** |
| best voxel shift | (0, 0, +1) — +99 % IoU | **(0, 0, 0)** |
| median surface distance | 0.200 m | **0.000 m** |
| recall vs future sweeps [1, 5, 10, 20] | — | 0.192 → 0.602 → 0.879 → 0.998 |

## 3. Firewall and freezing

Neither target dataset was read before freezing. Training and selection ran inside a file-access audit intercepting `open`, `numpy.load` and `numpy.fromfile`; **0 violations** across all runs (`neither_target_dataset_was_accessed = True`). The evaluator refuses to start without `frozen_manifest.json` and accepts no checkpoint or threshold argument — both come from the manifest. Checkpoint selection used KITTI-360 drive-0006 occupancy AP; the occupancy threshold, drive-0006 IoU; the semantic threshold, agreement with the frozen teacher on voxels where both halves of the future window agree — no human semantic label anywhere.

| seed | checkpoint | sha256 | source AP | AP/prev | AUROC | occupancy τ | semantic τ | teacher agreement | GPU-h | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | last | `5227c67201202478…` | 0.1147 | 1.69 | 0.6336 | -0.1250 | 0.00 | 0.4024 | 0.24 | 4.84 |
| 1 | last | `13640b7664997abb…` | 0.1127 | 1.66 | 0.6333 | -0.0938 | 0.00 | 0.4137 | 0.24 | 4.84 |
| 2 | last | `68e67f28a1e4b764…` | 0.1132 | 1.66 | 0.6323 | -0.1562 | 0.00 | 0.3901 | 0.24 | 4.84 |

## 4. Target results

### SemanticKITTI 08

**A · 5 past frames (causal)** — 163 clips, 3 seeds

| method | SC IoU % | precision % | recall % | AP | AP/prev | AUROC | pred/GT vol | Δ vs completion (paired 95 % CI, median seed) |
|---|---|---|---|---|---|---|---|---|
| **completion (ours, raw)** | 12.74 ± 1.03 | 15.88 | 39.15 | 0.1984 | 2.54 | 0.6507 | 2.38 | — |
| completion + OccAny pooling (3×3×3 max-pool) | 11.60 | 12.93 | 47.21 | — | — | — | 3.65 | +0.0113 [+0.0074, +0.0148] |
| completion + OccAny pooling (3×3×3 vote) | 12.76 | 15.33 | 37.70 | — | — | — | 2.46 | -0.0002 [-0.0008, +0.0003] |
| incremental mapper, no completion | 8.90 | 33.32 | 10.83 | — | — | — | 0.33 | +0.0383 [+0.0304, +0.0472] |
| mapper + fixed 0.4 m dilation | 15.41 | 23.58 | 30.80 | — | — | — | 1.31 | -0.0268 [-0.0364, -0.0162] |
| frozen five-frame G51-B, raw | 6.70 | 34.92 | 7.66 | — | — | — | 0.22 | +0.0604 [+0.0535, +0.0680] |
| frozen five-frame G51-B + 0.4 m dilation | 16.00 | 25.37 | 30.22 | — | — | — | 1.19 | -0.0326 [-0.0424, -0.0223] |
| editable-region fill | 7.85 | 7.86 | 98.86 | — | — | — | 12.57 | +0.0488 [+0.0383, +0.0587] |
| matched-density random | 7.45 | — | — | — | — | — | — | — |
| all-valid-occupied | 7.82 | 7.82 | 100.00 | — | — | — | 12.78 | +0.0491 [+0.0384, +0.0592] |
| OccAny (published, five-frame single camera) | 25.91 | 36.79 | 46.70 | — | — | — | — | *published, not re-run here* |

**A · all-past streaming (causal)** — 163 clips, 3 seeds

| method | SC IoU % | precision % | recall % | AP | AP/prev | AUROC | pred/GT vol | Δ vs completion (paired 95 % CI, median seed) |
|---|---|---|---|---|---|---|---|---|
| **completion (ours, raw)** | 12.83 ± 0.81 | 15.66 | 41.51 | 0.1711 | 2.19 | 0.6373 | 2.65 | — |
| completion + OccAny pooling (3×3×3 max-pool) | 11.49 | 11.85 | 49.01 | — | — | — | 4.14 | +0.0134 [+0.0090, +0.0168] |
| completion + OccAny pooling (3×3×3 vote) | 12.80 | 13.85 | 38.66 | — | — | — | 2.79 | +0.0003 [-0.0006, +0.0010] |
| incremental mapper, no completion | 9.64 | 25.85 | 13.32 | — | — | — | 0.52 | +0.0319 [+0.0191, +0.0403] |
| mapper + fixed 0.4 m dilation | 14.35 | 19.31 | 35.84 | — | — | — | 1.86 | -0.0152 [-0.0289, -0.0067] |
| frozen five-frame G51-B, raw | 6.70 | 34.92 | 7.66 | — | — | — | 0.22 | +0.0613 [+0.0489, +0.0723] |
| frozen five-frame G51-B + 0.4 m dilation | 16.00 | 25.37 | 30.22 | — | — | — | 1.19 | -0.0317 [-0.0434, -0.0196] |
| editable-region fill | 7.86 | 7.88 | 97.59 | — | — | — | 12.39 | +0.0497 [+0.0365, +0.0616] |
| matched-density random | 7.68 | — | — | — | — | — | — | — |
| all-valid-occupied | 7.82 | 7.82 | 100.00 | — | — | — | 12.78 | +0.0501 [+0.0370, +0.0620] |
| OccAny (published, five-frame single camera) | 25.91 | 36.79 | 46.70 | — | — | — | — | *published, not re-run here* |

**B · OccAny forward sampling (**non-causal**)** — 161 clips, 3 seeds

| method | SC IoU % | precision % | recall % | AP | AP/prev | AUROC | pred/GT vol | Δ vs completion (paired 95 % CI, median seed) |
|---|---|---|---|---|---|---|---|---|
| **completion (ours, raw)** | 13.94 ± 1.12 | 17.48 | 42.74 | 0.2006 | 2.54 | 0.6475 | 2.33 | — |
| completion + OccAny pooling (3×3×3 max-pool) | 12.66 | 14.25 | 53.14 | — | — | — | 3.73 | +0.0128 [+0.0088, +0.0172] |
| completion + OccAny pooling (3×3×3 vote) | 13.88 | 17.24 | 41.60 | — | — | — | 2.41 | -0.0002 [-0.0006, +0.0002] |
| incremental mapper, no completion | 9.52 | 32.19 | 11.91 | — | — | — | 0.37 | +0.0442 [+0.0317, +0.0557] |
| mapper + fixed 0.4 m dilation | 16.80 | 23.49 | 37.11 | — | — | — | 1.58 | -0.0286 [-0.0389, -0.0177] |
| frozen five-frame G51-B, raw | 6.70 | 34.92 | 7.66 | — | — | — | 0.22 | +0.0721 [+0.0618, +0.0823] |
| frozen five-frame G51-B + 0.4 m dilation | 16.00 | 25.37 | 30.22 | — | — | — | 1.19 | -0.0213 [-0.0309, -0.0116] |
| editable-region fill | 7.86 | 7.88 | 96.91 | — | — | — | 12.29 | +0.0608 [+0.0494, +0.0726] |
| matched-density random | 7.80 | — | — | — | — | — | — | — |
| all-valid-occupied | 7.89 | 7.89 | 100.00 | — | — | — | 12.67 | +0.0605 [+0.0488, +0.0726] |
| OccAny (published, five-frame single camera) | 25.91 | 36.79 | 46.70 | — | — | — | — | *published, not re-run here* |

### Occ3D-nuScenes val

**A · 5 past frames (causal)** — 1182 clips, 3 seeds

| method | SC IoU % | precision % | recall % | AP | AP/prev | AUROC | pred/GT vol | Δ vs completion (paired 95 % CI, median seed) |
|---|---|---|---|---|---|---|---|---|
| **completion (ours, raw)** | 30.89 ± 0.29 | 67.14 | 36.39 | 0.5313 | 2.31 | 0.7629 | 0.54 | — |
| completion + OccAny pooling (3×3×3 max-pool) | 33.53 | 54.52 | 46.56 | — | — | — | 0.85 | -0.0265 [-0.0312, -0.0218] |
| completion + OccAny pooling (3×3×3 vote) | 30.85 | 66.82 | 36.43 | — | — | — | 0.55 | +0.0004 [+0.0002, +0.0005] |
| incremental mapper, no completion | 9.99 | 65.08 | 10.55 | — | — | — | 0.16 | +0.2090 [+0.1965, +0.2220] |
| mapper + fixed 0.4 m dilation | 21.64 | 57.54 | 25.75 | — | — | — | 0.45 | +0.0925 [+0.0774, +0.1077] |
| frozen five-frame G51-B, raw | 7.93 | 62.89 | 8.32 | — | — | — | 0.13 | +0.2295 [+0.2158, +0.2433] |
| frozen five-frame G51-B + 0.4 m dilation | 20.94 | 56.41 | 24.99 | — | — | — | 0.44 | +0.0994 [+0.0867, +0.1129] |
| editable-region fill | 22.56 | 22.76 | 96.22 | — | — | — | 4.23 | +0.0833 [+0.0671, +0.0990] |
| matched-density random | 13.33 | — | — | — | — | — | — | — |
| all-valid-occupied | 22.95 | 22.95 | 100.00 | — | — | — | 4.36 | +0.0793 [+0.0627, +0.0955] |
| OccAny (published, five-frame single camera) | 23.55 | 36.09 | 40.39 | — | — | — | — | *published, not re-run here* |

**A · all-past streaming (causal)** — 1182 clips, 3 seeds

| method | SC IoU % | precision % | recall % | AP | AP/prev | AUROC | pred/GT vol | Δ vs completion (paired 95 % CI, median seed) |
|---|---|---|---|---|---|---|---|---|
| **completion (ours, raw)** | 31.98 ± 0.44 | 66.26 | 38.04 | 0.5227 | 2.28 | 0.7628 | 0.57 | — |
| completion + OccAny pooling (3×3×3 max-pool) | 33.56 | 53.10 | 47.69 | — | — | — | 0.90 | -0.0226 [-0.0281, -0.0171] |
| completion + OccAny pooling (3×3×3 vote) | 31.91 | 65.84 | 37.17 | — | — | — | 0.56 | +0.0006 [+0.0004, +0.0008] |
| incremental mapper, no completion | 11.70 | 60.52 | 12.66 | — | — | — | 0.21 | +0.2029 [+0.1904, +0.2155] |
| mapper + fixed 0.4 m dilation | 23.09 | 53.16 | 28.99 | — | — | — | 0.55 | +0.0889 [+0.0750, +0.1029] |
| frozen five-frame G51-B, raw | 7.93 | 62.89 | 8.32 | — | — | — | 0.13 | +0.2405 [+0.2267, +0.2535] |
| frozen five-frame G51-B + 0.4 m dilation | 20.94 | 56.41 | 24.99 | — | — | — | 0.44 | +0.1104 [+0.0979, +0.1230] |
| editable-region fill | 22.59 | 22.82 | 95.79 | — | — | — | 4.20 | +0.0939 [+0.0778, +0.1081] |
| matched-density random | 14.27 | — | — | — | — | — | — | — |
| all-valid-occupied | 22.95 | 22.95 | 100.00 | — | — | — | 4.36 | +0.0903 [+0.0739, +0.1051] |
| OccAny (published, five-frame single camera) | 23.55 | 36.09 | 40.39 | — | — | — | — | *published, not re-run here* |

**B · OccAny forward sampling (**non-causal**)** — 882 clips, 3 seeds

| method | SC IoU % | precision % | recall % | AP | AP/prev | AUROC | pred/GT vol | Δ vs completion (paired 95 % CI, median seed) |
|---|---|---|---|---|---|---|---|---|
| **completion (ours, raw)** | 27.56 ± 0.05 | 63.83 | 32.66 | 0.4822 | 2.24 | 0.7322 | 0.51 | — |
| completion + OccAny pooling (3×3×3 max-pool) | 30.63 | 51.57 | 43.00 | — | — | — | 0.83 | -0.0307 [-0.0355, -0.0259] |
| completion + OccAny pooling (3×3×3 vote) | 27.52 | 63.50 | 32.69 | — | — | — | 0.51 | +0.0003 [+0.0002, +0.0004] |
| incremental mapper, no completion | 7.17 | 60.01 | 7.53 | — | — | — | 0.13 | +0.2039 [+0.1914, +0.2167] |
| mapper + fixed 0.4 m dilation | 18.49 | 55.68 | 21.68 | — | — | — | 0.39 | +0.0906 [+0.0764, +0.1053] |
| frozen five-frame G51-B, raw | 7.93 | 62.89 | 8.32 | — | — | — | 0.13 | +0.1980 [+0.1847, +0.2119] |
| frozen five-frame G51-B + 0.4 m dilation | 20.94 | 56.41 | 24.99 | — | — | — | 0.44 | +0.0734 [+0.0593, +0.0883] |
| editable-region fill | 20.74 | 21.00 | 94.34 | — | — | — | 4.49 | +0.0682 [+0.0528, +0.0836] |
| matched-density random | 10.96 | — | — | — | — | — | — | — |
| all-valid-occupied | 21.52 | 21.52 | 100.00 | — | — | — | 4.65 | +0.0604 [+0.0446, +0.0760] |
| OccAny (published, five-frame single camera) | 23.55 | 36.09 | 40.39 | — | — | — | — | *published, not re-run here* |

## 5. OccAny comparison

The released OccAny checkpoint **could not be run here** — its model stack is missing `croco`, `dust3r`, `mast3r`, `depth_anything_3`, `diffusers` and `torchsparse` (two needing CUDA builds), plus 6.4 GB of weights and a GroundingDINO box-extraction pass over every evaluation clip. Its **official evaluator** (`SSCMetrics.get_score_completion`) and **official pooling** (`apply_majority_pooling`, separate mode) do run, and are used on our predictions: their `get_score_completion` was run on 193 of our real Occ3D predictions and returned **byte-identical TP/FP/FN** to our own counting, and Gate 8B verified the pooling bit-for-bit inside the OccAny environment.

**The published 23.55 % is post-processed.** `sh/compute_metric.sh` sets `USE_MAJORITY_POOLING=1` by default and the nuScenes 5-frame geometry entry passes `--geometry_only`, so their number carries a 3x3x3 max-pool that our raw number does not. Matching both the post-processing and their forward temporal sampling puts us at **30.51 %** against their 23.55 %; matching only the post-processing, on our causal protocol, **33.43 %**. Our subsample was checked for bias: prevalence 0.2295 over our 1 182 anchors against 0.2270 over all 6 019 val frames. Full audit in `occany_comparability.json`. Note the documented trap: OccAny's geometry-only pooling default is a 3×3×3 **max-pool** (a one-voxel dilation), not a majority vote — both are reported.

**Protocol difference.** OccAny's published five-frame setting is target-first and *forward-looking* (SemanticKITTI: the target is view 0 and the other four follow it at stride 5; nuScenes: five samples forward at interval 2). Our Protocol A is the mirror image — five frames **ending** at the target, no future observation. Protocol B reproduces their forward sampling for comparison only and is labelled non-causal throughout.

Full detail, including the exact dependency failures and what would be needed to run the
released model, is in [`occany_reproduction.md`](occany_reproduction.md).

## 6. Semantics (diagnostic)

### SemanticKITTI 08

| method | SSC mIoU % | SC IoU % | TP-cond. naming acc. | coverage miss | naming error | Δ SSC mIoU vs completion (paired 95 % CI) |
|---|---|---|---|---|---|---|
| completion | 2.65 | 12.32 | 0.3948 | 0.6296 | 0.2241 | — |
| mapper_native | 2.80 | 8.90 | 0.5648 | 0.8917 | 0.0471 | -0.0015 [-0.0031, +0.0000] |
| mapper_dilate | 3.82 | 15.41 | 0.5112 | 0.6920 | 0.1505 | -0.0117 [-0.0132, -0.0098] |
| frozen_5frame_raw | 2.81 | 6.70 | 0.6326 | 0.9234 | 0.0281 | -0.0016 [-0.0043, +0.0016] |
| frozen_5frame_dil | 5.11 | 16.00 | 0.5592 | 0.6978 | 0.1332 | -0.0246 [-0.0270, -0.0202] |

TP-conditioned naming accuracy: **all** true positives 0.3948; **observed** true positives 0.5644 (n = 1,988,856); **newly completed** true positives 0.3269 (n = 4,960,022).

### Occ3D-nuScenes val

| method | SSC mIoU % | SC IoU % | TP-cond. naming acc. | coverage miss | naming error | Δ SSC mIoU vs completion (paired 95 % CI) |
|---|---|---|---|---|---|---|
| completion | 4.42 | 30.89 | 0.3772 | 0.6361 | 0.2266 | — |
| mapper_native | 3.31 | 9.99 | 0.4696 | 0.8945 | 0.0560 | +0.0082 [+0.0073, +0.0091] |
| mapper_dilate | 5.42 | 21.64 | 0.4715 | 0.7425 | 0.1361 | -0.0129 [-0.0153, -0.0104] |
| frozen_5frame_raw | 2.45 | 7.93 | 0.4483 | 0.9168 | 0.0459 | +0.0168 [+0.0135, +0.0200] |
| frozen_5frame_dil | 4.80 | 20.94 | 0.4431 | 0.7501 | 0.1392 | -0.0067 [-0.0103, -0.0033] |

TP-conditioned naming accuracy: **all** true positives 0.3772; **observed** true positives 0.4804 (n = 1,245,836); **newly completed** true positives 0.3358 (n = 3,105,188).

### Classwise IoU — SemanticKITTI 08

| class | completion | mapper_native | mapper_dilate | frozen_5frame_dil |
|---|---|---|---|---|
| bicycle | 1.95 | 1.94 | 2.47 | 4.02 |
| bicyclist | 0.54 | 0.56 | 2.63 | 3.08 |
| building | 0.44 | 2.40 | 4.13 | 4.20 |
| car | 6.04 | 5.74 | 7.95 | 9.91 |
| fence | 1.28 | 1.34 | 2.42 | 2.89 |
| motorcycle | 1.49 | 1.52 | 2.43 | 5.90 |
| motorcyclist | 0.00 | 0.00 | 0.00 | 4.70 |
| other-ground | 0.16 | 0.15 | 0.07 | 0.07 |
| other-vehicle | 2.94 | 3.08 | 3.09 | 3.45 |
| parking | 1.05 | 1.05 | 1.18 | 1.26 |
| person | 1.26 | 1.27 | 2.13 | 2.58 |
| pole | 0.78 | 0.82 | 0.76 | 1.25 |
| road | 16.05 | 19.47 | 21.84 | 27.27 |
| sidewalk | 4.95 | 4.94 | 7.60 | 9.31 |
| terrain | 0.00 | 0.00 | 0.00 | 0.00 |
| traffic-sign | 0.69 | 0.73 | 0.63 | 1.35 |
| truck | 3.60 | 3.67 | 4.80 | 5.95 |
| trunk | 0.99 | 1.04 | 1.25 | 1.79 |
| vegetation | 6.13 | 3.46 | 7.14 | 8.13 |

### Classwise IoU — Occ3D-nuScenes val

| class | completion | mapper_native | mapper_dilate | frozen_5frame_dil |
|---|---|---|---|---|
| barrier | 2.49 | 2.42 | 2.35 | 1.88 |
| bicycle | 2.19 | 2.16 | 2.47 | 1.75 |
| bus | 6.19 | 6.24 | 8.87 | 7.37 |
| car | 6.27 | 6.92 | 10.09 | 9.43 |
| construction_vehicle | 2.21 | 2.21 | 3.52 | 2.87 |
| driveable_surface | 23.45 | 8.81 | 22.40 | 18.94 |
| manmade | 2.71 | 0.59 | 1.28 | 1.40 |
| motorcycle | 4.24 | 4.53 | 6.98 | 4.98 |
| other_flat | 1.06 | 1.04 | 0.91 | 0.80 |
| others | 0.07 | 0.07 | 0.07 | 0.05 |
| pedestrian | 1.33 | 1.30 | 1.32 | 1.13 |
| sidewalk | 3.88 | 4.30 | 9.11 | 8.08 |
| terrain | 0.13 | 0.16 | 0.38 | 0.47 |
| traffic_cone | 3.19 | 3.14 | 3.90 | 3.14 |
| trailer | 1.52 | 1.58 | 1.96 | 2.32 |
| truck | 6.23 | 6.32 | 8.31 | 8.19 |
| vegetation | 7.94 | 4.45 | 8.18 | 8.78 |

## 7. Efficiency

| quantity | value |
|---|---|
| trainable completion parameters | 986,114 (0.99 M) |
| total inference parameters | ≈1.71 B frozen (LingBot-Map + MoGe-2 + Trident-H) + 0.99 M trained |
| training GPU-hours (3 seeds) | 0.72 |
| training peak GPU | 4.84 GiB |
| per-frame mapper latency (SemanticKITTI 08) | 8.2 ms (p95 9.5) |
| per-frame mapper latency (Occ3D-nuScenes val) | 11.5 ms (p95 13.4) |
| completion latency (SemanticKITTI 08) | 45.0 ms (p95 45.1) |
| completion latency (Occ3D-nuScenes val) | 119.0 ms (p95 119.1) |
| evaluation peak GPU (SemanticKITTI 08) | 2.4 GiB |
| evaluation peak GPU (Occ3D-nuScenes val) | 6.0 GiB |

## 8. Reproduction

```bash
cd /home/minh/workspace/lingbot-map_fork
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1 GATE8_GPUS="1 2 3"
PY=/home/minh/anaconda3/envs/cu128/bin/python

tools/gate8c1/run_targets.sh                      # rebuild raw-LiDAR supervision
$PY tools/gate8c1/validate_targets.py             # the pre-training validation gate
tools/gate8c1/run_samples.sh                      # causal input + rebuilt target pairs
tools/gate8c1/run_train.sh                        # seeds 0,1,2 concurrently
$PY tools/gate8c1/selection.py                    # KITTI-360 drive 0006 only
$PY tools/gate8c1/freeze_manifest.py              # <- the firewall lifts here
for D in semantickitti occ3d; do for M in past5 stream occany_fwd; do for S in 0 1 2; do
  $PY tools/gate8c1/eval_target.py --dataset $D --mode $M --seed $S; done; done; done
$PY tools/gate8c1/aggregate.py && $PY tools/gate8c1/write_audit.py && $PY tools/gate8c1/report.py --write
$PY -m pytest tests/gate8c0 tests/gate8c1 tests/gate8 tests/gate8a -q
```

Seeds 0, 1, 2; code commit `9a648322e6dc`. Every checkpoint hash, threshold, source-cache
digest and firewall audit is in [`frozen_manifest.json`](frozen_manifest.json).

## 9. Tests

`python -m pytest tests/gate8c0 tests/gate8c1 tests/gate8 tests/gate8a tests/gate8b -q`

```
120 passed, 12 warnings in 11.78s
```

30 new Gate 8C-1 tests plus the Gate 8C-0, 8, 8A and 8B suites. The warnings are NumPy's
`np.fromstring` deprecation inside the frozen `sscbench_kitti360.adapter.parse_calibration`;
none originates in Gate 8C-1 code.

**Firewall — the brief's central requirement.** Five tests are parameterised over every
pre-firewall tool (`build_targets`, `build_samples`, `train`, `selection`,
`validate_targets`):

* `test_no_target_dataset_path_or_identifier_before_the_firewall` — no SemanticKITTI,
  Occ3D/nuScenes or SSCBench-label path or identifier appears in the source of any of them;
* `test_pre_firewall_modules_do_not_import_target_loaders` — nor in their import closure,
  so a target loader cannot arrive transitively;
* `test_sscbench_completion_labels_are_never_read` — `_1_1.npy` is unreachable from the
  Gate 8C-1 target path, the label Gate 8C-0 disqualified;
* `test_firewall_predicate_rejects_target_paths` and `test_runtime_file_audit_actually_fires`
  — the predicate rejects each forbidden pattern, and the runtime audit that wrapped
  training and selection genuinely raises when a forbidden path is opened (a passing audit
  with a broken interceptor would prove nothing);
* `test_target_evaluator_refuses_without_a_frozen_manifest` — the evaluator takes no
  checkpoint or threshold argument and will not start without `frozen_manifest.json`.

**Supervision.** `test_rebuilt_target_agrees_with_its_own_sweep_by_construction`,
`test_rebuilt_target_has_no_systematic_one_voxel_offset` (the +1 z test SSCBench failed at
+99 % IoU), `test_occupancy_recall_grows_with_future_sweeps`,
`test_conflicting_evidence_becomes_unknown_not_a_vote`,
`test_a_sweep_never_carves_the_voxel_it_measured` (deviation 2, asserted rather than
assumed).

**Causality and frozen method.**
`test_cached_samples_are_causal_and_use_only_future_frames_for_targets`,
`test_ground_truth_poses_and_future_lidar_are_absent_from_the_inference_path`,
`test_scale_is_fixed_once_and_applied_identically_to_depth_and_translation`,
`test_each_frame_is_integrated_exactly_once`,
`test_all_three_seeds_share_one_configuration_except_seed`,
`test_training_drives_and_validation_drive_are_disjoint`,
`test_protocol_offsets_are_causal_or_explicitly_labelled_non_causal`.

**OccAny honesty.** `test_occany_official_metric_agrees_with_our_counting` (their evaluator
against ours, exactly), `test_occany_pooling_default_is_a_max_pool_not_a_vote`,
`test_published_occany_numbers_are_quoted_not_recomputed`.

**Prior work.** `test_gate8c0_artifacts_remain_unchanged`.

## 10. Deviations and limitations

1. **The released OccAny model could not be re-run.** Its stack is missing `croco`,
   `dust3r`, `mast3r`, `depth_anything_3`, `diffusers` and `torchsparse`; two build CUDA
   extensions. Running it would also need 6.4 GB of weights and a GroundingDINO box pass
   over all 1 345 evaluation clips, then a generative diffusion model over them — several
   GPU-hours plus install risk, outside a gate whose instruction is to stop after
   reporting. **This was not attempted rather than attempted-and-failed**, and the
   comparison is therefore *published numbers vs ours*, with OccAny's own evaluator and
   pooling applied to our predictions. `occany_reproduction.md` records exactly what is
   missing and what would be required.

2. **A within-sweep carving exception, stated explicitly.** The brief's rule 8 sends
   occupied-and-free voxels to unknown. Applied naively that also fires *within* a single
   sweep, where a grazing ray passes through a voxel another ray of the same sweep
   terminated in; that made ~73 % of surface voxels self-conflicting and collapsed the
   target. A sweep therefore never carves a voxel it measured itself — standard occupancy
   mapping, and not a conflict resolution, since a measured return *is* a direct
   observation of occupancy. Genuine cross-sweep disagreement (dynamic objects, thin
   structure, occlusion boundaries) is still left unknown, at ~8 % of touched voxels.

3. **The future window is 20 *stream* frames, not 20 native frames.** Gate 8's privileged
   horizon is frozen at 20 frames of the stream, which for KITTI-360 is stride 5 in native
   frames — about 100 native frames, ~10 s. Gate 8C-0 showed the sensor leaves the 51.2 m
   box after 11–20 stream frames, so this is the horizon that actually covers the volume.
   A 20-*native*-frame window would cover only a fifth of it.

4. **Anchors are all eligible *stream* frames, not all raw frames.** The frozen LingBot
   stream for these drives is sampled at SSCBench-index stride 5, and re-streaming at
   stride 1 would change a frozen cache. "All eligible raw frames" is therefore realised as
   all eligible frames of the frozen stream: 1 358 training anchors versus the 1 276 that
   SSCBench labels would have allowed, with the three mechanical exclusions (scale warm-up,
   future-window availability, pose availability) recorded per drive in the audit.

5. **Drive 0006 targets are built at anchor stride 3** (590 of 1 768 eligible anchors) to
   bound build cost. The subsample is deterministic and was fixed before any model existed,
   so it cannot have been chosen against a result.

6. **`tools/gate8c1/select.py` had to be renamed `selection.py`.** Python puts a script's
   own directory first on `sys.path`, so a module named `select` shadows the standard
   library's and breaks `subprocess` for every other tool in that directory. It cost one
   failed validation-sample build. The same trap was recorded in Gate 8A.

7. **One crash during the validation-sample build**, fixed and re-run: the per-frame
   `T_cam_to_grid` is populated only for official SSCBench anchor frames, and Gate 8C-1
   deliberately anchors on frames that are not. For KITTI-360 that transform is a per-drive
   calibration constant, so it is now read from `DriveGeometry.rect_cam_to_velo`; it is
   byte-identical to the per-frame value on every frame that had one, so the 26 samples
   written before the fix remain valid.

8. **`np.fromstring` deprecation warning** from the frozen
   `sscbench_kitti360.adapter.parse_calibration`. Pre-existing, out of scope, parses
   correctly.



---

# Addendum — released OccAny checkpoint, run here (2026-09-04)

Added after the gate closed. Nothing in the gate body above was edited.

## Reproduction, all 4 819 official Occ3D-nuScenes val samples

Run `OccAny_5frames_nuscenes512_rot60_vpi10_fwd3_sTrans2` (EXP_ID 2 recipe, geometry-only,
`--recon_threshold 1.1`), scored by **their own** `compute_metrics_from_saved_voxels.py`:

| | precision | recall | SC IoU |
|---|---|---|---|
| with pooling (`USE_MAJORITY_POOLING=1`, their default) | 36.11 | 40.39 | **23.56** |
| **published** | 36.09 | 40.39 | **23.55** |
| raw, no pooling | 42.67 | 28.61 | **20.67** |

This settles the question §"The published 23.55 % is post-processed" raised: it is, and
**OccAny's raw geometry number is 20.67 %**.

Cross-checks: our scoring of their saved voxels over all 4 819 returns exactly
42.67 / 28.61 / 20.67, identical to their script; their stored `voxel_label` is
byte-identical to our target on every matched sample.

## Head-to-head, 882 samples both methods evaluate

Both scored by OccAny's evaluator, both using OccAny's own **pool-then-mask** order:

| method | SC IoU | precision | recall | pred/GT |
|---|---|---|---|---|
| OccAny, raw | 20.64 | 42.55 | 28.61 | 0.67 |
| OccAny, pooled *(their published protocol)* | 23.48 | 35.96 | 40.36 | 1.12 |
| ours, matched forward input, raw | **27.56** | 63.83 | 32.66 | 0.51 |
| ours, matched forward input, pooled | **40.47** | 55.75 | 59.62 | 1.07 |
| ours, causal 5 past frames, raw | 30.43 | 65.16 | 36.34 | 0.56 |
| ours, causal 5 past frames, pooled | 41.13 | 55.48 | 61.39 | 1.11 |

OccAny scores 20.64 / 23.48 on this subset against 20.67 / 23.56 on the full set, so the
subset is representative to 0.08 IoU. Matched-input deltas: **+6.92 raw**, **+16.99
pooled**.

## Correction to the gate body's 33.43 %

Line 173 quotes our pooled Occ3D number as 33.43 %. That applied the valid mask *before*
pooling; OccAny pools first and masks after (`eval_target.py:141`). On the same 250 samples:
**42.04** (their order) vs **33.26** (ours). The gate's pooled figure was therefore
*stricter on us* than OccAny's own convention. Raw numbers are unaffected, so the gate's
FAIL verdict and every threshold-free metric stand as written.

## Asymmetries that remain, all favouring OccAny

They trained on nuScenes; they tune the threshold per dataset (1.1 vs 2.5); our causal rows
give up all future observation. None of these is corrected for.

Records: `occany_headtohead_verified.json`, `occany_reproduction.md`,
figures `fig_occany_2d.png` / `fig_occany_3d.png`.
