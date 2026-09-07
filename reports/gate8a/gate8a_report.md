# Gate 8A — source-only prior and calibration ablation

**Verdict.** KITTI-360 AP (0.2600) sits within 4% of its own occupancy prevalence (0.2509), so the completion barely ranks unknown voxels better than chance there: threshold calibration and prior correction did not fix transfer, and the failure is one of representation, not of operating point.

**Recommended next action.** Stop. Do not scale the network or start an architecture search. The next gate should attack the domain gap in the *inputs* to the completion head -- map-state normalisation and depth/scale statistics across drives -- not the completion head itself.

**Did geometry transfer pass?** **No.** AP/prevalence 1.04; beats incremental mapper: yes; beats calibrated mapper: **no**; beats strongest dilation: yes; beats editable fill: **no**; beats matched density random: yes; beats all valid occupied: **no**

**Did semantic completion pass?** **No.** SSC mIoU 0.0173 vs the strongest non-completion semantic baseline frozen_5frame_dil at 0.0266; paired D -0.0093 [-0.0104, -0.0081]; semantic accuracy on newly completed true positives 0.2995 vs 0.3098 over all true positives.

**Evidence.**

1. **The crop sampler was not the cause.** Swapping the 70%-occupancy-centred sampler for a uniform one moves the occupancy prior the network sees by 0.0006 on SemanticKITTI and 0.0034 on Occ3D -- a 128x128x32 crop already covers a quarter of the benchmark volume, so centring on an occupied voxel barely shifts the prior. The four cells pair up by *loss* on every source metric, never by sampler.

2. **The loss moved calibration, not ranking.** Macro AP over the four cells spans only 0.4647-0.5018 while the IoU-optimal threshold moves from -0.94 to +0.56 log-odds. Removing the positive weighting, the 2x unknown up-weight, the focal modulation and Dice cuts SemanticKITTI ECE from 0.1600 to 0.0378 and predicted/GT occupied volume at threshold 0 from 3.05x to 0.77x, at an essentially unchanged best achievable IoU. Gate 8's over-prediction was a property of its objective and its fixed threshold, and both are fixable on the sources.

3. **Fixing them helps -- on the sources.** The selected configuration `cellB_last` gains +0.0305 macro full-grid IoU by moving the threshold from 0 to +0.5625, and ranks source voxels well: AUROC 0.8733 on SemanticKITTI and 0.8189 on Occ3D, AP/prevalence x4.12 and x2.97.

4. **On KITTI-360 the ranking collapses to chance.** AUROC 0.4928 on the full grid and 0.4863 on the editable region -- against 0.87 on both sources -- and AP 0.2600 against a prevalence of 0.2509, a ratio of x1.04. There is no ordering of unknown voxels left to threshold.

5. **No threshold rescues it, which is the point of measuring threshold-free.** The *oracle* threshold on KITTI-360 -- chosen with the answers in hand, which the locked protocol forbids -- is -16.00, i.e. declare every voxel occupied, and yields IoU 0.2509: exactly the all-valid-occupied line 0.2509. The source-locked threshold gives 0.1873, below that line by -0.0636 [-0.0714, -0.0561]. Gate 8's over-prediction is genuinely gone (predicted/GT occupied volume 1.43 against 2.1x in Gate 8) and it did not help.

6. **The map state handed to the completion head has already lost the signal.** The incremental mapper's own log-odds score KITTI-360 at AUROC 0.5067 and AP/prevalence x1.012, and its native prediction scores IoU 0.0360 there against 0.0964 on SemanticKITTI and 0.1170 on Occ3D. The frozen five-frame baseline degrades the same way (0.0363 raw). The completion head is being asked to complete a map that is itself near-uninformative on this domain, which is why changing the head's sampler, its loss and its threshold changes nothing here.

---

## 0. What this gate did, and what it was not allowed to do

Gate 8 trained a causal completion module that improved the incremental mapper on both
training sources *and* on held-out KITTI-360, and still scored below the trivial
"declare every valid voxel occupied" line on KITTI-360 (0.2098 vs 0.2509). Gate 8A asks
whether that is (a) the crop sampler's occupancy prior, (b) the occupancy loss pushing the
decision boundary, or (c) a representation that does not transfer. It changes exactly two
things -- the crop sampler and the occupancy objective -- and adds a threshold-free
evaluation so the decision boundary can be separated from the ranking.

Frozen and untouched (hashes in `artifacts/gate8a/stage0_audit.json`): LingBot-Map, MoGe-2
and the five-frame scale anchor, the Trident-H caches, the incremental mapper and map
representation, the completion U-Net architecture and its 32 input channels, the privileged
targets and the 20-frame future horizon, the training sources (SemanticKITTI 00/05/07 +
the first 100 Occ3D train scenes), the budget (6 000 steps, batch 4, crop 128x128x32,
seed 0), the union vocabulary and the 0.5 x future-teacher KL, and Gate 6's evaluation
masks, grids and metric code.

KITTI-360 was used **once**, after `configs/gate8a/frozen_selection.yaml` was written.
No KITTI-360 datum -- including its occupancy prevalence -- entered the sampler, the loss,
model selection, checkpoint selection, threshold calibration or any hyper-parameter. It is
reported as **unadapted transfer**, not as an untouched benchmark, because Gate 8 already
inspected a result on it.

## 1. Stage 1 — threshold-free completion evaluation

The evaluator now dumps the **final occupancy log-odds** (`apply_residual(base, residual)`)
rather than a binarized prediction, histogrammed per anchor over 1 024 bins on
[-16, +16] with 0.0 exactly on a bin boundary. Every threshold-dependent number in this
report -- IoU, precision, recall, predicted density, and the per-clip counts that feed the
bootstrap -- is recomputed from those histograms, so no threshold is baked into an
artifact. Brier score and ECE are accumulated exactly alongside.

Two regions are scored separately:

* **full** -- the complete valid evaluation grid (the benchmark's own `keep` mask);
* **edit** -- the *editable completion region*, which is not a new definition but
  `gate8.net.apply_residual`'s own gate, `abs(base_logodds) < 2.0`, read out by
  `gate8a.regions.editable_native`. A unit test asserts the mask is exactly the set of
  voxels the residual actually moves, and a second test asserts it follows a change to the
  lock constant. On Occ3D the 0.2 m prediction grid reduces to the 0.4 m evaluation grid
  under the frozen any-sub-voxel rule, so scores reduce by max, a coarse voxel is editable
  if any child is, and it is *forced occupied* if any child is locked occupied.

Baselines, all scored on byte-identical maps, masks and anchors in the same pass:
frozen five-frame G51-B (Gate 6's B-R and B-D count blocks), the incremental mapper at its
own rule, the mapper at a source-calibrated threshold on the same log-odds, the fixed 0.4 m
dilation, every valid voxel occupied, every *editable* voxel occupied with protected mapper
cells preserved, and matched-density random editable completion over five fixed seeds.
Raw non-dilated completion is the primary method throughout; dilation appears only as a
baseline.

## 2. Stage 2 — the controlled 2x2

### What each crop sampler actually shows the network

| training source | sampler | crop occ. prevalence | crop editable-region prevalence | whole-volume prevalence | whole-volume editable prevalence |
|---|---|---|---|---|---|
| sk_train | occ_centred | 0.0819 | 0.0777 | 0.0694 | 0.0652 |
| sk_train | uniform | 0.0813 | 0.0766 | 0.0688 | 0.0648 |
| occ3d_train | occ_centred | 0.2116 | 0.1999 | 0.2168 | 0.2043 |
| occ3d_train | uniform | 0.2082 | 0.1887 | 0.2201 | 0.2069 |

This is the gate's first substantive result and it removes one of the two candidate causes
outright. A 128x128x32 crop already covers a quarter of a 256x256x32 benchmark volume, so
centring 70 % of crops on a ground-truth-occupied unobserved voxel moves the occupancy
prior the network sees by well under one percentage point on either source. The occupancy
prior of Gate 8's training crops was never far from the benchmark's own.

### The four cells on source validation

| candidate | sampler | occ. loss | AP SK | AP Occ3D | macro AP | AP/prev SK | AP/prev Occ3D | own τ* | macro IoU @ τ* | macro IoU @ 0 | teacher KL (val) | train min | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cellA_best | occ_centred | focal_dice | 0.3186 | 0.6782 | 0.4984 | 4.07 | 2.95 | 0.500 | 0.3551 | 0.3263 | 1.142 | 25.5 | 6.22 |
| cellA_last | occ_centred | focal_dice | 0.3216 | 0.6806 | 0.5011 | 4.11 | 2.97 | 0.562 | 0.3550 | 0.3219 | 1.142 | 25.5 | 6.22 |
| cellB_best | uniform | focal_dice | 0.3223 | 0.6803 | 0.5013 | 4.12 | 2.96 | 0.562 | 0.3569 | 0.3260 | 1.105 | 25.8 | 6.22 |
| cellB_last | uniform | focal_dice | 0.3225 | 0.6810 | 0.5018 | 4.12 | 2.97 | 0.562 | 0.3565 | 0.3259 | 1.105 | 25.8 | 6.22 |
| cellC_best | occ_centred | bce | 0.2947 | 0.6347 | 0.4647 | 3.77 | 2.77 | -0.781 | 0.3482 | 0.3266 | 1.142 | 26.1 | 6.22 |
| cellC_last | occ_centred | bce | 0.2977 | 0.6462 | 0.4719 | 3.81 | 2.82 | -0.938 | 0.3491 | 0.3281 | 1.142 | 26.1 | 6.22 |
| cellD_best | uniform | bce | 0.3018 | 0.6376 | 0.4697 | 3.86 | 2.78 | -0.875 | 0.3510 | 0.3296 | 1.100 | 25.7 | 6.22 |
| cellD_last | uniform | bce | 0.3028 | 0.6384 | 0.4706 | 3.87 | 2.78 | -0.938 | 0.3500 | 0.3276 | 1.100 | 25.7 | 6.22 |

## 3. Stage 3 — source-only selection and calibration

Selection used SemanticKITTI 08 and Occ3D validation only, on the **full validation grids**
rather than on training crops, by macro-average AP, requiring AP > prevalence on both
sources; then one global final-logit threshold maximising the equally weighted mean of the
two per-source pooled full-grid binary IoUs. No semantic ground truth entered either step.
The same search was run on the mapper's own log-odds so the completion is not compared
against an uncalibrated baseline.

| candidate | source | AP | prevalence | AP/prev | τ* | full IoU | full P | full R | full density | edit IoU | edit P | edit R | edit density |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cellA_best | SemanticKITTI 08 | 0.3186 | 0.0782 | 4.07 | 0.500 | 0.2579 | 0.3295 | 0.5428 | 0.1289 | 0.2597 | 0.3470 | 0.5077 | 0.1070 |
|  | Occ3D-nuScenes val | 0.6782 | 0.2295 | 2.95 | 0.500 | 0.4523 | 0.6212 | 0.6247 | 0.2308 | 0.4569 | 0.6155 | 0.6394 | 0.2312 |
| cellA_last | SemanticKITTI 08 | 0.3216 | 0.0782 | 4.11 | 0.562 | 0.2574 | 0.3429 | 0.5080 | 0.1159 | 0.2592 | 0.3673 | 0.4682 | 0.0932 |
|  | Occ3D-nuScenes val | 0.6806 | 0.2295 | 2.97 | 0.562 | 0.4525 | 0.6305 | 0.6159 | 0.2242 | 0.4572 | 0.6251 | 0.6299 | 0.2243 |
| cellB_best | SemanticKITTI 08 | 0.3223 | 0.0782 | 4.12 | 0.562 | 0.2600 | 0.3356 | 0.5357 | 0.1249 | 0.2622 | 0.3556 | 0.4996 | 0.1028 |
|  | Occ3D-nuScenes val | 0.6803 | 0.2295 | 2.96 | 0.562 | 0.4538 | 0.6496 | 0.6008 | 0.2123 | 0.4587 | 0.6449 | 0.6136 | 0.2118 |
| cellB_last | SemanticKITTI 08 | 0.3225 | 0.0782 | 4.12 | 0.562 | 0.2593 | 0.3392 | 0.5240 | 0.1209 | 0.2615 | 0.3612 | 0.4864 | 0.0985 |
|  | Occ3D-nuScenes val | 0.6810 | 0.2295 | 2.97 | 0.562 | 0.4536 | 0.6449 | 0.6047 | 0.2152 | 0.4585 | 0.6400 | 0.6178 | 0.2149 |
| cellC_best | SemanticKITTI 08 | 0.2947 | 0.0782 | 3.77 | -0.781 | 0.2467 | 0.3291 | 0.4960 | 0.1179 | 0.2458 | 0.3487 | 0.4546 | 0.0954 |
|  | Occ3D-nuScenes val | 0.6347 | 0.2295 | 2.77 | -0.781 | 0.4497 | 0.6251 | 0.6158 | 0.2261 | 0.4542 | 0.6195 | 0.6298 | 0.2263 |
| cellC_last | SemanticKITTI 08 | 0.2977 | 0.0782 | 3.81 | -0.938 | 0.2502 | 0.3311 | 0.5062 | 0.1196 | 0.2503 | 0.3509 | 0.4662 | 0.0972 |
|  | Occ3D-nuScenes val | 0.6462 | 0.2295 | 2.82 | -0.938 | 0.4479 | 0.6068 | 0.6311 | 0.2387 | 0.4521 | 0.6006 | 0.6464 | 0.2395 |
| cellD_best | SemanticKITTI 08 | 0.3018 | 0.0782 | 3.86 | -0.875 | 0.2531 | 0.3365 | 0.5054 | 0.1175 | 0.2539 | 0.3585 | 0.4652 | 0.0949 |
|  | Occ3D-nuScenes val | 0.6376 | 0.2295 | 2.78 | -0.875 | 0.4488 | 0.6153 | 0.6238 | 0.2327 | 0.4531 | 0.6095 | 0.6385 | 0.2332 |
| cellD_last | SemanticKITTI 08 | 0.3028 | 0.0782 | 3.87 | -0.938 | 0.2542 | 0.3362 | 0.5106 | 0.1188 | 0.2552 | 0.3577 | 0.4711 | 0.0963 |
|  | Occ3D-nuScenes val | 0.6384 | 0.2295 | 2.78 | -0.938 | 0.4458 | 0.6020 | 0.6322 | 0.2411 | 0.4499 | 0.5957 | 0.6476 | 0.2420 |
| mapper | SemanticKITTI 08 | 0.1064 | 0.0782 | 1.36 | -16.000 | 0.0782 | 0.0782 | 1.0000 | 1.0000 | 0.0731 | 0.0731 | 1.0000 | 1.0000 |
|  | Occ3D-nuScenes val | 0.2821 | 0.2295 | 1.23 | -16.000 | 0.2295 | 0.2295 | 1.0000 | 1.0000 | 0.2226 | 0.2226 | 1.0000 | 1.0000 |

**Frozen before KITTI-360 was opened.** Selected `cellB_last` (checkpoint `artifacts/gate8a/checkpoints/cellB_uniform_focal_last.pt`, sha256 `b505fdc4facdd53a...`), macro AP 0.5018, single global threshold **+0.5625** on the final occupancy log-odds, macro full-grid IoU 0.3565. The incremental mapper's own calibrated threshold is -16.0000 (macro full-grid IoU 0.1539). Written to `configs/gate8a/frozen_selection.yaml`; the manifest with every candidate checkpoint hash is `artifacts/gate8a/frozen_manifest.json`.

## 4. Stage 4 — the one locked KITTI-360 evaluation

Run once, unadapted, from the frozen file: checkpoint and both thresholds were read out of `configs/gate8a/frozen_selection.yaml` and nothing was re-tuned afterwards. 1753 official anchors, 9.2 min, peak 14.9 GiB. KITTI-360 is **unadapted transfer**, not an untouched benchmark: Gate 8 already inspected a result on it.

### Threshold-free

| region | voxels | prevalence | AP | AP/prev | AUROC | Brier | ECE | IoU @ 0 | density @ 0 | oracle τ | IoU @ oracle τ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| completion_full | 549394259 | 0.2509 | 0.2600 | 1.04 | 0.4928 | 0.3531 | 0.3674 | 0.2058 | 0.5894 | -16.000 | 0.2509 |
| completion_edit | 518359270 | 0.2511 | 0.2583 | 1.03 | 0.4863 | 0.3486 | 0.3644 | 0.2042 | 0.5951 | -16.000 | 0.2511 |
| mapper_full | 549394259 | 0.2509 | 0.2539 | 1.01 | 0.5067 | 0.2587 | 0.2526 | 0.2476 | 0.9236 | -3.969 | 0.2513 |
| mapper_edit | 518359270 | 0.2511 | 0.2515 | 1.00 | 0.5009 | 0.2486 | 0.2426 | 0.2479 | 0.9493 | -16.000 | 0.2511 |

`artifacts/gate8a/fig_pr_kitti360.png` shows what those numbers look like: the PR curve is
flat on the prevalence line across the whole recall range, and the reliability curve runs
*backwards* -- voxels the network calls unlikely are occupied slightly more often than
voxels it calls likely. Compare `fig_pr_semantickitti.png` and `fig_pr_occ3d.png`, where
every checkpoint's PR curve stands well clear of prevalence and the BCE cells sit close to
the diagonal.

### At the locked source-selected threshold

| method | binary IoU | precision | recall | predicted density | pred/GT volume | Δ vs completion (paired 95% CI) |
|---|---|---|---|---|---|---|
| completion | 0.1873 | 0.2680 | 0.3835 | 0.3590 | 1.43 | — |
| mapper_native | 0.0360 | 0.2869 | 0.0396 | 0.0346 | 0.14 | +0.1513 [+0.1459, +0.1566] |
| mapper_score_ge_zero | 0.2476 | 0.2523 | 0.9289 | 0.9236 | 3.68 | -0.0602 [-0.0681, -0.0527] |
| mapper_calibrated | 0.2509 | 0.2509 | 1.0000 | 1.0000 | 3.99 | -0.0636 [-0.0714, -0.0561] |
| mapper_dilate | 0.1094 | 0.2854 | 0.1508 | 0.1325 | 0.53 | +0.0779 [+0.0702, +0.0855] |
| frozen_5frame_raw | 0.0363 | 0.3747 | 0.0386 | — | 0.10 | +0.1511 [+0.1470, +0.1554] |
| frozen_5frame_dil | 0.1321 | 0.3587 | 0.1729 | — | 0.48 | +0.0552 [+0.0500, +0.0608] |
| all_valid_occupied | 0.2509 | 0.2509 | 1.0000 | 1.0000 | 3.99 | -0.0636 [-0.0714, -0.0561] |
| editable_fill | 0.2506 | 0.2522 | 0.9762 | 0.9714 | 3.87 | -0.0633 [-0.0713, -0.0558] |
| random_editable_s0 | 0.1778 | 0.2565 | 0.3670 | 0.3589 | 1.43 | +0.0095 [+0.0053, +0.0138] |
| random_editable_s1 | 0.1779 | 0.2565 | 0.3670 | 0.3590 | 1.43 | — |
| random_editable_s2 | 0.1779 | 0.2565 | 0.3670 | 0.3589 | 1.43 | — |
| random_editable_s3 | 0.1778 | 0.2565 | 0.3670 | 0.3589 | 1.43 | — |
| random_editable_s4 | 0.1778 | 0.2565 | 0.3670 | 0.3590 | 1.43 | — |
| random_editable (mean of 5 seeds) | 0.1778 | — | — | — | — | sd 0.0000, range [0.1778, 0.1779] |

### Semantics

| method | SSC mIoU | binary IoU | TP-conditioned semantic accuracy | coverage miss | naming error | Δ SSC mIoU vs completion (paired 95% CI) |
|---|---|---|---|---|---|---|
| completion | 0.0173 | 0.1873 | 0.3098 | 0.6165 | 0.2647 | — |
| mapper_native | 0.0072 | 0.0360 | 0.4023 | 0.9604 | 0.0237 | +0.0100 [+0.0092, +0.0108] |
| mapper_dilate | 0.0189 | 0.1094 | 0.4219 | 0.8492 | 0.0871 | -0.0016 [-0.0025, -0.0007] |
| mapper_calibrated | 0.0070 | 0.2509 | 0.0377 | 0.0000 | 0.9623 | +0.0103 [+0.0094, +0.0110] |
| frozen_5frame_raw | 0.0088 | 0.0363 | 0.6245 | 0.9614 | 0.0145 | +0.0085 [+0.0074, +0.0094] |
| frozen_5frame_dil | 0.0266 | 0.1321 | 0.5959 | 0.8271 | 0.0699 | -0.0093 [-0.0104, -0.0081] |

**Semantic accuracy on newly completed true positives.** Of the 47,511,997 voxels that are genuinely occupied, that the completion predicts occupied and that the incremental mapper did *not*, 14,228,464 carry the right class -- 0.2995, against 0.3098 over all true positives and 1/19 = 0.0526 for a uniform guess.

### Classwise IoU

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

## 5. Decision rules, applied

**Geometry transfer**

| criterion | met | value |
|---|---|---|
| KITTI-360 AP meaningfully above prevalence (>1.5x) | **no** | AP 0.2600 / prevalence 0.2509 = x1.04 |
| beats the incremental mapper | yes |  |
| beats the source-calibrated mapper | **no** |  |
| beats the strongest fixed-dilation baseline | yes |  |
| beats the editable-region fill baseline | **no** |  |
| beats matched-density random completion | yes |  |
| gain is not an inflated occupied volume | yes | pred/GT volume 1.43 |

Geometry transfer **FAILED**.

**Semantic completion**

| criterion | met | value |
|---|---|---|
| SSC mIoU beats the strongest non-completion semantic baseline | **no** | 0.0173 vs 0.0266 (frozen_5frame_dil) |
| newly completed true positives carry semantic information | yes | accuracy 0.2995 vs 0.0526 chance |

Semantic completion **FAILED**.

### Binarization agreement

The semantic count block and the score histogram binarize the same tensor two ways (`final >= tau` versus the stored bin index). They agree to 2.27e-07 relative -- 23 voxels out of 2.4e8 -- and `aggregate.py` asserts a bound of 1e-5 on that. The residual disagreement is float32 bin-edge precision in the score histogram (~1.9e-6 near the +16 offset); scores within one ULP below tau bin as >= tau, seven orders of magnitude below any difference this report leans on.

## 6. Reproduction

```bash
cd /home/minh/workspace/lingbot-map_fork
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1 GATE8_GPUS="1 2 3"
PY=/home/minh/anaconda3/envs/cu128/bin/python

$PY tools/gate8a/stage0.py                       # hash the frozen surface
$PY tools/gate8a/sampler_stats.py --n 400        # what each sampler shows the network
tools/gate8a/run_ablation.sh                     # cells B, C, D (cell A is Gate 8's run)
tools/gate8a/run_source_eval.sh                  # score 8 checkpoints on both sources,
                                                 #   then write the frozen selection
tools/gate8a/run_heldout.sh                      # the ONE locked KITTI-360 run
$PY tools/gate8a/aggregate.py
$PY tools/gate8a/figures.py
$PY tools/gate8a/report.py --write
$PY -m pytest tests/gate6 tests/gate7a tests/gate7b tests/gate8 tests/gate8a -q
```

Seeds: 0 everywhere (training, crop sampling, the 10 000-draw paired bootstrap); the random
editable baseline uses seeds 0-4 with a per-anchor offset so no two anchors share a draw.
`tools/gate8a/run_heldout.sh` takes **no arguments** -- it reads the checkpoint and both
thresholds out of `configs/gate8a/frozen_selection.yaml`, which is the mechanism that keeps
the held-out read-out honest.

| stage | wall time | peak GPU |
|---|---|---|
| train cellA | 25.5 min | 6.22 GiB |
| train cellB | 25.8 min | 6.22 GiB |
| train cellC | 26.1 min | 6.22 GiB |
| train cellD | 25.7 min | 6.22 GiB |

## 7. Tests

`python -m pytest tests/gate6 tests/gate7a tests/gate7b tests/gate8 tests/gate8a -q`

```
239 passed, 1 warning in 231.02s (0:03:51)
```

212 prior tests (Gate 6: 63, Gate 7A: 68, Gate 7B: 56, Gate 8: 25) plus 27 new Gate 8A
tests, nothing skipped and nothing regressed. The one warning is numpy's `trapz`
deprecation inside the AUROC cross-check against scikit-learn.

The Gate 8A tests that carry the gate's integrity claims:

* `test_editable_mask_is_exactly_the_residual_gate` — the editable region is not a copy of
  the residual rule but the set of voxels `apply_residual` actually moves, checked on
  20 000 log-odds including both signs of the exact lock boundary;
  `test_editable_region_follows_a_change_in_the_lock` checks it tracks the constant.
* `test_editable_fill_preserves_every_protected_voxel`,
  `test_random_editable_preserves_protected_and_forced_voxels` — no baseline may change a
  voxel the residual cannot change, nor remove occupancy the residual cannot remove.
* `test_random_baseline_is_reproducible_and_seed_dependent`,
  `test_matched_density_hits_the_requested_count` — fixed seeds reproduce exactly, differ
  across seeds and across anchors, and hit the requested density to within 2 %.
* `test_sweep_reproduces_brute_force_counts_at_many_thresholds`,
  `test_threshold_sweep_uses_stored_continuous_scores_not_a_binarization`,
  `test_per_clip_counts_sum_to_the_pooled_sweep` — every threshold-dependent count comes
  out of stored continuous scores, one block answers different thresholds differently, and
  the per-clip bootstrap input sums to the pooled sweep exactly.
* `test_average_precision_matches_sklearn` — AP and AUROC agree with scikit-learn to 2e-3
  on identically quantised scores.
* `test_uniform_sampler_cannot_see_an_occupancy_label` — structural: the function's whole
  input is the RNG and the lattice shape, its body contains no indexing and no attribute
  access other than on the RNG. `test_uniform_sampler_is_blind_to_the_labels_behaviourally`
  confirms identical origins from an all-empty and an all-occupied volume.
* `test_bce_mask_excludes_invalid_and_protected_voxels` — flipping the target on invalid or
  protected voxels leaves the loss unchanged; flipping it inside the mask does not.
* `test_ablation_configs_differ_only_in_the_two_factors` — the 2x2 is a 2x2.
* `test_no_kitti360_anywhere_in_the_selection_path` — no KITTI-360 reference survives in the
  trainer, the sampler statistics or the selection tool except as a negation.

## 8. Deviations and failures

1. **`gate8/net.py` was refactored, not changed.** The evaluator needs the continuous final
   log-odds, which Gate 8's `Completer.complete` threw away. `complete` was split into
   `raw(q, grid) -> (final_logodds, probs)` plus the frozen `> 0` decision, so the score
   path and the Gate 8 decision path cannot drift apart. The 25 Gate 8 tests pass unchanged
   and `complete()` is now literally `raw()` followed by `> 0`. `stage0_audit.json` records
   the post-refactor hash; this is the only edit to a frozen file in the gate.

2. **The checkpoint pool is {best, last} per cell, not every 500-step checkpoint.** Gate 8's
   trainer keeps only the lowest-validation-loss checkpoint and the final one, and the brief
   says to reuse the Gate 8 run rather than retrain it. Selecting over per-500-step
   checkpoints would therefore have given the three new cells a larger pool than cell A. The
   Gate 8A trainer keeps exactly the same two, so all four cells enter selection with two
   candidates each, chosen by the same within-cell rule.

3. **Cell A was not retrained.** Its two checkpoints are Gate 8's own
   (`artifacts/gate8/checkpoints/completion_{best,last}.pt`). The Gate 8A trainer's
   `focal_dice` path reproduces Gate 8's objective term for term, and
   `configs/gate8a/cellA_occcentred_focal.yaml` is asserted by test to agree with
   `configs/gate8/completion.yaml` on every shared key.

4. **The editable region on Occ3D's coarse evaluation grid.** The residual acts at 0.2 m and
   the benchmark scores at 0.4 m under the frozen any-sub-voxel occupancy rule. A coarse
   voxel is therefore called editable if *any* of its eight children is editable, and
   *forced occupied* if any child is locked occupied -- occupancy the residual cannot
   remove, which no baseline is allowed to remove either. On SemanticKITTI and KITTI-360 the
   two grids coincide and the region is exactly `abs(base_logodds) < 2.0`.

5. **Scores are histogrammed, not stored per voxel.** Storing the final log-odds for every
   voxel of every anchor would be ~40 GB. They are accumulated per anchor into 1 024 bins on
   [-16, +16] with 0.0 on a bin boundary, which makes AP, PR, AUROC and every threshold
   sweep exact to a bin width of 0.03125 log-odds. Brier and ECE are accumulated exactly,
   not from the histogram. A test checks AP and AUROC against scikit-learn on identically
   quantised scores.

6. **`tools/gate8a/select.py` had to be renamed `selection.py`.** Python puts a script's own
   directory first on `sys.path`, so a module named `select` shadows the stdlib `select` and
   breaks `subprocess` for every other tool in the same directory. This cost one failed run
   of `sampler_stats.py`.

7. **The uniform sampler sometimes draws a crop with no valid future-teacher voxel**, in
   which case the teacher-KL term for that sample is zero because its mask is empty. That is
   the existing `gate8.losses.semantic_kl` behaviour under an empty mask, not a Gate 8A
   change, but it is worth knowing when reading the training logs.

8. **The frozen selection file was rewritten once, before KITTI-360 was opened, to fix a
   provenance label.** `selection.py` looked up the winning cell's training record by the
   candidate name (`cellB_last`) instead of its config tag (`cellB_uniform_focal`), so the
   first file recorded cell A's sampler, loss and config path next to cell B's checkpoint.
   The checkpoint path, its sha256 and both thresholds were correct and are byte-identical
   after the fix; the diff is three provenance lines. Selection is a deterministic function
   of the source score blocks, so re-running it reproduced the same choice.

9. **The mapper's source-calibrated threshold degenerates to "everything occupied"**
   (τ = −16.0, the bottom of the score range). This is the declared rule applied honestly,
   not a bug: the incremental map's log-odds carry very little ranking information
   (AUROC 0.552 on SemanticKITTI, and its IoU on Occ3D is maximised by predicting every
   voxel), so no single global threshold weighted equally across the two sources beats the
   trivial line. The consequence is that "incremental mapper with its own source-calibrated
   threshold" and "every valid voxel occupied" are the *same* prediction on the held-out
   set, and the report treats `mapper_native` and the 0.4 m dilation as the meaningful
   mapper baselines.

