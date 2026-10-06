# Is the frozen baseline comparable to OccAny? A like-for-like measurement

Frozen current geometry-only baseline (three seeds, threshold 0, source freeze verified) versus the released OccAny checkpoint's saved predictions, on **every sample OccAny scored**, with OccAny's labels, scorer and pooling applied identically to both. Primary comparison predeclared in [predeclaration.md](predeclaration.md). Verification: 19/19 checks pass ([verification.md](verification.md)).

## Answer

* **SemanticKITTI 08.** The quoted pair was ours 26.25 (median; mean 26.17) on our 163 anchors, raw vs OccAny 25.91 published (pooled) — not comparable. Like-for-like (all 805 OccAny samples, OccAny's frames, labels, placement and pooling): **ours 28.66 (seed mean; median 28.57) vs OccAny 25.92, Δ +2.74 [+2.05, +3.43] pp → ours higher.** Without pooling for either: 29.82 vs 25.20, Δ +4.61 [+3.73, +5.50] → higher. Per resampling block, ours is higher in 91 % of blocks.
* **Occ3D-nuScenes val.** The quoted pair was ours 31.56 (median; mean 31.22) on our 1,182 anchors, raw vs OccAny 23.55 published (pooled) — not comparable. Like-for-like (all 4,819 OccAny samples, OccAny's frames, labels, placement and pooling): **ours 40.09 (seed mean; median 40.63) vs OccAny 23.56, Δ +16.54 [+15.05, +18.03] pp → ours higher.** Without pooling for either: 25.13 vs 20.67, Δ +4.46 [+2.30, +6.62] → higher. Per resampling block, ours is higher in 92 % of blocks.

The numbers in the question were **not** evaluated on the whole validation sets: 163 of the 815 labelled SemanticKITTI 08 frames and 1,182 of the Occ3D val samples (1 in 5, CAM_FRONT), raw, with our five past frames. OccAny's published numbers use 805 / 4,819 samples, pooling, and five frames starting at the target.

## From the quoted comparison to the like-for-like one

Each row removes one difference; Δ is always on identical samples (bootstrap 95 % CI, pp).

**SemanticKITTI 08**

| step | ours mean | ours median | OccAny | Δ (95 % CI) | reading |
|---|---:|---:|---:|---|---|
| as quoted: ours raw, causal, our 163 anchors & label loader  vs  OccAny published (pooled) | 26.17 | 26.25 | 25.91 | — | not comparable |
| OccAny's samples (801 causal-eligible), labels & scorer; both raw; our causal frames | 26.32 | 26.40 | 25.21 | +1.11 [+0.30, +1.93] | higher |
| + OccAny's five frames (forward), all 805 samples; both raw | 29.82 | 29.70 | 25.20 | +4.61 [+3.73, +5.50] | higher |
| + OccAny's pooling for both  [PRIMARY, predeclared] | 28.66 | 28.57 | 25.92 | +2.74 [+2.05, +3.43] | higher |

**Occ3D-nuScenes val**

| step | ours mean | ours median | OccAny | Δ (95 % CI) | reading |
|---|---:|---:|---:|---|---|
| as quoted: ours raw, causal, our 1,182 anchors & label loader  vs  OccAny published (pooled) | 31.22 | 31.56 | 23.55 | — | not comparable |
| OccAny's samples (4,219 causal-eligible), labels & scorer; both raw; our causal frames, our placement | 31.64 | 31.94 | 20.66 | +10.98 [+9.01, +12.97] | higher |
| + OccAny's benchmark placement (static extrinsic) | 31.63 | 31.93 | 20.66 | +10.97 [+8.99, +12.96] | higher |
| + OccAny's five frames (forward), all 4,819 samples; both raw | 25.13 | 25.45 | 20.67 | +4.46 [+2.30, +6.62] | higher |
| + OccAny's pooling for both  [PRIMARY, predeclared] | 40.09 | 40.63 | 23.56 | +16.54 [+15.05, +18.03] | higher |

![ladder](figures/1_ladder.png)

## Supporting rows

**SemanticKITTI 08**

* Our deployed setting (causal frames, our placement) pooled, causal-eligible samples: 25.89 vs OccAny 25.93, Δ -0.04 [-0.77, +0.70] → not distinguishable. Forward frames on the same samples: 28.66 — the effect of seeing the future instead of the past on identical targets.
* Our official anchors that OccAny also scored (161), primary variant: 28.60 vs 25.92, Δ +2.67 [+1.95, +3.39]; OccAny scores 25.92 here vs 25.92 on all of its samples.
* Map-only reference (our raw map `S`, no network), primary placement, forward, raw: 11.40.

**Occ3D-nuScenes val**

* Our deployed setting (causal frames, our placement) pooled, causal-eligible samples: 40.91 vs OccAny 23.55, Δ +17.36 [+16.04, +18.66] → higher. Forward frames on the same samples: 40.05 — the effect of seeing the future instead of the past on identical targets.
* Our official anchors that OccAny also scored (900), primary variant: 40.02 vs 23.56, Δ +16.46 [+14.97, +17.99]; OccAny scores 23.56 here vs 23.56 on all of its samples.
* Map-only reference (our raw map `S`, no network), primary placement, forward, raw: 6.94.

![per-sample](figures/2_per_sample.png)

![operating points](figures/3_operating_points.png)

![blocks](figures/4_blocks.png)

## Maps: ours vs OccAny on the same samples

Six panels per sample, one viewpoint per row: (1) ground truth; (2, 3) what each method recovered (green) over what it left out (grey); (4, 5) the same plus voxels predicted where the ground truth is free (red); (6) recovered by only ours (blue), only OccAny (orange), both (grey). Setting: OccAny's five frames, stored label and placement. Our side is one seed (the median seed on the primary variant). Rows are **not curated**: they are the samples at fixed quantiles of the per-sample difference, so the first rows show where OccAny is better. Every drawn volume was re-counted and asserted equal to the evaluation. Figures showing all six samples: `5_maps_<dataset>_<raw|pooled>.png`; two-row splits: `_pair1..3`.

**SemanticKITTI 08** (seed 2)

| Δ-rank | sample | OccAny IoU raw / pooled | ours IoU raw / pooled |
|---:|---|---|---|
| 5 % | `08_003950` | 36.32 / 34.90 | 31.54 / 31.07 |
| 23 % | `08_000620` | 28.83 / 29.00 | 30.15 / 28.45 |
| 41 % | `08_002190` | 29.98 / 27.92 | 33.21 / 30.43 |
| 59 % | `08_003010` | 24.89 / 25.25 | 30.21 / 28.72 |
| 77 % | `08_002625` | 21.70 / 22.85 | 33.28 / 29.12 |
| 95 % | `08_000355` | 18.41 / 20.04 | 32.13 / 32.49 |

![semantickitti raw](figures/5_maps_semantickitti_raw_pair1.png)

![semantickitti pooled](figures/5_maps_semantickitti_pooled_pair1.png)

**Occ3D-nuScenes val** (seed 0)

| Δ-rank | sample | OccAny IoU raw / pooled | ours IoU raw / pooled |
|---:|---|---|---|
| 5 % | `scene-0771_2b8dd244a7204b07837a01ddc3fa1333` | 38.89 / 51.63 | 33.59 / 45.88 |
| 23 % | `scene-0919_6b849f66b0974bff9cd073b4c70b23cb` | 23.38 / 21.65 | 8.44 / 26.70 |
| 41 % | `scene-0909_a6307dcc4e5748bcb7b9c93bb9b0fc9b` | 3.45 / 7.68 | 21.88 / 20.09 |
| 59 % | `scene-0104_345ffff158a245bc80572dbde4344f28` | 25.54 / 27.94 | 19.06 / 46.72 |
| 77 % | `scene-0106_cd6b9c4674704ef6877790b0f0f95118` | 19.82 / 29.19 | 43.32 / 55.98 |
| 95 % | `scene-0101_160fa0f8e77b41409a98341093a7cace` | 16.57 / 20.02 | 37.75 / 56.94 |

![occ3d raw](figures/5_maps_occ3d_raw_pair1.png)

![occ3d pooled](figures/5_maps_occ3d_pooled_pair1.png)

## Interpretation (measurements vs reading)

* **SemanticKITTI 08 — measured.** OccAny's pooling moves OccAny 25.20 → 25.92 (+0.71 pp) and ours 29.82 → 28.66 (-1.16 pp). Raw operating points (all samples, forward): ours precision 39.91 / recall 54.11 / predicted-to-GT 1.36; OccAny 45.59 / 36.05 / 0.79. On the causal-eligible samples our raw IoU is 26.32 with our causal frames and 29.82 with OccAny's forward frames.
* **Occ3D-nuScenes val — measured.** OccAny's pooling moves OccAny 20.67 → 23.56 (+2.89 pp) and ours 25.13 → 40.09 (+14.96 pp). Raw operating points (all samples, forward): ours precision 79.07 / recall 26.91 / predicted-to-GT 0.34; OccAny 42.67 / 28.61 / 0.67. On the causal-eligible samples our raw IoU is 31.64 with our causal frames and 24.95 with OccAny's forward frames.
* **Reading — ranking.** Ours is higher on identical samples in every like-for-like row except: SemanticKITTI 08 `causal_eligible` causal|shared|pooled (Δ -0.04 [-0.77, +0.70]). Where the interval covers 0 the two systems are not distinguishable in that setting.
* **Reading — magnitude (SemanticKITTI 08).** Pooling favours OccAny: +0.71 pp for OccAny, -1.16 pp for ours, so the pooled Δ (+2.74) is smaller than the raw Δ (+4.61) and is the conservative figure.
* **Reading — magnitude (Occ3D-nuScenes val).** The gap depends on post-processing: pooling adds +14.96 pp to ours but +2.89 pp to OccAny, so the pooled Δ (+16.54) is much larger than the raw Δ (+4.46). A one-voxel dilation mostly helps a conservative, high-precision prediction; the raw-vs-raw Δ is the more conservative statement of the difference.
* **Hypothesis (not tested here).** On Occ3D-nuScenes val our raw IoU is lower with forward than with causal frames on the same targets. A plausible cause is that with the target first the four later frames mostly re-observe the far field ahead, while the target's near field is seen by one frame only; no ablation isolates this.

## What this does and does not establish

* **Measured:** on OccAny's own sample sets, labels, placement, frames, pooling and scorer, the frozen baseline's pooled IoU relative to OccAny's, with a paired interval over scenes / frame blocks. OccAny's pooled numbers on its full sets reproduce its published values (check 6), so the reference is the published system.
* **Not removed:** OccAny's per-dataset confidence thresholds vs our single frozen threshold; different training data (neither trained on these targets); different image preprocessing; three seeds vs one released model. The intervals cover sample variation, not these.
* **Not a clean held-out test:** both target sets have been evaluated many times in this project (RC5). This is a reporting measurement of a frozen model; no number here may be used to select anything.
* **Pooling** is OccAny's convention, not part of either model; the raw-vs-raw rows are the post-processing-free comparison.

## Files

* [predeclaration.md](predeclaration.md), [results_table.md](results_table.md), [analysis.json](analysis.json), [verification.md](verification.md)
* `eval/<dataset>_shardKofN.npz` — per-sample TP/FP/FN for every variant (OccAny scorer and GPU tally)
* code: `tools/occany_comparison/` (tests: `tests/occany_comparison/`)

```bash
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0a PYTHONPATH=$PWD
python tools/occany_comparison/cli/build_moge.py --dataset occ3d --device cuda:0
for k in 0 1 2 3; do python tools/occany_comparison/cli/build_clips.py --dataset occ3d --device cuda:$k --shard $k --shards 4 & done; wait
python tools/occany_comparison/cli/build_clips.py --dataset semantickitti --device cuda:0
for k in 0 1 2 3; do python tools/occany_comparison/cli/evaluate.py --dataset occ3d --device cuda:$k --shard $k --shards 4 & done; wait
for k in 0 1; do python tools/occany_comparison/cli/evaluate.py --dataset semantickitti --device cuda:$k --shard $k --shards 2 & done; wait
python tools/occany_comparison/cli/report.py          # analysis.json, tables, figures 1-4, verification
python tools/occany_comparison/cli/figures_3d.py --device cuda:0   # 3D maps (needs analysis.json)
python tools/occany_comparison/cli/report.py          # again, to include the maps
```
