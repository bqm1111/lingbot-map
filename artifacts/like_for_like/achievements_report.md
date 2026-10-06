# What we have measured against OccAny, as of 2026-09-18

Consolidated from the frozen artifacts, not from recollection. Every number below is read out of
`artifacts/occany_comparison/analysis.json` (geometry) or `artifacts/like_for_like/results_*.json` (semantics).
Superseded numbers are named as such at the end.

## 1. Geometry (scene-completion IoU) -- this is where we win

Like-for-like: every sample OccAny scored, OccAny's stored labels, OccAny's `SSCMetrics` scorer, OccAny's five
forward frames, OccAny's benchmark placement, and the same post-processing applied to both sides. Three seeds.

### SemanticKITTI 08 (805 samples)

| post-processing | seed 0 | seed 1 | seed 2 | ours (mean) | OccAny | delta |
|---|---:|---:|---:|---:|---:|---:|
| raw | 30.27 | 29.70 | 29.49 | **29.82** | 25.20 | **+4.61** |
| pooled | 29.05 | 28.35 | 28.57 | **28.66** | 25.92 | **+2.74** |

### Occ3D-nuScenes val (4819 samples)

| post-processing | seed 0 | seed 1 | seed 2 | ours (mean) | OccAny | delta |
|---|---:|---:|---:|---:|---:|---:|
| raw | 25.45 | 26.40 | 23.53 | **25.13** | 20.67 | **+4.46** |
| pooled | 40.63 | 40.70 | 38.94 | **40.09** | 23.56 | **+16.54** |

**Bootstrap CIs on the delta** (paired block bootstrap, 2,000 resamples): SemanticKITTI raw +4.61 [+3.73, +5.50],
pooled +2.74 [+2.05, +3.43]; Occ3D raw +4.46 [+2.30, +6.62], pooled +16.54 [+15.05, +18.03]. Ours is higher in
91 % (SK) and 92 % (Occ3D) of resampling blocks.

**Quote raw and pooled together.** Pooling is not neutral and it does not affect the two methods equally:

| | ours raw -> pooled | OccAny raw -> pooled | our raw precision / recall |
|---|---|---|---|
| SemanticKITTI | 29.82 -> 28.66 (**-1.16**) | 25.20 -> 25.92 (+0.72) | 0.399 / 0.541, pred/gt 1.33 |
| Occ3D | 25.13 -> 40.09 (**+14.96**) | 20.67 -> 23.56 (+2.89) | 0.791 / 0.270, pred/gt 0.34 |

Our raw Occ3D maps are under-complete, so 3x3x3 dilation buys recall cheaply; on SemanticKITTI we are already
over-complete, so it only adds false positives. The Occ3D pooled gap of +16.54 is real but is substantially a
post-processing effect. The defensible headline pair is the raw one: **+4.61** and **+4.46**.

**Sanity check on the comparison itself:** running OccAny's own saved predictions through our scorer reproduces
their published numbers -- SemanticKITTI 25.92 vs 25.91 published, Occ3D 23.56 vs 23.55. 19/19 verification checks pass.

## 2. Semantics (SSC mIoU) -- this is where we lose

### SemanticKITTI 08 -- a genuine like-for-like comparison

OccAny's published SemanticKITTI semantics are its `5frames` setting: one camera, five frames. That is exactly our
forward protocol, so these rows are comparable.

| | n | SC IoU | SSC mIoU | superclass mIoU |
|---|---:|---:|---:|---:|
| ours, B0, raw | 805 | 30.27 | 5.11 | 9.53 |
| ours, B3 seed mean, raw | 805 | 30.27 | 5.36 | 9.70 |
| ours, B3 seed mean, pooled | 805 | 30.27 | 5.38 | 9.64 |
| **OccAny, published (Tab. 5/6)** | 805 | — | **7.28** | **13.53** |

**We lose semantics by roughly 2 mIoU (5.38 vs 7.28) and 4 superclass mIoU**, while winning geometry.

### Occ3D-nuScenes -- no comparable published number exists yet

OccAny publishes nuScenes semantics **only** for its `surround` setting: six cameras, one timestep, and the full
360-degree label. Its `5frames` nuScenes setting (one camera, which is what our forward rows use, and what its saved
outputs on this machine are) has **no published semantic number**. Its nuScenes loader also ignores the rear half of
the grid unless the setting is surround (`voxel_label[:100] = 255`), so the two settings differ in ground truth too.

| ours | ground truth | n | SC IoU | B0 mIoU | B3 mIoU | B3 superclass |
|---|---|---:|---:|---:|---:|---:|
| forward, 1 camera, raw | front half | 4819 | 25.45 | 2.50 | 2.57 | 4.93 |
| forward, 1 camera, pooled | front half | 4819 | 25.45 | 2.58 | 2.62 | 5.08 |
| forward, 1 camera, raw | 360 deg | 4819 | 21.38 | 1.60 | 1.63 | 2.94 |
| forward, 1 camera, pooled | 360 deg | 4819 | 21.38 | 1.65 | 1.67 | 3.04 |
| **OccAny, published (Tab. 5/6)** | 360 deg, **6 cameras** | 4,819 | — | — | **6.66** | **10.32** |

On a common 360-degree ground truth our single-camera number is **1.67**, not 2.62: scoring one camera against the
front half only was flattering us. The gap to OccAny's 6.66 is therefore ~5.0, of which an unknown part is simply
the five cameras we do not use. The surround run now in progress measures that part.

## 3. Our deployed (causal) setting

The forward window is OccAny's, not ours -- it sees four frames *after* the target. Our deployed system is causal:

| | ours pooled | OccAny pooled | delta |
|---|---:|---:|---|
| SemanticKITTI, causal-eligible (801) | 25.89 | 25.93 | -0.04 [-0.77, +0.70], not distinguishable |
| Occ3D, causal-eligible (4,219) | 40.91 | 23.55 | +17.36 [+16.04, +18.66] |

Frozen causal semantics (the honest description of our system): SemanticKITTI B0 4.32 / B3 4.43, superclass 8.33 / 8.42;
Occ3D B0 3.71 / B3 3.80, superclass 6.63 / 6.76.

## 4. What is verified

- OccAny's saved predictions re-scored with our scorer reproduce their published SC IoU to 0.01 pp (both datasets).
- Our protocol-parameterised pipeline reproduces the frozen stored predictions array for array (`all_equal: true`).
- The 360-degree label reproduces OccAny's stored label voxel for voxel on the front half and restores the rear.
- Surround MoGe-2 depths reproduce cached frozen frames exactly (0.0 depth difference, identical masks).

## 5. Numbers that are superseded -- do not quote

- **Gate 8C-1 (Sept 3-7, `artifacts/gate8c1/`)**, including `fig_occany_*.png`: the cached targets predate the
  open-sky fix (commit `647bcf3`). Rebuilding drops completion SC IoU from 9.29 to 4.67 on a 6-anchor sample.
- **ours 26.25 (SK) / 31.56 (Occ3D) vs OccAny 25.91 / 23.55**: different sample sets, post-processing and temporal
  direction on each side. Superseded by section 1.
- **Occ3D semantic 2.57/2.62 vs OccAny 6.66**: different setting on both the input and the ground-truth side.
  Superseded by the 360-degree rows and, when it lands, the surround row.

## 6. In progress

The surround (six-camera) Occ3D evaluation, to produce the row that is actually comparable to OccAny's 6.66 / 10.32.
Clips complete (4,819/4,819); Trident-H token extraction is the critical path.
