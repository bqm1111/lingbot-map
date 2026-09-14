# Future semantic completion -- seed 0

## Source: KITTI-360 drive 0006, oracle-occupied voxels (590 anchors)

**unobserved at t -> later LiDAR-occupied (PRIMARY)** -- n = 609,435

| condition | cosine to future teacher | KL | top-1 agreement | mIoU (present classes) | cosine to B0 | classes predicted |
|---|---|---|---|---|---|---|
| B0 global mean | 0.4761 | 1.775 | 0.3939 | 0.0219 | 1.000 | 1 |
| B1 nearest map evidence | 0.4591 | 3.503 | 0.3813 | 0.0585 | 0.799 | 17 |
| B2 learned head | 0.4909 | 1.690 | 0.3876 | 0.0425 | 0.827 | 6 |
| ref: existing joint head | 0.4947 | 1.667 | 0.3888 | 0.0394 | 0.882 | 7 |

**map-observed, LiDAR-occupied (contrast)** -- n = 114,115

| condition | cosine to future teacher | KL | top-1 agreement | mIoU (present classes) | cosine to B0 | classes predicted |
|---|---|---|---|---|---|---|
| B0 global mean | 0.4047 | 1.990 | 0.3142 | 0.0175 | 1.000 | 1 |
| B1 nearest map evidence | 0.4314 | 6.271 | 0.3758 | 0.1014 | 0.422 | 18 |
| B2 learned head | 0.4558 | 1.832 | 0.3150 | 0.0489 | 0.617 | 7 |
| ref: existing joint head | 0.4748 | 1.776 | 0.3474 | 0.0518 | 0.637 | 7 |

Paired differences on the primary set (block bootstrap, 20-anchor blocks, 2 000 resamples):

| pair | Δ cosine [95 % CI] | Δ top-1 [95 % CI] | Δ KL [95 % CI] |
|---|---|---|---|
| B2-B0 | +0.0148 [-0.0043, +0.0356] | -0.0063 [-0.0412, +0.0290] | -0.0855 [-0.1547, -0.0160] |
| B2-B1 | +0.0318 [+0.0159, +0.0485] | +0.0063 [-0.0190, +0.0309] | -1.8129 [-2.0498, -1.5801] |
| B2-B2ref_existing_head | -0.0038 [-0.0181, +0.0099] | -0.0012 [-0.0204, +0.0168] | +0.0231 [-0.0195, +0.0674] |
| B1-B0 | -0.0170 [-0.0311, -0.0024] | -0.0126 [-0.0302, +0.0058] | +1.7274 [+1.4861, +1.9615] |

Top-1 agreement by distance to the nearest map voxel with teacher evidence (primary set):

| distance (voxels) | <=1 | 1-2 | 2-3 | 3-5 | 5-10 | 10-15 | >15 |
|---|---|---|---|---|---|---|---|
| share of voxels | 0.000 | 0.062 | 0.043 | 0.074 | 0.135 | 0.083 | 0.603 |
| B0 global mean | 0.000 | 0.407 | 0.414 | 0.430 | 0.440 | 0.433 | 0.371 |
| B1 nearest map evidence | 0.000 | 0.426 | 0.407 | 0.401 | 0.406 | 0.352 | 0.371 |
| B2 learned head | 0.000 | 0.377 | 0.368 | 0.387 | 0.382 | 0.382 | 0.392 |
| ref: existing joint head | 0.000 | 0.402 | 0.403 | 0.421 | 0.417 | 0.402 | 0.375 |

## Target repeatability (drive 0006, 295 anchors)

Causal-map teacher vector vs future teacher vector at the same LiDAR-occupied voxel: top-1 agreement **0.456**, cosine 0.511 (shuffled pairing 0.350 / 0.398), n = 26,740.

| evidence weight (rays) on the map side | (0,1] | (1,2] | (2,4] | (4,8] | (8,16] | >16 |
|---|---|---|---|---|---|---|
| top-1 agreement | 0.250 | 0.444 | 0.434 | 0.444 | 0.470 | 0.464 |

| future class | vegetation | road | man-made structure | car | other ground | building | sidewalk | fence |
|---|---|---|---|---|---|---|---|---|
| repeatability | 0.68 | 0.40 | 0.31 | 0.38 | 0.08 | 0.60 | 0.26 | 0.23 |

## Zero-shot transfer, past5, benchmark GT labels on GT-occupied voxels (seed 0, measurement only)

| dataset / population | n | B0 global mean top-1 / mIoU | B1 nearest map evidence top-1 / mIoU | B2 learned head top-1 / mIoU | ref: existing joint head top-1 / mIoU |
|---|---|---|---|---|---|
| semantickitti unobserved | 15,496,146 | 0.451 / 0.024 | 0.453 / 0.085 | 0.398 / 0.040 | 0.463 / 0.043 |
| semantickitti observed | 3,266,784 | 0.273 / 0.014 | 0.537 / 0.179 | 0.410 / 0.068 | 0.490 / 0.078 |
| occ3d unobserved | 8,716,093 | 0.238 / 0.014 | 0.286 / 0.079 | 0.314 / 0.036 | 0.337 / 0.035 |
| occ3d observed | 3,241,497 | 0.130 / 0.008 | 0.455 / 0.188 | 0.409 / 0.054 | 0.546 / 0.067 |

Paired Δ top-1 on unobserved voxels (scene bootstrap):

- semantickitti B2-B0: -0.0528 [-0.0718, -0.0339]
- semantickitti B2-B1: -0.0553 [-0.0908, -0.0146]
- semantickitti B1-B0: +0.0025 [-0.0265, +0.0249]
- occ3d B2-B0: +0.0763 [+0.0627, +0.0915]
- occ3d B2-B1: +0.0277 [+0.0144, +0.0408]
- occ3d B1-B0: +0.0486 [+0.0400, +0.0575]

## Supervision

Training key set 957,150 voxels over 1358 anchors; 3,247,317 further unobserved->future-observed->occupied voxels carry no teacher vector (22.8% coverage). 
- 0003: 88,439 key voxels (458/anchor), top-1 class share 0.790, top-2 0.914, entropy 0.83 nats, 16 classes
- 0007: 353,927 key voxels (622/anchor), top-1 class share 0.593, top-2 0.827, entropy 1.24 nats, 18 classes
- 0010: 514,784 key voxels (864/anchor), top-1 class share 0.412, top-2 0.699, entropy 1.75 nats, 18 classes
- 0006: 609,435 key voxels (1033/anchor), top-1 class share 0.394, top-2 0.606, entropy 1.79 nats, 18 classes

Training: 6000 steps, 77,049 head parameters, 19.9 min; final val KL 1.741 vs B0 1.840; mean train KL over the last 500 steps 1.016.

Verification: 8/10 checks pass; failing: projection_consistency, not_collapsed_to_mean.
