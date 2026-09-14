# Gate 9 -- depth-to-occupancy geometry audit: results

Raw five-frame observed-map metrics against the LiDAR measurement of the same five sweeps (GT relative poses, anchor SSC grid, camera-frustum mask). These are NOT completion IoU and are not comparable to benchmark completion numbers.

## 0006 (590 clips)

| condition | IoU | precision | recall | predicted occupied / clip (in mask) | occupied on LiDAR-free | false-free rate | IoU per clip median [p10, p90] |
|---|---|---|---|---|---|---|---|
| V_all | 1.000 | 1.000 | 1.000 | 9,874 | 0.000 | 0.0000 | 1.000 [1.000, 1.000] |
| V_cam | 0.852 | 1.000 | 0.852 | 8,416 | 0.000 | 0.0000 | 0.863 [0.784, 0.934] |
| LD | 0.438 | 0.532 | 0.711 | 13,180 | 0.468 | 0.0074 | 0.435 [0.398, 0.493] |
| PD | 0.075 | 0.108 | 0.196 | 18,006 | 0.892 | 0.0471 | 0.076 [0.030, 0.153] |
| GF | 0.083 | 0.130 | 0.185 | 14,053 | 0.870 | 0.0399 | 0.083 [0.031, 0.175] |
| Gs | 0.149 | 0.234 | 0.292 | 12,317 | 0.766 | 0.0692 | 0.150 [0.107, 0.211] |
| LD_band0 | 0.453 | 0.747 | 0.535 | 7,077 | 0.253 | 0.0230 | 0.453 [0.405, 0.516] |
| LD_band0.4 | 0.366 | 0.415 | 0.759 | 18,068 | 0.585 | 0.0045 | 0.364 [0.334, 0.415] |
| LD_nocarve | 0.434 | 0.527 | 0.711 | 13,324 | 0.473 | 0.0000 | 0.431 [0.395, 0.489] |
| LD_band0_nocarve | 0.451 | 0.740 | 0.536 | 7,149 | 0.260 | 0.0000 | 0.452 [0.404, 0.513] |
| LD_margin0 | 0.438 | 0.533 | 0.711 | 13,178 | 0.467 | 0.0078 | 0.435 [0.398, 0.493] |
| LD_margin0.1 | 0.438 | 0.532 | 0.711 | 13,178 | 0.468 | 0.0076 | 0.435 [0.398, 0.493] |
| LD_margin0.4 | 0.437 | 0.532 | 0.711 | 13,200 | 0.468 | 0.0069 | 0.435 [0.398, 0.492] |
| LD_gate | 0.374 | 0.543 | 0.546 | 9,923 | 0.457 | 0.0056 | 0.374 [0.326, 0.431] |
| PD_nocarve | 0.074 | 0.106 | 0.197 | 18,331 | 0.894 | 0.0000 | 0.075 [0.030, 0.152] |
| PD_band0_nocarve | 0.063 | 0.119 | 0.118 | 9,775 | 0.881 | 0.0000 | 0.061 [0.022, 0.130] |
| PD_margin0.4 | 0.075 | 0.108 | 0.197 | 18,028 | 0.892 | 0.0463 | 0.076 [0.030, 0.153] |
| PD_gate | 0.079 | 0.132 | 0.165 | 12,372 | 0.868 | 0.0282 | 0.079 [0.029, 0.171] |
| GF_nocarve | 0.082 | 0.127 | 0.187 | 14,580 | 0.873 | 0.0000 | 0.082 [0.031, 0.176] |

Same-surface endpoint support (LiDAR-occupied voxels holding an endpoint from >=1 / >=2 of the five frames): LD 0.643 / 0.281; PD 0.116 / 0.032

### LingBot depth at LiDAR-supported pixels

n = 12,040,167 stored pixels (20 % deterministic subsample). Signed error median -0.38 m; |error| median 0.58 m (mean 1.30); AbsRel median 0.078 (mean 0.112); relative error median -0.054; endpoint error median 0.68 m (p90 3.22); voxel displacement median |dx|,|dy|,|dz| = 3, 1, 0, total 5 voxels; in front 0.721, behind 0.279.

| within 0.1 m | 0.2 m | 0.4 m | 0.6 m | 1.0 m | exact voxel | within 0.2 m wrong voxel | within 0.4 m wrong voxel | within 0.6 m wrong voxel | endpoint on LiDAR-free | endpoint near LiDAR-occupied (±1 voxel) | beyond surface > 0.2 m | carve passes true endpoint | true endpoint voxel FREE in PD map | in GF map |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.082 | 0.164 | 0.319 | 0.452 | 0.637 | 0.0320 | 0.132 | 0.287 | 0.420 | 0.290 | 0.278 | 0.207 | 0.180 | 0.0444 | 0.0439 |

Scale decomposition (oracle scalars from LiDAR): AbsRel raw 0.078 -> after best per-clip scalar 0.046 -> per-frame scalar 0.039; within 0.2 m 0.164 -> 0.297 / 0.350; within 0.6 m 0.452 -> 0.616 / 0.650. Per-clip LiDAR/LingBot ratio median 1.068 [p10 0.967, p90 1.151].

**by range**

| bin | n | AbsRel median | |err| m median | signed m median | endpoint m median | within 0.2 | within 0.4 | within 0.6 | exact voxel | on LiDAR-free | carve passes true endpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-5 m | 2,515,953 | 0.071 | 0.28 | -0.14 | 0.36 | 0.293 | 0.548 | 0.730 | 0.0411 | 0.117 | 0.181 |
| 5-10 m | 5,329,965 | 0.073 | 0.50 | -0.36 | 0.58 | 0.178 | 0.356 | 0.516 | 0.0350 | 0.234 | 0.156 |
| 10-20 m | 3,049,572 | 0.086 | 1.14 | -0.85 | 1.25 | 0.082 | 0.164 | 0.247 | 0.0262 | 0.422 | 0.195 |
| 20-30 m | 708,744 | 0.104 | 2.46 | -1.70 | 2.61 | 0.039 | 0.078 | 0.117 | 0.0174 | 0.577 | 0.250 |
| >30 m | 435,933 | 0.136 | 5.24 | -3.73 | 5.45 | 0.018 | 0.037 | 0.056 | 0.0079 | 0.576 | 0.254 |

**by LiDAR depth discontinuity**

| bin | n | AbsRel median | |err| m median | signed m median | endpoint m median | within 0.2 | within 0.4 | within 0.6 | exact voxel | on LiDAR-free | carve passes true endpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| non-edge (<5 %) | 6,146,679 | 0.071 | 0.43 | -0.30 | 0.53 | 0.204 | 0.396 | 0.554 | 0.0361 | 0.254 | 0.154 |
| moderate (5-20 %) | 4,078,037 | 0.077 | 0.71 | -0.51 | 0.79 | 0.138 | 0.273 | 0.398 | 0.0315 | 0.304 | 0.165 |
| strong (>=20 %) | 1,807,936 | 0.138 | 1.56 | -0.56 | 1.85 | 0.085 | 0.162 | 0.230 | 0.0195 | 0.379 | 0.302 |
| isolated | 7,515 | 0.098 | 0.28 | +0.01 | 0.42 | 0.269 | 0.483 | 0.625 | 0.0317 | 0.100 | 0.317 |

**by local surface**

| bin | n | AbsRel median | |err| m median | signed m median | endpoint m median | within 0.2 | within 0.4 | within 0.6 | exact voxel | on LiDAR-free | carve passes true endpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| planar | 7,704,046 | 0.081 | 0.64 | -0.44 | 0.73 | 0.151 | 0.299 | 0.430 | 0.0309 | 0.299 | 0.171 |
| curved/cluttered | 3,616,586 | 0.074 | 0.55 | -0.32 | 0.67 | 0.169 | 0.326 | 0.458 | 0.0338 | 0.299 | 0.200 |
| isolated (<6 pts) | 719,535 | 0.073 | 0.28 | -0.15 | 0.40 | 0.272 | 0.502 | 0.667 | 0.0356 | 0.138 | 0.179 |

**by LingBot confidence**

| bin | n | AbsRel median | |err| m median | signed m median | endpoint m median | within 0.2 | within 0.4 | within 0.6 | exact voxel | on LiDAR-free | carve passes true endpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <1.5 (gated) | 788,806 | 0.159 | 3.91 | -0.34 | 4.34 | 0.028 | 0.055 | 0.083 | 0.0109 | 0.472 | 0.453 |
| 1.5-3 | 1,121,571 | 0.107 | 1.73 | -0.80 | 1.97 | 0.059 | 0.116 | 0.173 | 0.0221 | 0.492 | 0.323 |
| 3-6 | 2,355,294 | 0.085 | 0.95 | -0.61 | 1.10 | 0.103 | 0.203 | 0.298 | 0.0328 | 0.386 | 0.236 |
| 6-12 | 3,994,538 | 0.073 | 0.55 | -0.42 | 0.65 | 0.169 | 0.330 | 0.471 | 0.0364 | 0.259 | 0.149 |
| >=12 | 3,779,958 | 0.068 | 0.33 | -0.29 | 0.40 | 0.256 | 0.496 | 0.689 | 0.0343 | 0.164 | 0.078 |

**carving vs signed depth error**

| signed error bin (m) | n | carve passes true endpoint | true endpoint voxel FREE in PD map |
|---|---|---|---|
| [-1000000000.0, -2) | 1,326,815 | 0.000 | 0.0113 |
| [-2, -1) | 1,570,263 | 0.000 | 0.0092 |
| [-1, -0.5) | 2,267,164 | 0.000 | 0.0068 |
| [-0.5, -0.2) | 2,166,890 | 0.000 | 0.0043 |
| [-0.2, 0.2) | 2,353,446 | 0.002 | 0.0093 |
| [0.2, 0.5) | 861,010 | 0.778 | 0.0843 |
| [0.5, 1) | 623,954 | 1.000 | 0.1919 |
| [1, 2) | 409,832 | 1.000 | 0.2821 |
| [2, 1000000000.0) | 460,793 | 1.000 | 0.3268 |

Representative clips (PD IoU 10th / 50th / 90th percentile): {'bad': 1747, 'median': 1549, 'good': 16}

