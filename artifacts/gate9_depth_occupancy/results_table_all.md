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

## 0003 (65 clips)

| condition | IoU | precision | recall | predicted occupied / clip (in mask) | occupied on LiDAR-free | false-free rate | IoU per clip median [p10, p90] |
|---|---|---|---|---|---|---|---|
| V_all | 1.000 | 1.000 | 1.000 | 12,404 | 0.000 | 0.0000 | 1.000 [1.000, 1.000] |
| V_cam | 0.839 | 1.000 | 0.839 | 10,411 | 0.000 | 0.0000 | 0.838 [0.767, 0.937] |
| LD | 0.369 | 0.454 | 0.663 | 18,090 | 0.546 | 0.0110 | 0.368 [0.311, 0.415] |
| PD | 0.036 | 0.052 | 0.109 | 26,187 | 0.948 | 0.0275 | 0.036 [0.011, 0.074] |
| GF | 0.037 | 0.054 | 0.104 | 23,793 | 0.946 | 0.0217 | 0.038 [0.010, 0.081] |
| Gs | 0.120 | 0.179 | 0.269 | 18,606 | 0.821 | 0.0806 | 0.122 [0.084, 0.166] |
| LD_band0 | 0.383 | 0.671 | 0.472 | 8,728 | 0.329 | 0.0267 | 0.381 [0.338, 0.453] |
| LD_band0.4 | 0.312 | 0.355 | 0.722 | 25,238 | 0.645 | 0.0073 | 0.315 [0.263, 0.353] |
| LD_nocarve | 0.367 | 0.452 | 0.663 | 18,197 | 0.548 | 0.0000 | 0.367 [0.311, 0.412] |
| LD_band0_nocarve | 0.382 | 0.667 | 0.472 | 8,780 | 0.333 | 0.0000 | 0.380 [0.337, 0.447] |
| LD_margin0 | 0.369 | 0.454 | 0.663 | 18,086 | 0.546 | 0.0115 | 0.368 [0.312, 0.415] |
| LD_margin0.1 | 0.369 | 0.454 | 0.663 | 18,087 | 0.546 | 0.0112 | 0.368 [0.311, 0.415] |
| LD_margin0.4 | 0.369 | 0.454 | 0.663 | 18,104 | 0.546 | 0.0104 | 0.368 [0.311, 0.414] |
| LD_gate | 0.323 | 0.447 | 0.539 | 14,941 | 0.553 | 0.0089 | 0.326 [0.242, 0.384] |
| PD_nocarve | 0.036 | 0.051 | 0.109 | 26,394 | 0.949 | 0.0000 | 0.036 [0.011, 0.074] |
| PD_band0_nocarve | 0.027 | 0.052 | 0.055 | 13,060 | 0.948 | 0.0000 | 0.026 [0.008, 0.059] |
| PD_margin0.4 | 0.036 | 0.052 | 0.109 | 26,205 | 0.948 | 0.0269 | 0.036 [0.011, 0.074] |
| PD_gate | 0.035 | 0.054 | 0.088 | 20,209 | 0.946 | 0.0135 | 0.034 [0.008, 0.079] |
| GF_nocarve | 0.037 | 0.054 | 0.104 | 24,175 | 0.946 | 0.0000 | 0.037 [0.010, 0.081] |

Same-surface endpoint support (LiDAR-occupied voxels holding an endpoint from >=1 / >=2 of the five frames): LD 0.586 / 0.178; PD 0.053 / 0.006

### LingBot depth at LiDAR-supported pixels

n = 1,316,453 stored pixels (20 % deterministic subsample). Signed error median -0.84 m; |error| median 0.95 m (mean 1.98); AbsRel median 0.104 (mean 0.131); relative error median -0.098; endpoint error median 1.11 m (p90 4.88); voxel displacement median |dx|,|dy|,|dz| = 5, 2, 1, total 8 voxels; in front 0.892, behind 0.108.

| within 0.1 m | 0.2 m | 0.4 m | 0.6 m | 1.0 m | exact voxel | within 0.2 m wrong voxel | within 0.4 m wrong voxel | within 0.6 m wrong voxel | endpoint on LiDAR-free | endpoint near LiDAR-occupied (±1 voxel) | beyond surface > 0.2 m | carve passes true endpoint | true endpoint voxel FREE in PD map | in GF map |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.028 | 0.058 | 0.139 | 0.253 | 0.456 | 0.0088 | 0.049 | 0.131 | 0.245 | 0.335 | 0.163 | 0.087 | 0.080 | 0.0193 | 0.0168 |

Scale decomposition (oracle scalars from LiDAR): AbsRel raw 0.104 -> after best per-clip scalar 0.036 -> per-frame scalar 0.032; within 0.2 m 0.058 -> 0.317 / 0.361; within 0.6 m 0.253 -> 0.598 / 0.617. Per-clip LiDAR/LingBot ratio median 1.107 [p10 1.049, p90 1.190].

**by range**

| bin | n | AbsRel median | |err| m median | signed m median | endpoint m median | within 0.2 | within 0.4 | within 0.6 | exact voxel | on LiDAR-free | carve passes true endpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-5 m | 132,571 | 0.092 | 0.42 | -0.40 | 0.51 | 0.130 | 0.344 | 0.614 | 0.0122 | 0.181 | 0.042 |
| 5-10 m | 536,595 | 0.098 | 0.67 | -0.65 | 0.79 | 0.070 | 0.175 | 0.338 | 0.0086 | 0.238 | 0.048 |
| 10-20 m | 412,124 | 0.108 | 1.44 | -1.31 | 1.64 | 0.040 | 0.084 | 0.135 | 0.0088 | 0.390 | 0.104 |
| 20-30 m | 125,158 | 0.120 | 2.88 | -2.55 | 3.14 | 0.025 | 0.052 | 0.079 | 0.0088 | 0.540 | 0.144 |
| >30 m | 110,005 | 0.147 | 5.75 | -5.35 | 6.20 | 0.014 | 0.029 | 0.044 | 0.0054 | 0.553 | 0.125 |

**by LiDAR depth discontinuity**

| bin | n | AbsRel median | |err| m median | signed m median | endpoint m median | within 0.2 | within 0.4 | within 0.6 | exact voxel | on LiDAR-free | carve passes true endpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| non-edge (<5 %) | 509,318 | 0.096 | 0.64 | -0.60 | 0.77 | 0.075 | 0.194 | 0.370 | 0.0092 | 0.276 | 0.060 |
| moderate (5-20 %) | 595,537 | 0.105 | 1.13 | -1.08 | 1.28 | 0.048 | 0.110 | 0.196 | 0.0080 | 0.366 | 0.047 |
| strong (>=20 %) | 211,359 | 0.144 | 2.50 | -1.65 | 2.97 | 0.046 | 0.091 | 0.134 | 0.0100 | 0.389 | 0.223 |
| isolated | 239 | 0.093 | 0.39 | -0.20 | 0.60 | 0.180 | 0.347 | 0.498 | 0.0293 | 0.130 | 0.247 |

**by local surface**

| bin | n | AbsRel median | |err| m median | signed m median | endpoint m median | within 0.2 | within 0.4 | within 0.6 | exact voxel | on LiDAR-free | carve passes true endpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| planar | 1,030,051 | 0.107 | 0.97 | -0.88 | 1.12 | 0.050 | 0.128 | 0.244 | 0.0079 | 0.338 | 0.069 |
| curved/cluttered | 231,434 | 0.091 | 1.01 | -0.81 | 1.27 | 0.081 | 0.164 | 0.250 | 0.0124 | 0.355 | 0.133 |
| isolated (<6 pts) | 54,968 | 0.095 | 0.48 | -0.43 | 0.68 | 0.109 | 0.256 | 0.437 | 0.0104 | 0.191 | 0.074 |

**by LingBot confidence**

| bin | n | AbsRel median | |err| m median | signed m median | endpoint m median | within 0.2 | within 0.4 | within 0.6 | exact voxel | on LiDAR-free | carve passes true endpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <1.5 (gated) | 101,060 | 0.159 | 4.87 | -1.90 | 5.77 | 0.021 | 0.042 | 0.062 | 0.0062 | 0.374 | 0.371 |
| 1.5-3 | 139,724 | 0.127 | 2.75 | -2.33 | 3.22 | 0.032 | 0.065 | 0.097 | 0.0108 | 0.558 | 0.197 |
| 3-6 | 241,816 | 0.114 | 1.63 | -1.57 | 1.92 | 0.045 | 0.092 | 0.141 | 0.0107 | 0.483 | 0.102 |
| 6-12 | 323,927 | 0.100 | 1.03 | -1.03 | 1.21 | 0.060 | 0.131 | 0.215 | 0.0100 | 0.342 | 0.039 |
| >=12 | 509,926 | 0.096 | 0.59 | -0.59 | 0.69 | 0.077 | 0.207 | 0.412 | 0.0071 | 0.191 | 0.007 |

**carving vs signed depth error**

| signed error bin (m) | n | carve passes true endpoint | true endpoint voxel FREE in PD map |
|---|---|---|---|
| [-1000000000.0, -2) | 280,040 | 0.000 | 0.0068 |
| [-2, -1) | 287,169 | 0.000 | 0.0110 |
| [-1, -0.5) | 340,090 | 0.000 | 0.0066 |
| [-0.5, -0.2) | 203,369 | 0.000 | 0.0062 |
| [-0.2, 0.2) | 96,405 | 0.002 | 0.0089 |
| [0.2, 0.5) | 24,563 | 0.845 | 0.0575 |
| [0.5, 1) | 22,331 | 1.000 | 0.1064 |
| [1, 2) | 22,013 | 1.000 | 0.1494 |
| [2, 1000000000.0) | 40,473 | 1.000 | 0.2208 |

Representative clips (PD IoU 10th / 50th / 90th percentile): {'bad': 169, 'median': 40, 'good': 166}

## 0007 (190 clips)

| condition | IoU | precision | recall | predicted occupied / clip (in mask) | occupied on LiDAR-free | false-free rate | IoU per clip median [p10, p90] |
|---|---|---|---|---|---|---|---|
| V_all | 1.000 | 1.000 | 1.000 | 12,028 | 0.000 | 0.0000 | 1.000 [1.000, 1.000] |
| V_cam | 0.820 | 1.000 | 0.820 | 9,860 | 0.000 | 0.0000 | 0.827 [0.739, 0.914] |
| LD | 0.378 | 0.473 | 0.654 | 16,635 | 0.527 | 0.0116 | 0.379 [0.336, 0.422] |
| PD | 0.031 | 0.047 | 0.087 | 22,326 | 0.953 | 0.0166 | 0.029 [0.006, 0.062] |
| GF | 0.033 | 0.050 | 0.086 | 20,571 | 0.950 | 0.0135 | 0.030 [0.006, 0.071] |
| Gs | 0.125 | 0.189 | 0.271 | 17,222 | 0.811 | 0.0743 | 0.127 [0.070, 0.188] |
| LD_band0 | 0.382 | 0.686 | 0.463 | 8,123 | 0.314 | 0.0285 | 0.391 [0.326, 0.433] |
| LD_band0.4 | 0.324 | 0.372 | 0.717 | 23,213 | 0.628 | 0.0075 | 0.323 [0.282, 0.372] |
| LD_nocarve | 0.376 | 0.469 | 0.654 | 16,754 | 0.531 | 0.0000 | 0.377 [0.334, 0.421] |
| LD_band0_nocarve | 0.381 | 0.682 | 0.464 | 8,179 | 0.318 | 0.0000 | 0.390 [0.326, 0.430] |
| LD_margin0 | 0.378 | 0.473 | 0.654 | 16,631 | 0.527 | 0.0121 | 0.379 [0.336, 0.423] |
| LD_margin0.1 | 0.378 | 0.473 | 0.654 | 16,633 | 0.527 | 0.0119 | 0.379 [0.336, 0.423] |
| LD_margin0.4 | 0.378 | 0.472 | 0.654 | 16,648 | 0.528 | 0.0109 | 0.379 [0.336, 0.422] |
| LD_gate | 0.346 | 0.475 | 0.560 | 14,188 | 0.525 | 0.0100 | 0.348 [0.283, 0.407] |
| PD_nocarve | 0.031 | 0.047 | 0.087 | 22,536 | 0.953 | 0.0000 | 0.029 [0.006, 0.062] |
| PD_band0_nocarve | 0.024 | 0.048 | 0.045 | 11,222 | 0.952 | 0.0000 | 0.021 [0.004, 0.052] |
| PD_margin0.4 | 0.031 | 0.047 | 0.087 | 22,341 | 0.953 | 0.0162 | 0.029 [0.006, 0.062] |
| PD_gate | 0.031 | 0.051 | 0.075 | 17,615 | 0.949 | 0.0090 | 0.028 [0.005, 0.069] |
| GF_nocarve | 0.032 | 0.049 | 0.086 | 20,987 | 0.951 | 0.0000 | 0.030 [0.006, 0.070] |

Same-surface endpoint support (LiDAR-occupied voxels holding an endpoint from >=1 / >=2 of the five frames): LD 0.581 / 0.155; PD 0.044 / 0.006

### LingBot depth at LiDAR-supported pixels

n = 3,822,164 stored pixels (20 % deterministic subsample). Signed error median -1.05 m; |error| median 1.14 m (mean 2.29); AbsRel median 0.130 (mean 0.155); relative error median -0.125; endpoint error median 1.34 m (p90 5.76); voxel displacement median |dx|,|dy|,|dz| = 6, 3, 1, total 9 voxels; in front 0.911, behind 0.089.

| within 0.1 m | 0.2 m | 0.4 m | 0.6 m | 1.0 m | exact voxel | within 0.2 m wrong voxel | within 0.4 m wrong voxel | within 0.6 m wrong voxel | endpoint on LiDAR-free | endpoint near LiDAR-occupied (±1 voxel) | beyond surface > 0.2 m | carve passes true endpoint | true endpoint voxel FREE in PD map | in GF map |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.024 | 0.049 | 0.112 | 0.195 | 0.377 | 0.0085 | 0.041 | 0.103 | 0.187 | 0.296 | 0.146 | 0.070 | 0.063 | 0.0148 | 0.0137 |

Scale decomposition (oracle scalars from LiDAR): AbsRel raw 0.130 -> after best per-clip scalar 0.044 -> per-frame scalar 0.039; within 0.2 m 0.049 -> 0.284 / 0.324; within 0.6 m 0.195 -> 0.548 / 0.567. Per-clip LiDAR/LingBot ratio median 1.141 [p10 1.062, p90 1.237].

**by range**

| bin | n | AbsRel median | |err| m median | signed m median | endpoint m median | within 0.2 | within 0.4 | within 0.6 | exact voxel | on LiDAR-free | carve passes true endpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-5 m | 388,304 | 0.105 | 0.48 | -0.46 | 0.58 | 0.149 | 0.320 | 0.521 | 0.0188 | 0.180 | 0.043 |
| 5-10 m | 1,625,859 | 0.125 | 0.84 | -0.82 | 0.99 | 0.051 | 0.126 | 0.239 | 0.0076 | 0.221 | 0.047 |
| 10-20 m | 1,167,293 | 0.135 | 1.80 | -1.72 | 2.03 | 0.033 | 0.068 | 0.107 | 0.0080 | 0.342 | 0.072 |
| 20-30 m | 353,852 | 0.152 | 3.61 | -3.41 | 3.94 | 0.021 | 0.041 | 0.062 | 0.0070 | 0.459 | 0.097 |
| >30 m | 286,856 | 0.180 | 7.04 | -6.65 | 7.55 | 0.010 | 0.021 | 0.031 | 0.0038 | 0.492 | 0.106 |

**by LiDAR depth discontinuity**

| bin | n | AbsRel median | |err| m median | signed m median | endpoint m median | within 0.2 | within 0.4 | within 0.6 | exact voxel | on LiDAR-free | carve passes true endpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| non-edge (<5 %) | 1,382,545 | 0.119 | 0.78 | -0.75 | 0.91 | 0.065 | 0.156 | 0.285 | 0.0097 | 0.254 | 0.047 |
| moderate (5-20 %) | 1,787,444 | 0.133 | 1.35 | -1.31 | 1.55 | 0.038 | 0.085 | 0.147 | 0.0072 | 0.308 | 0.041 |
| strong (>=20 %) | 650,997 | 0.162 | 2.67 | -2.10 | 3.13 | 0.047 | 0.093 | 0.137 | 0.0097 | 0.353 | 0.159 |
| isolated | 1,178 | 0.119 | 0.54 | -0.33 | 0.83 | 0.175 | 0.301 | 0.408 | 0.0297 | 0.127 | 0.170 |

**by local surface**

| bin | n | AbsRel median | |err| m median | signed m median | endpoint m median | within 0.2 | within 0.4 | within 0.6 | exact voxel | on LiDAR-free | carve passes true endpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| planar | 2,838,663 | 0.133 | 1.15 | -1.07 | 1.31 | 0.040 | 0.099 | 0.185 | 0.0072 | 0.298 | 0.059 |
| curved/cluttered | 808,145 | 0.122 | 1.30 | -1.19 | 1.63 | 0.064 | 0.128 | 0.193 | 0.0112 | 0.314 | 0.077 |
| isolated (<6 pts) | 175,356 | 0.113 | 0.58 | -0.52 | 0.82 | 0.136 | 0.258 | 0.377 | 0.0185 | 0.177 | 0.075 |

**by LingBot confidence**

| bin | n | AbsRel median | |err| m median | signed m median | endpoint m median | within 0.2 | within 0.4 | within 0.6 | exact voxel | on LiDAR-free | carve passes true endpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <1.5 (gated) | 225,002 | 0.181 | 5.88 | -3.30 | 6.81 | 0.019 | 0.037 | 0.056 | 0.0060 | 0.406 | 0.303 |
| 1.5-3 | 384,540 | 0.155 | 3.70 | -3.42 | 4.26 | 0.028 | 0.055 | 0.082 | 0.0089 | 0.491 | 0.144 |
| 3-6 | 792,071 | 0.129 | 1.83 | -1.77 | 2.14 | 0.047 | 0.094 | 0.142 | 0.0113 | 0.390 | 0.088 |
| 6-12 | 1,118,546 | 0.126 | 1.19 | -1.18 | 1.42 | 0.057 | 0.120 | 0.191 | 0.0099 | 0.279 | 0.036 |
| >=12 | 1,302,005 | 0.126 | 0.75 | -0.75 | 0.86 | 0.056 | 0.146 | 0.289 | 0.0061 | 0.178 | 0.006 |

**carving vs signed depth error**

| signed error bin (m) | n | carve passes true endpoint | true endpoint voxel FREE in PD map |
|---|---|---|---|
| [-1000000000.0, -2) | 1,034,830 | 0.000 | 0.0054 |
| [-2, -1) | 956,779 | 0.000 | 0.0082 |
| [-1, -0.5) | 917,203 | 0.000 | 0.0086 |
| [-0.5, -0.2) | 420,918 | 0.000 | 0.0099 |
| [-0.2, 0.2) | 240,067 | 0.002 | 0.0101 |
| [0.2, 0.5) | 66,859 | 0.842 | 0.0586 |
| [0.5, 1) | 55,756 | 1.000 | 0.0935 |
| [1, 2) | 46,430 | 1.000 | 0.1245 |
| [2, 1000000000.0) | 83,322 | 1.000 | 0.1656 |

Representative clips (PD IoU 10th / 50th / 90th percentile): {'bad': 289, 'median': 7, 'good': 238}

## 0010 (199 clips)

| condition | IoU | precision | recall | predicted occupied / clip (in mask) | occupied on LiDAR-free | false-free rate | IoU per clip median [p10, p90] |
|---|---|---|---|---|---|---|---|
| V_all | 1.000 | 1.000 | 1.000 | 11,844 | 0.000 | 0.0000 | 1.000 [1.000, 1.000] |
| V_cam | 0.866 | 1.000 | 0.866 | 10,259 | 0.000 | 0.0000 | 0.867 [0.813, 0.942] |
| LD | 0.393 | 0.470 | 0.704 | 17,734 | 0.530 | 0.0092 | 0.393 [0.355, 0.445] |
| PD | 0.026 | 0.035 | 0.085 | 28,394 | 0.965 | 0.0219 | 0.021 [0.009, 0.058] |
| GF | 0.026 | 0.038 | 0.081 | 25,567 | 0.962 | 0.0167 | 0.021 [0.008, 0.067] |
| Gs | 0.121 | 0.173 | 0.286 | 19,566 | 0.827 | 0.0855 | 0.122 [0.086, 0.176] |
| LD_band0 | 0.420 | 0.698 | 0.513 | 8,712 | 0.302 | 0.0250 | 0.419 [0.372, 0.480] |
| LD_band0.4 | 0.320 | 0.358 | 0.753 | 24,927 | 0.642 | 0.0062 | 0.322 [0.287, 0.363] |
| LD_nocarve | 0.390 | 0.466 | 0.704 | 17,893 | 0.534 | 0.0000 | 0.392 [0.352, 0.439] |
| LD_band0_nocarve | 0.419 | 0.693 | 0.514 | 8,780 | 0.307 | 0.0000 | 0.418 [0.371, 0.479] |
| LD_margin0 | 0.393 | 0.470 | 0.704 | 17,730 | 0.530 | 0.0097 | 0.393 [0.355, 0.445] |
| LD_margin0.1 | 0.393 | 0.470 | 0.704 | 17,732 | 0.530 | 0.0094 | 0.393 [0.355, 0.445] |
| LD_margin0.4 | 0.392 | 0.470 | 0.704 | 17,752 | 0.530 | 0.0087 | 0.393 [0.355, 0.444] |
| LD_gate | 0.347 | 0.466 | 0.576 | 14,633 | 0.534 | 0.0076 | 0.345 [0.288, 0.417] |
| PD_nocarve | 0.025 | 0.035 | 0.085 | 28,748 | 0.965 | 0.0000 | 0.021 [0.009, 0.057] |
| PD_band0_nocarve | 0.020 | 0.036 | 0.044 | 14,492 | 0.964 | 0.0000 | 0.015 [0.006, 0.046] |
| PD_margin0.4 | 0.026 | 0.035 | 0.085 | 28,416 | 0.965 | 0.0215 | 0.021 [0.009, 0.058] |
| PD_gate | 0.025 | 0.037 | 0.068 | 21,570 | 0.963 | 0.0100 | 0.019 [0.007, 0.061] |
| GF_nocarve | 0.026 | 0.037 | 0.082 | 26,128 | 0.963 | 0.0000 | 0.021 [0.008, 0.066] |

Same-surface endpoint support (LiDAR-occupied voxels holding an endpoint from >=1 / >=2 of the five frames): LD 0.634 / 0.236; PD 0.041 / 0.007

### LingBot depth at LiDAR-supported pixels

n = 3,975,327 stored pixels (20 % deterministic subsample). Signed error median -0.99 m; |error| median 1.09 m (mean 2.26); AbsRel median 0.124 (mean 0.158); relative error median -0.118; endpoint error median 1.28 m (p90 5.57); voxel displacement median |dx|,|dy|,|dz| = 5, 3, 1, total 9 voxels; in front 0.917, behind 0.083.

| within 0.1 m | 0.2 m | 0.4 m | 0.6 m | 1.0 m | exact voxel | within 0.2 m wrong voxel | within 0.4 m wrong voxel | within 0.6 m wrong voxel | endpoint on LiDAR-free | endpoint near LiDAR-occupied (±1 voxel) | beyond surface > 0.2 m | carve passes true endpoint | true endpoint voxel FREE in PD map | in GF map |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.018 | 0.040 | 0.099 | 0.184 | 0.390 | 0.0079 | 0.032 | 0.091 | 0.176 | 0.412 | 0.157 | 0.069 | 0.064 | 0.0225 | 0.0197 |

Scale decomposition (oracle scalars from LiDAR): AbsRel raw 0.124 -> after best per-clip scalar 0.037 -> per-frame scalar 0.032; within 0.2 m 0.040 -> 0.312 / 0.360; within 0.6 m 0.184 -> 0.604 / 0.630. Per-clip LiDAR/LingBot ratio median 1.138 [p10 1.075, p90 1.212].

**by range**

| bin | n | AbsRel median | |err| m median | signed m median | endpoint m median | within 0.2 | within 0.4 | within 0.6 | exact voxel | on LiDAR-free | carve passes true endpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-5 m | 406,161 | 0.110 | 0.49 | -0.47 | 0.60 | 0.101 | 0.272 | 0.503 | 0.0120 | 0.215 | 0.042 |
| 5-10 m | 1,678,617 | 0.118 | 0.80 | -0.77 | 0.93 | 0.045 | 0.116 | 0.233 | 0.0079 | 0.316 | 0.042 |
| 10-20 m | 1,223,199 | 0.126 | 1.69 | -1.59 | 1.93 | 0.026 | 0.056 | 0.091 | 0.0079 | 0.478 | 0.078 |
| 20-30 m | 371,618 | 0.142 | 3.38 | -3.15 | 3.73 | 0.016 | 0.032 | 0.048 | 0.0065 | 0.662 | 0.103 |
| >30 m | 295,732 | 0.172 | 6.66 | -6.19 | 7.20 | 0.009 | 0.019 | 0.028 | 0.0036 | 0.635 | 0.111 |

**by LiDAR depth discontinuity**

| bin | n | AbsRel median | |err| m median | signed m median | endpoint m median | within 0.2 | within 0.4 | within 0.6 | exact voxel | on LiDAR-free | carve passes true endpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| non-edge (<5 %) | 1,785,697 | 0.116 | 0.78 | -0.74 | 0.93 | 0.050 | 0.133 | 0.258 | 0.0086 | 0.356 | 0.049 |
| moderate (5-20 %) | 1,571,835 | 0.125 | 1.32 | -1.29 | 1.49 | 0.028 | 0.068 | 0.127 | 0.0064 | 0.437 | 0.033 |
| strong (>=20 %) | 616,467 | 0.164 | 3.04 | -2.08 | 3.56 | 0.039 | 0.078 | 0.116 | 0.0095 | 0.510 | 0.186 |
| isolated | 1,328 | 0.137 | 0.68 | -0.16 | 0.98 | 0.139 | 0.276 | 0.374 | 0.0233 | 0.180 | 0.325 |

**by local surface**

| bin | n | AbsRel median | |err| m median | signed m median | endpoint m median | within 0.2 | within 0.4 | within 0.6 | exact voxel | on LiDAR-free | carve passes true endpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| planar | 2,983,602 | 0.127 | 1.12 | -1.03 | 1.29 | 0.032 | 0.085 | 0.171 | 0.0066 | 0.411 | 0.058 |
| curved/cluttered | 807,755 | 0.111 | 1.14 | -1.00 | 1.43 | 0.053 | 0.119 | 0.195 | 0.0114 | 0.453 | 0.082 |
| isolated (<6 pts) | 183,970 | 0.115 | 0.57 | -0.51 | 0.79 | 0.099 | 0.222 | 0.361 | 0.0134 | 0.233 | 0.081 |

**by LingBot confidence**

| bin | n | AbsRel median | |err| m median | signed m median | endpoint m median | within 0.2 | within 0.4 | within 0.6 | exact voxel | on LiDAR-free | carve passes true endpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <1.5 (gated) | 277,239 | 0.199 | 6.21 | -2.22 | 7.13 | 0.019 | 0.037 | 0.054 | 0.0069 | 0.489 | 0.372 |
| 1.5-3 | 346,377 | 0.144 | 3.31 | -2.87 | 3.83 | 0.030 | 0.059 | 0.088 | 0.0101 | 0.614 | 0.170 |
| 3-6 | 649,763 | 0.126 | 2.04 | -1.97 | 2.37 | 0.037 | 0.074 | 0.113 | 0.0104 | 0.582 | 0.077 |
| 6-12 | 1,073,200 | 0.118 | 1.23 | -1.22 | 1.44 | 0.042 | 0.093 | 0.157 | 0.0093 | 0.444 | 0.029 |
| >=12 | 1,628,748 | 0.120 | 0.73 | -0.73 | 0.85 | 0.045 | 0.131 | 0.274 | 0.0056 | 0.266 | 0.007 |

**carving vs signed depth error**

| signed error bin (m) | n | carve passes true endpoint | true endpoint voxel FREE in PD map |
|---|---|---|---|
| [-1000000000.0, -2) | 976,596 | 0.000 | 0.0101 |
| [-2, -1) | 987,543 | 0.000 | 0.0132 |
| [-1, -0.5) | 1,082,866 | 0.000 | 0.0113 |
| [-0.5, -0.2) | 459,172 | 0.000 | 0.0078 |
| [-0.2, 0.2) | 205,074 | 0.002 | 0.0107 |
| [0.2, 0.5) | 51,958 | 0.824 | 0.0728 |
| [0.5, 1) | 44,156 | 1.000 | 0.1434 |
| [1, 2) | 45,941 | 1.000 | 0.1912 |
| [2, 1000000000.0) | 122,021 | 1.000 | 0.2419 |

Representative clips (PD IoU 10th / 50th / 90th percentile): {'bad': 298, 'median': 487, 'good': 247}

