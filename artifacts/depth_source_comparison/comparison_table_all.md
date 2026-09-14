# Controlled depth-source comparison: results

Raw five-frame observed maps vs the LiDAR measurement of the same five sweeps; KITTI-360 GT relative poses, Gate 9 clips, lattice, grid, evaluation mask and frozen mapper. Not completion IoU; says nothing about Occ3D numbers.

Masks: **S1 (primary)** = Gate 9 LiDAR support AND MoGe-2 valid AND LingBot conf >= 1.5; S0 (secondary) = support AND MoGe-2 valid. IoU pools voxel counts over clips; depth metrics pool the Gate 9 deterministic 20 % pixel subsample inside the mask; bias = median signed error; endpoint = |z - z_LiDAR| x |ray|.

## 0006 (590 clips)

Support: Gate 9 LiDAR support 60,195,655 px; MoGe mask removes 0.03 %, LingBot gate 6.56 %, S1 keeps 56,234,710 (93.42 %). Mapper 1-60 m range rejects of S1 rays: LD 0.000 %, LB-M 0.000 %, MG 0.001 %, LB-O 0.000 %.

| primary S1 | IoU | precision | recall | occupied in LiDAR-free (fraction / per clip) |
|---|---|---|---|---|
| LD | 0.374 | 0.543 | 0.546 | 0.457 / 4,530 |
| LB-M | 0.079 | 0.132 | 0.165 | 0.868 / 10,738 |
| MG | 0.088 | 0.140 | 0.189 | 0.860 / 11,425 |
| LB-O | 0.141 | 0.236 | 0.260 | 0.764 / 8,279 |

Gap closed (S1, denominator IoU_LD - IoU_LB-M = 0.295): MG +2.8 % [+1.6, +4.2]; LB-O +21.0 % [+18.4, +23.8] (95 % clip-block bootstrap).

| depth on S1 | n px | bias rel (median) | bias m (median) | AbsRel median / mean | endpoint median | p90 | within 0.2 m | within 0.6 m | exact 0.2 m voxel |
|---|---|---|---|---|---|---|---|---|---|
| LB-M | 11,248,756 | -0.056 | -0.38 | 0.076 / 0.101 | 0.64 | 2.45 | 0.173 | 0.478 | 0.0335 |
| MG | 11,248,756 | -0.056 | -0.38 | 0.080 / 0.106 | 0.67 | 2.72 | 0.167 | 0.461 | 0.0344 |
| LB-O | 11,248,756 | +0.000 | +0.00 | 0.043 / 0.075 | 0.37 | 1.91 | 0.317 | 0.653 | 0.0733 |

| paired per clip (S1) | pooled IoU diff [95 % CI] | per-clip mean | median [p10, p90] | clips improved |
|---|---|---|---|---|
| MG - LB-M | +0.008 [+0.005, +0.012] | +0.009 | +0.008 [-0.036, +0.056] | 59 % |
| LB-O - LB-M | +0.062 [+0.053, +0.072] | +0.059 | +0.052 [-0.007, +0.135] | 86 % |
| LB-O - MG | +0.054 [+0.043, +0.065] | +0.049 | +0.046 [-0.021, +0.126] | 80 % |
| LD - MG | +0.287 [+0.275, +0.299] | +0.278 | +0.280 [+0.195, +0.358] | 100 % |
| LD - LB-O | +0.233 [+0.226, +0.239] | +0.228 | +0.232 [+0.158, +0.290] | 100 % |

| range (S1) | n px | LB-M endpoint med / within 0.2 / within 0.6 / bias | MG | LB-O |
|---|---|---|---|---|
| 0-5 m | 2,497,829 | 0.36 / 0.295 / 0.735 / -0.036 | 0.37 / 0.290 / 0.715 / -0.032 | 0.23 / 0.450 / 0.831 / +0.010 |
| 5-10 m | 5,273,254 | 0.57 / 0.180 / 0.520 / -0.055 | 0.59 / 0.172 / 0.506 / -0.056 | 0.30 / 0.362 / 0.731 / +0.001 |
| 10-20 m | 2,860,653 | 1.20 / 0.085 / 0.256 / -0.067 | 1.32 / 0.079 / 0.234 / -0.072 | 0.69 / 0.170 / 0.451 / -0.006 |
| 20-30 m | 502,653 | 2.36 / 0.043 / 0.127 / -0.084 | 2.62 / 0.038 / 0.112 / -0.086 | 1.47 / 0.079 / 0.231 / -0.022 |
| >30 m | 114,367 | 4.60 / 0.018 / 0.055 / -0.126 | 4.96 / 0.017 / 0.053 / -0.131 | 3.04 / 0.039 / 0.119 / -0.068 |

| secondary S0 (no LingBot gate) | IoU | precision | recall | occupied in LiDAR-free |
|---|---|---|---|---|
| LD | 0.438 | 0.532 | 0.710 | 0.468 |
| LB-M | 0.075 | 0.108 | 0.196 | 0.892 |
| MG | 0.081 | 0.114 | 0.216 | 0.886 |
| LB-O | 0.126 | 0.181 | 0.292 | 0.819 |

S0 gap closed: MG +1.6 %, LB-O +14.1 %. S0 depth: LB-M endpoint 0.68 m, within 0.2 0.164; MG endpoint 0.71 m, within 0.2 0.158; LB-O endpoint 0.41 m, within 0.2 0.297

| original Gate 9 conditions (reproduced exactly, Gate 9 support) | IoU | precision | recall |
|---|---|---|---|
| Gate 9 LD | 0.438 | 0.532 | 0.711 |
| Gate 9 PD (= LB-M) | 0.075 | 0.108 | 0.196 |
| Gate 9 LD_gate | 0.374 | 0.543 | 0.546 |
| Gate 9 PD_gate (= LB-M, gated) | 0.079 | 0.132 | 0.165 |

Scalars median [p10, p90]: MoGe clip scale 20.43 [16.42, 24.23]; LB-O scalar 21.84 [17.43, 25.85]; LB-O / MoGe scale 1.057 [0.982, 1.133]. Representative clips: {'bad': 1747, 'median': 1549, 'good': 16}

## 0003 (65 clips)

Support: Gate 9 LiDAR support 6,583,554 px; MoGe mask removes 0.06 %, LingBot gate 7.68 %, S1 keeps 6,077,213 (92.31 %). Mapper 1-60 m range rejects of S1 rays: LD 0.000 %, LB-M 0.000 %, MG 0.009 %, LB-O 0.001 %.

| primary S1 | IoU | precision | recall | occupied in LiDAR-free (fraction / per clip) |
|---|---|---|---|---|
| LD | 0.323 | 0.447 | 0.539 | 0.553 / 8,259 |
| LB-M | 0.035 | 0.054 | 0.088 | 0.946 / 19,120 |
| MG | 0.045 | 0.070 | 0.114 | 0.930 / 18,912 |
| LB-O | 0.115 | 0.184 | 0.234 | 0.816 / 12,879 |

Gap closed (S1, denominator IoU_LD - IoU_LB-M = 0.289): MG +3.7 % [+1.9, +7.2]; LB-O +27.8 % [+24.0, +31.7] (95 % clip-block bootstrap).

| depth on S1 | n px | bias rel (median) | bias m (median) | AbsRel median / mean | endpoint median | p90 | within 0.2 m | within 0.6 m | exact 0.2 m voxel |
|---|---|---|---|---|---|---|---|---|---|
| LB-M | 1,215,253 | -0.099 | -0.83 | 0.102 / 0.118 | 1.03 | 3.65 | 0.061 | 0.269 | 0.0090 |
| MG | 1,215,253 | -0.098 | -0.80 | 0.103 / 0.120 | 1.03 | 4.01 | 0.077 | 0.280 | 0.0124 |
| LB-O | 1,215,253 | +0.000 | +0.00 | 0.033 / 0.060 | 0.35 | 2.35 | 0.343 | 0.642 | 0.0640 |

| paired per clip (S1) | pooled IoU diff [95 % CI] | per-clip mean | median [p10, p90] | clips improved |
|---|---|---|---|---|
| MG - LB-M | +0.011 [+0.005, +0.021] | +0.014 | +0.010 [-0.022, +0.048] | 69 % |
| LB-O - LB-M | +0.080 [+0.069, +0.092] | +0.080 | +0.084 [+0.039, +0.114] | 100 % |
| LB-O - MG | +0.069 [+0.050, +0.085] | +0.066 | +0.067 [+0.009, +0.112] | 92 % |
| LD - MG | +0.278 [+0.271, +0.284] | +0.268 | +0.267 [+0.199, +0.327] | 100 % |
| LD - LB-O | +0.209 [+0.199, +0.221] | +0.201 | +0.211 [+0.130, +0.255] | 100 % |

| range (S1) | n px | LB-M endpoint med / within 0.2 / within 0.6 / bias | MG | LB-O |
|---|---|---|---|---|
| 0-5 m | 132,032 | 0.51 / 0.130 / 0.617 / -0.088 | 0.49 / 0.186 / 0.619 / -0.084 | 0.15 / 0.605 / 0.915 / +0.006 |
| 5-10 m | 531,155 | 0.79 / 0.070 / 0.341 / -0.096 | 0.79 / 0.088 / 0.347 / -0.096 | 0.22 / 0.469 / 0.814 / +0.001 |
| 10-20 m | 394,589 | 1.61 / 0.041 / 0.137 / -0.101 | 1.60 / 0.049 / 0.160 / -0.100 | 0.61 / 0.195 / 0.494 / +0.000 |
| 20-30 m | 104,661 | 2.98 / 0.026 / 0.082 / -0.114 | 3.18 / 0.027 / 0.083 / -0.114 | 1.35 / 0.088 / 0.253 / -0.007 |
| >30 m | 52,816 | 5.71 / 0.012 / 0.038 / -0.150 | 5.71 / 0.014 / 0.045 / -0.142 | 2.68 / 0.038 / 0.120 / -0.043 |

| secondary S0 (no LingBot gate) | IoU | precision | recall | occupied in LiDAR-free |
|---|---|---|---|---|
| LD | 0.369 | 0.454 | 0.662 | 0.546 |
| LB-M | 0.036 | 0.051 | 0.109 | 0.949 |
| MG | 0.043 | 0.060 | 0.129 | 0.940 |
| LB-O | 0.108 | 0.158 | 0.256 | 0.842 |

S0 gap closed: MG +2.0 %, LB-O +21.7 %. S0 depth: LB-M endpoint 1.11 m, within 0.2 0.058; MG endpoint 1.12 m, within 0.2 0.073; LB-O endpoint 0.40 m, within 0.2 0.317

| original Gate 9 conditions (reproduced exactly, Gate 9 support) | IoU | precision | recall |
|---|---|---|---|
| Gate 9 LD | 0.369 | 0.454 | 0.663 |
| Gate 9 PD (= LB-M) | 0.036 | 0.052 | 0.109 |
| Gate 9 LD_gate | 0.323 | 0.447 | 0.539 |
| Gate 9 PD_gate (= LB-M, gated) | 0.035 | 0.054 | 0.088 |

Scalars median [p10, p90]: MoGe clip scale 28.52 [21.22, 32.29]; LB-O scalar 31.61 [24.43, 35.63]; LB-O / MoGe scale 1.104 [1.066, 1.184]. Representative clips: {'bad': 169, 'median': 40, 'good': 166}

## 0007 (190 clips)

Support: Gate 9 LiDAR support 19,115,940 px; MoGe mask removes 0.03 %, LingBot gate 5.90 %, S1 keeps 17,985,298 (94.09 %). Mapper 1-60 m range rejects of S1 rays: LD 0.000 %, LB-M 0.000 %, MG 0.045 %, LB-O 0.006 %.

| primary S1 | IoU | precision | recall | occupied in LiDAR-free (fraction / per clip) |
|---|---|---|---|---|
| LD | 0.346 | 0.475 | 0.560 | 0.525 / 7,451 |
| LB-M | 0.031 | 0.051 | 0.075 | 0.949 / 16,715 |
| MG | 0.040 | 0.064 | 0.093 | 0.936 / 16,250 |
| LB-O | 0.123 | 0.201 | 0.241 | 0.799 / 11,527 |

Gap closed (S1, denominator IoU_LD - IoU_LB-M = 0.315): MG +2.7 % [-0.1, +5.8]; LB-O +29.1 % [+25.1, +33.8] (95 % clip-block bootstrap).

| depth on S1 | n px | bias rel (median) | bias m (median) | AbsRel median / mean | endpoint median | p90 | within 0.2 m | within 0.6 m | exact 0.2 m voxel |
|---|---|---|---|---|---|---|---|---|---|
| LB-M | 3,596,469 | -0.126 | -1.03 | 0.129 / 0.144 | 1.26 | 4.63 | 0.051 | 0.204 | 0.0087 |
| MG | 3,596,469 | -0.126 | -1.00 | 0.130 / 0.149 | 1.27 | 5.09 | 0.057 | 0.215 | 0.0095 |
| LB-O | 3,596,469 | +0.000 | +0.00 | 0.041 / 0.075 | 0.44 | 2.86 | 0.302 | 0.578 | 0.0515 |

| paired per clip (S1) | pooled IoU diff [95 % CI] | per-clip mean | median [p10, p90] | clips improved |
|---|---|---|---|---|
| MG - LB-M | +0.008 [-0.000, +0.018] | +0.009 | +0.003 [-0.022, +0.052] | 57 % |
| LB-O - LB-M | +0.092 [+0.073, +0.110] | +0.094 | +0.085 [+0.042, +0.162] | 99 % |
| LB-O - MG | +0.083 [+0.063, +0.103] | +0.084 | +0.077 [+0.025, +0.154] | 97 % |
| LD - MG | +0.306 [+0.279, +0.330] | +0.302 | +0.311 [+0.213, +0.379] | 100 % |
| LD - LB-O | +0.223 [+0.206, +0.238] | +0.218 | +0.215 [+0.161, +0.274] | 100 % |

| range (S1) | n px | LB-M endpoint med / within 0.2 / within 0.6 / bias | MG | LB-O |
|---|---|---|---|---|
| 0-5 m | 386,630 | 0.58 / 0.149 / 0.522 / -0.101 | 0.55 / 0.167 / 0.553 / -0.096 | 0.17 / 0.569 / 0.865 / +0.010 |
| 5-10 m | 1,615,736 | 0.99 / 0.051 / 0.240 / -0.122 | 0.97 / 0.055 / 0.249 / -0.120 | 0.27 / 0.404 / 0.731 / +0.003 |
| 10-20 m | 1,134,427 | 2.01 / 0.033 / 0.107 / -0.132 | 2.09 / 0.038 / 0.117 / -0.139 | 0.75 / 0.164 / 0.427 / -0.003 |
| 20-30 m | 306,342 | 3.85 / 0.020 / 0.062 / -0.148 | 4.14 / 0.022 / 0.064 / -0.155 | 1.66 / 0.069 / 0.204 / -0.018 |
| >30 m | 153,334 | 7.17 / 0.009 / 0.025 / -0.185 | 7.55 / 0.011 / 0.034 / -0.186 | 3.17 / 0.038 / 0.112 / -0.057 |

| secondary S0 (no LingBot gate) | IoU | precision | recall | occupied in LiDAR-free |
|---|---|---|---|---|
| LD | 0.378 | 0.473 | 0.653 | 0.527 |
| LB-M | 0.031 | 0.047 | 0.087 | 0.953 |
| MG | 0.038 | 0.057 | 0.104 | 0.943 |
| LB-O | 0.115 | 0.172 | 0.258 | 0.828 |

S0 gap closed: MG +2.0 %, LB-O +24.2 %. S0 depth: LB-M endpoint 1.33 m, within 0.2 0.049; MG endpoint 1.34 m, within 0.2 0.055; LB-O endpoint 0.49 m, within 0.2 0.284

| original Gate 9 conditions (reproduced exactly, Gate 9 support) | IoU | precision | recall |
|---|---|---|---|
| Gate 9 LD | 0.378 | 0.473 | 0.654 |
| Gate 9 PD (= LB-M) | 0.031 | 0.047 | 0.087 |
| Gate 9 LD_gate | 0.346 | 0.475 | 0.560 |
| Gate 9 PD_gate (= LB-M, gated) | 0.031 | 0.051 | 0.075 |

Scalars median [p10, p90]: MoGe clip scale 29.89 [20.76, 35.19]; LB-O scalar 34.72 [24.06, 39.73]; LB-O / MoGe scale 1.145 [1.070, 1.225]. Representative clips: {'bad': 289, 'median': 7, 'good': 238}

## 0010 (199 clips)

Support: Gate 9 LiDAR support 19,882,441 px; MoGe mask removes 0.01 %, LingBot gate 6.98 %, S1 keeps 18,493,520 (93.01 %). Mapper 1-60 m range rejects of S1 rays: LD 0.000 %, LB-M 0.002 %, MG 0.006 %, LB-O 0.018 %.

| primary S1 | IoU | precision | recall | occupied in LiDAR-free (fraction / per clip) |
|---|---|---|---|---|
| LD | 0.347 | 0.466 | 0.576 | 0.534 / 7,811 |
| LB-M | 0.025 | 0.037 | 0.068 | 0.963 / 20,768 |
| MG | 0.028 | 0.042 | 0.077 | 0.958 / 20,756 |
| LB-O | 0.116 | 0.179 | 0.250 | 0.821 / 13,576 |

Gap closed (S1, denominator IoU_LD - IoU_LB-M = 0.323): MG +1.0 % [-0.4, +2.7]; LB-O +28.5 % [+26.2, +31.0] (95 % clip-block bootstrap).

| depth on S1 | n px | bias rel (median) | bias m (median) | AbsRel median / mean | endpoint median | p90 | within 0.2 m | within 0.6 m | exact 0.2 m voxel |
|---|---|---|---|---|---|---|---|---|---|
| LB-M | 3,697,876 | -0.119 | -0.98 | 0.122 / 0.138 | 1.20 | 4.23 | 0.041 | 0.194 | 0.0079 |
| MG | 3,697,876 | -0.120 | -0.98 | 0.122 / 0.140 | 1.18 | 4.72 | 0.054 | 0.216 | 0.0106 |
| LB-O | 3,697,876 | -0.000 | -0.00 | 0.034 / 0.067 | 0.36 | 2.35 | 0.334 | 0.645 | 0.0736 |

| paired per clip (S1) | pooled IoU diff [95 % CI] | per-clip mean | median [p10, p90] | clips improved |
|---|---|---|---|---|
| MG - LB-M | +0.003 [-0.001, +0.008] | +0.004 | +0.000 [-0.015, +0.032] | 51 % |
| LB-O - LB-M | +0.092 [+0.083, +0.101] | +0.094 | +0.093 [+0.048, +0.141] | 100 % |
| LB-O - MG | +0.089 [+0.076, +0.100] | +0.090 | +0.093 [+0.042, +0.140] | 97 % |
| LD - MG | +0.319 [+0.296, +0.343] | +0.317 | +0.312 [+0.247, +0.387] | 100 % |
| LD - LB-O | +0.231 [+0.214, +0.247] | +0.227 | +0.230 [+0.162, +0.285] | 100 % |

| range (S1) | n px | LB-M endpoint med / within 0.2 / within 0.6 / bias | MG | LB-O |
|---|---|---|---|---|
| 0-5 m | 404,341 | 0.60 / 0.101 / 0.504 / -0.106 | 0.50 / 0.166 / 0.619 / -0.088 | 0.16 / 0.585 / 0.910 / +0.007 |
| 5-10 m | 1,660,142 | 0.93 / 0.045 / 0.235 / -0.115 | 0.90 / 0.056 / 0.254 / -0.114 | 0.23 / 0.445 / 0.806 / +0.002 |
| 10-20 m | 1,170,324 | 1.89 / 0.026 / 0.091 / -0.122 | 1.99 / 0.027 / 0.090 / -0.131 | 0.60 / 0.194 / 0.498 / -0.001 |
| 20-30 m | 314,492 | 3.63 / 0.015 / 0.044 / -0.138 | 3.94 / 0.016 / 0.053 / -0.151 | 1.35 / 0.084 / 0.246 / -0.014 |
| >30 m | 148,577 | 6.54 / 0.006 / 0.020 / -0.167 | 7.37 / 0.009 / 0.028 / -0.189 | 2.65 / 0.044 / 0.130 / -0.035 |

| secondary S0 (no LingBot gate) | IoU | precision | recall | occupied in LiDAR-free |
|---|---|---|---|---|
| LD | 0.393 | 0.470 | 0.704 | 0.530 |
| LB-M | 0.026 | 0.035 | 0.085 | 0.965 |
| MG | 0.028 | 0.039 | 0.091 | 0.961 |
| LB-O | 0.106 | 0.148 | 0.271 | 0.852 |

S0 gap closed: MG +0.7 %, LB-O +21.8 %. S0 depth: LB-M endpoint 1.28 m, within 0.2 0.040; MG endpoint 1.26 m, within 0.2 0.052; LB-O endpoint 0.40 m, within 0.2 0.312

| original Gate 9 conditions (reproduced exactly, Gate 9 support) | IoU | precision | recall |
|---|---|---|---|
| Gate 9 LD | 0.393 | 0.470 | 0.704 |
| Gate 9 PD (= LB-M) | 0.026 | 0.035 | 0.085 |
| Gate 9 LD_gate | 0.347 | 0.466 | 0.576 |
| Gate 9 PD_gate (= LB-M, gated) | 0.025 | 0.037 | 0.068 |

Scalars median [p10, p90]: MoGe clip scale 26.73 [21.12, 30.88]; LB-O scalar 30.60 [23.54, 35.89]; LB-O / MoGe scale 1.137 [1.079, 1.200]. Representative clips: {'bad': 298, 'median': 487, 'good': 247}

