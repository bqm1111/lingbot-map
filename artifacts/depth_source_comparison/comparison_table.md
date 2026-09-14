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

