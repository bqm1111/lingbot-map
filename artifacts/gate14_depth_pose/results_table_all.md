# Gate 14 Stage 1 -- depth x pose observed geometry: results

Observed five-frame maps vs the five-sweep LiDAR measurement inside the camera frusta (Gate 9 protocol). LBM = LingBot x causal MoGe-2 clip scale; NEW = MUSt3R-512 (official code) x causal MoGe-2 clip scale; L = LingBot local-reset pose + pred_K; G = GT pose + calibrated camera. shared = LingBot conf >= 1.5 AND MUSt3R-valid pixels; native = each source's own support. LD-G = LiDAR camera depth + GT pose (Gate 9 LD). Voxel range = anchor-camera z.

## 0006 (590 clips)

Support: shared 149,117,803 px = 83.7 % of LBM native; NEW native / LBM native = 1.03; lattice inside the MUSt3R crop 85.3 %. Scales: s_clip median 20.43, s_new median 1.390.

| condition | IoU | precision | recall | predicted occ / ref occ | occupied in LiDAR-free | false-free |
|---|---|---|---|---|---|---|
| LBM-L|shared | 0.088 | 0.187 | 0.144 | 0.77 | 0.813 | 0.1148 |
| LBM-G|shared | 0.080 | 0.128 | 0.175 | 1.37 | 0.872 | 0.0364 |
| NEW-L|shared | 0.090 | 0.174 | 0.157 | 0.90 | 0.826 | 0.1127 |
| NEW-G|shared | 0.078 | 0.124 | 0.176 | 1.42 | 0.876 | 0.0421 |
| LBM-L|native | 0.091 | 0.187 | 0.152 | 0.81 | 0.813 | 0.1217 |
| LBM-G|native | 0.083 | 0.130 | 0.185 | 1.42 | 0.870 | 0.0399 |
| NEW-L|native | 0.082 | 0.124 | 0.192 | 1.55 | 0.876 | 0.1856 |
| NEW-G|native | 0.068 | 0.091 | 0.206 | 2.25 | 0.909 | 0.0607 |
| LD-G|lidar | 0.438 | 0.532 | 0.711 | 1.33 | 0.468 | 0.0074 |

Factors (shared): pose_cost_LBM -0.0085 [-0.0174, +0.0004]; pose_cost_NEW -0.0118 [-0.0209, -0.0023]; depth_gain_GT -0.0015 [-0.0049, +0.0018]; deployable_gain +0.0018 [-0.0000, +0.0038]; gap_closed +0.0051 [-0.0001, +0.0109]; gap_closed_GT -0.0041 [-0.0136, +0.0052]

Factors (native): pose_cost_LBM -0.0089 [-0.0178, +0.0001]; pose_cost_NEW -0.0141 [-0.0214, -0.0062]; depth_gain_GT -0.0150 [-0.0192, -0.0113]; deployable_gain -0.0098 [-0.0124, -0.0072]; gap_closed -0.0284 [-0.0360, -0.0208]; gap_closed_GT -0.0423 [-0.0545, -0.0316]

| depth (shared LiDAR-supported pixels) | n | AbsRel | signed rel | abs m | endpoint med / p75 / p90 / p95 | within 0.2 / 0.4 / 0.6 / 1.0 m | exact voxel | endpoint on LiDAR-free |
|---|---|---|---|---|---|---|---|---|
| LBM shared | 10,211,366 | 0.075 | -0.056 | 0.55 | 0.63 / 1.26 / 2.41 / 3.66 | 0.173 / 0.337 / 0.478 / 0.673 | 0.0341 | 0.285 |
| NEW shared | 10,211,366 | 0.080 | -0.056 | 0.59 | 0.68 / 1.36 / 2.70 / 4.21 | 0.165 / 0.321 / 0.455 / 0.646 | 0.0345 | 0.274 |
| LBM native | 11,251,361 | 0.076 | -0.056 | 0.54 | 0.64 / 1.27 / 2.45 / 3.75 | 0.173 / 0.338 / 0.478 / 0.672 | 0.0335 | 0.277 |
| NEW native | 10,958,852 | 0.083 | -0.057 | 0.64 | 0.73 / 1.57 / 3.64 / 6.42 | 0.156 / 0.303 / 0.430 / 0.611 | 0.0329 | 0.292 |

| range (shared) | LBM-L IoU / occ-in-free | LBM-G | NEW-L | NEW-G | LD-G | endpoint med LBM / NEW | within 0.2 LBM / NEW |
|---|---|---|---|---|---|---|---|
| 0-5 m | 0.151 / 0.742 | 0.126 / 0.804 | 0.147 / 0.762 | 0.122 / 0.814 | 0.475 / 0.354 | 0.34 / 0.36 | 0.303 / 0.289 |
| 5-10 m | 0.111 / 0.806 | 0.096 / 0.869 | 0.115 / 0.814 | 0.094 / 0.872 | 0.450 / 0.475 | 0.55 / 0.59 | 0.186 / 0.179 |
| 10-20 m | 0.063 / 0.876 | 0.068 / 0.905 | 0.068 / 0.879 | 0.065 / 0.909 | 0.430 / 0.517 | 1.18 / 1.25 | 0.086 / 0.082 |
| 20-30 m | 0.025 / 0.916 | 0.033 / 0.934 | 0.029 / 0.917 | 0.034 / 0.930 | 0.411 / 0.518 | 2.34 / 2.63 | 0.043 / 0.039 |
| >30 m | 0.004 / 0.918 | 0.004 / 0.945 | 0.004 / 0.941 | 0.005 / 0.930 | 0.406 / 0.450 | 4.58 / 4.98 | 0.018 / 0.021 |

| edge (shared) | n | AbsRel LBM / NEW | endpoint med LBM / NEW | within 0.2 LBM / NEW | exact voxel LBM / NEW |
|---|---|---|---|---|---|
| non-edge (<5 %) | 5,297,898 | 0.069 / 0.071 | 0.51 / 0.53 | 0.209 / 0.205 | 0.0376 / 0.0396 |
| moderate (5-20 %) | 3,698,459 | 0.075 / 0.083 | 0.75 / 0.83 | 0.144 / 0.132 | 0.0328 / 0.0315 |
| strong (>=20 %) | 1,212,221 | 0.121 / 0.137 | 1.39 / 1.54 | 0.102 / 0.097 | 0.0231 / 0.0215 |

Registration (L vs G): {"rel_translation_err_m_median": 1.3197809167624386, "rel_translation_err_m_p90": 2.595253290064184, "rel_rotation_err_deg_median": 0.720139363983878, "rel_rotation_err_deg_p90": 2.4125987594405047, "gt_translation_m_median": 10.032262579615674, "pose_endpoint_displacement_m_median": 1.0660591125488281, "pose_endpoint_displacement_m_p90": 2.3746613025665284, "reprojection_px_median": 22.312347412109375, "reprojection_px_p90": 72.83089981079108, "reprojection_valid_fraction": 0.48995018092357023, "pose_displacement_by_range_median_m": {"0-5 m": 0.8683733344078064, "5-10 m": 0.882872462272644, "10-20 m": 1.2033019065856934, "20-30 m": 1.953981876373291, ">30 m": 3.235743284225464}}

Promotion (frozen rule, shared): {"C1_iou_gain": false, "C2_free_space": false, "C3_not_density": false, "C4_range_consistency": false, "promote": false, "values": {"iou_LBM-L": 0.08838503061957088, "iou_NEW-L": 0.09016438152868778, "gain": 0.0017793509091168919, "relative": 0.020131813007743582, "occ_in_free_LBM-L": 0.8134343731049754, "occ_in_free_NEW-L": 0.8258010846833012, "major_bands": ["0-5 m", "5-10 m", "10-20 m", "20-30 m", ">30 m"], "band_ious": {"0-5 m": [0.15085458595034132, 0.14722384644803582], "5-10 m": [0.11113097479813314, 0.11475362118287599], "10-20 m": [0.06346919845449865, 0.06824143901746244], "20-30 m": [0.024569950808634007, 0.028683298051244633], ">30 m": [0.004428450138931769, 0.003938843766951567]}, "native_gain": -0.009845812723271494}, "stage1_guide": {"depth_gain_GT": -0.0014580319715767759, "depth_gain_GT_relative": -0.018259719699548666, "depth_substantially_better_GT": false, "iou_LD-G": 0.4375842986259205, "iou_NEW-G": 0.0783916040342166, "pose_cost_NEW": -0.01177277749447117, "residual_depth_gap_GT": 0.3591926945917039}, "rule_sha256": "d46c5c41dde27c77675b56706c3bc54173c4a9b13a2022b7167fada7d53d9f5a"}

## 0003 (65 clips)

Support: shared 14,045,435 px = 84.1 % of LBM native; NEW native / LBM native = 1.20; lattice inside the MUSt3R crop 85.3 %. Scales: s_clip median 28.52, s_new median 1.700.

| condition | IoU | precision | recall | predicted occ / ref occ | occupied in LiDAR-free | false-free |
|---|---|---|---|---|---|---|
| LBM-L|shared | 0.041 | 0.085 | 0.074 | 0.87 | 0.915 | 0.0816 |
| LBM-G|shared | 0.035 | 0.052 | 0.095 | 1.83 | 0.948 | 0.0178 |
| NEW-L|shared | 0.043 | 0.083 | 0.082 | 0.99 | 0.917 | 0.0753 |
| NEW-G|shared | 0.029 | 0.043 | 0.080 | 1.85 | 0.957 | 0.0216 |
| LBM-L|native | 0.044 | 0.087 | 0.080 | 0.92 | 0.913 | 0.0889 |
| LBM-G|native | 0.037 | 0.054 | 0.104 | 1.92 | 0.946 | 0.0217 |
| NEW-L|native | 0.043 | 0.066 | 0.108 | 1.65 | 0.934 | 0.1397 |
| NEW-G|native | 0.027 | 0.036 | 0.096 | 2.68 | 0.964 | 0.0338 |
| LD-G|lidar | 0.369 | 0.454 | 0.663 | 1.46 | 0.546 | 0.0110 |

Factors (shared): pose_cost_LBM -0.0065 [-0.0167, +0.0023]; pose_cost_NEW -0.0140 [-0.0225, -0.0040]; depth_gain_GT -0.0058 [-0.0112, -0.0014]; deployable_gain +0.0017 [-0.0004, +0.0040]; gap_closed +0.0050 [-0.0013, +0.0122]; gap_closed_GT -0.0173 [-0.0349, -0.0039]

Factors (native): pose_cost_LBM -0.0065 [-0.0160, +0.0022]; pose_cost_NEW -0.0158 [-0.0227, -0.0062]; depth_gain_GT -0.0102 [-0.0144, -0.0055]; deployable_gain -0.0009 [-0.0018, +0.0012]; gap_closed -0.0027 [-0.0056, +0.0038]; gap_closed_GT -0.0306 [-0.0451, -0.0159]

| depth (shared LiDAR-supported pixels) | n | AbsRel | signed rel | abs m | endpoint med / p75 / p90 / p95 | within 0.2 / 0.4 / 0.6 / 1.0 m | exact voxel | endpoint on LiDAR-free |
|---|---|---|---|---|---|---|---|---|
| LBM shared | 1,104,837 | 0.102 | -0.100 | 0.91 | 1.03 / 1.98 / 3.63 / 5.43 | 0.057 / 0.143 / 0.266 / 0.485 | 0.0088 | 0.341 |
| NEW shared | 1,104,837 | 0.117 | -0.108 | 1.02 | 1.17 / 2.36 / 4.64 / 7.05 | 0.077 / 0.161 / 0.255 / 0.436 | 0.0080 | 0.350 |
| LBM native | 1,215,393 | 0.102 | -0.099 | 0.88 | 1.03 / 1.97 / 3.65 / 5.53 | 0.061 / 0.148 / 0.269 / 0.485 | 0.0090 | 0.332 |
| NEW native | 1,193,845 | 0.121 | -0.110 | 1.10 | 1.27 / 2.78 / 6.46 / 11.08 | 0.072 / 0.151 / 0.239 / 0.408 | 0.0077 | 0.363 |

| range (shared) | LBM-L IoU / occ-in-free | LBM-G | NEW-L | NEW-G | LD-G | endpoint med LBM / NEW | within 0.2 LBM / NEW |
|---|---|---|---|---|---|---|---|
| 0-5 m | 0.070 / 0.883 | 0.048 / 0.928 | 0.068 / 0.893 | 0.045 / 0.934 | 0.399 / 0.449 | 0.50 / 0.52 | 0.120 / 0.175 |
| 5-10 m | 0.062 / 0.888 | 0.040 / 0.946 | 0.069 / 0.885 | 0.036 / 0.952 | 0.354 / 0.574 | 0.77 / 0.83 | 0.067 / 0.101 |
| 10-20 m | 0.032 / 0.941 | 0.037 / 0.951 | 0.036 / 0.938 | 0.029 / 0.962 | 0.350 / 0.605 | 1.58 / 1.86 | 0.041 / 0.042 |
| 20-30 m | 0.019 / 0.954 | 0.024 / 0.964 | 0.017 / 0.963 | 0.015 / 0.977 | 0.370 / 0.566 | 2.93 / 3.95 | 0.026 / 0.019 |
| >30 m | 0.007 / 0.953 | 0.012 / 0.959 | 0.005 / 0.957 | 0.005 / 0.974 | 0.387 / 0.456 | 5.63 / 7.38 | 0.012 / 0.009 |

| edge (shared) | n | AbsRel LBM / NEW | endpoint med LBM / NEW | within 0.2 LBM / NEW | exact voxel LBM / NEW |
|---|---|---|---|---|---|
| non-edge (<5 %) | 426,074 | 0.096 / 0.105 | 0.73 / 0.79 | 0.072 / 0.108 | 0.0087 / 0.0083 |
| moderate (5-20 %) | 536,838 | 0.104 / 0.121 | 1.24 / 1.42 | 0.047 / 0.061 | 0.0079 / 0.0075 |
| strong (>=20 %) | 141,875 | 0.129 / 0.158 | 2.47 / 2.99 | 0.053 / 0.044 | 0.0121 / 0.0086 |

Registration (L vs G): {"rel_translation_err_m_median": 2.356499188676855, "rel_translation_err_m_p90": 4.904503966746042, "rel_rotation_err_deg_median": 0.5623173002084313, "rel_rotation_err_deg_p90": 1.7312434669591583, "gt_translation_m_median": 16.321612916894296, "pose_endpoint_displacement_m_median": 1.9981575012207031, "pose_endpoint_displacement_m_p90": 4.410773372650147, "reprojection_px_median": 23.17878532409668, "reprojection_px_p90": 127.94499969482442, "reprojection_valid_fraction": 0.46090745358930396, "pose_displacement_by_range_median_m": {"0-5 m": 1.691192865371704, "5-10 m": 1.6580822467803955, "10-20 m": 1.8012651801109314, "20-30 m": 2.354450225830078, ">30 m": 3.6034083366394043}}

Promotion (frozen rule, shared): {"C1_iou_gain": false, "C2_free_space": false, "C3_not_density": false, "C4_range_consistency": false, "promote": false, "values": {"iou_LBM-L": 0.04137336708609025, "iou_NEW-L": 0.04302807373309006, "gain": 0.0016547066469998137, "relative": 0.0399944883276403, "occ_in_free_LBM-L": 0.9146291420168929, "occ_in_free_NEW-L": 0.917146841601681, "major_bands": ["0-5 m", "5-10 m", "10-20 m", "20-30 m", ">30 m"], "band_ious": {"0-5 m": [0.07033189834745174, 0.06786026650203221], "5-10 m": [0.06154105878103554, 0.06876752327963159], "10-20 m": [0.03214989899478644, 0.035664102313819585], "20-30 m": [0.018801006098112662, 0.017435637659016446], ">30 m": [0.007401154640418444, 0.004996003197442046]}, "native_gain": -0.0008640214547416744}, "stage1_guide": {"depth_gain_GT": -0.005782882152164142, "depth_gain_GT_relative": -0.16605693249940734, "depth_substantially_better_GT": false, "iou_LD-G": 0.36904345015219675, "iou_NEW-G": 0.029041813602015114, "pose_cost_NEW": -0.013986260131074946, "residual_depth_gap_GT": 0.3400016365501816}, "rule_sha256": "d46c5c41dde27c77675b56706c3bc54173c4a9b13a2022b7167fada7d53d9f5a"}

## 0007 (190 clips)

Support: shared 43,826,489 px = 84.0 % of LBM native; NEW native / LBM native = 1.13; lattice inside the MUSt3R crop 85.3 %. Scales: s_clip median 29.89, s_new median 1.785.

| condition | IoU | precision | recall | predicted occ / ref occ | occupied in LiDAR-free | false-free |
|---|---|---|---|---|---|---|
| LBM-L|shared | 0.047 | 0.097 | 0.085 | 0.87 | 0.903 | 0.0833 |
| LBM-G|shared | 0.031 | 0.048 | 0.079 | 1.65 | 0.952 | 0.0114 |
| NEW-L|shared | 0.044 | 0.088 | 0.081 | 0.92 | 0.912 | 0.0824 |
| NEW-G|shared | 0.032 | 0.050 | 0.080 | 1.61 | 0.950 | 0.0173 |
| LBM-L|native | 0.050 | 0.099 | 0.091 | 0.92 | 0.901 | 0.0889 |
| LBM-G|native | 0.033 | 0.050 | 0.086 | 1.71 | 0.950 | 0.0135 |
| NEW-L|native | 0.044 | 0.072 | 0.099 | 1.36 | 0.928 | 0.1309 |
| NEW-G|native | 0.030 | 0.042 | 0.093 | 2.23 | 0.958 | 0.0266 |
| LD-G|lidar | 0.378 | 0.473 | 0.654 | 1.38 | 0.527 | 0.0116 |

Factors (shared): pose_cost_LBM -0.0165 [-0.0267, -0.0072]; pose_cost_NEW -0.0122 [-0.0194, -0.0046]; depth_gain_GT +0.0009 [-0.0028, +0.0045]; deployable_gain -0.0034 [-0.0078, +0.0003]; gap_closed -0.0103 [-0.0240, +0.0009]; gap_closed_GT +0.0027 [-0.0083, +0.0125]

Factors (native): pose_cost_LBM -0.0169 [-0.0275, -0.0077]; pose_cost_NEW -0.0139 [-0.0211, -0.0063]; depth_gain_GT -0.0032 [-0.0083, +0.0017]; deployable_gain -0.0061 [-0.0115, -0.0021]; gap_closed -0.0186 [-0.0355, -0.0061]; gap_closed_GT -0.0091 [-0.0252, +0.0046]

| depth (shared LiDAR-supported pixels) | n | AbsRel | signed rel | abs m | endpoint med / p75 / p90 / p95 | within 0.2 / 0.4 / 0.6 / 1.0 m | exact voxel | endpoint on LiDAR-free |
|---|---|---|---|---|---|---|---|---|
| LBM shared | 3,267,925 | 0.130 | -0.127 | 1.11 | 1.27 / 2.43 / 4.59 / 7.04 | 0.047 / 0.110 / 0.198 / 0.392 | 0.0082 | 0.298 |
| NEW shared | 3,267,925 | 0.124 | -0.118 | 1.10 | 1.27 / 2.82 / 5.35 / 7.97 | 0.077 / 0.163 / 0.257 / 0.419 | 0.0119 | 0.283 |
| LBM native | 3,597,162 | 0.129 | -0.126 | 1.08 | 1.26 / 2.42 / 4.63 / 7.22 | 0.051 / 0.117 / 0.204 / 0.395 | 0.0087 | 0.289 |
| NEW native | 3,469,060 | 0.128 | -0.119 | 1.18 | 1.37 / 3.16 / 6.63 / 10.86 | 0.073 / 0.155 / 0.245 / 0.399 | 0.0115 | 0.293 |

| range (shared) | LBM-L IoU / occ-in-free | LBM-G | NEW-L | NEW-G | LD-G | endpoint med LBM / NEW | within 0.2 LBM / NEW |
|---|---|---|---|---|---|---|---|
| 0-5 m | 0.073 / 0.876 | 0.042 / 0.933 | 0.063 / 0.895 | 0.039 / 0.937 | 0.399 / 0.434 | 0.58 / 0.47 | 0.129 / 0.198 |
| 5-10 m | 0.065 / 0.882 | 0.035 / 0.951 | 0.059 / 0.897 | 0.034 / 0.951 | 0.380 / 0.545 | 0.97 / 0.86 | 0.047 / 0.097 |
| 10-20 m | 0.042 / 0.923 | 0.031 / 0.957 | 0.039 / 0.928 | 0.032 / 0.956 | 0.369 / 0.580 | 1.97 / 2.18 | 0.033 / 0.039 |
| 20-30 m | 0.018 / 0.954 | 0.021 / 0.966 | 0.023 / 0.942 | 0.026 / 0.956 | 0.377 / 0.550 | 3.81 / 4.40 | 0.021 / 0.019 |
| >30 m | 0.006 / 0.950 | 0.008 / 0.969 | 0.007 / 0.943 | 0.013 / 0.944 | 0.366 / 0.458 | 7.11 / 7.33 | 0.008 / 0.012 |

| edge (shared) | n | AbsRel LBM / NEW | endpoint med LBM / NEW | within 0.2 LBM / NEW | exact voxel LBM / NEW |
|---|---|---|---|---|---|
| non-edge (<5 %) | 1,169,748 | 0.121 / 0.106 | 0.88 / 0.77 | 0.059 / 0.115 | 0.0087 / 0.0155 |
| moderate (5-20 %) | 1,615,607 | 0.133 / 0.130 | 1.52 / 1.51 | 0.036 / 0.059 | 0.0071 / 0.0104 |
| strong (>=20 %) | 482,241 | 0.152 / 0.180 | 2.70 / 3.09 | 0.052 / 0.043 | 0.0105 / 0.0084 |

Registration (L vs G): {"rel_translation_err_m_median": 3.254805354371809, "rel_translation_err_m_p90": 7.737732930955748, "rel_rotation_err_deg_median": 0.45449054342264655, "rel_rotation_err_deg_p90": 1.1695719214733173, "gt_translation_m_median": 20.425775426617122, "pose_endpoint_displacement_m_median": 2.430157423019409, "pose_endpoint_displacement_m_p90": 6.980825042724612, "reprojection_px_median": 22.641450881958008, "reprojection_px_p90": 114.06985168457031, "reprojection_valid_fraction": 0.40625912441224393, "pose_displacement_by_range_median_m": {"0-5 m": 1.7560455799102783, "5-10 m": 2.06355357170105, "10-20 m": 2.179027795791626, "20-30 m": 2.710920572280884, ">30 m": 4.008515357971191}}

Promotion (frozen rule, shared): {"C1_iou_gain": false, "C2_free_space": false, "C3_not_density": false, "C4_range_consistency": false, "promote": false, "values": {"iou_LBM-L": 0.047303642501251186, "iou_NEW-L": 0.04389380016367915, "gain": -0.003409842337572036, "relative": -0.072084138921899, "occ_in_free_LBM-L": 0.9031734688158095, "occ_in_free_NEW-L": 0.9120770792895476, "major_bands": ["0-5 m", "5-10 m", "10-20 m", "20-30 m", ">30 m"], "band_ious": {"0-5 m": [0.07275481982527153, 0.06273381915398214], "5-10 m": [0.06511916022372101, 0.059018377125494244], "10-20 m": [0.04159438988636109, 0.039441194267328054], "20-30 m": [0.018392051939304356, 0.02275020830206646], ">30 m": [0.005692468203275626, 0.0066099273710273024]}, "native_gain": -0.006114622305216186}, "stage1_guide": {"depth_gain_GT": 0.0009208392217874249, "depth_gain_GT_relative": 0.029939537206528404, "depth_substantially_better_GT": false, "iou_LD-G": 0.37788130245775675, "iou_NEW-G": 0.03167746767049415, "pose_cost_NEW": -0.012216332493185003, "residual_depth_gap_GT": 0.3462038347872626}, "rule_sha256": "d46c5c41dde27c77675b56706c3bc54173c4a9b13a2022b7167fada7d53d9f5a"}

## 0010 (199 clips)

Support: shared 47,067,278 px = 83.9 % of LBM native; NEW native / LBM native = 1.10; lattice inside the MUSt3R crop 85.3 %. Scales: s_clip median 26.73, s_new median 1.471.

| condition | IoU | precision | recall | predicted occ / ref occ | occupied in LiDAR-free | false-free |
|---|---|---|---|---|---|---|
| LBM-L|shared | 0.057 | 0.108 | 0.109 | 1.01 | 0.892 | 0.1101 |
| LBM-G|shared | 0.025 | 0.036 | 0.074 | 2.05 | 0.964 | 0.0139 |
| NEW-L|shared | 0.057 | 0.098 | 0.120 | 1.23 | 0.902 | 0.0995 |
| NEW-G|shared | 0.022 | 0.031 | 0.068 | 2.19 | 0.969 | 0.0122 |
| LBM-L|native | 0.060 | 0.109 | 0.117 | 1.08 | 0.891 | 0.1191 |
| LBM-G|native | 0.026 | 0.038 | 0.081 | 2.16 | 0.962 | 0.0167 |
| NEW-L|native | 0.054 | 0.077 | 0.152 | 1.97 | 0.923 | 0.1660 |
| NEW-G|native | 0.021 | 0.027 | 0.086 | 3.15 | 0.973 | 0.0250 |
| LD-G|lidar | 0.393 | 0.470 | 0.704 | 1.50 | 0.530 | 0.0092 |

Factors (shared): pose_cost_LBM -0.0325 [-0.0375, -0.0269]; pose_cost_NEW -0.0352 [-0.0391, -0.0310]; depth_gain_GT -0.0031 [-0.0066, +0.0002]; deployable_gain -0.0004 [-0.0029, +0.0019]; gap_closed -0.0013 [-0.0086, +0.0058]; gap_closed_GT -0.0084 [-0.0179, +0.0005]

Factors (native): pose_cost_LBM -0.0335 [-0.0386, -0.0277]; pose_cost_NEW -0.0326 [-0.0368, -0.0282]; depth_gain_GT -0.0052 [-0.0091, -0.0023]; deployable_gain -0.0061 [-0.0095, -0.0037]; gap_closed -0.0182 [-0.0285, -0.0112]; gap_closed_GT -0.0142 [-0.0249, -0.0063]

| depth (shared LiDAR-supported pixels) | n | AbsRel | signed rel | abs m | endpoint med / p75 / p90 / p95 | within 0.2 / 0.4 / 0.6 / 1.0 m | exact voxel | endpoint on LiDAR-free |
|---|---|---|---|---|---|---|---|---|
| LBM shared | 3,356,547 | 0.122 | -0.120 | 1.04 | 1.20 / 2.23 / 4.18 / 6.27 | 0.038 / 0.098 / 0.189 / 0.413 | 0.0074 | 0.415 |
| NEW shared | 3,356,547 | 0.123 | -0.120 | 1.06 | 1.22 / 2.43 / 4.84 / 7.43 | 0.057 / 0.127 / 0.217 / 0.414 | 0.0084 | 0.425 |
| LBM native | 3,698,088 | 0.122 | -0.119 | 1.02 | 1.20 / 2.24 / 4.23 / 6.39 | 0.041 / 0.103 / 0.194 / 0.413 | 0.0079 | 0.406 |
| NEW native | 3,609,186 | 0.126 | -0.120 | 1.14 | 1.31 / 2.80 / 6.42 / 10.86 | 0.054 / 0.120 / 0.206 / 0.391 | 0.0083 | 0.434 |

| range (shared) | LBM-L IoU / occ-in-free | LBM-G | NEW-L | NEW-G | LD-G | endpoint med LBM / NEW | within 0.2 LBM / NEW |
|---|---|---|---|---|---|---|---|
| 0-5 m | 0.106 / 0.839 | 0.039 / 0.946 | 0.096 / 0.861 | 0.034 / 0.953 | 0.415 / 0.457 | 0.60 / 0.54 | 0.087 / 0.139 |
| 5-10 m | 0.072 / 0.881 | 0.030 / 0.961 | 0.076 / 0.886 | 0.025 / 0.968 | 0.375 / 0.565 | 0.91 / 0.88 | 0.043 / 0.073 |
| 10-20 m | 0.046 / 0.919 | 0.023 / 0.970 | 0.047 / 0.924 | 0.020 / 0.975 | 0.371 / 0.589 | 1.86 / 2.00 | 0.026 / 0.029 |
| 20-30 m | 0.025 / 0.945 | 0.014 / 0.978 | 0.027 / 0.946 | 0.014 / 0.978 | 0.404 / 0.533 | 3.58 / 4.09 | 0.015 / 0.013 |
| >30 m | 0.016 / 0.932 | 0.007 / 0.976 | 0.015 / 0.943 | 0.007 / 0.975 | 0.413 / 0.431 | 6.47 / 7.48 | 0.007 / 0.007 |

| edge (shared) | n | AbsRel LBM / NEW | endpoint med LBM / NEW | within 0.2 LBM / NEW | exact voxel LBM / NEW |
|---|---|---|---|---|---|
| non-edge (<5 %) | 1,500,498 | 0.116 / 0.111 | 0.89 / 0.87 | 0.046 / 0.076 | 0.0079 / 0.0098 |
| moderate (5-20 %) | 1,440,052 | 0.125 / 0.130 | 1.44 / 1.52 | 0.027 / 0.041 | 0.0062 / 0.0065 |
| strong (>=20 %) | 415,530 | 0.150 / 0.173 | 2.91 / 3.46 | 0.042 / 0.045 | 0.0101 / 0.0101 |

Registration (L vs G): {"rel_translation_err_m_median": 1.9722201244301334, "rel_translation_err_m_p90": 4.049820052156443, "rel_rotation_err_deg_median": 0.45346698398985896, "rel_rotation_err_deg_p90": 1.3333637535195682, "gt_translation_m_median": 12.646904822496639, "pose_endpoint_displacement_m_median": 1.6872581243515015, "pose_endpoint_displacement_m_p90": 3.7205147743225098, "reprojection_px_median": 22.82651710510254, "reprojection_px_p90": 101.18697509765633, "reprojection_valid_fraction": 0.5141023618937511, "pose_displacement_by_range_median_m": {"0-5 m": 1.1095643043518066, "5-10 m": 1.255684733390808, "10-20 m": 1.6341665983200073, "20-30 m": 2.309122681617737, ">30 m": 3.571722626686096}}

Promotion (frozen rule, shared): {"C1_iou_gain": false, "C2_free_space": false, "C3_not_density": false, "C4_range_consistency": false, "promote": false, "values": {"iou_LBM-L": 0.05741690891797393, "iou_NEW-L": 0.05698975596696304, "gain": -0.0004271529510108882, "relative": -0.007439497511458183, "occ_in_free_LBM-L": 0.8919022984423018, "occ_in_free_NEW-L": 0.9021680723434885, "major_bands": ["0-5 m", "5-10 m", "10-20 m", "20-30 m", ">30 m"], "band_ious": {"0-5 m": [0.1058586240584917, 0.09625307497586097], "5-10 m": [0.07221394144729423, 0.07571479920269254], "10-20 m": [0.04552697417563958, 0.04729528920743079], "20-30 m": [0.025169013361711554, 0.027447876409616108], ">30 m": [0.015508684863523574, 0.014920767344888064]}, "native_gain": -0.00605247181956424}, "stage1_guide": {"depth_gain_GT": -0.003093272649977167, "depth_gain_GT_relative": -0.12423512787392281, "depth_substantially_better_GT": false, "iou_LD-G": 0.39274194464320195, "iou_NEW-G": 0.02180526211159449, "pose_cost_NEW": -0.03518449385536855, "residual_depth_gap_GT": 0.37093668253160744}, "rule_sha256": "d46c5c41dde27c77675b56706c3bc54173c4a9b13a2022b7167fada7d53d9f5a"}

