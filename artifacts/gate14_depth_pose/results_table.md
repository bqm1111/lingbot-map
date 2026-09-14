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

