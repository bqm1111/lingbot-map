# Delayed-supervision pilot — 0006 development evaluation (590 anchors, geometry seed 0, head seed 0)

mIoU in %. Semantic regions score GT-occupied voxels; SSC regions score all labelled voxels under the frozen occupancy.

## mIoU by region

| region | B0 | B0g | B1 | B2 | B2s | voxels |
|---|---|---|---|---|---|---|
| observed in frame t | 15.21 | 15.21 | 15.21 | 15.21 | 15.21 | 156,937 |
| observed only in t-4..t-1 | 17.38 | 17.38 | 17.38 | 17.38 | 17.38 | 277,746 |
| never observed (GT occupancy) | 8.39 | 8.35 | 8.60 | 8.66 | 8.60 | 7,834,276 |
| **never observed ∩ predicted occupied (primary)** | 8.80 | 8.70 | 9.13 | 9.21 | 9.12 | 2,743,212 |
| … geometry observed (S or F) | 9.72 | 9.63 | 10.13 | 10.90 | 10.75 | 476,001 |
| … geometry unobserved | 7.73 | 7.70 | 8.03 | 7.99 | 7.95 | 2,267,211 |
| … later revealed (0006 delayed target) | 7.06 | 7.04 | 7.10 | 7.33 | 7.30 | 559,021 |
| … later revealed, no H_t camera return | 5.97 | 5.89 | 5.94 | 6.35 | 6.26 | 375,035 |
| … not revealed within 10 s | 9.17 | 9.06 | 9.60 | 9.67 | 9.55 | 2,184,191 |
| SSC mIoU, all labelled voxels | 1.55 | 1.48 | 1.67 | 1.60 | 1.59 | 558,778,307 |
| SSC mIoU, voxels without evidence | 1.39 | 1.32 | 1.51 | 1.45 | 1.43 | 553,275,690 |


## Overall accuracy by region

| region | B0 | B0g | B1 | B2 | B2s |
|---|---|---|---|---|---|
| observed in frame t | 62.34 | 62.34 | 62.34 | 62.34 | 62.34 |
| observed only in t-4..t-1 | 62.70 | 62.70 | 62.70 | 62.70 | 62.70 |
| never observed (GT occupancy) | 49.91 | 49.94 | 51.07 | 50.95 | 50.74 |
| **never observed ∩ predicted occupied (primary)** | 46.01 | 46.11 | 47.09 | 46.03 | 45.84 |
| … geometry observed (S or F) | 40.75 | 40.35 | 41.47 | 42.79 | 42.78 |
| … geometry unobserved | 47.11 | 47.32 | 48.26 | 46.71 | 46.48 |
| … later revealed (0006 delayed target) | 45.53 | 45.92 | 46.67 | 46.09 | 45.98 |
| … later revealed, no H_t camera return | 42.37 | 42.85 | 43.60 | 42.94 | 42.85 |
| … not revealed within 10 s | 46.13 | 46.16 | 47.19 | 46.02 | 45.80 |


## Paired differences (mIoU points, 95 % block bootstrap, 20-anchor blocks)

| pair | region | Δ | CI | blocks won |
|---|---|---|---|---|
| B2 − B1 | **never observed ∩ predicted occupied (primary)** | +0.08 | [-0.11, +0.37] | 11/29 |
| B2 − B1 | … geometry observed (S or F) | +0.77 | [+0.37, +1.45] |  |
| B2 − B1 | … geometry unobserved | -0.04 | [-0.19, +0.18] |  |
| B2 − B1 | … later revealed (0006 delayed target) | +0.23 | [+0.05, +0.38] |  |
| B2 − B1 | … later revealed, no H_t camera return | +0.41 | [+0.14, +0.83] |  |
| B2 − B1 | … not revealed within 10 s | +0.07 | [-0.19, +0.47] |  |
| B2 − B1 | SSC mIoU, all labelled voxels | -0.07 | [-0.10, -0.01] |  |
| B2 − B0 | **never observed ∩ predicted occupied (primary)** | +0.41 | [+0.16, +0.78] | 17/29 |
| B2 − B0 | … geometry observed (S or F) | +1.18 | [+0.71, +1.95] |  |
| B2 − B0 | … geometry unobserved | +0.26 | [+0.07, +0.51] |  |
| B2 − B0 | … later revealed (0006 delayed target) | +0.27 | [+0.01, +0.46] |  |
| B2 − B0 | … later revealed, no H_t camera return | +0.38 | [-0.27, +0.87] |  |
| B2 − B0 | … not revealed within 10 s | +0.50 | [+0.16, +1.02] |  |
| B2 − B0 | SSC mIoU, all labelled voxels | +0.05 | [+0.01, +0.13] |  |
| B1 − B0 | **never observed ∩ predicted occupied (primary)** | +0.33 | [+0.19, +0.48] | 26/29 |
| B1 − B0 | … geometry observed (S or F) | +0.41 | [+0.20, +0.65] |  |
| B1 − B0 | … geometry unobserved | +0.29 | [+0.16, +0.45] |  |
| B1 − B0 | … later revealed (0006 delayed target) | +0.05 | [-0.13, +0.21] |  |
| B1 − B0 | … later revealed, no H_t camera return | -0.03 | [-0.69, +0.40] |  |
| B1 − B0 | … not revealed within 10 s | +0.42 | [+0.24, +0.67] |  |
| B1 − B0 | SSC mIoU, all labelled voxels | +0.12 | [+0.08, +0.17] |  |
| B0g − B0 | **never observed ∩ predicted occupied (primary)** | -0.09 | [-0.25, +0.04] | 10/29 |
| B0g − B0 | … geometry observed (S or F) | -0.10 | [-0.28, +0.11] |  |
| B0g − B0 | … geometry unobserved | -0.03 | [-0.20, +0.12] |  |
| B0g − B0 | … later revealed (0006 delayed target) | -0.01 | [-0.21, +0.12] |  |
| B0g − B0 | … later revealed, no H_t camera return | -0.08 | [-0.43, +0.15] |  |
| B0g − B0 | … not revealed within 10 s | -0.11 | [-0.30, +0.05] |  |
| B0g − B0 | SSC mIoU, all labelled voxels | -0.07 | [-0.12, -0.03] |  |
| B2 − B0g | **never observed ∩ predicted occupied (primary)** | +0.51 | [+0.22, +0.92] | 19/29 |
| B2 − B0g | … geometry observed (S or F) | +1.28 | [+0.78, +2.16] |  |
| B2 − B0g | … geometry unobserved | +0.29 | [+0.08, +0.59] |  |
| B2 − B0g | … later revealed (0006 delayed target) | +0.29 | [+0.08, +0.51] |  |
| B2 − B0g | … later revealed, no H_t camera return | +0.46 | [-0.17, +1.13] |  |
| B2 − B0g | … not revealed within 10 s | +0.61 | [+0.22, +1.19] |  |
| B2 − B0g | SSC mIoU, all labelled voxels | +0.12 | [+0.07, +0.23] |  |
| B2s − B1 | **never observed ∩ predicted occupied (primary)** | -0.01 | [-0.21, +0.27] | 11/29 |
| B2s − B1 | … geometry observed (S or F) | +0.62 | [+0.20, +1.23] |  |
| B2s − B1 | … geometry unobserved | -0.08 | [-0.23, +0.12] |  |
| B2s − B1 | … later revealed (0006 delayed target) | +0.20 | [+0.02, +0.36] |  |
| B2s − B1 | … later revealed, no H_t camera return | +0.32 | [+0.08, +0.75] |  |
| B2s − B1 | … not revealed within 10 s | -0.05 | [-0.35, +0.35] |  |
| B2s − B1 | SSC mIoU, all labelled voxels | -0.08 | [-0.12, -0.03] |  |
| B2s − B0 | **never observed ∩ predicted occupied (primary)** | +0.32 | [+0.08, +0.66] | 17/29 |
| B2s − B0 | … geometry observed (S or F) | +1.03 | [+0.61, +1.74] |  |
| B2s − B0 | … geometry unobserved | +0.22 | [+0.02, +0.46] |  |
| B2s − B0 | … later revealed (0006 delayed target) | +0.25 | [-0.02, +0.44] |  |
| B2s − B0 | … later revealed, no H_t camera return | +0.29 | [-0.34, +0.78] |  |
| B2s − B0 | … not revealed within 10 s | +0.37 | [+0.05, +0.88] |  |
| B2s − B0 | SSC mIoU, all labelled voxels | +0.04 | [+0.00, +0.11] |  |


## Decision criteria — B2 (decision)

| criterion | value | pass |
|---|---|---|
| K1 Δ vs B1 > 0, CI lo > 0 | +0.08 [-0.11, +0.37] | False |
| K2 Δ vs B0 ≥ +2.00, CI lo > 0 | +0.41 [+0.16, +0.78] | False |
| K3 evidence path identical | 590/590 | True |
| K4 ≥ 5 classes gain; gain without top class > 0 | 14 classes; without road: +0.32 | True |
| K5 SSC mIoU ≥ B0 | 1.60 vs 1.55 | True |
| K6 occupancy identical |  | True |


## Decision criteria — B2s (addendum A sensitivity, not a criterion)

| criterion | value | pass |
|---|---|---|
| K1 Δ vs B1 > 0, CI lo > 0 | -0.01 [-0.21, +0.27] | False |
| K2 Δ vs B0 ≥ +2.00, CI lo > 0 | +0.32 [+0.08, +0.66] | False |
| K3 evidence path identical | 590/590 | True |
| K4 ≥ 5 classes gain; gain without top class > 0 | 14 classes; without road: +0.22 | True |
| K5 SSC mIoU ≥ B0 | 1.59 vs 1.55 | True |
| K6 occupancy identical |  | True |


**Decision: Revise the semantic mechanism or supervision.**


## Per-class IoU on the primary region (%), GT share and prediction share (%)

| class | B0 | B0g | B1 | B2 | B2s | GT share | B0 pred share | B0g pred share | B1 pred share | B2 pred share | B2s pred share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| car | 9.06 | 8.68 | 9.78 | 9.86 | 9.87 | 1.29 | 3.88 | 4.08 | 3.39 | 2.83 | 2.77 |
| bicycle | 0.20 | 0.18 | 0.25 | 0.00 | 0.00 | 0.00 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 |
| motorcycle | 3.38 | 3.39 | 4.70 | 5.20 | 4.09 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 |
| truck | 7.96 | 7.01 | 8.44 | 8.96 | 8.66 | 0.08 | 0.71 | 0.82 | 0.68 | 0.57 | 0.59 |
| other vehicle | 0.93 | 0.75 | 1.02 | 1.34 | 1.22 | 0.13 | 0.41 | 0.46 | 0.40 | 0.32 | 0.34 |
| person | 0.82 | 0.79 | 1.07 | 1.51 | 1.61 | 0.00 | 0.06 | 0.08 | 0.04 | 0.04 | 0.04 |
| road | 17.52 | 17.06 | 17.85 | 19.54 | 19.50 | 11.21 | 3.69 | 3.75 | 3.51 | 4.15 | 4.18 |
| parking | 3.89 | 4.16 | 4.75 | 5.33 | 5.33 | 6.47 | 2.28 | 2.60 | 2.38 | 3.49 | 3.48 |
| sidewalk | 8.33 | 7.86 | 7.70 | 8.31 | 8.39 | 5.85 | 2.79 | 2.81 | 2.29 | 3.34 | 3.42 |
| other ground | 3.17 | 3.46 | 3.73 | 3.97 | 3.97 | 3.69 | 3.12 | 2.90 | 3.07 | 4.11 | 4.22 |
| building | 35.05 | 35.14 | 35.03 | 32.34 | 32.15 | 17.67 | 22.00 | 21.40 | 22.12 | 17.68 | 17.46 |
| fence | 4.50 | 4.51 | 4.44 | 4.48 | 4.39 | 0.92 | 2.99 | 3.44 | 2.77 | 2.99 | 3.13 |
| vegetation | 52.77 | 52.91 | 53.98 | 53.34 | 53.27 | 38.68 | 49.51 | 50.39 | 50.92 | 51.89 | 51.58 |
| terrain | 0.26 | 0.17 | 0.29 | 0.30 | 0.32 | 6.57 | 0.44 | 0.42 | 0.38 | 0.58 | 0.59 |
| pole | 0.03 | 0.05 | 0.06 | 0.03 | 0.02 | 0.01 | 0.92 | 0.64 | 0.63 | 0.65 | 0.68 |
| traffic sign | 0.05 | 0.21 | 0.08 | 0.07 | 0.07 | 0.00 | 0.08 | 0.07 | 0.05 | 0.05 | 0.05 |
| other structure | 8.89 | 8.63 | 9.41 | 9.57 | 9.60 | 7.23 | 6.68 | 5.74 | 7.07 | 7.02 | 7.15 |
| other object | 1.53 | 1.66 | 1.70 | 1.64 | 1.66 | 0.18 | 0.42 | 0.38 | 0.29 | 0.29 | 0.31 |


## Primary region by distance to the nearest evidence (mIoU / accuracy %)

| distance | voxels | B0 | B0g | B1 | B2 | B2s |
|---|---|---|---|---|---|---|
| 0-1 m | 1,106,487 | 12.39 / 52.72 | 12.28 / 52.54 | 12.97 / 53.66 | 13.13 / 53.20 | 12.97 / 53.03 |
| 1-2 m | 557,765 | 9.82 / 52.35 | 9.73 / 51.79 | 10.25 / 53.31 | 10.16 / 51.31 | 10.06 / 51.08 |
| 2-4 m | 459,431 | 7.72 / 45.94 | 7.78 / 46.02 | 7.97 / 46.90 | 8.15 / 45.41 | 8.09 / 45.23 |
| 4-8 m | 221,659 | 4.23 / 30.56 | 4.38 / 31.42 | 4.22 / 31.53 | 4.54 / 30.90 | 4.47 / 30.69 |
| >8 m | 397,870 | 3.07 / 27.11 | 3.20 / 28.55 | 3.24 / 28.97 | 3.26 / 27.86 | 3.26 / 27.66 |


## Primary region by time until a later camera view (mIoU / accuracy %)

| time to reveal | voxels | B0 | B0g | B1 | B2 | B2s |
|---|---|---|---|---|---|---|
| 0-1 s | 251,165 | 9.26 / 52.34 | 9.17 / 52.38 | 9.43 / 53.44 | 9.70 / 52.83 | 9.62 / 52.67 |
| 1-2 s | 154,820 | 7.09 / 47.26 | 7.23 / 47.45 | 6.88 / 48.23 | 7.29 / 47.80 | 7.28 / 47.69 |
| 2-4 s | 108,370 | 5.08 / 33.55 | 4.63 / 34.52 | 5.67 / 34.95 | 5.80 / 34.80 | 5.78 / 34.75 |
| 4-10 s | 44,666 | 3.58 / 30.37 | 3.64 / 31.94 | 3.63 / 31.66 | 3.46 / 29.71 | 3.47 / 29.70 |
| not revealed | 2,184,191 | 9.17 / 46.13 | 9.06 / 46.16 | 9.60 / 47.19 | 9.67 / 46.02 | 9.55 / 45.80 |


## Label-free: agreement with the 0006 delayed target class (all non-evidence target voxels)

| method | voxels | agreement % | mIoU vs target % |
|---|---|---|---|
| B0 | 2,206,103 | 41.87 | 6.60 |
| B0g | 2,206,103 | 42.10 | 6.65 |
| B1 | 2,206,103 | 42.44 | 6.50 |
| B2 | 2,206,103 | 44.00 | 6.95 |
| B2s | 2,206,103 | 43.89 | 6.99 |


## Occupancy

TP/FP/FN [6172202, 27026109, 12021922], IoU 13.65 %, identical to Gate 15 seed 0 per anchor: True. B0 reproduces Gate 15 per anchor: {'primary_per_anchor_equal': True, 'ssc_full_per_anchor_equal': True}. Geodesic spread reaches 99.9 % of completed non-evidence voxels.
