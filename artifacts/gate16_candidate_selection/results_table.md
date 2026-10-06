# Gate 16 results — 0006 (590 anchors), geometry seed 0; B1/B2 pilot seed 0; B3/B3geom seeds 0/1/2 (mean (per seed))

## Semantic mIoU %

| region | B0 | B1 | B2 | B3 | B3geom |
|---|---|---|---|---|---|
| observed in frame t | 15.21 | 15.21 | 15.21 | 15.21 (15.21 / 15.21 / 15.21) | 15.21 (15.21 / 15.21 / 15.21) |
| observed only earlier | 17.38 | 17.38 | 17.38 | 17.38 (17.38 / 17.38 / 17.38) | 17.38 (17.38 / 17.38 / 17.38) |
| never observed (GT occupancy) | 8.39 | 8.60 | 8.66 | 8.85 (8.85 / 8.88 / 8.83) | 8.68 (8.68 / 8.68 / 8.70) |
| completed_unobserved (primary) | 8.80 | 9.13 | 9.21 | 9.38 (9.39 / 9.37 / 9.37) | 9.28 (9.27 / 9.28 / 9.29) |
| geometry observed, semantically unsupported | 9.72 | 10.13 | 10.90 | 10.70 (10.73 / 10.68 / 10.68) | 10.28 (10.26 / 10.28 / 10.31) |
| geometry unobserved | 7.73 | 8.03 | 7.99 | 8.16 (8.16 / 8.15 / 8.17) | 8.14 (8.14 / 8.14 / 8.14) |
| later revealed | 7.06 | 7.10 | 7.33 | 7.29 (7.29 / 7.29 / 7.30) | 7.23 (7.23 / 7.23 / 7.23) |
| later revealed, no H_t camera return | 5.97 | 5.94 | 6.35 | 6.22 (6.21 / 6.23 / 6.23) | 6.13 (6.12 / 6.14 / 6.13) |
| not revealed | 9.17 | 9.60 | 9.67 | 9.92 (9.94 / 9.90 / 9.91) | 9.78 (9.77 / 9.78 / 9.79) |
| primary ∩ candidate-hit | 23.41 | 25.75 | 26.54 | 27.08 (27.17 / 27.00 / 27.06) | 25.96 (25.89 / 26.00 / 25.99) |
| primary ∩ candidate-miss | 0.00 | 0.05 | 0.04 | 0.08 (0.08 / 0.08 / 0.09) | 0.07 (0.07 / 0.07 / 0.08) |
| SSC full | 1.55 | 1.67 | 1.60 | 1.67 (1.67 / 1.67 / 1.67) | 1.66 (1.66 / 1.66 / 1.67) |
| SSC without evidence | 1.39 | 1.51 | 1.45 | 1.51 (1.51 / 1.51 / 1.51) | 1.50 (1.50 / 1.50 / 1.50) |


## Paired differences (mIoU points; seed-pooled B3 vs single-seed controls; 95 % block bootstrap)

| pair | region | Δ | CI |
|---|---|---|---|
| B3 − B0 | completed_unobserved (primary) | +0.58 | [+0.34, +0.90] |
| B3 − B0 | geometry observed, semantically unsupported | +0.98 | [+0.40, +2.11] |
| B3 − B0 | geometry unobserved | +0.43 | [+0.30, +0.59] |
| B3 − B0 | later revealed | +0.24 | [+0.09, +0.48] |
| B3 − B0 | later revealed, no H_t camera return | +0.25 | [-0.40, +0.95] |
| B3 − B0 | primary ∩ candidate-hit | +3.66 | [+2.15, +4.50] |
| B3 − B0 | primary ∩ candidate-miss | +0.08 | [+0.05, +0.11] |
| B3 − B0 | SSC full | +0.12 | [+0.07, +0.20] |
| B3 − B1 | completed_unobserved (primary) | +0.25 | [+0.06, +0.52] |
| B3 − B1 | geometry observed, semantically unsupported | +0.57 | [+0.09, +1.61] |
| B3 − B1 | geometry unobserved | +0.14 | [+0.04, +0.24] |
| B3 − B1 | later revealed | +0.19 | [+0.04, +0.43] |
| B3 − B1 | later revealed, no H_t camera return | +0.28 | [+0.02, +0.90] |
| B3 − B1 | primary ∩ candidate-hit | +1.32 | [-0.44, +2.43] |
| B3 − B1 | primary ∩ candidate-miss | +0.03 | [+0.00, +0.06] |
| B3 − B1 | SSC full | +0.00 | [-0.04, +0.06] |
| B3 − B2 | completed_unobserved (primary) | +0.17 | [-0.01, +0.33] |
| B3 − B2 | geometry observed, semantically unsupported | -0.21 | [-0.56, +0.35] |
| B3 − B2 | geometry unobserved | +0.17 | [-0.03, +0.32] |
| B3 − B2 | later revealed | -0.04 | [-0.21, +0.25] |
| B3 − B2 | later revealed, no H_t camera return | -0.13 | [-0.28, +0.17] |
| B3 − B2 | primary ∩ candidate-hit | +0.53 | [-0.52, +1.27] |
| B3 − B2 | primary ∩ candidate-miss | +0.04 | [+0.02, +0.07] |
| B3 − B2 | SSC full | +0.07 | [+0.02, +0.14] |
| B3geom − B0 | completed_unobserved (primary) | +0.48 | [+0.30, +0.76] |
| B3geom − B0 | geometry observed, semantically unsupported | +0.56 | [+0.20, +1.78] |
| B3geom − B0 | geometry unobserved | +0.41 | [+0.28, +0.56] |
| B3geom − B0 | later revealed | +0.18 | [+0.04, +0.42] |
| B3geom − B0 | later revealed, no H_t camera return | +0.16 | [-0.46, +0.84] |
| B3geom − B0 | primary ∩ candidate-hit | +2.55 | [+1.59, +3.65] |
| B3geom − B0 | primary ∩ candidate-miss | +0.07 | [+0.05, +0.10] |
| B3geom − B0 | SSC full | +0.12 | [+0.06, +0.20] |
| B3geom − B1 | completed_unobserved (primary) | +0.15 | [+0.03, +0.36] |
| B3geom − B1 | geometry observed, semantically unsupported | +0.15 | [-0.15, +1.31] |
| B3geom − B1 | geometry unobserved | +0.11 | [+0.03, +0.18] |
| B3geom − B1 | later revealed | +0.13 | [-0.03, +0.40] |
| B3geom − B1 | later revealed, no H_t camera return | +0.19 | [-0.10, +0.87] |
| B3geom − B1 | primary ∩ candidate-hit | +0.21 | [-0.89, +1.39] |
| B3geom − B1 | primary ∩ candidate-miss | +0.02 | [-0.00, +0.05] |
| B3geom − B1 | SSC full | -0.01 | [-0.04, +0.05] |
| B3geom − B2 | completed_unobserved (primary) | +0.07 | [-0.16, +0.26] |
| B3geom − B2 | geometry observed, semantically unsupported | -0.62 | [-0.88, +0.16] |
| B3geom − B2 | geometry unobserved | +0.15 | [-0.06, +0.30] |
| B3geom − B2 | later revealed | -0.10 | [-0.27, +0.21] |
| B3geom − B2 | later revealed, no H_t camera return | -0.22 | [-0.42, +0.12] |
| B3geom − B2 | primary ∩ candidate-hit | -0.58 | [-1.61, +0.75] |
| B3geom − B2 | primary ∩ candidate-miss | +0.03 | [+0.01, +0.06] |
| B3geom − B2 | SSC full | +0.06 | [-0.01, +0.13] |
| B3 − B3geom | completed_unobserved (primary) | +0.10 | [-0.01, +0.24] |
| B3 − B3geom | geometry observed, semantically unsupported | +0.41 | [+0.00, +0.69] |
| B3 − B3geom | geometry unobserved | +0.02 | [-0.03, +0.08] |
| B3 − B3geom | later revealed | +0.06 | [+0.00, +0.12] |
| B3 − B3geom | later revealed, no H_t camera return | +0.09 | [-0.01, +0.21] |
| B3 − B3geom | primary ∩ candidate-hit | +1.11 | [-0.26, +1.55] |
| B3 − B3geom | primary ∩ candidate-miss | +0.01 | [-0.01, +0.02] |
| B3 − B3geom | SSC full | +0.01 | [-0.01, +0.03] |


## Per class (primary): coverage vs association

| class | GT voxels | nearest hit % | any-of-16 % | B3 selected correct % | B0 IoU % | B3 IoU % | larger gap |
|---|---|---|---|---|---|---|---|
| car | 35,497 | 33.41 | 47.17 | 33.81 | 9.06 | 10.91 | candidate absence |
| bicycle | 69 | 2.90 | 4.35 | 0.00 | 0.20 | 0.26 | candidate absence |
| motorcycle | 189 | 7.41 | 14.81 | 2.65 | 3.38 | 5.61 | candidate absence |
| truck | 2,236 | 71.60 | 84.48 | 67.10 | 7.96 | 9.29 | association |
| other vehicle | 3,470 | 3.86 | 13.72 | 2.50 | 0.93 | 1.24 | candidate absence |
| person | 99 | 12.12 | 42.42 | 10.44 | 0.82 | 1.00 | candidate absence |
| road | 307,633 | 19.60 | 28.45 | 22.96 | 17.52 | 18.25 | candidate absence |
| parking | 177,605 | 4.99 | 14.05 | 5.82 | 3.89 | 4.83 | candidate absence |
| sidewalk | 160,432 | 10.91 | 22.33 | 14.20 | 8.33 | 7.94 | candidate absence |
| other ground | 101,243 | 5.55 | 15.81 | 6.46 | 3.17 | 4.04 | candidate absence |
| building | 484,817 | 58.42 | 77.06 | 49.67 | 35.05 | 35.62 | association |
| fence | 25,193 | 18.13 | 31.95 | 16.02 | 4.50 | 4.40 | candidate absence |
| vegetation | 1,060,981 | 78.81 | 88.84 | 84.08 | 52.77 | 54.18 | candidate absence |
| terrain | 180,188 | 0.28 | 1.43 | 0.52 | 0.26 | 0.20 | candidate absence |
| pole | 294 | 2.72 | 14.63 | 1.13 | 0.03 | 0.02 | candidate absence |
| traffic sign | 54 | 1.85 | 11.11 | 1.85 | 0.05 | 0.00 | candidate absence |
| other structure | 198,329 | 15.79 | 30.96 | 12.77 | 8.89 | 9.46 | candidate absence |
| other object | 4,883 | 5.28 | 11.98 | 3.44 | 1.53 | 1.59 | candidate absence |


'larger gap' compares candidate absence (1 − any-of-16) with the association gap (any-of-16 − B3 selected correct); descriptive, not a criterion.


## By distance to nearest evidence (primary)

| bin | voxels | candidate-hit % | B0 acc % | B3 top-weight acc % | B3 readout acc % | B0 mIoU % | B3 mIoU % |
|---|---|---|---|---|---|---|---|
| 0-1 m | 1,106,487 | 65.77 | 52.72 | 53.69 | 54.57 | 12.39 | 13.38 |
| 1-2 m | 557,765 | 65.17 | 52.35 | 52.86 | 54.12 | 9.82 | 10.48 |
| 2-4 m | 459,431 | 56.95 | 45.94 | 46.78 | 47.47 | 7.72 | 7.98 |
| 4-8 m | 221,659 | 38.13 | 30.56 | 32.02 | 32.11 | 4.23 | 4.38 |
| >8 m | 397,870 | 33.89 | 27.11 | 28.26 | 28.47 | 3.07 | 3.21 |


## By reveal delay (primary)

| bin | voxels | candidate-hit % | B0 acc % | B3 top-weight acc % | B3 readout acc % | B0 mIoU % | B3 mIoU % |
|---|---|---|---|---|---|---|---|
| 0-1 s | 251,165 | 62.48 | 52.34 | 53.51 | 54.26 | 9.26 | 9.70 |
| 1-2 s | 154,820 | 54.53 | 47.26 | 48.84 | 48.92 | 7.09 | 7.20 |
| 2-4 s | 108,370 | 39.77 | 33.55 | 35.01 | 35.28 | 5.08 | 5.65 |
| 4-10 s | 44,666 | 36.83 | 30.37 | 30.45 | 30.40 | 3.58 | 3.44 |
| not revealed | 2,184,191 | 58.21 | 46.13 | 46.96 | 47.82 | 9.17 | 9.92 |


**Decision C.**
