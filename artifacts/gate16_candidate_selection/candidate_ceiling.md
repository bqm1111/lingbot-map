# Candidate ceiling (0006, reconstructed with the frozen candidate rule; evaluation only; frozen before any B3 score)

| region | voxels | C0 nearest-candidate acc % | C16 any-of-16 hit % | oracle selection acc % |
|---|---|---|---|---|
| completed_unobserved (primary) | 2,743,212 | 46.00 | 57.32 | 57.32 |
| geometry observed, semantically unsupported | 476,001 | 40.57 | 54.27 | 54.27 |
| geometry unobserved | 2,267,211 | 47.14 | 57.96 | 57.96 |
| later revealed | 559,021 | 45.52 | 53.83 | 53.83 |
| later revealed, no H_t camera return | 375,035 | 42.39 | 49.88 | 49.88 |
| not revealed | 2,184,191 | 46.13 | 58.21 | 58.21 |


Primary mIoU: nearest candidate 8.77 %, oracle selection 14.90 %; on candidate-hit voxels nearest 23.34 %.

Candidate-miss fraction of the primary region: 42.68 %. History-wide evidence ceiling (context only): 97.49 %.


**Selection headroom** H = 57.32 − 46.00 = 11.31 points; **threshold** T = N + 0.5 H = **51.66 %**.


## Per class (primary)

| class | GT voxels | nearest hit % | any-of-16 hit % |
|---|---|---|---|
| car | 35,497 | 33.41 | 47.17 |
| bicycle | 69 | 2.90 | 4.35 |
| motorcycle | 189 | 7.41 | 14.81 |
| truck | 2,236 | 71.60 | 84.48 |
| other vehicle | 3,470 | 3.86 | 13.72 |
| person | 99 | 12.12 | 42.42 |
| road | 307,633 | 19.60 | 28.45 |
| parking | 177,605 | 4.99 | 14.05 |
| sidewalk | 160,432 | 10.91 | 22.33 |
| other ground | 101,243 | 5.55 | 15.81 |
| building | 484,817 | 58.42 | 77.06 |
| fence | 25,193 | 18.13 | 31.95 |
| vegetation | 1,060,981 | 78.81 | 88.84 |
| terrain | 180,188 | 0.28 | 1.43 |
| pole | 294 | 2.72 | 14.63 |
| traffic sign | 54 | 1.85 | 11.11 |
| other structure | 198,329 | 15.79 | 30.96 |
| other object | 4,883 | 5.28 | 11.98 |


## Per distance to nearest evidence

| bin | voxels | nearest % | any-of-16 % |
|---|---|---|---|
| 0-1 m | 1,106,487 | 52.70 | 65.77 |
| 1-2 m | 557,765 | 52.36 | 65.17 |
| 2-4 m | 459,431 | 45.95 | 56.95 |
| 4-8 m | 221,659 | 30.57 | 38.13 |
| >8 m | 397,870 | 27.11 | 33.89 |


## Per reveal delay

| bin | voxels | nearest % | any-of-16 % |
|---|---|---|---|
| 0-1 s | 251,165 | 52.30 | 62.48 |
| 1-2 s | 154,820 | 47.28 | 54.53 |
| 2-4 s | 108,370 | 33.55 | 39.77 |
| 4-10 s | 44,666 | 30.38 | 36.83 |
| not revealed | 2,184,191 | 46.13 | 58.21 |
