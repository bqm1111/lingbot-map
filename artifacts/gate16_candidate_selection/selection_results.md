# Stage 1 — candidate selection (primary region, 0006)

N (nearest candidate) 46.00 %, O (oracle) 57.32 %, H 11.31 points, T 51.66 %.

| method | readout acc % | top-weight candidate acc % | candidate-hit acc % | candidate-miss acc % | recovered fraction R |
|---|---|---|---|---|---|
| C0 nearest candidate | 46.00 |  |  |  | 0.00 |
| B0 nearest evidence (EDT) | 46.01 | 46.00 | 80.27 | 0.00 | 0.00 |
| B1 (pilot, seed 0) | 47.09 | 45.82 | 82.04 | 0.15 | 0.10 |
| B2 (pilot, seed 0) | 46.03 | 45.40 | 80.24 | 0.10 | 0.00 |
| B3_s0 | 47.76 | 47.19 | 83.16 | 0.22 | 0.16 |
| B3_s1 | 47.65 | 46.83 | 82.98 | 0.21 | 0.15 |
| B3_s2 | 47.66 | 46.76 | 82.97 | 0.24 | 0.15 |
| **B3 mean** | 47.69 | 46.93 | 83.04 | 0.23 | **0.15** |
| B3geom_s0 | 47.48 | 46.38 | 82.69 | 0.21 | 0.13 |
| B3geom_s1 | 47.48 | 46.28 | 82.68 | 0.21 | 0.13 |
| B3geom_s2 | 47.48 | 46.37 | 82.68 | 0.22 | 0.13 |
| **B3geom mean** | 47.48 | 46.34 | 82.68 | 0.21 | **0.13** |
| oracle selection | 57.32 |  | 100.00 | 0.00 | 1.00 |


B3 − B0 accuracy: +1.68 [+1.50, +1.88]; on candidate-hit +2.77 [+2.43, +3.09] (seed-pooled, 20-anchor blocks).


B3geom − B0 accuracy: +1.47 [+1.27, +1.68]; on candidate-hit +2.41 [+2.10, +2.70] (seed-pooled, 20-anchor blocks).


## Label-free agreement with the delayed candidate target (0006 delayed-target voxels, not evidence) and GT diagnostic there

| method | voxels | top-1 = best % | top-3 ∋ best % | mean weight on best % | KL(target‖w) | top-weight class = GT % | delayed-best class = GT % | nearest class = GT % |
|---|---|---|---|---|---|---|---|---|
| B0 | 2,653,083 | 7.60 | 17.04 | 7.60 | n/a | 48.13 | 53.52 | 48.13 |
| B1 | 2,653,083 | 8.98 | 21.20 | 6.67 | 0.9942 | 48.32 | 53.52 | 48.13 |
| B2 | 2,653,083 | 20.18 | 33.01 | 17.24 | 6.7711 | 49.26 | 53.52 | 48.13 |
| B3_s0 | 2,653,083 | 30.81 | 47.47 | 8.58 | 0.2992 | 50.97 | 53.52 | 48.13 |
| B3_s1 | 2,653,083 | 31.69 | 48.62 | 8.57 | 0.2985 | 50.64 | 53.52 | 48.13 |
| B3_s2 | 2,653,083 | 32.11 | 48.57 | 8.57 | 0.2970 | 50.75 | 53.52 | 48.13 |
| B3geom_s0 | 2,653,083 | 12.80 | 28.64 | 6.39 | 0.3552 | 49.23 | 53.52 | 48.13 |
| B3geom_s1 | 2,653,083 | 13.00 | 28.86 | 6.39 | 0.3553 | 49.39 | 53.52 | 48.13 |
| B3geom_s2 | 2,653,083 | 13.12 | 29.10 | 6.40 | 0.3550 | 49.42 | 53.52 | 48.13 |


## Criteria

| criterion | value | pass |
|---|---|---|
| S1 seed-mean acc ≥ T, all seeds > B0, CI lo > 0 | 47.69 vs T 51.66; R 0.15; seeds [1.75, 1.65, 1.65]; CI [+1.50, +1.88] | False |
| S2 candidate-hit gain ≥ +2, all seeds, CI lo > 0 | +2.77; seeds [2.89, 2.71, 2.7]; CI [+2.43, +3.09] | True |
| M1 mIoU gain ≥ +1, all seeds, CI lo > 0, > B2, ≥ 5 classes | +0.58; seeds [0.59, 0.58, 0.58]; CI [+0.34, +0.90]; B3−B2 +0.17; 13 classes | False |
| Occ occupancy identical |  | True |
| D delayed target learned but GT selection worse | top-1 B3 31.54 vs B1 8.98; acc-gain CI upper +1.88 | False |


**Decision C.**
