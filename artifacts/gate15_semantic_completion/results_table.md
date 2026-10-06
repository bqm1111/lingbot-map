# Gate 15 results -- KITTI-360 0006 (590 anchors, final heads, seeds 0/1/2)

mIoU in %, per seed and mean. Semantic-only regions score GT-occupied voxels; SSC regions score all labelled voxels with the frozen occupancy.

| region | method | mIoU seeds 0/1/2 | mIoU mean | mean class acc | overall acc | GT classes predicted | classes ≥1 % of preds | dominant pred class | voxels (seed 0) |
|---|---|---|---|---|---|---|---|---|---|
| oracle_observed | B0 | 16.91 / 16.91 / 16.91 | 16.91 | 33.24 | 62.57 | 18.0/18 | 9.0 | vegetation 55.38 | 434,683 |
| oracle_observed | B1 | 9.55 / 9.89 / 9.72 | 9.72 | 15.72 | 59.81 | 8.0/18 | 6.3 | vegetation 55.35 | 434,683 |
| oracle_unobserved | B0 | 8.39 / 8.39 / 8.39 | 8.39 | 17.01 | 49.91 | 18.0/18 | 9.0 | vegetation 50.63 | 7,834,276 |
| oracle_unobserved | B1 | 5.62 / 5.54 / 5.97 | 5.71 | 10.08 | 43.61 | 8.0/18 | 6.0 | vegetation 57.23 | 7,834,276 |
| completed_observed | B0 | 14.86 / 16.21 / 16.13 | 15.73 | 32.29 | 55.71 | 17.0/17 | 9.0 | vegetation 53.19 | 275,149 |
| completed_observed | B1 | 9.35 / 9.44 / 9.18 | 9.32 | 15.66 | 52.41 | 8.0/17 | 6.7 | vegetation 53.38 | 275,149 |
| completed_unobserved **(primary)** | B0 | 8.80 / 8.96 / 9.11 | 8.96 | 20.00 | 46.37 | 18.0/18 | 9.0 | vegetation 49.86 | 2,743,212 |
| completed_unobserved **(primary)** | B1 | 6.12 / 6.06 / 7.43 | 6.54 | 11.37 | 39.79 | 7.7/18 | 6.3 | vegetation 43.11 | 2,743,212 |
| ssc_full | B0 | 1.55 / 1.55 / 1.65 | 1.58 | 7.86 | 17.55 | 18.0/18 | 9.3 | vegetation 45.40 | 558,778,307 |
| ssc_full | B1 | 0.90 / 1.17 / 1.34 | 1.14 | 5.82 | 15.22 | 8.3/18 | 6.0 | other ground 47.64 | 558,778,307 |
| ssc_no_evidence | B0 | 1.39 / 1.35 / 1.42 | 1.39 | 6.69 | 16.48 | 18.0/18 | 9.7 | vegetation 45.50 | 553,275,690 |
| ssc_no_evidence | B1 | 0.84 / 1.15 / 1.31 | 1.10 | 5.55 | 14.14 | 8.3/18 | 6.0 | other ground 48.55 | 553,275,690 |
| oracle_observed | teacher readout | 16.91 | 16.91 | 33.24 | 62.57 | 18/18 | 9 | vegetation 55.38 | 434,683 |


## Promotion rule (frozen)

| criterion | value | pass |
|---|---|---|
| P1 Δ mIoU completed_unobserved (B1−B0) | per seed -2.68, -2.90, -1.68; mean -2.42; CI [-3.06, -1.74] (need ≥ +2.00, all > 0, CI lo > 0) | False |
| P2 Δ SSC mIoU ssc_full | per seed -0.65, -0.38, -0.31; mean -0.44; CI [-0.56, -0.31] | False |
| P3 breadth / collapse | 3 classes gain (need ≥ 5); B1 dominant 43.11 (limit 60.00); B1 classes ≥ 1 % 6.3 (need ≥ 5) | False |
| P4 occupancy identical | seed 0: True; seed 1: True; seed 2: True | True |
| D teacher poor on observed | accuracy 62.57, mIoU 16.91 (D if < 40 / < 15) | False |


**Decision B.**


## Propagation distance (completed_unobserved, mean over seeds)

| distance to nearest evidence | voxels (seed 0) | B0 mIoU | B1 mIoU | B0 acc | B1 acc |
|---|---|---|---|---|---|
| 0-1 m | 1,106,487 | 12.59 | 8.61 | 53.15 | 50.91 |
| 1-2 m | 557,765 | 9.88 | 7.44 | 52.35 | 43.82 |
| 2-4 m | 459,431 | 7.79 | 5.94 | 46.36 | 35.41 |
| 4-8 m | 221,659 | 4.30 | 3.29 | 31.11 | 23.84 |
| >8 m | 397,870 | 3.08 | 1.66 | 27.27 | 16.27 |


## Label-free held-out evidence (hidden (8,8,4) blocks, cosine to the teacher at hidden evidence voxels)

| seed | hidden voxels | cos B1 | cos B0 (nearest visible) | class agreement B1 | class agreement B0 |
|---|---|---|---|---|---|
| 0 | 5,294,676 | 0.9226 | 0.9799 | 70.43 | 84.16 |
| 1 | 5,294,676 | 0.9221 | 0.9799 | 68.14 | 84.16 |
| 2 | 5,294,676 | 0.9234 | 0.9799 | 68.52 | 84.16 |


## Per-class IoU on completed_unobserved (mean over seeds)

| class | B0 | B1 | B1 − B0 | GT share % | B1 pred share % |
|---|---|---|---|---|---|
| car | 10.41 | 0.10 | -10.31 | 1.29 | 0.03 |
| bicycle | 0.19 | 0.00 | -0.19 | 0.00 | 0.00 |
| motorcycle | 3.74 | 0.00 | -3.74 | 0.01 | 0.00 |
| truck | 8.18 | 0.00 | -8.18 | 0.08 | 0.00 |
| other vehicle | 1.11 | 0.36 | -0.75 | 0.13 | 1.50 |
| person | 0.89 | 0.00 | -0.89 | 0.00 | 0.00 |
| road | 17.47 | 29.35 | 11.89 | 11.21 | 8.33 |
| parking | 3.91 | 7.28 | 3.37 | 6.47 | 8.13 |
| sidewalk | 8.21 | 0.00 | -8.21 | 5.85 | 0.00 |
| other ground | 3.13 | 5.09 | 1.95 | 3.69 | 30.63 |
| building | 34.90 | 21.73 | -13.17 | 17.67 | 6.64 |
| fence | 4.67 | 0.00 | -4.67 | 0.92 | 0.00 |
| vegetation | 53.29 | 52.24 | -1.05 | 38.68 | 43.11 |
| terrain | 0.23 | 0.00 | -0.23 | 6.57 | 0.00 |
| pole | 0.03 | 0.00 | -0.03 | 0.01 | 0.00 |
| traffic sign | 0.05 | 0.00 | -0.05 | 0.00 | 0.00 |
| other structure | 9.00 | 1.53 | -7.47 | 7.23 | 1.64 |
| other object | 1.80 | 0.00 | -1.80 | 0.18 | 0.00 |
