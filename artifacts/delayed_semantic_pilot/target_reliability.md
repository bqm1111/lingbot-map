# Target reliability (label-free gate, then a labelled 0006 diagnostic)

## G1 registration, training drives pooled (point-to-plane, m)

| offset (frames) | points | median | p90 |
|---|---|---|---|
| 1 | 750384 | 0.027 | 0.073 |
| 5 | 206792 | 0.055 | 0.137 |
| 10 | 59911 | 0.069 | 0.184 |
| 20 | 0 | — | — |


## G1 per drive (strict reading) with perturbation controls (median after +0.2 m shift)

| drive | offset | points | median | p90 | z+0.2 median | y+0.2 median | per-drive pass |
|---|---|---|---|---|---|---|---|
| 0003 | 1 | 114368 | 0.028 | 0.071 | 0.215 | 0.033 | True |
| 0003 | 5 | 31268 | 0.043 | 0.124 | 0.221 | 0.044 | True |
| 0003 | 10 | 15937 | 0.086 | 0.180 | 0.245 | 0.084 | True |
| 0007 | 1 | 266596 | 0.030 | 0.079 | 0.215 | 0.036 | True |
| 0007 | 5 | 55250 | 0.041 | 0.123 | 0.210 | 0.046 | True |
| 0007 | 10 | 15555 | 0.045 | 0.137 | 0.173 | 0.069 | True |
| 0010 | 1 | 369420 | 0.026 | 0.069 | 0.203 | 0.038 | True |
| 0010 | 5 | 120274 | 0.065 | 0.143 | 0.203 | 0.098 | True |
| 0010 | 10 | 28419 | 0.076 | 0.207 | 0.148 | 0.108 | False |
| 0006 | 1 | 282989 | 0.024 | 0.072 | 0.186 | 0.046 | True |
| 0006 | 5 | 143588 | 0.061 | 0.156 | 0.139 | 0.117 | True |
| 0006 | 10 | 33980 | 0.090 | 0.217 | 0.167 | 0.103 | False |


## G3 cross-view repeatability (first vs second later view, top-1 %)

| tier | train pairs | train agree | train shuffled | 0006 pairs | 0006 agree | 0006 shuffled |
|---|---|---|---|---|---|---|
| tier A | 637,886 | 75.16 | 38.06 | 413,041 | 75.87 | 35.95 |
| tier B | 1,265,965 | 73.31 | 36.46 | 1,019,563 | 72.67 | 32.70 |


## Coverage (G2), reobservation (G4), student evidence vs target

| drive | mean |D|/anchor | tier A | tier B | tier A seen by an H_t camera % | H_t view vs target % | student evidence vs target % | tier A time to first view |
|---|---|---|---|---|---|---|---|
| 0003 | 2952 | 1045 | 1907 | 87.38 | 78.42 | 53.61 | A 0-1 s 133,095, A 1-2 s 55,482, A 2-4 s 11,942, A 4-10 s 1,164 |
| 0007 | 3431 | 1017 | 2414 | 84.03 | 72.22 | 69.21 | A 0-1 s 399,821, A 1-2 s 143,743, A 2-4 s 25,401, A 4-10 s 9,796 |
| 0010 | 3381 | 1215 | 2166 | 86.70 | 63.54 | 42.89 | A 0-1 s 472,332, A 1-2 s 187,094, A 2-4 s 52,400, A 4-10 s 12,096 |
| 0006 | 4497 | 1258 | 3239 | 84.57 | 67.10 | 61.48 | A 0-1 s 510,122, A 1-2 s 166,238, A 2-4 s 57,843, A 4-10 s 7,837 |


Training target class counts (argmax): vegetation 2,182,510, road 716,395, other ground 383,721, building 303,178, car 225,203, parking 221,647, fence 115,810, sidewalk 104,494, other structure 71,031, terrain 63,700, other vehicle 52,512, pole 37,814, truck 25,111, other object 11,614, traffic sign 9,761, bicycle 9,034, person 2,336, motorcycle 792


## LABELLED DIAGNOSTIC (0006 semantic GT, never used by a rule): target class vs GT

| set | voxels | accuracy % | mIoU % |
|---|---|---|---|
| D_A | 577,769 | 62.72 | 19.68 |
| D_B | 1,430,463 | 63.00 | 18.39 |
| D_view0 | 2,008,232 | 61.88 | 16.62 |
| evidence_reobserved | 55,214 | 62.62 | 20.00 |


Gate: G1 True (strict per drive False), G2 True (14 classes), G3 A True / B True, G4 True → **targets usable**, delayed loss tiers [1, 2].
