# Does feature supervision complete unobserved space better than class supervision?

Three conditions differing in one loss term; trained on KITTI-360 drives 0003/0007/0010,
selected on drive 0006, evaluated once on SemanticKITTI and Occ3D-nuScenes. `A = L_occ`,
`B = L_occ + 0.5·L_KL` (the adopted objective, retrained as the matched control),
`C = L_occ + 0.5·L_feature` (cosine distance to a frozen SAM-H embedding projected to 64
dimensions). Nine runs, 6 000 steps, final-step checkpoints, no selection. A/B/C for a seed
drew byte-identical files and crops (24 000 draws verified).

## 1. Did C improve later-measured completion relative to B?

Yes, consistently, and by very little. Primary metric — SC IoU on map-unobserved voxels that
a future sweep later measured occupied or free (590 drive-0006 anchors, 412 k such voxels per
anchor, 6.7 % prevalence):

| | seed 0 | seed 1 | seed 2 | median |
|---|---|---|---|---|
| A | 0.1080 | 0.1118 | 0.1114 | **0.1114** |
| B | 0.1037 | 0.1069 | 0.1031 | **0.1037** |
| C | 0.1078 | 0.1097 | 0.1078 | **0.1078** |

C − B = +0.0041 / +0.0028 / +0.0047. Every seed positive; every bootstrap CI (contiguous
20-anchor blocks) excludes zero. That is **+0.4 pp**, not the 1–2 pp that would be useful.

The result that matters more sits next to it: **both auxiliary objectives lose to occupancy
loss alone in-domain.** C − A = −0.0002 / −0.0021 / −0.0036, B − A = −0.0043 / −0.0049 /
−0.0083. C recovers most of what B gives up; it buys nothing over A.

## 2. Did any gain come with additional false occupancy?

Partly. At its own selected threshold C predicts 1.81× the occupied volume of ground truth,
against 1.72× for B and 1.70× for A. Relative to B, C adds 191 k true positives and 1.35 M
false positives, raising recall 0.2623 → 0.2739 while precision moves only 0.1505 → 0.1511.
The IoU gain is real rather than a threshold artefact, but it comes from filling more space
at roughly constant precision, not from completing more precisely.

## 3. Did the improvement transfer?

**No.** C − B changes sign across seeds on both target datasets.

| median IoU | A | B | C | C − B per seed |
|---|---|---|---|---|
| SemanticKITTI (163) | 0.2255 | 0.2295 | **0.2341** | +0.0085, −0.0053, +0.0057 |
| Occ3D (1 182) | 0.2854 | 0.3134 | **0.3139** | +0.0140, −0.0267, +0.0134 |

Mean C − B is +0.0030 and +0.0002; seed spread is several times the effect. Model-free
baselines reproduced exactly (mapper 0.0890/0.0999, dilate 0.1541/0.2164), so the spread is
the models, not the evaluator.

The transfer table contains a sharper result than the one the study set out to measure:
**auxiliary supervision hurts in-domain and helps transfer.** B − A is −0.006 on drive 0006
but +0.0079 on SemanticKITTI and +0.0352 on Occ3D — every seed, every dataset. Predicting a
frozen teacher costs completion accuracy where you train and pays it back where you don't.

## 4. What was learned about feature supervision?

C's feature head barely learned. Held-out cosine agreement on drive 0006 is 0.2085–0.2143,
against **0.1990 for a constant predictor that outputs the training-set mean feature**. The
margin is ~0.012. Predicted features have a per-dimension spread of 0.030 against the
target's 0.101 — roughly a third. The head learned mostly where the mean is.

Two mechanisms are visible in the training logs. The auxiliary objectives do not push equally:
median ‖∇L_aux‖/‖∇L_occ‖ at the shared trunk is **4.2–4.5 for B** and **1.9–2.2 for C**, at
the same nominal weight 0.5. And final occupancy loss orders A < C < B on all three seeds —
C interferes less because it pulls less. So "C beats B" is confounded with "C perturbs the
trunk half as much", and the +0.4 pp may be that rather than the representation.

SAM's letterboxing is a second limit: a 376×1408 image fills ~17 of 64 encoder rows, so the
teacher is vertically coarse (~22 native px per cell) before any lifting.

## 5. What remains unresolved?

Whether feature supervision helps at matched *gradient* magnitude rather than matched loss
weight. Whether a non-collapsing head would behave differently. Whether the transfer benefit
of auxiliary supervision involves the teacher at all, or is just regularisation. C's
semantic-label metrics are unavailable — it never trains the 25-class head and no validated
decoder exists for SAM features; only B's SSC mIoU (0.0319 / 0.0422) is a measurement.
"Map-unobserved" describes the mapper's causal state, not visibility, and later-measured-free
can mean a moving object left.

## 6. Single next step

Rerun C at an auxiliary weight that equalises trunk gradient magnitude with B (≈1.1 rather
than 0.5, from the measured 4.3/2.0 ratio). Until the two objectives push equally hard, the
C-vs-B comparison measures optimisation, not information content — and that is one training
sweep, not a new research direction.
