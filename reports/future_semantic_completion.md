# Future semantic completion: can a causal head predict the semantics of unseen, later-occupied voxels?

Minimal viability experiment, one seed. Frozen geometry (adopted Gate 8C-1 checkpoint,
seed 0), a 77 k-parameter head on `concat(trunk d1, causal input)`, trained with the
validated `KL(teacher || student)` on the key set only, evaluated on **oracle-occupied**
voxels so geometry errors never enter. Target = the cached Trident-H class-probability
vector lifted by the frozen future mapper (25-d union simplex; continuous, fixed
vocabulary -- no language-aligned dense embedding is cached, and this is the
representation's stated limit). Numbers: `artifacts/future_semantic_completion/results_table.md`.

## 1. Is the experiment causally valid?

Yes. All 1,948 sample files have `input_frames ≤ t < target_frames ≤ t+20`; erasing every
target array leaves the 32-channel input bit-identical (7/7 files probed). All 1,566,585
supervision voxels are `observed = False` at `t` and LiDAR-occupied; on drive 0006 the
sample-rule set equals the measurement-mask set (`later_measured & measured_occ`) voxel
for voxel (609,435, symmetric difference 0). The frozen network has 0 parameters
requiring grad, a semantic backward leaves no gradient on it, its state digest is
identical before/after training, and the occupancy residual is bit-identical. The
stacked transfer input reproduces the gate's own occupancy path exactly (max diff 0.0).
Targets are opened only after the manifest was frozen; nothing was selected on them.

## 2. How much semantic supervision exists?

957,150 key voxels over 1,358 training anchors (~700/anchor); 609,435 on drive 0006
(1,033/anchor). Only **22.8 %** of unobserved→future-observed→occupied voxels carry a
teacher vector at all. 18 classes present; pooled top-1 share 0.447 (vegetation), top-2
0.669, entropy 1.71 nats -- not one- or two-class dominated overall, but the drives differ
sharply (0003: 79 % vegetation; 0010: 41 % building). The decisive number is
**repeatability**: where the causal map already holds a teacher vector for a LiDAR-occupied
voxel and the future window lifts another, the two agree on the class only **0.456** of the
time (cosine 0.51; shuffled pairing 0.35 / 0.40), and this does not improve with evidence
weight on either side (0.44 at 1–2 rays, 0.46 at >16 rays) -- the disagreement is
systematic, not averaging noise. Per class: vegetation 0.68, building 0.60, road 0.40,
man-made structure 0.31, other ground 0.08. The within-window half-agreement filter that
already prunes the targets does not remove this.

## 3. Does the learned head beat trivial baselines?

No, not in the sense the decision rule requires. On the primary set: cosine B2 0.491 vs B0
0.476 (Δ +0.015, CI [−0.004, +0.036]) and B1 0.459; KL B2 1.690 vs B0 1.775 (Δ −0.086,
CI [−0.155, −0.016]); top-1 B2 0.388 vs B0 **0.394** (the majority-class rate) and B1
0.381. The frozen checkpoint's existing joint head is indistinguishable from B2 on every
metric. B1 has the best mIoU (0.059 vs B2 0.043) because it commits to 17 classes; B2
predicts 6 and never names `man-made structure` (21 % of targets). Within 1–2 voxels of map
evidence, B1 reaches only 0.426 against B0's 0.407: the map's own labels next door barely
predict the future label. Training KL fell to ~1.0 while validation stayed at 1.74: the
head learned drive-level class priors, not transferable voxel semantics.

## 4. Can semantics be predicted in unseen space?

Not measurably beyond the prior, and not because the head cannot fit. B2's cosine to its
own target (0.491) is already at the target's own repeatability (0.51); B2's predictions
sit at cosine 0.83 to the global mean with 0.22× the target variance. Any causal predictor
of *this* target is capped near the majority rate, because the target itself agrees with
an independent view of the same voxel only ~46 % of the time. Classification: **Case C**
(collapsed toward the mean). Verification flags exactly this: 8/10, failing
`projection_consistency` and `not_collapsed_to_mean`.

## 5. Does it transfer?

Measured once, seed 0, benchmark GT labels on GT-occupied voxels, after freezing; no
decision rests on it. SemanticKITTI unobserved (15.5 M voxels): top-1 B0 0.451, B1 0.453,
**B2 0.398** (−0.053 vs B0, CI excludes 0), existing head 0.463. Occ3D unobserved (8.7 M):
B0 0.238, B1 0.286, B2 0.314 (+0.076 vs B0), existing head 0.337. On *observed* voxels the
map's own lifted labels (B1) reach 0.537 / 0.455 top-1 and 0.18–0.19 mIoU against human
GT -- the input semantics are usable; it is the future *target* that is not repeatable.
B2's transfer sign follows the class prior it learned (vegetation/building), consistent
with Case C, not with a semantic-completion capability.

## 6. What is the single next decision?

Fix the target before touching the model. Build a **view-consistent semantic target**: keep
a voxel's teacher vector only when the *causal* map's lifted vector (where present) and
the future window's vector agree, or -- for voxels without causal evidence -- when two
disjoint future sub-windows agree *and* the fused vector is confident (max ≥ τ, chosen on
drive 0006 by the existing coverage × agreement rule). Then re-measure repeatability with
the same `diagnose_targets.py`; only if it rises well above the majority rate (say ≥ 0.6)
re-run this exact head. No larger model, no new teacher, no occupancy change.
