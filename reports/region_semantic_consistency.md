# Region semantic consistency: is the persistent semantic unit a 3D region rather than a voxel?

No model trained. Regions are geometry-only: the raw-LiDAR occupied grid cut into cubic
blocks of side 3 / 5 / 9 / 15 voxels (0.6 / 1.0 / 1.8 / 3.0 m), 26-connected components
inside a block. Semantic evidence is the per-frame future teacher replay of the previous
study, in the interleaved frame-disjoint groups A / B / C; a unit's group vector is the
mapper-weighted mean of that group's observations at its voxels; C never enters A or B.
Scored on the unobserved→future-observed→occupied population (2.02 M voxels on 0006,
4.20 M on the training drives) and, as a robustness check, on all future-observed
LiDAR-occupied voxels. 10/10 verification checks. Numbers: `results_table.md`.

## 1. What regions were constructed?

Block-bounded connected components of occupied voxels. Median occupied voxels per
validated region 6 / 14 / 38 / 80, median extent 0.6 / 1.0 / 1.8 / 3.0 m; mean teacher
observations pooled per region in A+B 6.5 / 11 / 20 / 33 (voxel: 2.1). Regions with A&B
evidence: 10.6 k / 8.6 k / 7.1 k / 6.1 k on 0006 (voxels: 18.1 k). Region membership is
deterministic and uses no semantic quantity.

## 2. Are region semantics more repeatable than voxel semantics?

No. Independent A-vs-B top-1 repeatability on 0006: voxel **0.561**, regions 0.557 /
0.552 / 0.539 / 0.525; cosine 0.622 → 0.619 / 0.616 / 0.611 / 0.602; three-way A=B=C
0.398 → 0.399 / 0.387 / 0.370 / 0.365. Training drives pooled: voxel 0.698, regions
0.680 / 0.672 / 0.658 / 0.647. On the matched voxels (those with their own A&B evidence)
inheriting the region vector changes top-1 from 0.561 to 0.569–0.575 and cosine from 0.62
to 0.63–0.67 — a marginal smoothing gain, not a change of representation. The same holds
on the all-future-observed population (0.561 → 0.552 … 0.518).

The mechanism is explicit: the mean number of *frames* contributing to a unit's A+B
evidence is 2.07 for voxels and still 2.11 / 2.13 / 2.19 / 2.25 for regions up to 3 m.
Aggregation pools many rays of the same one or two views; it never combines independent
views. Regions that do collect ≥ 4 frames agree *less* (0.42–0.48 on 0006), because they
are larger and more heterogeneous.

## 3. Is the improvement larger than local semantic-prior effects?

There is no improvement to test, and the controls show why. True-minus-R1 (within-anchor
shuffle, size-stratified) is +0.159 for voxels and +0.146 … +0.167 for regions — flat.
True-minus-R2 (local shuffle within 0.8 m z-bands × 4 m cells) is **+0.014** for voxels
and +0.012 … +0.030 for regions: swapping the B vector with that of a neighbouring unit
of similar height reproduces almost all of the agreement at every scale. Whatever is
repeatable in the lifted teacher is local class homogeneity, not unit identity.

## 4. What is the coverage and diversity cost?

Coverage is the one thing regions buy: eligible voxels lying in a unit with A&B evidence
rise from 0.9 % (voxel) to 3.0 / 5.6 / 11.8 / 21.7 %, and with any evidence from 30 % to
42 / 51 / 62 / 73 %. Diversity is unchanged (0006: 8 classes with ≥ 200 units at every
scale but 6 at 3 m; entropy 1.90 → 1.91; dominant share 0.33 → 0.36). Aggregation neither
collapses classes nor sharpens identity — it spreads the same noisy single-view labels
over more voxels.

## 5. Does the result hold across drives and classes?

Yes. Per drive, region minus voxel top-1 at 3 m: 0003 −0.07, 0007 +0.03, 0010 −0.07,
0006 −0.04; the R1 margin stays within ±0.03 of the voxel margin on every drive. Per class
on 0006, region aggregation changes repeatability by at most a few points in either
direction (vegetation 0.73 → 0.69, road 0.63 → 0.55, building 0.48 → 0.50, man-made
structure 0.56 → 0.51, car 0.46 → 0.42–0.52, sidewalk 0.10 → 0.13, other ground 0.28 →
0.23); the hard classes stay hard.

## 6. Is there a useful region scale?

No scale improves stability. If coverage alone were the goal, the 0.6 m scale is the
only one that does not lose repeatability (0.557 vs 0.561) while tripling A&B coverage;
but it does not make the representation more persistent.

## 7. Single decision

**Decision C — region semantics remain unstable.** The instability lives in the
view-conditioned lifted observations themselves: each surface patch receives essentially
one view per group from the frozen lifting, two independent views of the same patch agree
on the class only ~50–56 % of the time on 0006/0007, and geometry-based aggregation at
any scale up to 3 m cannot add views it does not have. Stop treating the lifted Trident-H
voxel/region vector as the core persistent 3D semantic representation, and rethink the
2D-to-3D semantic representation itself.
