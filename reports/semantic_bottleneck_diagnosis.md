# Where the semantic completion error comes from

**Run 2026-10-20.** Raw: `artifacts/waymo_arm/results/bottleneck_B0.json`. Code:
`tools/waymo_arm/cli/bottleneck.py`. Population: the canonical zero-shot predictions,
SemanticKITTI sequence 08 (801 anchors) and Occ3D-nuScenes validation (4,219 anchors),
method **B0**, geometry seed 0. Both the KITTI-360 arm (the deployed system) and the Waymo
arm are decomposed, so the diagnosis can be checked against a second geometry.

This report answers **Q3**: *is the dominant remaining bottleneck geometric completion,
teacher quality, evidence coverage, or unobserved-region semantic inference?*

---

## 1. Measured evidence — the partition

Every GT-occupied valid voxel falls in exactly one bucket; the buckets sum to the whole
population, asserted per sample in code. `occ_false` sits outside the partition because a
false positive has no GT label, but it still costs IoU.

### SemanticKITTI, KITTI-360 arm (the deployed system)

93,298,722 GT-occupied voxels; 73,588,119 false positives.

| bucket | voxels | share of GT occupied |
|---|---:|---:|
| `occ_miss` — GT occupied, predicted empty | 49,245,689 | **52.8 %** |
| `ev_right` — occupied, has evidence, label right | 2,934,511 | 3.1 % |
| `ev_wrong` — occupied, has evidence, label wrong | 2,286,917 | 2.5 % |
| `noev_right` — occupied, no evidence, label right | 15,363,584 | 16.5 % |
| `noev_wrong` — occupied, no evidence, label wrong | 23,468,021 | **25.2 %** |

Derived, all from the table above:

* **Occupancy recall 47.2 %, precision 37.5 %** (44,053,033 true positives against
  73,588,119 false positives).
* **Semantic evidence reaches 5.6 % of GT-occupied voxels** and 11.9 % of the correctly
  occupied ones. The other 88.1 % of correct geometry carries a label nothing observed.
* **Where the teacher speaks directly it is 56.2 % correct** (2,934,511 / 5,221,428). B0's
  label at an evidence voxel *is* that voxel's own teacher readout — its nearest evidence is
  itself at distance 0 — so this bucket is the 2D teacher's own error rate, not a transfer
  artefact.
* **Where it must extrapolate, the nearest-evidence copy is 39.6 % correct**
  (15,363,584 / 38,831,605).

### Occ3D-nuScenes, KITTI-360 arm

46,118,669 GT-occupied voxels; 4,604,698 false positives.

| bucket | voxels | share |
|---|---:|---:|
| `occ_miss` | 29,917,647 | **64.9 %** |
| `ev_right` | 1,219,640 | 2.6 % |
| `ev_wrong` | 1,735,396 | 3.8 % |
| `noev_right` | 3,961,747 | 8.6 % |
| `noev_wrong` | 9,284,239 | **20.1 %** |

Occupancy recall 35.1 %, precision 77.9 %. Evidence reaches 6.4 % of GT-occupied voxels,
18.2 % of correct geometry. Teacher correctness where it speaks **41.3 %**; extrapolation
**29.9 %**.

### Internal consistency

The true-positive counts implied by `occ_miss` reproduce, exactly, the `correct_occupied`
counts from the independent Phase 4 scorer: SemanticKITTI 44,053,033 (KITTI-360) and
39,683,185 (Waymo); Occ3D 16,201,022 and 11,914,775. The two tools share no accumulation
code, so this is a real cross-check on both.

---

## 2. Measured evidence — the oracles

Bucket sizes say where the voxels are, not what fixing a stage would buy. Each oracle
substitutes **GT labels into one bucket only**, leaves the predicted occupancy mask
untouched, and the full confusion is recomputed.

Every oracle is confined to GT-occupied voxels on purpose. Substituting the GT label on a
false positive sets it to "free" and so deletes the false positive — such an oracle scores
occupancy repair as a semantic gain. An earlier draft of this tool did exactly that and
reported an `oracle_semantics` of 44.79 on a SemanticKITTI subset where the correct figure
was 23.32; the 21-point difference was entirely false-positive deletion. False positives are
reported but never repaired here.

| | SemanticKITTI | | Occ3D | |
|---|---:|---:|---:|---:|
| | mIoU | gain | mIoU | gain |
| B0 as deployed | 4.32 | — | 3.71 | — |
| `oracle_teacher` — perfect labels on evidence voxels | 5.94 | **+1.63** | 5.59 | **+1.89** |
| `oracle_extrapolation` — perfect labels on non-evidence occupied voxels | 21.82 | **+17.50** | 16.20 | **+12.49** |
| `oracle_semantics` — perfect labels on all correct geometry | 23.88 | +19.56 | 19.87 | +16.16 |

Two readings, both load-bearing:

1. **Perfecting the 2D teacher where it already speaks is worth under 2 mIoU points** on
   either benchmark. Teacher accuracy is not the binding constraint.
2. **`oracle_semantics` reaches 23.88 / 19.87 out of 100.** Perfect semantics on the geometry
   we already get right leaves roughly **76 and 80 points of the metric on the table**, and
   those belong to occupancy — 52.8 % / 64.9 % of GT surface never predicted, plus 1.7
   false positives per true positive on SemanticKITTI.

---

## 3. Measured evidence — distance, and what it rules out

Label accuracy on correctly-occupied voxels against distance to the nearest evidence voxel.
The right-hand columns exclude evidence voxels themselves, which sit at distance 0 and would
otherwise mix the teacher's own accuracy into the extrapolation curve.

| bin | SK all, n | acc | SK no-evidence, n | acc |
|---|---:|---:|---:|---:|
| 0–1 m | 24,448,451 | 47.67 % | 19,227,023 | **45.35 %** |
| 1–2 m | 7,142,365 | 42.92 % | 7,142,365 | 42.92 % |
| 2–4 m | 6,352,417 | 37.20 % | 6,352,417 | 37.20 % |
| 4–8 m | 3,332,624 | 21.09 % | 3,332,624 | 21.09 % |
| > 8 m | 2,777,176 | 18.46 % | 2,777,176 | 18.46 % |

**This is not primarily a distance problem.** 19.2 M of the 38.8 M non-evidence correct
voxels sit *within one metre* of evidence, and there the copy is still wrong 55 % of the
time. Accuracy does decay with distance — 45.4 % to 18.5 % — but the mass is at the near
end, so a scheme that only improved far extrapolation would address a minority of the error.

**And it bounds what better extrapolation can win.** At distance 0 the teacher itself is
56.2 % correct; one metre away the copy is 45.4 % correct. The copy rule loses about 11
points relative to its source. The source is the ceiling, so the `oracle_extrapolation`
figure of +17.50 — computed with **GT** labels — is *not* reachable by a better extrapolator
fed by this teacher. It locates the headroom; it does not say a better copier can reach it.

### Error character

| | intra-superclass share of errors |
|---|---:|
| SemanticKITTI, teacher errors (`ev_wrong`) | 35.1 % |
| SemanticKITTI, extrapolation errors (`noev_wrong`) | 28.5 % |
| Occ3D, teacher errors | 38.1 % |
| Occ3D, extrapolation errors | 34.3 % |

Using OccAny's own superclass mapping. Roughly a third of errors stay inside a superclass
(road/sidewalk, car/truck) and are candidates for genuine ambiguity; **about two thirds cross
superclass boundaries** (building vs vegetation) and are not defensible as ambiguity.

---

## 4. The brief's five categories, scored

On SemanticKITTI, KITTI-360 arm:

| category | extent | oracle value |
|---|---|---|
| **Incorrect occupancy** | 52.8 % of GT surface missed; precision 37.5 % | owns ~76 of 100 metric points |
| **Missing semantic evidence** | 94.4 % of GT-occupied voxels, 88.1 % of correct geometry | — (see §5) |
| **Incorrect 2D teacher prediction** | 43.8 % error rate over 5.6 % of voxels | **+1.63** |
| **Incorrect association / extrapolation** | 60.4 % error rate over 94.4 % of voxels | **+17.50**, but capped by the teacher |
| **Potential semantic ambiguity** | ≤ ~⅓ of semantic errors | not separately isolable |

Ranked: **geometric completion first, by a wide margin; then the coupled pair of evidence
coverage and unobserved-region inference; teacher accuracy last.**

---

## 5. Hypotheses, and why coverage and extrapolation cannot be separated here

"Missing semantic evidence" and "incorrect extrapolation" are two descriptions of the same
voxels: a voxel with no evidence *must* be extrapolated. The oracle separates the headroom
from the teacher's share, but it cannot say whether reaching that headroom calls for **more
coverage** or **better inference from existing coverage**. Three readings remain live:

* **H-coverage.** Extend the evidence field; extrapolation then has less to do.
  *Contrary evidence already on file:* Gate 18's temporal ladder saturates — evidence voxels
  go 17,887 (5 frames) → 18,289 (20 frames), +2.2 %, while T3 mIoU moves 8.796 → 8.845 over
  four times the history. Accumulating more of the same observations does not expand coverage.
* **H-inference.** Replace nearest-neighbour copying with a learned rule. *Contrary evidence:*
  B3 over B0 is +0.25 on SemanticKITTI with the CI containing zero (PAPER C), and §3 shows
  the copy already runs within 11 points of its source.
* **H-source.** The teacher's 56.2 % / 41.3 % correctness is the real ceiling on any
  evidence-propagating scheme, so the path runs through a better label source rather than a
  better propagator. Consistent with §2's reading 1 being *small* — fixing the teacher where
  it speaks is cheap precisely because it speaks so rarely — and with §3's bound.

These are hypotheses. §4's ranking is measured; this section's three-way split is not.

---

## 6. Statistical uncertainty

The partition and the oracles are **pooled counts over 801 / 4,219 anchors** with no
resampling, so no interval is attached to them. At these voxel counts (10⁷–10⁸) sampling
noise on a share is negligible; **anchor-level correlation is not**, and that is what a
bootstrap would quantify. The oracle gains in §2 should be read as point estimates on this
anchor set, not as population estimates with stated precision.

Two things do carry across-condition evidence instead. The ranking in §4 holds on **both
benchmarks and both geometry arms** — four independent decompositions. And the Waymo arm, a
genuinely different geometry, moves every term in the expected direction without changing the
order: `oracle_teacher` +1.91 / +2.51, `oracle_extrapolation` +13.78 / +7.73.

Figures quoted from earlier gates (Gate 18, Gate 21, PAPER C) carry whatever uncertainty
their own reports state; they are cited, not recomputed.

---

## 7. Confounding factors

* **Geometry seed 0 only.** The canonical prediction pipeline fixes `GEOMETRY_SEED = 0`;
  §2's oracles are therefore one geometry, not a seed mean. The frozen three-seed spread on
  SemanticKITTI SC IoU is ±0.32, small against the effects here, but the oracles were not
  re-run per seed.
* **`evidence` is reduced to the evaluation grid by an any-subvoxel rule.** On Occ3D one
  native 0.4 m voxel counts as having evidence if any of its eight 0.2 m subvoxels does, so
  Occ3D's coverage figures are the more generous of the two.
* **GT-occupied is the OccAny valid mask**, not the official SemanticKITTI invalid mask used
  by the frozen geometry evaluator. §1's recall is therefore not the number in
  `waymo_vs_kitti360_report.md` §2, and should not be compared with it.
* **Teacher correctness here is measured against the benchmark's own label set** after the
  frozen 18-phrase readout and class mapping. A readout or mapping error is charged to the
  teacher by this decomposition. The Stage A audit's 63.9 % and Gate 18's 62.6 % sit above
  §1's 56.2 % — different populations, different datasets, and the brief's instruction not
  to generalise those numbers applies in this direction too.
* **Superclass mapping is OccAny's**, chosen so the ambiguity split is not ours to tune.

---

## 8. Unsupported interpretations — explicitly not claimed

* **Not claimed:** that improving the 2D teacher is worthless. It is worth under 2 mIoU
  *within the current evidence footprint*. A teacher with broader coverage is a different
  intervention and is not measured here.
* **Not claimed:** that `oracle_extrapolation` = +17.50 is achievable. It uses GT labels.
  §3 bounds what a teacher-fed extrapolator can reach, and that bound is far below it.
* **Not claimed:** that the ~⅓ intra-superclass errors *are* genuine ambiguity. Superclass
  membership makes a confusion a candidate; it does not establish that the voxel was
  ambiguous.
* **Not claimed:** causal attribution between coverage and inference. §5 says why the data
  here cannot separate them.
* **Not claimed:** that these proportions transfer to other datasets, sensors or teachers.

---

## 9. Decisions and next actions

**Q3 answered: the dominant bottleneck is geometric completion**, and among the semantic
terms it is unobserved-region inference coupled to evidence coverage — not teacher accuracy.

Consequences for the project:

1. **Stop spending on teacher quality** as a route to mIoU. Measured ceiling under 2 points.
2. **Semantic mIoU is gated by occupancy** in a way the paper must state: perfect semantics
   on today's geometry gives 23.88 mIoU on SemanticKITTI. Any semantic method evaluated on
   SSC mIoU is working inside that envelope, and comparisons that do not hold occupancy fixed
   are reporting occupancy.
3. **Evidence coverage cannot be bought with more history** (Gate 18). Expanding it needs a
   different observation source or a different lifting rule, not a longer clip.
4. **The next semantic experiment must be scored on a fixed occupancy mask**, as Phase 4's
   `shared_correct` population does, or its result will be an occupancy result.

Whether privileged 3D semantic supervision can lift the coverage ceiling is taken up in
`reports/privileged_semantic_supervision_feasibility.md`.
