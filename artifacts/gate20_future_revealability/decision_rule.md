# Gate 20 — pre-registered decision rule

Written and hashed **before** the full run. Thresholds are not moved after the results are seen.

Commit at pre-registration: `598cd228620f1d4f5db9010aedca0392c926edb5`
Date: 2026-10-01

## 0. What is being decided

Whether ordinary future driving observation provides enough high-quality semantic supervision for the
geometry our causal system correctly completes but has never observed. A pass licenses building the
smallest possible semantic-completion learner on that supervision; a fail stops the branch.

## 1. Population (fixed before the run)

For an anchor `t`, in the frozen SSCBench-KITTI-360 velodyne grid of `t` ((256, 256, 32), 0.2 m,
origin (0, −25.6, −2) m):

```
P0 = sem_gt in 1..18                       (GT-occupied with a class)
   & O                                     (frozen CompletionUNet, geometry seed 0, occupied)
   & ~E                                    (no Trident-H evidence voxel anywhere in t−4..t)
```

`P0` is **not a new mask**: it is exactly the frozen region `completed_never_observed`
(`tools/delayed_semantic/regions.py`), the primary region of the pilot, Gate 16, Gate 17 and Gate 18.

```
P1         = P0 & (B0 nearest-evidence class != sem_gt)
P0_correct = P0 & (B0 nearest-evidence class == sem_gt)
```

Primary drives: **0003, 0007, 0010** (the training drives admitted by the frozen pipeline). Drive **0006**
is reported separately and is *never* used to set or move a threshold.

## 2. Reveal levels (fixed before the run)

Three nested levels. Each is evaluated per horizon Δ with the voxel's own first qualifying future
observation time, so the Δ-curve is monotone by construction.

| level | rule | role |
|---|---|---|
| `geometric` | ≥1 future LiDAR return falls in the voxel within Δ | loose diagnostic — the ceiling |
| **`visible`** | that return is **camera-visible** (z ≥ 1 m, inside 1408×376, passes the frozen 4 px z-buffer test with tol max(0.5 m, 0.05 z)) **and** carries a valid bilinear Trident-H token | **DECISION** |
| `deployable` | `visible` **and** the voxel is selected by the frozen delayed-target rule (tier A/B occupancy + ≥1 s static span + no cross-sweep conflict) | what a label-free selector would actually keep |

**`visible` is the decision level.** It is the evidence rule the existing semantic lifting pipeline already
uses, applied to a future frame and registered into the grid of `t` with GT poses — conditions 1–5 of the
brief's section 3, and nothing looser. `geometric` is reported only to show which filter costs what;
`deployable` adds the frozen occupancy-persistence selection, which exists to establish *occupancy at t*
without labels — a job GT already does inside `P0`, so it is reported but does not decide.

Pose registration, the z-buffer and the token lookup are reused verbatim from
`tools/delayed_semantic/targets.py`; this gate adds no new coordinate-transform or lifting code.

## 3. Horizons (fixed before the run)

Δ ∈ {0, 1, 2, 4, 6, 10} s, filtered on real KITTI-360 image timestamps, not frame counts. The privileged
window is the frozen one (`t+1..t+20`, ≤ 10 s), so **"eventual" ≡ Δ = 10 s** — there is no unbounded future
in this audit. At Δ = 0 revealability must be exactly 0.

## 4. Future-target quality

On revealed `P0` voxels with a GT label, using only views inside the horizon:
top-1 accuracy, mIoU over classes 1..18 with GT support (`semantic_metrics.summarize(semantic_only=True)`,
the frozen definition — class 0 excluded), thing mIoU, stuff mIoU. Readouts: `first`, `latest`,
`mean_probability`; **`mean_probability` decides** (it is the frozen delayed-target readout).

Reference, read-only: Gate 18's current-evidence teacher = 62.57 top-1 / 16.91 mIoU.

## 5. Thresholds

### Outcome A — strong support. **All** of:

| id | condition |
|---|---|
| A1 | eventual `P0` revealability ≥ **50 %** |
| A2 | eventual `P1` revealability ≥ **40 %** |
| A3 | eventual `P0` revealability pooled over the 4–8 m and >8 m distance bins ≥ **25 %** |
| A4 | future-target top-1 ≥ **55 %** and mIoU ≥ **12.0** on revealed `P0` |
| A5 | ≥ **12 of 18** classes have eventual `P0` revealability ≥ 20 % |

### Outcome C — fail. **Any** of:

| id | condition |
|---|---|
| C1 | eventual `P0` revealability < **20 %** |
| C2 | eventual `P1` revealability < **15 %** |
| C3 | eventual `P0` revealability pooled over 4–8 m and >8 m < **10 %** |
| C4 | future-target top-1 < **40 %** or mIoU < **8.0** on revealed `P0` |

### Outcome B — anything else.

All thresholds are read on the **micro** aggregate (total revealed / total target voxels) over the
**training drives**, at the **`visible`** level, with the **`mean_probability`** readout. Macro
(mean per-anchor) is reported alongside and does not decide. 0006 does not decide.

### Why these numbers

- **A1/C1 (50 % / 20 %).** A learner trained on pseudo-labels covering under a fifth of the population it
  must predict is supervised on a biased minority; the unsupervised majority is exactly today's failure
  case, so under 20 % cannot be the mechanism of a paper. Above half, the supervision is representative
  enough that the remainder can plausibly be reached by generalisation.
- **A2/C2 (40 % / 15 %).** `P1` is the only population where future supervision could teach something B0
  does not already produce. It is set below `P0` because `P1` voxels are expected to be harder and further
  away, and demanding parity would be rigged against the hypothesis.
- **A3/C3 (25 % / 10 %).** Gate 18 established that propagation collapses with distance from evidence. If
  future supervision only covers the near-evidence shell, it supervises the part B0 already handles. 10 %
  of the far population is the floor at which the idea stops addressing the actual problem.
- **A4/C4.** A pseudo-label must beat what B0 already achieves on this population (8.80 mIoU, pilot) to be
  worth training on; 8.0 mIoU is therefore the floor, and 12.0 — about 70 % of the teacher's own 16.91 on
  what it directly sees — is the level at which the target is close to as good as this teacher ever gets.
  40 % top-1 is chance-dominated for 18 classes with vegetation at ~50 % prior; 55 % is within 8 pp of the
  teacher's direct-evidence 62.57 %.
- **A5 (12/18 classes).** Guards against the expected failure mode: plentiful road/building/vegetation
  reveals and no cars, poles, signs or terrain. Two thirds of the vocabulary is the bar for "not degenerate".

## 6. Statistics

Block bootstrap over anchors: 20-anchor temporal blocks, 2,000 resamples, seed 0, counts summed within a
resample and the ratio recomputed from the sums (never averaging per-anchor ratios). Reported as 95 %
percentile intervals. A threshold is judged on the point estimate; the interval is reported so a borderline
call is visible as borderline.

## 7. Deliberately not consulted before this file was hashed

- The size and composition of the frozen pilot region `completed_later_revealed`, which measures a closely
  related quantity on 0006. It is read **after** the run, as a reproduction check (validation #8).
- Any Gate 20 result.

## 8. What each outcome licenses

- **A** → build the smallest possible semantic-completion learner on this supervision.
- **B** → proceed only if a region/object formulation can turn partial reveals into useful supervision;
  the region analysis of section 9 is the evidence for that, not a new architecture.
- **C** → stop this research branch. Recommend a different source of semantic priors. Do not build the
  completion model.
