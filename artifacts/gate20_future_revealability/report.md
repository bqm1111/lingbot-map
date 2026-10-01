# OUTCOME B — future revealability is useful but incomplete

Training drives 0003/0007/0010, 1276 anchors, **8,379,519** P0 voxels. Of the geometry the frozen causal system correctly completes but has never observed, **27.2%** becomes genuinely observable again within the 10 s privileged window, **34.5%** of the subset B0 currently gets wrong, and **46.4%** of the population more than 4 m from any causal semantic evidence. The future teacher on the voxels it does reach scores **60.45%** top-1 / **19.01** mIoU, against the frozen current-evidence teacher's 62.57% / 16.91. Median time to first reveal **0.53 s**, 90th percentile **1.58 s**; **72.8%** is never revealed.

## 1. Headline

| | P0 | P1 | P0 that B0 already gets right |
|---|---:|---:|---:|
| geometric — revealed by 10 s | 89.73 | 88.12 | 90.93 |
| in_frustum — revealed by 10 s | 29.77 | 37.23 | 24.15 |
| visible — revealed by 10 s | 27.21 | 34.48 | 21.74 |
| deployable — revealed by 10 s | 14.39 | 14.41 | 14.37 |
| population (voxels) | 8,379,519 | 3,599,060 | 4,780,459 |

Decision level is **`visible`** and decision readout **`mean_probability`** (`decision_rule.md` §2, §4), on the micro aggregate over the training drives.

## 2. Pre-registered decision rule

Hashed before the run: `086812519dae2fedc737976c398b4a5bc370e0db83f948490d09841984c0df9b`

| id | condition | value | met |
|---|---|---:|---|
| A1 | P0 eventual ≥ 50% | 27.21 | no |
| A2 | P1 eventual ≥ 40% | 34.48 | no |
| A3 | >4 m eventual ≥ 25% | 46.38 | yes |
| A4 | top-1 ≥ 55% and mIoU ≥ 12.0 | 60.45 / 19.01 | yes |
| A5 | ≥ 12/18 classes ≥ 20% | 15 | yes |
| C1 | P0 eventual < 20% | 27.21 | no |
| C2 | P1 eventual < 15% | 34.48 | no |
| C3 | >4 m eventual < 10% | 46.38 | no |
| C4 | top-1 < 40% or mIoU < 8.0 | 60.45 / 19.01 | no |

**Outcome B.** No threshold was moved after the results were seen.

## 3. Population

```
P0 = sem_gt in 1..18  &  frozen CompletionUNet occupied  &  no Trident-H evidence in t-4..t
P1 = P0 & (frozen B0 nearest-evidence class != sem_gt)
```
P0 is bit-identical to the frozen region `completed_never_observed` — the primary region of the pilot, Gate 16, Gate 17 and Gate 18 — asserted on every anchor.

| | anchors | P0 | P1 | P1 share of P0 | B0 mIoU on P0 |
|---|---:|---:|---:|---:|---:|
| training drives | 1276 | 8,379,519 | 3,599,060 | 43.0% | 11.31 |
| 0006 (held out) | 590 | 2,743,212 | 1,481,145 | 54.0% | 8.80 |

## 4. Horizon curve

Micro = total revealed / total target voxels; macro = mean of the per-anchor ratios. 95 % intervals are a 20-anchor block bootstrap, 2,000 resamples, seed 0.

**P0** (N_target = 8,379,519)

| Δ (s) | geometric micro | in_frustum micro | visible micro | deployable micro | visible macro | visible 95 % CI | visible N_revealed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | [0.00, 0.00] | 0 |
| 1 | 47.50 | 18.10 | 16.09 | 8.37 | 16.03 | [15.56, 16.57] | 1,348,560 |
| 2 | 75.76 | 27.17 | 24.62 | 12.94 | 24.47 | [23.87, 25.35] | 2,063,321 |
| 4 | 87.05 | 29.41 | 26.86 | 14.18 | 27.06 | [25.89, 27.85] | 2,250,540 |
| 6 | 88.86 | 29.60 | 27.05 | 14.30 | 27.28 | [26.05, 28.12] | 2,266,875 |
| 10 | 89.73 | 29.77 | 27.21 | 14.39 | 27.43 | [26.21, 28.31] | 2,280,255 |

**P1** (N_target = 3,599,060)

| Δ (s) | geometric micro | in_frustum micro | visible micro | deployable micro | visible macro | visible 95 % CI | visible N_revealed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | [0.00, 0.00] | 0 |
| 1 | 45.65 | 21.38 | 19.13 | 7.27 | 19.61 | [18.28, 20.04] | 688,540 |
| 2 | 71.44 | 32.73 | 29.94 | 12.07 | 30.19 | [28.89, 31.11] | 1,077,572 |
| 4 | 84.65 | 36.63 | 33.87 | 14.07 | 33.78 | [32.43, 35.38] | 1,218,956 |
| 6 | 87.13 | 37.00 | 34.24 | 14.30 | 34.10 | [32.71, 35.84] | 1,232,295 |
| 10 | 88.12 | 37.23 | 34.48 | 14.41 | 34.36 | [32.90, 36.15] | 1,240,929 |

![A](figures/A_horizon_curve.png)

### Which filter costs the coverage

| level | revealed by 10 s | lost at this step |
|---|---:|---:|
| a future LiDAR sweep puts a return in the voxel | 89.73% | — |
| ... and that return lands inside a future image | 29.77% | **−59.96 pp** (camera field of view) |
| ... and it is the nearest surface on its ray (z-buffer) | 27.21% | −2.56 pp (occlusion) |
| ... and the label-free target selector keeps it | 14.39% | −12.83 pp (occupancy/persistence rules) |

The drive re-measures almost all of this geometry — **89.7%** of P0 gets a future LiDAR return. The supervision is lost at the **camera**: 60.0 percentage points disappear because the return falls outside the 1408x376 perspective frustum, which is 2.6 pp more than occlusion costs. This is a property of the sensor rig, not of the trajectory or of the scene.

## 5. P0 vs P1

| | P0 | P1 | P0 B0-correct |
|---|---:|---:|---:|
| geometric eventual | 89.73 | 88.12 | 90.93 |
| in_frustum eventual | 29.77 | 37.23 | 24.15 |
| visible eventual | 27.21 | 34.48 | 21.74 |
| deployable eventual | 14.39 | 14.41 | 14.37 |
| median reveal time (s) | 0.53 | 0.53 | 0.53 |
| p90 reveal time (s) | 1.58 | 2.08 | 1.58 |
| never revealed (%) | 72.79 | 65.52 | 78.26 |

## 6. Distance to causal semantic evidence

Same distance definition as Gate 18 and the pilot (Euclidean distance transform to the nearest evidence voxel centre).

| distance | P0 voxels | geometric | in frustum | **visible** | deployable | P1 visible | median reveal (s) | p90 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-1 m | 3,104,972 | 87.85 | 18.25 | 17.31 | 10.21 | 18.40 | 0.53 | 1.02 |
| 1-2 m | 1,865,647 | 89.49 | 26.38 | 24.54 | 14.16 | 30.65 | 0.53 | 1.02 |
| 2-4 m | 1,873,879 | 90.66 | 33.88 | 30.58 | 16.16 | 40.21 | 0.53 | 1.58 |
| 4-8 m | 810,313 | 91.35 | 49.32 | 44.73 | 18.69 | 53.93 | 1.02 | 2.08 |
| >8 m | 724,708 | 94.12 | 55.35 | 48.22 | 23.44 | 55.43 | 1.58 | 3.12 |

- pooled **> 4 m** (4–8 m and > 8 m, 1,535,021 voxels): **46.38%** [43.22, 49.34]
- pooled **> 8 m** (724,708 voxels): **48.22%** [43.79, 52.90]

![B](figures/B_distance.png)

## 7. Class, thing and stuff

GT classes are used here for **auditing only**; no supervision in this gate is built from them.

| class | | P0 voxels | revealed | revealability | median reveal (s) | P1 revealability |
|---|---|---:|---:|---:|---:|---:|
| vegetation | stuff | 4,277,449 | 941,870 | 22.02 | 0.53 | 17.23 |
| road | stuff | 1,405,345 | 542,098 | 38.57 | 0.53 | 50.59 |
| terrain | stuff | 1,140,833 | 347,614 | 30.47 | 0.53 | 30.57 |
| building | stuff | 489,592 | 113,994 | 23.28 | 0.53 | 8.64 |
| sidewalk | stuff | 425,491 | 143,448 | 33.71 | 0.53 | 35.05 |
| other-ground | stuff | 192,038 | 53,160 | 27.68 | 0.53 | 28.62 |
| parking | stuff | 125,620 | 36,919 | 29.39 | 1.02 | 31.16 |
| other-structure | stuff | 101,308 | 30,828 | 30.43 | 0.53 | 29.76 |
| car | thing | 99,391 | 32,257 | 32.45 | 0.53 | 38.26 |
| other-vehicle | thing | 49,859 | 14,795 | 29.67 | 0.53 | 28.76 |
| fence | stuff | 45,237 | 16,821 | 37.18 | 0.53 | 39.21 |
| other-object | stuff | 15,353 | 4,087 | 26.62 | 0.53 | 26.76 |
| truck | thing | 6,479 | 1,168 | 18.03 | 0.53 | 10.43 |
| pole | stuff | 2,659 | 705 | 26.51 | 0.53 | 27.21 |
| bicycle | thing | 1,635 | 275 | 16.82 | 0.53 | 20.16 |
| motorcycle | thing | 593 | 67 | 11.30 | 0.53 | 11.71 |
| person | thing | 516 | 122 | 23.64 | 0.53 | 23.33 |
| traffic-sign | stuff | 121 | 27 | 22.31 | 0.53 | 22.64 |

- **thing** (158,473 voxels): **30.72%**
- **stuff** (8,221,046 voxels): **27.14%**
- classes with ≥ 20 % revealability: **15 of 18** with GT support

![C](figures/C_per_class.png)

## 8. Quality of the future semantic target

On revealed P0 voxels with a GT label, using only the views inside the horizon. mIoU is the frozen definition (classes 1..18 with GT support; class 0 excluded).

| readout | voxels | top-1 | mIoU | thing mIoU | stuff mIoU |
|---|---:|---:|---:|---:|---:|
| first | 2,280,255 | 60.10 | 18.42 | 14.47 | 20.40 |
| latest | 2,280,255 | 59.12 | 18.49 | 14.83 | 20.33 |
| mean_probability **(decision)** | 2,280,255 | 60.45 | 19.01 | 15.29 | 20.87 |
| *Gate 18 current-evidence teacher* | — | *62.57* | *16.91* | *32.58* | *17.83* |
| *frozen B0 on all of P0* | 8,379,519 | 57.05 | 11.31 | — | — |

**The comparison that matters: B0 and the future target on exactly the same voxels.**

| on the 2,280,255 revealed P0 voxels | top-1 | mIoU | thing mIoU | stuff mIoU |
|---|---:|---:|---:|---:|
| frozen B0 (what we produce today) | 45.58 | 8.45 | 2.51 | 11.42 |
| future pseudo-target | 60.45 | 19.01 | 15.29 | 20.87 |
| difference | +14.87 | **+10.56** | +12.78 | +9.45 |

B0 scores **12.63** mIoU on the part that is *never* revealed and only **8.45** on the revealed part — the supervision is concentrated exactly where the current method is weakest.

**Moving surfaces.** The `visible` level deliberately drops the frozen selector's static-persistence test, so a voxel can be "revealed" by a surface that moved into it after `t`. Restricting the same target to the voxels that selector keeps is the check:

| on the deployable-selected voxels | voxels | top-1 | mIoU | thing mIoU | stuff mIoU |
|---|---:|---:|---:|---:|---:|
| frozen B0 | 1,205,540 | 56.99 | 10.11 | 3.85 | 13.24 |
| future pseudo-target | 1,205,540 | 61.57 | 19.11 | 16.62 | 20.36 |

**What that is worth over all of P0**, with no learning at all:

| prediction on all of P0 | top-1 | mIoU | Δ mIoU vs B0 |
|---|---:|---:|---:|
| frozen B0 everywhere | 57.05 | 11.31 | — |
| future target where revealed, B0 elsewhere | 61.10 | 14.20 | +2.89 |
| GT where revealed, B0 elsewhere (perfect fit, no generalisation) | 71.86 | 23.81 | +12.50 |

More future views of the same voxel (`mean_probability`, Δ = 10 s):

| future views | voxels | top-1 | mIoU |
|---|---:|---:|---:|
| 1 | 1,691,529 | 60.41 | 18.29 |
| 2 | 440,347 | 61.20 | 19.73 |
| 3-5 | 141,103 | 58.49 | 22.62 |
| >5 | 7,276 | 61.50 | 22.29 |

On **P1** only — the voxels B0 gets wrong — the future target scores **46.56%** top-1 / **14.43** mIoU (1,240,929 voxels).

Across horizons (decision readout):

| Δ (s) | 0 | 1 | 2 | 4 | 6 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| top-1 | 0.00 | 62.04 | 60.67 | 60.39 | 60.40 | 60.45 |
| mIoU | — | 19.63 | 19.25 | 19.01 | 19.00 | 19.01 |
| voxels | 0 | 1,348,560 | 2,063,321 | 2,250,540 | 2,266,875 | 2,280,255 |

## 9. Region analysis

GT connected components (per-class, 6-connectivity — Gate 17's definition, reused unchanged) with at least 8 hidden P0 voxels: **102,702** (3,902 thing, 98,800 stuff).

| share of the component's hidden support revealed by 10 s | all | thing | stuff |
|---|---:|---:|---:|
| ≥ 10% | 49.07 | 50.56 | 49.02 |
| ≥ 25% | 43.46 | 46.10 | 43.36 |
| ≥ 50% | 33.32 | 35.75 | 33.23 |
| ≥ 75% | 20.09 | 19.25 | 20.13 |

This is the weakest-assumption version of the region question: a component whose hidden support is mostly revealed could be supervised as an object even if individual voxels are missed. It is a connected-component analysis, not instance annotation — KITTI-360 instance labels are not available on this machine (see Gate 17's `plan.md`), so adjacent same-class objects merge, which makes these numbers an upper bound.

![E](figures/E_region_coverage.png)

## 10. Time to reveal

| | median (s) | p90 (s) | never revealed (%) |
|---|---:|---:|---:|
| P0 | 0.53 | 1.58 | 72.79 |
| P1 | 0.53 | 2.08 | 65.52 |
| P0_correct | 0.53 | 1.58 | 78.26 |

![D](figures/D_reveal_time.png)

## 11. 0006, held out from every threshold

| | P0 | geometric | in frustum | **visible** | deployable | future top-1 | future mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| training drives | 8,379,519 | 89.73 | 29.77 | 27.21 | 14.39 | 60.45 | 19.01 |
| 0006 | 2,743,212 | 92.15 | 34.49 | 31.94 | 20.38 | 53.33 | 16.88 |

Per drive:

| drive | anchors | P0 | visible eventual | deployable eventual | future mIoU |
|---|---:|---:|---:|---:|---:|
| 0003 | 193 | 1,076,037 | 24.98 | 12.83 | 15.04 |
| 0007 | 529 | 4,405,165 | 25.73 | 13.81 | 14.46 |
| 0010 | 554 | 2,898,317 | 30.29 | 15.84 | 19.19 |
| 0006 (dev) | 590 | 2,743,212 | 31.94 | 20.38 | 16.88 |

## 12. Leakage and validation

| # | check | result |
|---|---|---|
| 1 | horizon 0 revealability is exactly zero, every anchor, every level | pass |
| 2 | P0 count on 0006 equals the frozen `completed_never_observed` (2,743,212) | pass |
| 2b | B0 on P0 reproduces the pilot's 8.80 mIoU on 0006 | pass |
| 3 | P0 ∩ evidence = ∅ (asserted per anchor) | pass |
| 4 | the `visible` level reproduces the frozen delayed-target cache's first-view times, view counts and readout | pass |
| 5 | the causal-state digest is unchanged after the future is opened | pass |
| 6 | a rebuild is bit-identical | pass |
| 7 | grid dims/origin/voxel size, floor binning round-trip, pose identity | pass |
| 8 | frozen `completed_later_revealed` size (559,021) and bit-exact 0006 GT masks | pass |
| levels | geometric ⊇ in_frustum ⊇ visible ⊇ deployable on every voxel | pass |

All checks: **pass**. Full record: `logs/validation.json`.

The future is opened strictly after the causal state is computed and digested, and the digest is re-checked afterwards; the causal state itself is computed inside a file audit that forbids LiDAR, GT poses, delayed targets and human labels. Future information can therefore only answer *when* a hidden voxel becomes observable and *what* it would then say.

## 13. Interpretation

**The supervision exists, it is good, and it covers about a quarter of the problem.** Five findings, in
the order that matters.

**1. The drive re-measures this geometry; the camera does not see it.** 89.7% of P0 gets a
future LiDAR return — the trajectory really does carry the vehicle past almost all of the space it
completed blind. But only 29.8% of those returns land inside a future image, and
27.2% survive the occlusion test with a usable Trident-H token. The field of view costs
60.0 percentage points — far more than occlusion's
2.6 pp. The ceiling on this idea is set by the sensor rig, not by the scene
or the route.

**2. Reveals are immediate or never.** Median time to first reveal is **0.53 s** — the very
next stream frame — and the 90th percentile is **1.58 s**. Horizons beyond 2 s add almost
nothing, and **72.8%** of P0 is never revealed inside the 10 s window. This is the mirror image
of Gate 18's finding that the past leaves the forward volume: the future is in front of you, but the window
in which a hidden voxel is simultaneously ahead of the vehicle and inside the frustum is about a second and
a half wide. Extending the horizon is not the lever; widening the field of view would be.

**3. The coverage is best exactly where the current method is worst.** Revealability *rises* with distance
from causal semantic evidence — 17.3% at 0–1 m to 48.2% beyond 8 m — and P1, the
subset B0 gets wrong, is better covered (34.5%) than P0 as a whole (27.2%). This is
the opposite of the failure mode the gate was designed to catch. Gate 18 showed propagation collapses with
distance; future observation is densest precisely there, because voxels far from any evidence are mostly
voxels *ahead* of the vehicle, beyond the reconstruction's depth range but inside the grid.

**4. No class collapse.** Thing 30.7% against stuff 27.1%, and
**15 of 18** classes with GT support clear 20% revealability. The expected bias — plentiful
road, building and vegetation, no cars, poles or signs — does not appear.

**5. Where the target exists it is much better than what we produce today.** On the same
2,280,255 revealed voxels the future pseudo-target scores **19.01 mIoU /
60.45% top-1** against B0's **8.45 / 45.58%** — a gap of
**+10.56 mIoU**. On revealed P1, where B0 is wrong by construction, the future
target is right 46.6% of the time. Substituting it wherever it exists, with no learning at
all, lifts mIoU over *all* of P0 from 11.31 to **14.20** (+2.89);
a learner that fit this supervision perfectly and generalised nothing would reach 23.81
(+12.50). For scale, every learned semantic mechanism tried so far moved the same
metric by less than half a point (B1 +0.33, B2 +0.41, B4 −0.58, B5 −2.04).

**One number that looks bad and is not.** The future target's thing mIoU is 15.29, well under
the current-evidence teacher's 32.58. Two things rule out the obvious worry. The
static-persistence filter — the frozen selector's defence against a surface that moved into the voxel after
`t` — barely changes it (16.62), so moving objects are not the explanation. And B0 on the same
voxels scores 2.51, so the future target is still several times better on objects than what we
produce today. The gap to the teacher is a population effect: Gate 18 measured the teacher on surfaces it was
looking straight at, while these are object voxels that were *hidden* at `t` — boundaries, far cars, partly
occluded shapes — and things are only 1.9% of P0 to begin with.

**Why this is B and not A.** The pre-registered coverage bars were 50% of P0 and 40% of P1; the measured
27.2% and 34.5% miss them. 72.8% of the population this mechanism is
supposed to solve receives no future view at all, so future observation cannot be the *sole* source of
semantic supervision — a model trained on it is supervised on a minority of its own target population, and
that minority is systematically the far, ahead-of-vehicle part rather than the occluded part. The decision
rule therefore licenses proceeding only if partial reveals can be turned into useful supervision, which is
what the region analysis speaks to: **43%** of the 102,702 hidden GT components get at
least a quarter of their hidden support revealed and **33%** get at least half, so
region-level consistency is available as an auxiliary signal rather than a replacement.

**One caveat stated plainly.** Future evidence here is lifted from raw LiDAR returns registered with GT
poses, while causal evidence is lifted from the model's own predicted geometry. The future side therefore
has better geometry than the deployed system ever would, which makes every number in this report an upper
bound on what a real pipeline could harvest. The direction of that bias is the safe one for a falsification
test — the hypothesis survived a test tilted in its favour, and the honest coverage figure is somewhere at
or below 27.2%.

## 14. Recommended next experiment

**Train the smallest possible semantic-completion head on future-revealed pseudo-labels only, and test
whether it generalises to the never-revealed part of the same population.**

That last clause is the experiment. Coverage of 27.2% is enough to train on and not enough to
deploy on, so the only question that decides the paper is whether supervision drawn from the revealed
27.2% transfers to the 72.8% that never becomes observable. Nothing measured in
this gate answers it, and no further audit can.

Concretely, and with no new architecture: reuse the frozen `CompletionUNet` trunk and a small semantic head
(the Gate 15 head is the obvious candidate), supervise it **only** on training-drive P0 voxels that this
gate marks as future-revealed, using the `mean_probability` target, and evaluate on 0006's P0 split into its
revealed and never-revealed halves — reporting B0 on each half as the control. Pre-register the
never-revealed half as the gate before training, because the revealed half is where the supervision lives
and a gain there proves only that the model can fit its own targets. The headroom is bounded above by the
23.81 mIoU figure in section 8 and below by 14.20 (pure substitution, no learning), so a
result inside that band is a real mechanism and a result at or under B0's 11.31 on the
never-revealed half closes the branch.

Two things worth recording but **not** worth doing now. The field-of-view finding in section 4 implies a
large, cheap coverage lever — KITTI-360 carries a second perspective camera and two fisheyes, and the
60.0 pp lost to the frustum is partly recoverable from them. That is data
engineering, not the paper's mechanism, and it should only be spent after the generalisation question is
answered. Likewise, the region-level signal (33% of hidden components at least half revealed)
is a plausible second loss term, but adding it before the voxel-level question is settled would confound
the one test that matters.

---

Version `gate20.1` · commit `598cd22` · `decision_rule.md` hashed before the run · 1866 anchors · no training, no model change.
