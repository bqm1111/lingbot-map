# Accumulated reconstruction vs genuine completion

**Measured 2026-10-20.** Raw: `artifacts/method_vs_data/results/observed_strict.json`
(and `..._pooled.json`). Code: `tools/method_vs_data/observed_strict.py`. 801 shared
SemanticKITTI-08 anchors, 33 blocks, Track O, OccAny's own stored labels and placement.

Brief §9. This is the decomposition the method claim rests on, and it **revises** the earlier
reading of the anchor-frame split.

> ## Headline
>
> **Our advantage over OccAny sits in geometry the permitted input never observed**, not in
> geometry we remembered from earlier frames. Voxels only an earlier frame saw are just
> **6.18 %** of the GT surface; the 89.06 % no permitted frame ever saw carries the lead —
> **+8.54** SC IoU raw [+7.01, +10.06], **+4.77** pooled [+3.25, +6.23].
>
> Under OccAny's own published (pooled) convention the remembered-space advantage **vanishes**
> (+0.65, CI [−0.33, +1.74]), leaving never-observed completion as the *only* region with a
> significant positive advantage. §8.
>
> The earlier conclusion — *"we beat it by remembering, not by completing"* — was **too strong
> and is withdrawn.** §5 explains why the anchor-frame split looked that way.

---

## 1. Populations, and why they are legitimate for both methods

Per brief §9, over every valid voxel:

| region | definition |
|---|---|
| **anchor_observed** | a GT raycast from the **anchor** camera reaches it, up to and including the first GT-occupied voxel |
| **earlier_only** | not anchor-observed, but reached the same way from one of the **four earlier** permitted frames — accumulated reconstruction |
| **never_observed** | reached from **none** of the five permitted frames — only here is a correct prediction completion of unobserved space |

Built from **ground-truth occupancy and ground-truth poses** of the five permitted frames, not
from either method's reconstruction or pose estimate. So the partition is a property of the
*permitted input* and is a common denominator both methods are scored against, as the brief
requires.

**Pose chain, gated per sample.** Grid = the anchor's velodyne frame, so
`T_cam(j)→grid = inv(M_anchor) @ M_j @ cam_to_velo` with `M_n = P_cam2_c2w(n) @ inv(cam_to_velo)`.
For the anchor this must collapse to `cam_to_velo` exactly and must equal the transform stored
in the canonical prediction; both are asserted on every one of the 801 samples.

### Validation against the existing evaluator

On the `all` population the new tool reproduces `tools/observability_split` **exactly**:

| | strict tool | existing | diff |
|---|---:|---:|---:|
| ours, raw | 26.396948 | 26.396948 | 0 |
| OccAny 1-frame, raw | 20.319705 | 20.319705 | 0 |
| delta | +6.0772 | +6.0772 | 0 |
| 95 % CI | [+4.8150, +7.3325] | [+4.8150, +7.3325] | 0 |

Bit-identical, bootstrap included. The partition is new; the scoring is not.

---

## 2. Composition — how little the five frames see

| region | share of valid | share of GT surface | GT-occupied voxels |
|---|---:|---:|---:|
| anchor_observed | 32.69 % | **4.76 %** | 4,440,970 |
| earlier_only | 24.81 % | **6.18 %** | 5,766,773 |
| never_observed | 42.50 % | **89.06 %** | 83,090,979 |
| all | 100 % | 100 % | 93,298,722 |

GT-occupied surface each permitted frame sees on its own:

| frame | t−4 | t−3 | t−2 | t−1 | **t (anchor)** |
|---|---:|---:|---:|---:|---:|
| share of GT surface | 4.59 % | 4.88 % | 5.16 % | 5.11 % | **4.76 %** |

**The five-frame window barely expands coverage.** One frame sees ~4.8 % of the GT surface;
all five together see 10.94 % (4.76 + 6.18). The clip spans ~2 s at 3–5 m/s, so the frames
look at almost the same scene. **89.06 % of the SSC ground-truth surface is never observed by
any permitted frame** — and that is the population that decides the benchmark.

---

## 3. Results — raw convention

Δ = ours (causal, 5 frames, seed 0) − OccAny 1-frame causal. Paired block bootstrap, 25
contiguous anchors per block, 2,000 resamples, seed 0. `*` = CI excludes zero.

| region | ours IoU | P | R | OccAny IoU | P | R | Δ | 95 % CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| all | 26.40 | 37.45 | 47.22 | 20.32 | 55.64 | 24.25 | **+6.08** | [+4.82, +7.33]* |
| anchor_observed | 11.47 | 12.74 | 53.48 | 15.76 | 19.95 | 42.88 | **−4.29** | [−5.08, −3.60]* |
| earlier_only | 17.95 | 20.57 | 58.50 | 12.97 | 35.25 | 17.02 | **+4.98** | [+3.57, +6.64]* |
| never_observed | 30.07 | 46.37 | 46.10 | 21.53 | 69.66 | 23.75 | **+8.54** | [+7.01, +10.06]* |

Three findings, all with intervals clear of zero:

1. **We lose where the anchor camera looks** (−4.29). OccAny's single-view, 2D-grounded
   geometry is better on directly visible surfaces. Our precision there is 12.74 against their
   19.95.
2. **We win on remembered geometry** (+4.98) — accumulation does help, and their 1-frame run
   structurally cannot see these voxels (recall 17.02 against our 58.50).
3. **We win most on never-observed geometry** (+8.54), and because that region holds 89.06 %
   of the surface, it is what produces the aggregate +6.08.

Across every region the pattern is the same: **they are more precise, we recall far more.**

---

## 4. Which of the three mechanisms explains the advantage

Brief §9 asks whether the advantage is (1) temporal accumulation, (2) geometric completion, or
(3) both. Two independent measurements agree:

**The map-only ablation.** The accumulated five-frame hard map scored on its own:

| benchmark | hard map alone | + CompletionUNet | completion adds | OccAny best released |
|---|---:|---:|---:|---:|
| SemanticKITTI | **8.79** | 26.32 | **+17.53** (67 %) | 25.21 |
| Occ3D | **9.82** | 31.63 | **+21.81** (69 %) | 20.66 |

Accumulation alone is **16.4 points below** OccAny on SemanticKITTI. It cannot be what wins.

**The strict split.** The region our lead lives in is the one no permitted frame observed, and
the region accumulation uniquely provides (`earlier_only`) is only 6.18 % of the surface.

**Answer: (2), learned geometric completion**, with accumulation a small and
convention-dependent contributor. Its contribution is bounded three ways — it supplies the
8.79 floor; it uniquely supplies only 6.18 % of the surface; and the advantage it yields there
is +4.98 raw but **+0.65 with a CI containing zero** under OccAny's own pooled convention
(§8).

---

## 5. Why the earlier anchor-frame split read differently, and the reconciliation

The 2026-10-16 split reported our lead as "entirely out-of-frustum" (+16.16 pooled on
out-of-view, −2.70 observed), which was read as *winning by remembering*. The two measurements
are arithmetically consistent:

| | share of GT surface |
|---|---:|
| anchor-frame: observed | 4.669 % |
| anchor-frame: occluded + out_of_view | 95.331 % |
| strict: anchor_observed | 4.760 % |
| strict: earlier_only + never_observed | 95.240 % |

The 0.091 pp difference is exactly the difference between the two anchor-observed definitions
(the existing one intersects with the frustum mask; the strict one does not, and rays may enter
a voxel whose centre projects marginally outside the image rectangle). Nothing else differs.

**The error was interpretive, not numerical.** "Out of frustum" was taken to imply "seen
earlier", because a voxel behind the car at *t* is naturally imagined as having been in front
of it at *t−4*. The strict split shows that is mostly false here: out-of-frustum voxels are
overwhelmingly **never observed**, because 2 s of forward motion does not bring the rear of the
scene into an earlier forward-facing view. The brief's warning — *"a voxel outside the anchor
camera frustum is not necessarily unobserved"* — cuts the other way too: it is usually
*un*observed, not remembered.

---

## 6. Statistical uncertainty

* Every Δ is a paired block bootstrap on the **difference**, 25-anchor blocks, 2,000 resamples,
  seed 0; both methods always resampled on the same anchors. All four intervals exclude zero.
* **Seed variation is a separate quantity and is not in these intervals.** Our strict-split
  numbers are **seed 0 only**, because the canonical prediction pipeline fixes
  `GEOMETRY_SEED = 0`. Our three-seed spread on SemanticKITTI SC IoU is ±0.30, small against
  the effects here but not included.
* OccAny is a single released checkpoint, so it has no seed spread at all.
* The map-only figures in §4 are pooled counts without resampling; `map_only` is bit-identical
  across two independently trained arms (8.820500920980095 on the 163-anchor frozen evaluator),
  which is the validity check available for it.

---

## 7. Confounding factors and limits

* **OccAny gets one frame, we get five.** The comparator is their only causal mode. The
  asymmetry favours us and is why §3's `earlier_only` row is partly structural — a 1-frame
  method cannot see those voxels at all. Their forward window is worth only +1.9 SC IoU to
  them, which bounds what a causal 5-frame OccAny would add. **The `never_observed` row is the
  one least affected by this**, since no frame of either method's input observed those voxels.
* **"Never observed" means not reached by a GT raycast** from the five poses at 0.1 m steps,
  pixel stride 2, out to 80 m. It does **not** mean the network had no information: a voxel
  behind a visible wall, or under the road, is unobserved yet strongly constrained by context.
  Correct predictions there are completion in the brief's sense — *predicted without direct
  observation* — not hallucination from nothing.
* **89 % is a property of SSC ground truth**, which includes extensive occluded and
  below-surface structure, not evidence that the task is impossible.
* **SemanticKITTI only.** The strict split needs GT poses for all five frames; the machinery
  exists for sequence 08. **Occ3D is not done**, so §3 is single-benchmark. The map-only
  ablation in §4 does cover both.
* **Raw convention in §3, pooled in §8, and they disagree on one row.** Pooling is strongly
  asymmetric — +3.74 SC IoU to OccAny against −0.05 to us — and it is what flips `earlier_only`
  from significant to not. Both conventions must be read; quoting only the raw table would
  overstate the accumulation contribution.
* **Different training corpora.** Nothing here attributes anything to architecture; see
  `matched_data_control_feasibility.md`.

---

## 8. Pooled convention — and it sharpens the conclusion

Same populations, same anchors, same bootstrap; geometry dilation applied to both methods.
**This is OccAny's own published convention** (their published 0.2591 matches our pooled
scorer to five digits), so it is arguably the primary table rather than a secondary one.

| region | ours IoU | P | R | OccAny IoU | P | R | Δ | 95 % CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| all | 26.34 | 33.08 | 56.39 | 24.06 | 46.08 | 33.49 | **+2.28** | [+1.23, +3.33]* |
| anchor_observed | 9.02 | 9.50 | 64.06 | 11.72 | 12.98 | 54.67 | **−2.70** | [−3.20, −2.29]* |
| earlier_only | 14.57 | 15.73 | 66.35 | 13.92 | 25.99 | 23.06 | **+0.65** | [−0.33, +1.74] |
| never_observed | 32.37 | 43.85 | 55.29 | 27.60 | 62.50 | 33.08 | **+4.77** | [+3.25, +6.23]* |

### Validation gate, pooled

| | strict tool | existing | diff |
|---|---:|---:|---:|
| all, ours | 26.342443 | 26.342443 | 0 |
| all, OccAny 1-frame | 24.061954 | 24.061954 | 0 |
| all, delta and CI | +2.2805 [+1.2330, +3.3259] | +2.2805 [+1.2330, +3.3259] | 0 |
| anchor_observed vs their `observed` | −2.6995 [−3.2010, −2.2853] | −2.7006 [−3.2014, −2.2901] | 0.001 |

Bit-identical on `all`; the 0.001 on the observed row is the 0.091 pp definitional difference
of §5, as expected.

### The finding: accumulation's contribution does not survive dilation

**`earlier_only` falls from +4.98\* (raw) to +0.65 with a CI containing zero (pooled).** The raw
advantage in remembered space was largely a dilation artefact: OccAny's 1-frame recall in that
region is 17.02 raw and 23.06 pooled, and dilation is worth +3.74 SC IoU to them overall
against −0.05 to us, because our predictions are already dense.

So under the convention OccAny publishes in:

* **accumulated reconstruction buys us nothing statistically distinguishable from zero**
  (+0.65, CI ∋ 0);
* **completion of never-observed geometry buys +4.77** (CI clear of zero), over 89.06 % of the
  surface;
* **we remain worse on what the anchor camera sees** (−2.70, CI clear of zero).

This is a **stronger** version of §4's conclusion, not a weaker one: in both conventions the
only region with a robust advantage is the never-observed one, and under the pooled convention
it is the *only* region with a significant positive advantage at all.

**Note the direction of caution.** A CI containing zero is not evidence that accumulation
contributes nothing — it is failure to detect a contribution at this sample size. The raw
convention does detect one (+4.98, CI clear of zero). The defensible statement is that the
accumulation advantage is **convention-dependent and small**, while the never-observed
advantage is **robust across both conventions**.
