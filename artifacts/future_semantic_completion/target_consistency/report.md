# Target consistency: is there a persistent voxel-level semantic identity under the per-view Trident-H observations?

No model trained. The frozen future mapper was replayed **one future frame at a time** for
all 1,948 anchors (4 drives), so any frame-disjoint grouping of the 20-frame horizon can be
formed exactly (the sum over all frames reproduces the cached target to u8 precision on
every anchor). Two groupings: *contiguous* (A = t+1..7, B = 8..14, C = 15..20) and
*interleaved* (offset mod 3). Candidates are decided and fused from A+B only, validated on
C only; verified by perturbation (10/10 checks, `verification.json`). Thresholds inspected
on drive 0006 only. Numbers: `results_table.md`; figures: `figures/`.

## 1. Is the current target unstable?

Yes, reproduced with independent future evidence. The existing rule applied to A+B and
validated on C gives top-1 **0.530**, cosine 0.589 on drive 0006 (within-anchor shuffled
reference 0.381); raw group-vs-group agreement is 0.51–0.56 on 0006, 0.45–0.52 on 0007,
0.73 on 0010 and 0.90 on 0003 (79 % vegetation). The previous 0.456 (causal vs future) is
the same phenomenon measured across a longer baseline.

## 2. Does future-only multi-view consensus create a more stable target?

Only nominally. Requiring top-1 agreement between two independent future groups (F1) lifts
the independent C repeatability to **0.698** / cosine 0.729 on 0006 and 0.907 / 0.910 on the
training drives. But the within-anchor shuffled reference rises with it (0.375 → 0.514 on
0006; 0.601 → 0.823 train): the voxel-specific margin over shuffling is +0.15 → +0.18 on
0006 and +0.09 → +0.08 on the training drives. Confidence (τ) and similarity (γ) thresholds
add nothing beyond F1 (0.70–0.72 on 0006 for every τ, γ). The contiguous grouping cannot
be evaluated at all: A∧B∧C support is 0–133 voxels per drive.

## 3. What is the coverage cost?

Severe, and structural. 92–95 % of voxels that carry any teacher evidence receive it from
**exactly one** future frame; two frame-disjoint groups coincide on 0.45–0.89 % of eligible
voxels and three on ≤ 0.2 %. F1 therefore retains 10,131 of 2,018,736 eligible voxels on
0006 (**0.50 %**, ~17 per anchor; 2.2 % of the existing target) and 19,742 of 4,204,467 on
the training drives (0.47 %). The nearest evidence from another frame lies more than 2 m
away for 39–57 % of observations, and where it lies 1–3 voxels away it agrees no better
than same-voxel evidence (0.52 vs 0.52 on 0006, 0.44 vs 0.45 on 0007) — the single-view
structure is not endpoint scatter hiding a consensus next door; the per-view labels
themselves disagree at any scale.

## 4. Does filtering destroy semantic diversity?

Largely. Classes with ≥ 500 retained voxels fall from 14 to 4 on 0006 and from 17 to 3 on
the training drives; entropy 1.79 → 1.61 (0006) and 1.71 → 1.28 (train); dominant-class
share 0.40 → 0.43 and 0.45 → 0.47. On 0006 only *vegetation* keeps ≥ 500 validated voxels
(repeatability 0.80); everything else is measured on tens to hundreds.

## 5. Is the improvement genuinely voxel-specific?

Partly, but no more than before filtering. Against the pooled class prior (0.19–0.25) and
the C majority rate (0.32–0.35) every filter is far above chance; against the within-anchor
shuffle — the reference that respects local class homogeneity — the margin stays at
+0.08–0.18 whether or not agreement is enforced. Filtering selects homogeneous regions and
anchors; it does not sharpen per-voxel identity. The causal-map diagnostic agrees: the F1
target matches the nearest causal teacher voxel at 0.67 when that voxel is 1–2 voxels away,
0.44 overall, and 55 % of retained voxels are > 3 m from any causal evidence.

## 6. What target definition should be frozen, if any?

None. The simplest rule that meets the repeatability floor is F1 on the interleaved
grouping (two frame-disjoint future groups agree on the class; target = their weighted
fusion), but at 0.5 % coverage, 3–4 supported classes and ~15 voxels per anchor it is not a
learning problem.

## 7. Single decision

**Decision B — the target only becomes stable after severe coverage/class collapse.** Do
not retrain. Voxel-wise Trident-H teacher supervision is not currently viable: the frozen
lifting yields one view per voxel for > 90 % of covered voxels, so independent consensus
exists for well under 1 % of the supervision population, and where it exists it mostly
certifies large homogeneous regions.
