# Teacher-alignment control: does scene-aligned class supervision transfer better than shuffled supervision?

Two conditions, three seeds, 6 000 steps, identical initialisation and the historical B
runs' 24 000 paired data draws. **R** is supervised by the future teacher's class
vector at its own voxel. **S** receives the same vectors permuted within per-crop strata of
(z-layer, map-observed flag), so class multiset, support, occupancy targets and inputs are
unchanged. Both arms receive the same auxiliary gradient budget at the shared trunk:
`g = g_occ + ρ·‖g_occ‖·g_aux/‖g_aux‖` with ρ = 3.564, the pooled median of
‖∇(0.5·L_KL)‖/‖∇L_occ‖ over shared parameters on 32 minibatches of the historical B
seed-0 checkpoint.
Numbers: `artifacts/teacher_alignment_study/comparison_table.md`; ten verification checks
pass (`verification.json`).

## 1. Was local teacher correspondence disrupted?

Moderately, not fully. Over training the shuffle moved 46 % of supervision vectors, changed
the argmax class of 31 %, with mean L1 0.58 between aligned and delivered vectors. Strata
are strongly class-homogeneous: 14 % of supervision sat in strata where every vector shares
an argmax and a permutation changes nothing, and even where it changes, S still receives
the correct (z, observed) class prior. So S removes voxel-level correspondence while
keeping the coarse vertical/observability class structure. S is "coarse semantics", not "no semantics".

## 2. Were gradient budgets matched?

Yes. Achieved ratio equals ρ on 5 999/6 000 steps of every run; the single exception is
step 1, where the zero-initialised occupancy head gives ‖g_occ‖ = 0 by construction. The
raw ratio medians (3.4–3.9) are close to ρ, so c ≈ 1 (0.90–1.06). The fp32 check reproduces joint backprop to 1e-6 at
c = 1; heads are unscaled. Task-gradient cosines are near zero in both arms (+0.01 to +0.05).
R and S are indistinguishable in every panel of `figures/gradient_budget.png`.

## 3. Did aligned supervision improve source completion or transfer?

No. On drive 0006 later-measured IoU, R−S is −0.006, −0.002, +0.005 (each seed's block CI
excludes zero, signs disagree; mean −0.001). On observed voxels the arms are within 0.004.
On the targets the sign flips again: SemanticKITTI −0.022, −0.019, +0.013; Occ3D −0.036,
−0.001, +0.038. The per-seed transfer differences follow the frozen thresholds
(R 0.00/−0.31/−0.31; S −0.22/−0.28/−0.09): the more negative threshold predicts more, and
the sparse targets reward recall. Threshold-free, S is equal or slightly ahead: AUROC and
oracle-threshold IoU favour S on five of six dataset×seed cells (Occ3D oracle IoU −0.023,
−0.018, −0.004; SemanticKITTI −0.011, −0.023, +0.001). Both arms sit at or below reference B
in-domain and near B on the targets; A remains best in-domain.

## 4. What happened to semantic predictions?

Aligned supervision is learnable and S is partly learnable. Top-1 agreement with the aligned
teacher on drive 0006 is 0.41/0.41/0.40 for R and 0.40/0.39/0.38 for S — a 1.5 pp gap, far
smaller than the 31 % argmax disruption, which means the network in S mostly predicts the
stratum prior it can still learn. On SemanticKITTI, R has higher SSC mIoU (0.033 vs 0.031,
0.033 vs 0.031, 0.032 vs 0.027) and higher naming accuracy on the common set (+1.5, +1.0,
+2.3 pp). On Occ3D the two are within 0.004 mIoU; the seed-2 naming outlier (S 0.50 vs R
0.37) is a composition effect — S's high threshold shrinks the common set to road voxels —
and not evidence of better naming. Semantic quality is low in both arms (mIoU ≈ 0.03–0.04).

## 5. Explanations supported and unresolved

Supported: with the trunk gradient budget held fixed, voxel-level alignment of the class
signal adds nothing measurable to occupancy completion in-domain or on the targets; the
occupancy effect of the auxiliary loss is carried by something the shuffle preserves — the
gradient magnitude itself and/or the stratum-level class prior. The small semantic gain
of R does not translate into occupancy.

Unresolved: (i) the shuffle left coarse class structure intact, so this study cannot say
whether *all* semantic content is irrelevant; (ii) the transfer differences are dominated by
threshold selection on 590 source anchors, and the source-only rule is noisier than the
effects being measured; (iii) three seeds with per-seed sign flips bound the effect but do
not estimate it — the mechanism result is **inconclusive on direction, conclusive that R does
not beat S** at this budget.

## 6. Single next step

Replace the source-only global threshold with a threshold-free or calibrated selection
(e.g. match predicted density to the source occupancy prior) and re-score the twelve frozen
checkpoints already on disk. This costs no training, removes the one factor that currently
determines the sign of every transfer difference, and decides whether the residual S ≥ R
ordering is real before any further supervision variant is trained.

