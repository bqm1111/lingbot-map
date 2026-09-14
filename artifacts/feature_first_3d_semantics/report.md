# Feature-first 3D semantics: are Trident-H's dense features more view-consistent than its class probabilities?

No model trained. Trident-H's pre-classification dense feature was hooked, cached for all
3,162 KITTI-360 frames of the four drives, lifted into 3D by the frozen mapper at exactly
the voxels, rays and weights of the probability path, and compared with the class vectors
on matched voxels under interleaved frame-disjoint groups A/B/C. Two populations
(unobserved→future-observed→occupied; all future-observed occupied) agree throughout.
12/12 verification checks. Numbers: `results_table.md`; figures: `figures/`.

## 1. What dense semantic representation does Trident-H expose?

Inside `get_trident_seg` the frozen stack computes `clip_features = clip.encode_image(...)`
— CLIP ViT-H-14 (laion2b) image tokens after Trident's DINO/SAM correlation refinement —
L2-normalises them and takes `logits = f · text_embeddingsᵀ` with the mean-templated CLIP
text embeddings of the vocabulary. Bilinearly interpolated, scaled by 40 and soft-maxed,
those logits are the official coarse probabilities; SAM refinement then rewrites region
scores (6.9 % of pixel labels change). The candidate is therefore the unit-normalised
token feature: 1024-d, 34 × 128 tokens per frame, genuinely language-aligned (any CLIP text
embedding compares by cosine), one linear map from the official classifier. Its
reproduction of the coarse probabilities was checked on every frame: max |Δp| 0.0037 (fp16
tokens), argmax agreement 99.93 %; re-extraction is bit-identical. There is no better
candidate: nothing later in the stack is continuous, and nothing earlier is aligned.

## 2. Are latent features more view-consistent than class probabilities?

No. On drive 0006 (18,051 matched A/B voxels) the same-surface feature cosine is 0.918 —
but the **local-shuffle control** (swap B with another voxel in the same anchor, 0.8 m height
band and 4 m cell) gives 0.915: margin +0.004, AUC 0.520. The class-probability vector
gives the same picture (0.762 vs 0.748, margin +0.015, AUC 0.514), as does P_ref. Every
drive repeats it (all AUCs 0.50–0.54). The feature does separate itself from *distant*
controls better than the probabilities do — random-in-drive AUC 0.85 vs 0.79, same-class-
other-location AUC 0.75 vs 0.57 — so it carries more than the class identity, but what it
carries is local appearance context shared with its neighbours, not the identity of the
surface element. Per class the local-shuffle AUC is 0.50–0.52 for both representations,
including cars, road, sidewalk and fences.

## 3. Does feature-first fusion beat classify-first fusion?

No. With identical rays, weights and text head, fusing the feature and classifying once
agrees with the held-out view on 0.540 of 2,592 A&B&C voxels on 0006 against 0.542 for
averaging the per-view probabilities (0.783 vs 0.787 on 0010, 0.537 vs 0.540 on 0007,
0.957 vs 0.961 on 0003); the two pipelines make the same decision on 95 % of voxels, and
the three held-out rotations and the larger population agree. Whether the held-out view
is judged by its own per-view classification or by classifying its fused feature changes
nothing (±0.002).

## 4. Does adding independent views help?

Marginally and inconsistently. One view vs the held-out view: 0.519 (0006); two views:
0.540; A+B with ≥ 3 contributing frames: 0.571 (n = 387) on 0006 but 0.751 vs 0.788 on
0010. Feature cosine to the held-out view moves from 0.916 (one view) to 0.929 (two) —
the same size as the local-shuffle baseline shift. Nothing accumulates.

## 5. Does the result hold across drives and classes?

Yes, uniformly: margins over the local shuffle are ≤ 0.016 for both representations on
all four drives, and fusion-order differences are ≤ 0.005 everywhere. Drive 0006 and 0007
(the hard drives) sit at 0.54 held-out agreement with either pipeline; the vegetation-heavy
0003 at 0.96. Per class on 0006, feature-first and classify-first agree within ±0.02 for
every class with support.

## 6. Is the representation suitable for open-vocabulary 3D semantics?

The feature is language-aligned and can be queried directly: on the fused A+B features of
0006, the best of 16 fixed free-text queries maps to the same vocabulary class as the
vocabulary decision for building (99 %), vegetation (95 %), car (95 %), sidewalk (92 %) and
fence (70 %); the failures are my query→class mapping (‘a house’ for man-made structure,
‘a pavement’ for road), not the feature. Text querying works — but the queried quantity is
only as persistent as the single views behind it.

## 7. Single decision

**Decision C — dense features are no more persistent than final class outputs.** The
view-to-view instability is already present in the latent feature (same-surface similarity
indistinguishable from a local neighbour's), so moving the semantic decision after fusion
cannot recover a stable 3D identity. Stop using this lifted Trident-H representation as the
core 3D semantic signal; the teacher's per-view observations, or the 2D-to-3D
correspondence that lifts them, have to be re-evaluated before any model complexity is
added.
