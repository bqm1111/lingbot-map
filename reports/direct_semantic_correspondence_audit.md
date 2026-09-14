# Direct semantic-correspondence audit: Trident-H's view dependence, or the 2D→3D lifting?

No model trained. Physical 3D points are raw Velodyne returns placed in the KITTI-360
world with the ground-truth pose chain, projected into rectified `image_00` of other
stream frames with the calibrated `K` and `cam0_to_world`, and visibility-checked against
each frame's own LiDAR z-buffer. The dense Trident-H feature (the language-aligned
1024-d token grid validated in the previous study) is sampled directly at the projected
pixel with the official interpolation rule. The semantic mapper is never consulted to
define a correspondence; it is replayed only to be audited. 12/12 verification checks.
Numbers: `results_table.md`; figures: `figures/`; visual panels: `panels/`.

## 1. How were true physical correspondences constructed?

Source frame A: every 5th stream frame of 0006 (356 frames; every 10th on the training
drives); partner frames B at stream offsets ±1…±12; 1,500 returns per sweep, depth
3–50 m, 12-px border. Visibility: the minimum depth of the frame's own sweep in a
7×7-pixel window must lie within 0.3 m + 1 % of the projected depth (strict; loose
0.6 m + 2 %), and a window with no return rejects the point as unverifiable. Of 7.45 M
(X, B) candidates on 0006, 51 % are behind the camera or out of range, 4 % outside the
image, 1.5 % have no LiDAR reference, 15 % are occluded, 1.8 % lie in front of the
reference; **1.60 M strict pairs** remain (2.06 M over the four drives). Calibration was
checked by round trip (7.6e-5 px) and the frames' own-sweep residuals are ≤ 0.7 m.

## 2. Are Trident features stable at verified same physical points?

Partly. On 0006 the same-point feature cosine is 0.942, per-view class agreement 0.729
(SAM-refined labels 0.740), probability cosine 0.877 — against 0.568 / 0.762 for the
previous mapper-defined same-voxel pairs at comparable frame gaps. But within a single
image a *different* point 0.3–1.5 m away scores 0.974 (class agreement 0.783): a
verified view change moves the feature more than moving a metre along the surface.
Per class, same-point agreement is 0.91 road, 0.88 vegetation, 0.74 building, 0.69 car,
0.59 sidewalk/fence, 0.25–0.30 other ground / man-made structure / terrain, 0.21 pole.

## 3. Do same points separate from nearby different points?

Weakly. The cross-view local control (nearby point in B) gives cosine 0.932 → margin
+0.009, **AUC 0.561** (previous mapper-based 0.520); probability AUC 0.553 (0.514);
class agreement 0.729 vs 0.675. Against a random point the AUC is 0.90 and against a
same-class point 0.69, so the feature carries class and context, but at a metre scale
the same point is barely more similar to itself across views than to its neighbour. The
same holds on 0003/0007/0010 (AUC 0.55–0.56).

## 4. How much does viewpoint change matter?

Systematically and strongly, with correspondence held exact: class agreement 0.814 →
0.733 → 0.615 for baselines < 5 / 5–15 / > 15 m; 0.798 → 0.690 → 0.473 for viewing-angle
change < 10° / 10–30° / > 30°; feature cosine 0.970 → 0.900 and the local AUC 0.62 → 0.52
over the same bins; adjacent stream frames (±1) give 0.80. Depth matters less
(0.78 → 0.60 from 3–10 m to 35–50 m) and visibility residual not at all (0.72 vs 0.74).

## 5. Was the previous mapper correspondence wrong?

Yes, grossly. Replaying the mapper's frozen inputs for 298 k verified pairs with
|offset| ≤ 3: the mapper has *any* frame-B observation in the 0.2 m voxel of X for only
**5.5 %** of true points; when it does, those pixels lie 84 px (median) from the true
projection, within one token only 3.7 % of the time. LingBot depth at the true pixel is
off by 11.5 % (median), and the LingBot relative pose alone reprojects the point 93 px
(median) from where it is; the endpoint lands 5.0 m from X. On the pairs where the
mapper has observations in both frames, same-voxel class agreement is 0.534 vs 0.692 for
the same points matched directly. The pose error grows along the stream — 30 px in the
first 100 frames, 156 px beyond frame 1000 (mapper agreement 0.65 → 0.32, direct 0.75
flat) — consistent with LingBot streaming poses drifting beyond their RoPE range: the
per-step relative translation error is 20–30 % of the step early in a drive and exceeds
100 % after ~300–600 stream frames on all four drives. The single-view-per-voxel
structure, the 0.5 % multi-view support and the 0.42–0.56 same-voxel agreement of every
previous study are consequences of this misregistration, not properties of the teacher.

## 6. Can tiny pixel errors explain the instability?

No. Shifting the B sample by 2 px changes the feature by 1 − cos = 0.0004, by 5 px 0.003,
by one token (11 px) 0.010, against 0.058 for a verified view change — a one-token
error explains 18 % of it. Trident's feature is spatially smooth; its cross-view change
is not a reprojection artefact.

## 7. Single decision

**Decision C — both contribute; the correspondence pipeline is the dominant bottleneck.**
Direct verified correspondence raises class agreement from 0.57 to 0.73 (0.80 at adjacent
frames) and, more importantly, the mapper misses the true observation in 95 % of cases,
which is why no multi-view fusion or consistency test could work through it. Trident's
own view dependence is real and bounds what precise correspondence can deliver (local
AUC 0.56, agreement falling to 0.47 beyond 30°), but that limit is only reachable once
observations are registered correctly. Recommended next step: replace the mapper's
semantic association with precise, visibility-aware projective correspondence from
registered geometry — and, because the stream's LingBot poses drift beyond ~300 frames,
that registration must not come from the current streaming pose chain.
