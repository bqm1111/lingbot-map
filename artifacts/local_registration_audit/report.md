# Local registration audit: does removing long-horizon streaming drift give consistent causal geometry?

No model trained. Same five-frame causal clips (t-4 .. t; 590 anchors on 0006, 454 on
0003/0007/0010), frozen LingBot-Map checkpoint `204754b` throughout. Conditions:
**S** stream cache as used today; **L** LingBot re-run on the five images alone (KV cache and
RoPE counter cleared, clip-local numbering, clip MoGe-2 scale); **Lp** L poses + S depth;
**G** GT relative poses + S depth; **GLd** GT poses + L depth; **Gs** GT poses + S depth
rescaled by a per-clip LiDAR ratio (diagnostic oracle, not deployable). 12/12 verification
checks: L bit-identical fresh vs after another clip; S prefix re-run reproduces the stream cache
exactly; identical landmarks for every condition; GT transforms round-trip to 3e-12 m.
Numbers: `results_table.md`; figures: `figures/`, maps: `figures/maps/`.

## 1. How bad is the current five-frame streaming registration?

Bad. On 0006, relative translation error of frames t-1..t-4 to the anchor is 5.2 m median
(53 % of the true motion; 84 % at t-1), rotation 2.0°, translation direction 19° off.
Verified LiDAR points reproject **128 px** median (p90 470 px) from their true position and
land 4.7 m away in the anchor camera. 0007 is worse (254 px, 12.5 m); 0003 is milder (68 px).

## 2. Does local reset recover short-window relative pose?

Largely, not fully. L cuts the relative translation error to 1.3 m (12.6 % of motion),
rotation to 0.7°, reprojection to **46.5 px** (p90 124) and pose-only endpoint error to 1.1 m on
0006; 47–50 px on the other drives. It is not G (0 px): L's translation magnitude is ~13–17 %
wrong and its direction carries a constant ~5.9° bias on every offset and drive — a systematic
camera-model offset rather than drift.

## 3. How much of the geometry error is caused by pose?

A minority. Raw five-frame map IoU against the LiDAR of the same five sweeps (0006): S 0.054,
**G 0.061**, Lp 0.068, GLd 0.079, **L 0.090**, oracle-scale Gs 0.072. Replacing the pose with
ground truth while keeping the depth moves IoU by +0.007 and precision from 0.14 to 0.21; even
with GT pose and LiDAR-corrected scale the map is 29 % precise and 9 % complete, and 19–39 % of
occupied voxels sit in LiDAR-measured free space on every drive. GT pose with L depth (0.079) is
worse than L's own pose with the same depth (0.090): LingBot's depth and pose are mutually
consistent in its own gauge, so correct metric motion on imperfect depth does not sharpen the
map. Depth itself differs between S and L (AbsRel S vs L 0.20, correlation 0.99, scale ratio
1.1–1.3); L's depth is closer to LiDAR on 0006 (AbsRel 0.081 vs 0.123) but not on 0007 (0.129
vs 0.087), so the L-over-S map gain is part pose, part re-predicted depth. Dilated maps against
the future target stay at 0.04–0.07 IoU under every condition; the frozen head, trained on
S-style inputs, is unchanged at 0.09–0.12 (compatibility only).

## 4. Does the problem worsen with global stream position?

Yes for S, no for L. 0006 reprojection S 84 → 116 → 165 px from early to late stream, L 44 → 46
→ 48 px; relative translation S 0.31 → 0.67, L 0.12 → 0.13; raw map IoU S 0.063 → 0.048,
L 0.083 → 0.088. Same pattern on 0007 and 0010 (S 87 → 245 px, L 49 → 46 px).

## 5. Does corrected registration restore multi-view semantic agreement?

Partly (0006, 140 k verified same-point landmarks inside clips). Two-frame same-voxel support:
S 0.8 %, L 6.6 %, G 1.1 %. Class agreement where supported: S 0.52 (local control 0.49), L 0.71
(0.62), G 0.73 (0.68); feature-cosine margin over a nearby point: S +0.002, L +0.019, G +0.014.
Registration restores agreement to the level of the direct audit, but support remains tiny
because depth errors still scatter endpoints out of the 0.2 m voxel even under GT pose.

## 6. What geometry representation should now be frozen?

L — local-reset five-frame LingBot inference. It dominates S on every pose, reprojection,
stationarity, raw-geometry and semantic-support measure and is deployable. It does not make
the geometry good.

## 7. Single decision

**Decision C — pose is not the dominant remaining geometry bottleneck.** Long-horizon streaming
drift is real and local reset removes most of it (128 → 47 px), but ground-truth pose on
LingBot depth leaves the five-frame map almost as poor as before (IoU 0.054 → 0.061; 0.072 even
with oracle scale), so depth/reconstruction limits the map. Next step: diagnose the
depth-to-occupancy geometry under GT pose — where LingBot depth places surfaces relative to
LiDAR (per-pixel error by range, edge and surface type) and how the mapper's confidence gate,
occupied band and free-space carving convert that error into false occupied and missed voxels.
Semantics stays paused.
