# Gate 14 — can a stronger causal camera-only depth source move toward the oracle?

**Stage 1 decision: C — the candidate depth (MUSt3R-512) is not substantially better than LB-M.
It fails the frozen promotion rule on every criterion; Stage 2 completion training stays locked.**

Stage 1 is a raw observed-geometry audit with no training. Everything except the depth values and
the pose source is Gate 9's: the same 1,044 five-frame clips (0006 decision drive; 0003 / 0007 /
0010 replayed unchanged after the code freeze), the anchor SSC grid at 0.2 m, the frozen hard
mapper, the five-sweep LiDAR reference and evaluation masks. SemanticKITTI and Occ3D-nuScenes
were not accessed. Verification passes 20/20 (`verification.md`).

**The 2 × 2 result (0006, shared support, observed-map IoU):**

| | LingBot local-reset pose (L) | KITTI-360 GT pose (G) |
|---|---|---|
| **LB-M** (LingBot × causal MoGe-2 scale, the deployed baseline) | **0.088** | 0.080 |
| **NEW** (MUSt3R-512 × causal MoGe-2 scale) | **0.090** | 0.078 |
| **LD** (LiDAR camera depth, oracle reference) | — | **0.438** |

Factorisation on 0006 (shared support, clip-block bootstrap 95 % CI):

| quantity | value | 95 % CI | reading |
|---|---|---|---|
| depth_gain_GT = IoU(NEW-G) − IoU(LBM-G) | −0.0015 | [−0.005, +0.002] | MUSt3R depth is not better |
| deployable_gain = IoU(NEW-L) − IoU(LBM-L) | +0.0018 | [−0.000, +0.004] | no practical gain |
| pose_cost_LBM = IoU(LBM-G) − IoU(LBM-L) | −0.0085 | [−0.017, +0.000] | GT pose does not help LB-M |
| pose_cost_NEW = IoU(NEW-G) − IoU(NEW-L) | −0.0118 | [−0.021, −0.002] | GT pose does not help NEW either |
| gap_closed = deployable_gain / (LD-G − LBM-L) | +0.5 % | [−0.0 %, +1.1 %] | 99.5 % of the oracle gap remains |
| gap_closed_GT | −0.4 % | [−1.4 %, +0.5 %] | |

The replay drives tell the same story:

| drive | LBM-L | LBM-G | NEW-L | NEW-G | LD-G | depth_gain_GT | deployable_gain | promote |
|---|---|---|---|---|---|---|---|---|
| 0006 (590) | 0.088 | 0.080 | 0.090 | 0.078 | 0.438 | −0.002 | +0.002 | no |
| 0003 (65) | 0.041 | 0.035 | 0.043 | 0.029 | 0.369 | −0.006 | +0.002 | no |
| 0007 (190) | 0.047 | 0.031 | 0.044 | 0.032 | 0.378 | +0.001 | −0.003 | no |
| 0010 (199) | 0.057 | 0.025 | 0.057 | 0.022 | 0.393 | −0.003 | −0.000 | no |

Both camera sources leave the observed map at 0.03–0.09 IoU against 0.37–0.44 for LiDAR depth.

## Candidate and protocol

**Candidate** (`candidate_config.yaml`). MUSt3R-512 run with the official `naver/must3r` code
(commit 0c0d76b, cloned code only), checkpoint `MUSt3R_512.pth` (MD5 matches Naver's published
checksum; the weights were already on disk as OccAny's base model), loaded strictly, frozen. Per
clip it receives only the five rectified images of frames t−4..t through the official preprocessing
(centre crop to the 16:5 bucket, 512×160) and the official sequential memory schedule [2, 1, 1, 1];
no future image, no intrinsics, no LiDAR, no labels, no target-domain calibration. The per-view
optical-axis depth (`pts3d_local[..., 2]`) is nearest-neighbour resampled onto the 140×518 LingBot
lattice through the official crop/resize matrix (85 % of lattice pixels lie inside the crop). It is
scale-ambiguous; its metric scale comes from the same causal MoGe-2 clip-scale estimator LB-M uses
(no confidence gate, since MUSt3R confidence is not on LingBot's scale). Its training data
(NOTICE) contains no KITTI-360, SemanticKITTI or nuScenes; it does include Virtual KITTI 2.

**Conditions.** LBM-L = the frozen deployable input (it equals the Gate 12/13 H/M input map bit for
bit on all 1,044 clips); LBM-G = LB-M with GT poses and the calibrated camera (equals Gate 9's GF
exactly); NEW-L / NEW-G likewise with MUSt3R depth; LD-G = Gate 9 LD = Gate 13's oracle input.
Under L, depth enters the mapper in the LingBot gauge with the mapper scale forced to `s_clip`
(applied once to depth and translation, as in deployment); under G, metric depth with scale 1.

**Supports.** `S_shared` = LingBot confidence ≥ 1.5 AND MUSt3R-valid (inside crop, finite), built
from validity alone before any depth array; it keeps 84 % of LB-M's native rays. Primary
comparisons use it. `S_native` is each source's own support (NEW has 3 % more rays than LB-M on
0006, all outside LB-M's confidence gate). On native support NEW is *worse* (0.082 vs 0.091 under L;
0.068 vs 0.083 under G), so the shared-support comparison is, if anything, generous to MUSt3R.

**Promotion rule** (`promotion_rule.json`, written 00:16 on 2026-09-14, before the first MUSt3R
occupancy record at 00:41). All four criteria on 0006, shared support:

| criterion | required | measured | pass |
|---|---|---|---|
| C1 IoU gain | ≥ +0.05 abs or ≥ +50 % rel | +0.0018 (+2.0 %) | no |
| C2 free-space | occupied-in-LiDAR-free down ≥ 0.05 | 0.813 → 0.826 (up) | no |
| C3 not density | C1 on identical rays | fails | no |
| C4 range consistency | NEW-L > LBM-L in every major band | worse at 0–5 m and > 30 m | no |

## Answers

1. **Is the candidate depth genuinely more accurate than LB-M under GT pose?** No. On the shared
   LiDAR-supported pixels of 0006: AbsRel 0.080 vs 0.075, median endpoint error 0.68 m vs 0.63 m,
   p90 2.70 m vs 2.41 m, within 0.2 m 16.5 % vs 17.3 %, exact 0.2 m voxel 3.45 % vs 3.41 %, identical
   near bias (−5.6 %). On the replay drives MUSt3R is slightly better within 0.2 m (5.7–7.7 % vs
   3.8–5.7 %) but has heavier tails (p90 4.6–5.4 m vs 3.6–4.6 m). The two sources are the same kind
   of depth with the same bias; MUSt3R is not a step toward LiDAR.
2. **Observed-map IoU gain under GT pose:** −0.0015 [−0.005, +0.002] on 0006; −0.006 / +0.001 /
   −0.003 on 0003 / 0007 / 0010. None is positive with a CI excluding zero.
3. **Gain under deployable local-reset pose:** +0.0018 [−0.000, +0.004] on 0006 (+2 % relative);
   +0.002 / −0.003 / −0.000 on the replay drives. Closes 0.5 % of the LD-G gap on 0006.
4. **Does pose become a larger bottleneck after depth improves?** The question is moot because depth
   did not improve, but the pose factor itself is informative: **GT pose lowers observed-map IoU on
   every drive and both depths** (pose cost −0.009 to −0.033 for LB-M, −0.012 to −0.035 for NEW).
   LingBot's local poses share the depth's ~6–13 % under-scale, so the five frames are mutually
   consistent in the LingBot gauge; metric GT translations break that consistency and scatter the
   already-noisy surfaces (precision 0.187 → 0.128, predicted occupancy 0.77× → 1.37× the reference
   on 0006). The L-vs-G pose difference is real (relative translation error 1.3 m median on 0006,
   2.0–3.3 m on the replay drives; 0.5–0.7° rotation; 1.1–2.4 m median endpoint displacement), but
   with this depth quality it is not what limits the map. Pose remains secondary.
5. **Fraction of the LD-G oracle gap closed by NEW-L:** +0.5 % [−0.0, +1.1] on 0006; +0.5 / −1.0 /
   −0.1 % on 0003 / 0007 / 0010.
6. **Does the candidate reduce occupied predictions in LiDAR-free space?** No. On shared support
   under L: 0.813 → 0.826 (0006), 0.915 → 0.917, 0.903 → 0.912, 0.892 → 0.902. LD-G sits at
   0.47–0.55.
7. **Where by range does the improvement occur?** Nowhere consistently. On 0006 NEW-L is +0.004 at
   5–10, +0.005 at 10–20 and +0.004 at 20–30 m, but −0.004 at 0–5 m and 0 beyond 30 m; the replay
   drives show ±0.005 swings in both directions. LD-G is 0.41–0.48 in every band.
8. **Does it improve edges as well as smooth surfaces?** Neither. On 0006 within-0.2 m is 20.5 % vs
   20.9 % on non-edge pixels, 13.2 % vs 14.4 % at moderate discontinuities and 9.7 % vs 10.2 % at
   strong ones; on the replay drives MUSt3R gains on non-edge pixels (+3–6 points) and loses on
   strong edges. Its endpoint errors at strong edges are the largest of any stratum (1.5–2.9 m median).
9. **Did it pass the predeclared promotion criterion?** No — 0 of 4 criteria on 0006, and none on
   any replay drive.
10. **Stage 2:** not run. `stage2_guard.require_promotion()` raises `Stage2Locked`; no Stage 2
    artifact exists.
11. **Single dominant remaining bottleneck:** observed depth accuracy, unchanged from Gate 13.
    Two frozen monocular/multi-view models (LingBot, MUSt3R) plus MoGe-2 (depth-source
    comparison) all land at 0.03–0.09 observed-map IoU with ~6–13 % near bias, 0.6–1.3 m median
    endpoint error and > 80 % of occupied voxels in LiDAR-free space, against LiDAR depth's 0.37–0.44
    through the same mapper. The failure is shared across frozen camera-only depth models at this
    resolution and range, not specific to LingBot.

## Decision

**C — the candidate depth is not substantially better.** NEW-G ≈ LBM-G on all four drives
(|depth_gain_GT| ≤ 0.006, CIs spanning zero on three), and NEW-L ≈ LBM-L. Decision B does not apply
(no GT-pose gain for pose to absorb; pose cost is negative for both depths). Decision D does not
apply (NEW does not improve under either pose). MUSt3R-512 is rejected as a replacement for LB-M
without completion training, and the mapper was not tuned to rescue it.

**Next experiment (single).** Stop screening frozen depth models. The evidence across Gates 9–14 is
that frozen camera-only depth at this resolution cannot place surfaces within one 0.2 m voxel, and
Gate 13 shows the completion model would use accurate geometry if it had it. The one source-side
experiment that addresses the residual directly is a **LiDAR-supervised depth refinement of the
frozen LB-M depth on KITTI-360 training drives only** (0003/0007/0010; a small per-pixel residual
network on the LingBot depth, confidence and image, supervised by the same raw-LiDAR camera depth
used in LD; frozen after training; evaluated with the Gate 9 observed-map protocol on 0006 under L
pose, then, only if it passes the same promotion rule, the frozen Gate 13 completion recipe). This
keeps the deployable pipeline camera-only at inference, uses only source-domain LiDAR for
supervision, and tests whether the depth error is learnable rather than intrinsic.

## Limitations

- Only one candidate was tested, as the brief required; DA3 was not available offline and
  Depth Anything V2 (single-frame, ViT-S) had no prior evidence of being stronger.
- MUSt3R ran at its 512×160 bucket, which crops 15 % of the lattice rows; the shared support
  removes those rays for both sources, so this does not bias the comparison, but MUSt3R may do
  better at higher resolution than tested here.
- MUSt3R's confidence was not used as a gate (no defensible threshold on LingBot's scale). On its
  native validity it is worse than LB-M on its native gate, so a confidence gate could only narrow
  its support, not raise its geometry.
- The metric scale for MUSt3R came from the LB-M estimator applied to MUSt3R depth
  (s_new ≈ 1.39 on 0006, 1.47–1.79 on the replay drives). Any scalar error is shared with LB-M's own
  6–13 % under-scale; the depth-source comparison showed a LiDAR-fitted scalar recovers only 21–29 %
  of the LiDAR gap, so scale cannot account for the missing 0.35 IoU.
- These are raw observed-map metrics under the Gate 9 protocol, not completion IoU, and say
  nothing about the historical Occ3D result.

## Verification and process notes

20/20 checks pass. Two things were amended and recorded:
- Before the code freeze, the reprojection-error diagnostic (a secondary registration statistic)
  divided by a clamped depth for endpoints behind the anchor camera, giving absurd values. It was
  restricted to endpoints in front of the camera and 0006 was re-audited; every occupancy count,
  scalar and other pixel column is identical to the first run (`logs/results_table_0006_prefix_reproj.md`).
  The corrected values are 22 px median / 73 px p90 on 0006.
- The first verification run failed the deterministic-rerun check because `np.array_equal` treats
  NaN ≠ NaN in the pixel table (NaN marks pixels outside the MUSt3R crop). The MUSt3R rerun matched
  bit for bit; the check was made NaN-aware (`logs/verification_attempt1_19of20.md`).

## Files

`results_table.md` (0006), `results_table_all.md`, `verification.md`, `candidate_config.yaml`,
`promotion_rule.json`, `promotion_result.json`, `code_freeze.json`, `analysis[_all].json`,
`tables/{observed,factors,depth}_*.csv`, `figures/1_2_3_iou_gain_posecost_*.png`,
`figures/4_5_endpoint_range_*.png`, `figures/maps/6_*_t*.png` (Gate 9 rule representatives:
LB-M IoU at the 10th / 50th / 90th percentile), `logs/`. MUSt3R depth cache and per-clip records:
`/media/SSD1/MINH_DATASETS/lingbot_gate14_must3r/`.

## Reproduce

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python; T=tools/gate14_depth_pose; L=artifacts/gate14_depth_pose/logs
R=/media/SSD1/MINH_DATASETS/lingbot_gate14_must3r; M=$R/src/must3r
git clone --recursive https://github.com/naver/must3r.git $M && git -C $M checkout 0c0d76b   # code only; weights already on disk
# promotion_rule.json is written before any MUSt3R occupancy result
$PY $T/make_manifest.py --drives 2013_05_28_drive_0006_sync 2013_05_28_drive_0003_sync 2013_05_28_drive_0007_sync 2013_05_28_drive_0010_sync
for d in 0006 0003 0007 0010; do PYTHONPATH=$M:$M/dust3r:$M/dust3r/croco /home/minh/workspace/third_party/occany_env/bin/python $T/must3r_infer.py --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
for s in 0 1 2 3; do $PY $T/audit14.py --drive 2013_05_28_drive_0006_sync --shard $s --shards 4 --device cuda:$s > $L/audit_0006_s$s.log 2>&1 & done; wait
$PY $T/analyze14.py                       # promotion_result.json; code frozen here (code_freeze.json)
for d in 0003 0007 0010; do $PY $T/audit14.py --drive 2013_05_28_drive_${d}_sync --device cuda:1 > $L/audit_${d}.log 2>&1; done
$PY $T/analyze14.py --drives 2013_05_28_drive_0006_sync 2013_05_28_drive_0003_sync 2013_05_28_drive_0007_sync 2013_05_28_drive_0010_sync
$PY $T/verify14.py --device cuda:0
$PY $T/stage2_guard.py                    # exits 3: Stage 2 locked
```
