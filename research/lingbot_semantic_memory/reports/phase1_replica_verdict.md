# Phase 1 on Replica — verdict

## Verdict: `FAIL_CONSISTENCY` (independent replication of the KITTI result)

Best GCT gain in cross-view consistency: **+7.0 %** against the pre-registered **+15 %**.
On KITTI it was +7.2 %. Two datasets that differ in almost every respect — outdoor vs
indoor, translation- vs rotation-dominated motion, sparse LiDAR-derived vs dense
rendered depth, 407 vs 777 patches per frame — land within 0.2 percentage points of each
other. The hypothesis fails the same way in both, for the same reason.

**Recommendation: stop, with more confidence than after KITTI alone.**

---

## 1. What was run

The **identical** protocol, thresholds and code path as the KITTI study; only the
dataset changed. No threshold was reinterpreted, and the pre-registration in
`README.md` was not edited after seeing the KITTI numbers.

| | |
|---|---|
| dataset | Replica, Nice-SLAM / iMAP layout, `data/Replica` (read-only), 8 scenes × 2000 frames |
| training frames | **384** — six 64-frame chunks, one per scene: `office0:983`, `office1:978`, `office2:935`, `room0:912`, `room1:884`, `room2:904` |
| held-out frames | **192** — three chunks of scene **`office4`**: `341`, `970`, `1618` |
| scene overlap | none; `office3` was left unused so the train/val budget matches KITTI exactly |
| stride | **2** (see §2) |
| resolution | 1200×680 → **518×294**, a pure resize, no crop → **21×37 = 777** patches |
| intrinsics | `cam_params.json`: fx = fy = 600, cx = 599.5, cy = 339.5, depth divisor 6553.5 |
| labels | **none exist in this release** (see §3) |
| seed | 0 |

Same frozen LingBot checkpoint (SHA-256 `ee665103348e07e6…`, 0 trainable), same frozen
DINOv2 ViT-B/14-reg teacher, same parameter-matched probes (1 998 425 / 1 999 300), same
4000-step schedule, no augmentation.

### Commands

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD CUDA_HOME=/usr/local/cuda-12.8 FLASHINFER_CUDA_ARCH_LIST="12.0a"
PY=/home/minh/anaconda3/envs/cu128/bin/python

$PY -m pytest research/lingbot_semantic_memory/tests -q            # 50 passed

$PY -m research.lingbot_semantic_memory.run_phase1 \
    --config research/lingbot_semantic_memory/configs/phase1_replica.yaml --smoke

$PY -m research.lingbot_semantic_memory.run_phase1 \
    --config research/lingbot_semantic_memory/configs/phase1_replica.yaml \
    --device cuda:0 --seed 0

$PY -m research.lingbot_semantic_memory.report_tables \
    --metrics research/lingbot_semantic_memory/outputs/phase1/metrics.json \
    --compare "Replica (indoor)=research/lingbot_semantic_memory/outputs/phase1_replica/metrics.json"
```

## 2. Two things measured rather than assumed

**Pose convention.** Replica's `traj.txt` holds row-major 4×4 camera-to-world matrices.
The widely copied Nice-SLAM loader applies `c2w[:3,1] *= -1; c2w[:3,2] *= -1`, which is
correct for its OpenGL raycasting and **wrong** for OpenCV projection. Rather than pick
one, both were tested by warping frame *i*'s ground-truth depth into frame *j* and
comparing against frame *j*'s own depth:

| convention | median relative depth error | fraction within 5 % |
|---|---:|---:|
| **`traj` as c2w, OpenCV axes (used)** | **0.0004** | **99.6 %** |
| `traj` as c2w with the Nice-SLAM y/z flip | 0.0980 | 25.3 % |
| `traj` as w2c, as-is | 0.0854 | 32.8 % |
| `traj` as w2c, flipped | 0.0581 | 44.5 % |

Adopting the flip out of habit would have injected ~10 % depth error into every oracle
correspondence and quietly corrupted the geometry ablation.
`tests/test_replica.py::test_traj_is_camera_to_world_in_opencv_axes_not_the_niceslam_flip`
pins this against real data.

**Frame spacing.** Replica moves **9 mm and 0.63° per frame** — at stride 1, "gap 1"
would be two near-identical views and the short-gap band would be meaningless. Measured
motion by gap fixed the choice of **stride 2**:

| sampled gap | 1 | 2 | 3 | 8 | 16 | 24 |
|---|---:|---:|---:|---:|---:|---:|
| rotation | 1.25° | 2.5° | 3.8° | 10° | 21° | 31° |
| translation | 0.018 m | 0.037 m | 0.055 m | 0.14 m | 0.28 m | 0.39 m |

For contrast, KITTI at gap 24 is ~34 m of mostly-forward translation with little
rotation. The two datasets probe genuinely different motion regimes, which is the point.

## 3. Replica has no semantic labels — and what replaced them

This release ships RGB, depth, trajectories and an RGB-coloured mesh. There is no
semantic annotation anywhere (the mesh carries only `red/green/blue` vertex
properties), so the label-based boundary metric used on KITTI cannot run.

Rather than drop the metric or silently swap in something unvalidated, a **label-free
boundary definition** was added: adjacent patches are *within-region* when their
ground-truth relative depth differs by < 2 % and *across-boundary* when it differs by
> 10 %, with the gap dropping ambiguous slanted surfaces. It was then **validated on
KITTI**, where both definitions can be computed:

| representation | KITTI label-boundary margin | KITTI depth-boundary margin |
|---|---:|---:|
| `lingbot_encoder` | 0.1177 | 0.1019 |
| `lingbot_gct_b04` | 0.1238 | 0.1032 |
| `lingbot_gct_mid` | 0.1179 | 0.0993 |
| `lingbot_gct_b17` | 0.1026 | 0.0839 |
| `lingbot_gct_final` | 0.1043 | 0.0811 |
| teacher | 0.1295 | 0.1197 |

Same ranking, same conclusion. The label-free metric is a sound proxy, so using it on
Replica is justified rather than merely convenient. Text-query / mIoU remains
**unavailable** on both datasets — still no text-to-DINO bridge in this repository.

## 4. Results — 100 % teacher, predicted geometry

288 held-out frame pairs, 79 998 valid correspondences.

| representation | params | teacher cos | cos (centred) | diversity | **cross-view cos** | vs encoder | margin | recall@1 | depth-boundary margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lingbot_encoder` | 1 998 425 | 0.6105 | 0.4911 | 0.6545 | 0.7986 | — | 0.3910 | 0.2009 | 0.1368 |
| `lingbot_gct_b04` | 1 999 300 | 0.6066 | 0.5071 | 0.7150 | 0.7901 | **−1.1 %** | **0.4308** | 0.2053 | **0.1534** |
| `lingbot_gct_mid` | 1 999 300 | **0.6162** | 0.5023 | 0.6609 | 0.8070 | **+1.0 %** | 0.4008 | 0.2026 | 0.1442 |
| `lingbot_gct_b17` | 1 999 300 | 0.5474 | 0.4256 | 0.6714 | 0.8320 | **+4.2 %** | 0.3847 | 0.1964 | 0.1350 |
| `lingbot_gct_final` | 1 999 300 | 0.5155 | 0.3920 | 0.6581 | 0.8545 | **+7.0 %** | 0.3873 | 0.2011 | 0.1042 |
| `external_dino_teacher` (ref.) | 0 | 1.0000 | 1.0000 | 0.7772 | 0.7265 | — | 0.4101 | 0.1823 | 0.0838 |

**Gate arithmetic** (required: consistency ≥ 1.15, fidelity ≥ 0.95):

| representation | consistency | fidelity | crit 1 | crit 2 |
|---|---:|---:|:--:|:--:|
| `lingbot_gct_b04` | 0.989 | 0.994 | ✗ | ✓ |
| `lingbot_gct_mid` | 1.010 | 1.009 | ✗ | ✓ |
| `lingbot_gct_b17` | 1.042 | 0.897 | ✗ | ✗ |
| `lingbot_gct_final` | **1.070** | 0.844 | ✗ | ✗ |

Same shape as KITTI: the layers that keep semantics do not move consistency, and the
layer that moves consistency most loses ~16 % of teacher fidelity. `lingbot_gct_b04` is
actually **worse** than the encoder on consistency here (−1.1 %).

### 4.1 The anisotropy artefact reproduces

Raw tokens, no probe:

| representation | cross-view cos | vs encoder | negative control | **margin** | vs encoder | recall@1 | diversity |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lingbot_encoder` | 0.8579 | — | 0.6466 | **0.2113** | — | 0.1993 | 0.4030 |
| `lingbot_gct_b04` | 0.8889 | +3.6 % | 0.7343 | 0.1546 | −26.8 % | 0.2030 | 0.3110 |
| `lingbot_gct_mid` | 0.9341 | +8.9 % | 0.8702 | 0.0639 | −69.8 % | 0.1833 | 0.1470 |
| `lingbot_gct_b17` | 0.9589 | +11.8 % | 0.9046 | 0.0544 | −74.3 % | 0.1746 | 0.1178 |
| `lingbot_gct_final` | 0.9896 | **+15.4 %** | 0.9761 | **0.0135** | **−93.6 %** | 0.1736 | 0.0301 |
| `external_dino_teacher` | 0.7265 | — | 0.3164 | 0.4101 | — | 0.1823 | 0.7774 |

Again the raw metric clears +15 % while the margin collapses by 93.6 %, diversity falls
13-fold and recall@1 *drops*. Two independent domains, the same artefact, the same
magnitude. This is now a property of the GCT stack, not of driving scenes.

### 4.2 Predicted vs oracle geometry

| representation | predicted | oracle | Δ | predicted recall@1 | oracle recall@1 |
|---|---:|---:|---:|---:|---:|
| `lingbot_encoder` | 0.7986 | 0.8638 | +0.0651 | 0.2009 | 0.3350 |
| `lingbot_gct_mid` | 0.8070 | 0.8765 | +0.0696 | 0.2026 | 0.3396 |
| `lingbot_gct_final` | 0.8545 | 0.8960 | +0.0416 | 0.2011 | 0.3200 |

| correspondence set | matches | valid fraction |
|---|---:|---:|
| predicted | 79 998 | 0.368 |
| oracle | **170 360** | **0.761** |

Replica's oracle geometry is far richer than KITTI's (76.1 % of patches valid vs
31.8 %), because rendered depth is dense and exact and the baselines are short. Even so,
the GCT gain under oracle geometry is only **+3.7 %** (KITTI: +3.0 %). Better geometry
shrinks the effect on both datasets — so `FAIL_CONSISTENCY` is the right verdict and
`FAIL_GEOMETRY` is not.

### 4.3 Teacher budget — the one place the datasets disagree

| representation | @10 % | @25 % | @100 % | retention |
|---|---:|---:|---:|---:|
| `lingbot_encoder` | 0.5305 | 0.5831 | 0.6105 | 86.9 % |
| `lingbot_gct_b04` | 0.5294 | 0.5743 | 0.6066 | 87.3 % |
| `lingbot_gct_mid` | **0.5579** | **0.5990** | **0.6162** | **90.5 %** ✓ |
| `lingbot_gct_b17` | 0.4965 | 0.5320 | 0.5474 | **90.7 %** ✓ |
| `lingbot_gct_final` | 0.4691 | 0.5042 | 0.5155 | **91.0 %** ✓ |

**`SPARSE_PASS`: met on Replica** (91.0 % ≥ 90 % for the best representation), where on
KITTI every representation sat at 85–88 % and it was not met. All three deeper GCT
layers clear the bar while the encoder and block 4 do not, so this is a real ordering,
not noise.

**Read it carefully, because it is easy to over-claim.** Retention is measured against
each representation's *own* 100 % ceiling. `lingbot_gct_final` retains 91.0 % of 0.5155,
i.e. **0.4691 absolute** — which is *worse* than the encoder's 10 %-budget result of
**0.5305**. Deeper GCT tokens are more label-efficient in the relative sense and still
less useful in the absolute sense. The one representation that is better on both counts
is `lingbot_gct_mid`: 0.5579 absolute at 10 % (best of any representation) with 90.5 %
retention, and it costs nothing in fidelity (ratio 1.009). That is the only genuinely
positive finding in either study, and it is about label efficiency — **not** about
cross-view consistency, which is what the gate tested.

### 4.4 Depth-confidence filtering — resolves a KITTI anomaly

| representation | no filter | keep top 75 % conf. | Δ (Replica) | Δ (KITTI) |
|---|---:|---:|---:|---:|
| `lingbot_encoder` | 0.7986 | 0.8114 | **+0.0127** | −0.0097 |
| `lingbot_gct_mid` | 0.8070 | 0.8222 | **+0.0152** | −0.0100 |
| `lingbot_gct_final` | 0.8545 | 0.8679 | **+0.0134** | −0.0078 |

On KITTI, confidence filtering was uniformly *unhelpful* and I flagged it as an
unexplained anomaly. On Replica it uniformly **helps**. The natural reading: LingBot's
depth confidence carries real signal indoors, where the whole scene is within a few
metres, but not on outdoor driving frames dominated by sky and far structure where
confidence is low almost everywhere useful. The filter is domain-dependent; it reorders
nothing on either dataset.

### 4.5 Temporal band

| representation | short (1–3) | long (8–24) | gap 1 | gap 24 |
|---|---:|---:|---:|---:|
| `lingbot_encoder` | 0.8357 | 0.5635 | 0.8919 | 0.4610 |
| `lingbot_gct_b04` | 0.8314 | **0.5274** | 0.8908 | **0.4032** |
| `lingbot_gct_mid` | 0.8448 | 0.5669 | 0.9001 | 0.4472 |
| `lingbot_gct_final` | **0.8876** | **0.6438** | 0.9254 | 0.5079 |

Under large rotation (gap 24 ≈ 31°), `lingbot_gct_b04` is clearly **worse** than the
frame-independent encoder (0.4032 vs 0.4610). Early cross-frame attention does not help
across wide viewpoint change indoors; only the deepest layer does, and it pays for it in
fidelity.

### 4.6 Efficiency

LingBot **61.2 ms/frame** (vs 44.1 on KITTI — 777 vs 407 tokens), teacher 1.36 ms/frame,
probe 1.60 ms/frame, cache 9.09 GiB for 576 frames (16.2 MiB/frame), peak GPU 14.44 GiB,
15 probes trained in 460 s.

## 5. Cross-dataset comparison

| representation | KITTI consistency | KITTI fidelity | Replica consistency | Replica fidelity |
|---|---:|---:|---:|---:|
| `lingbot_gct_b04` | 1.009 | 0.999 | 0.989 | 0.994 |
| `lingbot_gct_mid` | 1.017 | 0.990 | 1.010 | 1.009 |
| `lingbot_gct_b17` | 1.049 | 0.841 | 1.042 | 0.897 |
| `lingbot_gct_final` | **1.072** | 0.842 | **1.070** | 0.844 |

| quantity | KITTI (outdoor) | Replica (indoor) |
|---|---:|---:|
| patch grid | 11×37 | 21×37 |
| motion regime | translation-dominated (~34 m @ gap 24) | rotation-dominated (~31° @ gap 24) |
| valid corr., predicted | 0.160 | 0.368 |
| valid corr., oracle | 0.318 | 0.761 |
| best GCT consistency gain | **+7.2 %** | **+7.0 %** |
| best GCT gain, oracle geometry | +3.0 % | +3.7 % |
| raw-token gain, `gct_final` | +22.6 % | +15.4 % |
| raw-token **margin**, `gct_final` | **−94.6 %** | **−93.6 %** |
| `SPARSE_PASS` | no (87.9 %) | **yes (91.0 %)** |
| confidence filtering | hurts (−0.010) | helps (+0.013) |
| **verdict** | `FAIL_CONSISTENCY` | `FAIL_CONSISTENCY` |

## 6. Limitations and anomalies

1. **Still one teacher and one seed per configuration.** The cross-dataset agreement is
   a much stronger check than a second seed would have been, but no variance estimate
   exists for within-dataset contrasts of ~1 %.
2. **`office3` unused.** Held back so the train/val frame budget matched KITTI exactly.
3. **Held-out is a single scene** (`office4`, 192 frames from three chunks). Scene-level
   diversity in the validation set is lower than KITTI's, where one long sequence covers
   varied streets.
4. **Anomaly — the teacher has the *worst* depth-boundary margin on Replica** (0.0838,
   below every probe) while it had the *best* on KITTI (0.1197, above every probe). The
   depth-discontinuity definition is structural, not semantic; indoor scenes are full of
   depth edges that are not semantic edges (a wall receding behind a chair), so DINO
   features legitimately do not break there. This is a limitation of the label-free
   proxy indoors, not evidence that the probes beat the teacher semantically.
5. **Anomaly — `gct_b04` inverts between datasets** (+0.9 % on KITTI, −1.1 % on
   Replica), and is the worst representation at long gaps on Replica while having the
   best margin and boundary margin. Early cross-frame attention appears to help under
   forward translation and hurt under rotation. Single-seed, unexplained, and worth
   knowing before anyone picks block 4.
6. **Replica is rendered, not photographed** — no motion blur beyond what is baked in,
   no exposure change, no rolling shutter. It is an easy domain for geometry, which is
   exactly why its oracle correspondence yield is 76 %.

## 7. Verdict and recommendation

**`FAIL_CONSISTENCY`, replicated.** Best GCT consistency gain +7.0 % against a required
+15 %; +3.7 % under oracle geometry. The KITTI conclusion was not a property of driving
scenes, forward motion, or sparse LiDAR depth — it reproduces to within 0.2 percentage
points in an indoor, rotation-dominated, densely-supervised domain.

The one thing Replica adds that KITTI could not: **`SPARSE_PASS` is met**, and
`lingbot_gct_mid` is the only representation that is better than the encoder on *both*
absolute sparse-teacher performance (0.5579 vs 0.5305 at a 10 % budget) and fidelity
(ratio 1.009) at zero consistency cost. That is a real, small, positive result about
**label efficiency** — a different claim from the one the gate tested, and not enough to
reopen it.

**Per the gate: stop.** Do not proceed to geometry-aware frame selection, persistent
semantic memory, 3D target retrieval, or paper writing on the geometric-consistency
premise.

**Single next action justified by the evidence, unchanged and now better supported:**
the binding constraint is **geometry, not layer choice**. Predicted geometry recovers
only 36.8 % of patches on Replica and 16.0 % on KITTI, against oracle's 76.1 % and
31.8 %, and predicted-vs-oracle recall@1 differs by 1.7× and 2.7×. That gap dwarfs every
representation-level difference measured in either study. If the streaming
open-vocabulary mapping goal is still worth pursuing, characterise where predicted
depth/pose correspondence yield collapses as a function of baseline and rotation, and
read features from `lingbot_gct_mid` — the one layer that costs nothing in fidelity,
clears `SPARSE_PASS`, and is the strongest representation at a 10 % teacher budget.
