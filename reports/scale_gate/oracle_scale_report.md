# Phase 3 — oracle scale diagnosis

**Decision: proceed to Phase 4, but with a small prize.** Scale is *necessary* — without
it occupancy IoU is exactly 0.0000. A single global constant recovers **81.8 %** of the
oracle result, leaving only **+0.0091 IoU** for any per-clip method to win.

Artifacts: `artifacts/scale_gate/oracle_metrics_per_clip.csv`,
`oracle_summary.json`, `scale_targets_{val,train}.csv`,
`figures/oracle_scale_diagnosis.png`.

## 1. Does one scalar align both depth and translation?

**Yes, decisively.** Over 924 clips the depth-derived and pose-derived scales agree to a
median `|log(s_depth/s_pose)|` of **0.024** (train) and **0.029** (val) — about 2.5 %.
The p90 is 0.065; nothing approaches the 0.20 reliability threshold.

| split | clips | reliable | agreement median | agreement p90 | s range | s median |
|---|---:|---:|---:|---:|---|---:|
| train (00–07, 09, 10) | 763 | 745 | 0.0239 | 0.0650 | 13.9 – 75.4 | 27.4 |
| val (08) | 163 | 162 | 0.0288 | 0.0675 | 14.6 – 38.2 | 26.3 |

Within a clip the per-frame depth scale is almost constant: the median standard deviation
of per-frame `log s` is **0.018** (1.8 %). One scalar per 5-frame clip is well posed.

This is the finding that clears the plan's stop condition — the coordinate contract,
the depth convention and the pose direction are all mutually consistent. Had they not
been, this number would be large and Phase 4 would have been forbidden.

18 train clips and 1 val clip were flagged unreliable (insufficient motion or LiDAR
pixels); they are listed by id and reason in `scale_targets_summary.json` and excluded
from target fitting, never silently dropped.

## 2. How much of the occupancy failure is scale?

Held-out sequence 08, 163 clips, identical grid / evaluator / frame selection / confidence
filter for every row. Bootstrap: 10 000 resamples, seed 0.

| method | deployable | IoU | precision | recall | median \|log s err\| | ΔIoU vs raw [95 % CI] |
|---|---|---:|---:|---:|---:|---|
| `raw_canonical` | yes | **0.0000** | 0.004 | 0.000 | 3.268 | — |
| `global_median_train` | yes | **0.0410** | 0.256 | 0.047 | 0.131 | +0.0410 [+0.0385, +0.0434] * |
| `oracle_pose` | ORACLE | 0.0485 | 0.296 | 0.055 | 0.014 | +0.0485 [+0.0461, +0.0509] * |
| `oracle_joint` | ORACLE | 0.0501 | 0.303 | 0.057 | 0.000 | +0.0501 [+0.0478, +0.0524] * |
| `oracle_depth` | ORACLE | **0.0514** | 0.308 | 0.059 | 0.014 | +0.0514 [+0.0491, +0.0536] * |

\* CI excludes zero.

### The denominator: what this protocol can achieve at all

Absolute IoU on this benchmark is not interpretable on its own. The ground truth is
accumulated from multi-pass LiDAR **including future frames**, while the protocol observes
one forward-facing camera over five causal frames. Replacing LingBot's depth *and* poses
with ground truth, through the identical clips, grid, voxeliser and metric
(`tools/scale_gate/eval_ceiling.py`):

| | IoU | precision | recall |
|---|---:|---:|---:|
| **ceiling — GT LiDAR depth + GT poses** | **0.1255** | 0.938 | 0.128 |

A perfect sensor recovers only 12.8 % of the labelled voxels here. Every method should
therefore be read as a fraction of 0.1255, never as a bare number:

| method | IoU | % of ceiling |
|---|---:|---:|
| `raw_canonical` | 0.0000 | 0.0 % |
| `global_median_train` | 0.0410 | **32.6 %** |
| `combined_mlp` (learned) | 0.0437 | 34.8 % |
| `depth_mlp` (learned) | 0.0440 | 35.0 % |
| `oracle_pose` | 0.0485 | 38.6 % |
| `oracle_joint` | 0.0501 | 39.9 % |
| `oracle_depth` | 0.0514 | **40.9 %** |
| ceiling | 0.1255 | 100 % |

This splits the loss into three gaps of very different size:

| gap | size | nature |
|---|---:|---|
| 0 → 32.6 % | 32.6 pts | **scale** — closed by one free constant |
| 32.6 % → 40.9 % | **8.3 pts** | **per-clip scale** — the entire prize this gate competes for |
| 40.9 % → 100 % | **59.1 pts** | depth precision and unrecovered geometry |

The third gap is **seven times** the second. Even a perfect per-clip scalar leaves the
method at 41 % of what the same pipeline achieves with a real sensor. Precision is the
visible symptom: 0.256–0.308 for scaled LingBot against the ceiling's 0.938.

**Scale is the difference between nothing and something.** In canonical units the
reconstruction is ~27× too small, lands almost entirely in one grid cell, and scores
IoU 0.0000 with precision 0.004. Applying any sane scalar fixes that.

**But per-clip refinement is a small effect.** The decision-relevant contrasts:

| contrast | mean ΔIoU | 95 % CI | win rate |
|---|---:|---|---:|
| `global_median` − `raw` | +0.0410 | [+0.0385, +0.0434] * | 100.0 % |
| `oracle_joint` − `global_median` | **+0.0091** | [+0.0063, +0.0119] * | 68.1 % |
| `oracle_depth` − `global_median` | +0.0104 | [+0.0076, +0.0133] * | 74.8 % |
| `oracle_depth` − `oracle_pose` | +0.0029 | [+0.0020, +0.0039] * | 66.0 % |

A deployment-legal constant, fitted only on source-training clips, already reaches
**81.8 %** of the joint-oracle IoU. Its typical scale error is 14 % (median `|log err|`
0.131), and that 14 % costs only 0.009 IoU.

## 3. Which oracle definition is best?

`oracle_depth` (0.0514) beats `oracle_joint` (0.0501) and `oracle_pose` (0.0485), all
separations significant. This is expected: the downstream metric is voxel occupancy built
from **depth**, so a scalar fitted to depth residuals optimises the thing being measured;
the pose-fitted scalar is the physically independent check, not the best regressor.

The joint solution is kept as the training target because it is the one physically
constrained by both signals, and because using the depth-only target would make the
learned model's evaluation partly circular.

## 4. Why did the previous SemanticKITTI oracle (0.081) not exceed the causal result (0.087)?

Three concrete reasons, all verifiable in the earlier artifacts:

1. **It was not a per-clip oracle.** `Z_oracle_trajectory_scale` fits **one Sim(3) scale
   per 500-frame chunk** via `umeyama_sim3` over the whole trajectory. A single scalar
   across 500 frames cannot track the 14.6–38.2 range this study measures *within* seq 08.
2. **Sim(3) alignment absorbs scale**, which the plan explicitly warns against. The
   quantity it minimises is trajectory ATE after scaling, not depth or relative-translation
   residual, so it is not the scalar that optimises reconstruction.
3. **The comparison mixed horizons.** The 0.087 headline is `iou_h100` for the best causal
   config while the 0.081 oracle figure is `iou_h20`. At matched horizon the ordering
   reverses: 0.0869 vs 0.0841 at h=100, 0.0840 vs 0.0813 at h=20 — the oracle is *below*
   the running anchor in both, which is the real anomaly and is explained by (1) and (2).

This study's per-clip oracle does not have that defect: it beats its deployable baseline
on every definition, with CIs excluding zero.

## 5. Is training a learned scale predictor justified?

**Yes, narrowly.** The oracle improves on the deployable baseline by +0.0091 IoU with a CI
excluding zero, so the plan's condition ("if oracle scale improves occupancy, proceed to
Phase 4") is met. But the prize is bounded at 18 % of the oracle result, and any learned
model must be judged against that ceiling rather than against raw canonical scale.

## 6. Caveat on absolute levels

These IoU values (0.04–0.05) are lower than the previous study's 0.087 because a 5-frame
clip at stride 5 accumulates 5 frames, against that study's 20-frame causal history. Fewer
accumulated frames means lower recall and lower IoU. The comparison **within** this table
is like-for-like; the comparison **to** 0.087 is not, and no claim is made on it.

**A consistency check that supports the pipeline.** That earlier history sweep, on the same
grid, measured LingBot at 0.0207 (h=1), **0.0523 (h=5)**, 0.0840 (h=20) and 0.0869 (h=100).
This study's `oracle_depth` of **0.0514** on 5-frame clips lands essentially on the
independent h=5 point. The low absolute number is the protocol, not a defect in this
implementation.
