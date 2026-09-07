# Gate 0 — final report: metric scale for LingBot-Map

## Recommendation: `USE_SIMPLE_SCALE_BASELINE_AND_PROCEED`

Scale is **necessary but not the bottleneck**. Against the protocol ceiling of **0.1255**
(GT LiDAR depth + GT poses through the identical pipeline): canonical units reach 0 %, a
free source-fitted constant reaches **32.6 %**, a learned predictor **34.8 %**, and a
per-clip oracle **40.9 %**. So the entire remaining scale prize is **8.3 points of
ceiling**, while **59.1 points** lie beyond scale altogether. Coordinates are sound, so
`STOP_AND_FIX_COORDINATES` does not apply, and an 8.3-point prize does not justify
`ESCALATE_TO_METRIC_HEAD_FINETUNING`.

## 1. What was implemented

`scale_gate/` (library): `scale.py` robust log-space estimators, `kitti.py` calibration /
manifests / LiDAR projection, `features.py` inference-legal features, `config.py`
config + provenance. `tools/scale_gate/`: `preflight`, `prepare_manifest`,
`project_lidar`, `cache_lingbot`, `build_scale_targets`, `eval_oracle_scale`,
`train_scale`, `eval_scale`, `eval_ceiling`, `plot_oracle`. `tests/scale_gate/`: **49 tests, all passing**.

## 2. Exact commands run

```bash
export PYTHONPATH=$PWD CUDA_HOME=/usr/local/cuda-12.8 FLASHINFER_CUDA_ARCH_LIST="12.0a"
PY=/home/minh/anaconda3/envs/cu128/bin/python

$PY tools/scale_gate/preflight.py        --config configs/scale_gate/semantickitti.yaml
$PY tools/scale_gate/prepare_manifest.py --config configs/scale_gate/semantickitti.yaml
$PY tools/scale_gate/project_lidar.py    --config configs/scale_gate/semantickitti.yaml --split train --qa-only --qa-frames 20
$PY tools/scale_gate/project_lidar.py    --config configs/scale_gate/semantickitti.yaml --split val
$PY tools/scale_gate/project_lidar.py    --config configs/scale_gate/semantickitti.yaml --split train
$PY tools/scale_gate/cache_lingbot.py    --config configs/scale_gate/semantickitti.yaml --split smoke
$PY tools/scale_gate/cache_lingbot.py    --config configs/scale_gate/semantickitti.yaml --split val
$PY tools/scale_gate/cache_lingbot.py    --config configs/scale_gate/semantickitti.yaml --split train
$PY tools/scale_gate/build_scale_targets.py --config configs/scale_gate/semantickitti.yaml
$PY tools/scale_gate/eval_oracle_scale.py   --config configs/scale_gate/semantickitti.yaml --split val
$PY tools/scale_gate/train_scale.py --config configs/scale_gate/semantickitti.yaml --model combined_mlp --overfit 16
$PY tools/scale_gate/train_scale.py --config configs/scale_gate/semantickitti.yaml --model combined_mlp --smoke
for m in global_median depth_mlp image_mlp combined_mlp; do
  $PY tools/scale_gate/train_scale.py --config configs/scale_gate/semantickitti.yaml --model $m; done
for m in depth_mlp image_mlp combined_mlp; do
  $PY tools/scale_gate/eval_scale.py --config configs/scale_gate/semantickitti.yaml \
     --checkpoint artifacts/scale_gate/scale_models/${m}_best.pt; done
$PY tools/scale_gate/eval_ceiling.py --config configs/scale_gate/semantickitti.yaml --split val
$PY tools/scale_gate/plot_oracle.py --config configs/scale_gate/semantickitti.yaml
$PY -m pytest tests/scale_gate -q
```

## 3. Data used and excluded

| split | sequences | clips | frames | note |
|---|---|---:|---:|---|
| train | 00–07, 09, 10 | 763 | 3 815 | 745 reliable; 18 excluded |
| val | 08 | 163 | 815 | 162 reliable; 1 excluded |

Clips are 5 frames at stride 5 (≈ 2 Hz), non-overlapping, chronological. Model selection
used an **inner** split (seqs 09, 10) so seq 08 was never seen during training or
checkpointing. Exclusions are per-clip with reasons in `scale_targets_summary.json`;
none were silent. Semantic labels were never used — the occupancy target is binary.

## 4. Assumptions

* One positive scalar per clip; long-window drift is out of scope (plan constraint).
* Depth and camera translation always receive the same scalar; rotations and intrinsics never.
* LiDAR is treated as metric ground truth for depth targets, `poses.txt` as metric for pose targets.
* Clips whose depth- and pose-derived scales disagree by `|log ratio| > 0.20` are unreliable.
* IoU is compared **within** this study's 5-frame protocol; not against the previous 20-frame numbers.

## 5. Conventions (verified, not assumed)

Depth is **optical-axis z depth**; `depth_conf` is `1+exp(x)`, a precision > 1; poses are
stored as explicitly named `pred_pose_c2w` (camera-to-world) after inverting the
world-to-camera output of `pose_encoding_to_extri_intri`; axes are OpenCV. `P2` is a
projection matrix — `K = P2[:3,:3]`, baseline → `t_cam2`; `poses.txt` is cam0-to-world and
is offset to cam2. Preprocessing is a pure resize for every KITTI resolution, asserted at
load time. Full detail in `preflight_inventory.md`.

**Calibration QA: PASS.** 20 deterministic overlays across 10 sequences at both native and
processed resolution (`artifacts/scale_gate/qa/semantickitti_projection/`); LiDAR locks
onto poles, facades and vehicles, the near→far gradient is correct, sky is empty,
17 000–20 000 valid pixels per frame over 2–80 m.

## 6. Scale metrics

| split | agreement median | p90 | per-frame log-s std | s range | s median |
|---|---:|---:|---:|---|---:|
| train | 0.0239 | 0.0650 | 0.018 | 13.9 – 75.4 | 27.4 |
| val | 0.0288 | 0.0675 | 0.018 | 14.6 – 38.2 | 26.3 |

One scalar is physically coherent: depth- and pose-derived estimates agree to ~2.5 %.

## 7. Occupancy metrics — held-out sequence 08

| method | deployable | IoU | P | R | median \|log s err\| | ΔIoU vs `global_median` [95 % CI] |
|---|---|---:|---:|---:|---:|---|
| `raw_canonical` | yes | 0.0000 | 0.004 | 0.000 | 3.268 | −0.0410 |
| `global_median_train` (s = 27.35) | yes | 0.0410 | 0.256 | 0.047 | 0.131 | — |
| `image_mlp` | yes | 0.0407 | — | — | 0.094 | −0.0003 [−0.0032, +0.0026] |
| `combined_mlp` | yes | **0.0437** | — | — | 0.074 | **+0.0027 [+0.0000, +0.0053]** |
| `depth_mlp` | yes | **0.0440** | — | — | 0.077 | **+0.0030 [+0.0002, +0.0058]** |
| `oracle_pose` | ORACLE | 0.0485 | 0.296 | 0.055 | 0.014 | +0.0075 |
| `oracle_joint` | ORACLE | 0.0501 | 0.303 | 0.057 | 0.000 | +0.0091 |
| `oracle_depth` | ORACLE | 0.0514 | 0.308 | 0.059 | 0.014 | +0.0104 |
| **ceiling** — GT LiDAR + GT poses | reference | **0.1255** | 0.938 | 0.128 | — | +0.0845 |

### Reading these numbers

Absolute IoU is not interpretable here: the ground truth is accumulated from multi-pass
LiDAR **including future frames**, while the protocol sees one forward camera over five
causal frames, so even a perfect sensor recovers just 12.8 % of the labelled voxels
(`tools/scale_gate/eval_ceiling.py`, `artifacts/scale_gate/ceiling_summary.json`).
Read every method as a fraction of the 0.1255 ceiling:

| method | IoU | % of ceiling |
|---|---:|---:|
| `raw_canonical` | 0.0000 | 0.0 % |
| `global_median_train` (free) | 0.0410 | **32.6 %** |
| `combined_mlp` (learned) | 0.0437 | 34.8 % |
| `depth_mlp` (learned) | 0.0440 | 35.0 % |
| `oracle_depth` (cheating) | 0.0514 | **40.9 %** |
| ceiling | 0.1255 | 100 % |

| gap | size | nature |
|---|---:|---|
| 0 → 32.6 % | 32.6 pts | **scale** — closed by one free constant |
| 32.6 % → 40.9 % | **8.3 pts** | per-clip scale: the whole prize this gate competes for |
| 40.9 % → 100 % | **59.1 pts** | depth precision and unrecovered geometry |

The third gap is seven times the second, which is what makes the recommendation
straightforward. Precision is the visible symptom: 0.26–0.31 for scaled LingBot against
the ceiling's 0.94.

**Do not quote 0.044 without the ceiling beside it** — alone it reads as a method that does
not work. The defensible phrasing is *"35 % of what the same pipeline achieves with
ground-truth LiDAR and ground-truth poses."*

## 8. Does the learned model pass?

Against plan 4.6, for `combined_mlp` (the proposed minimal model, 608 621 params):

| criterion | result |
|---|---|
| statistically beats the source-trained global median on held-out sequences | **yes** — Wilcoxon signed-rank p = 0.0009; bootstrap CI [+0.0000, +0.0053] |
| improves downstream occupancy toward the oracle | **yes** — recovers 29.9 % of the +0.0091 headroom |
| uses no target-domain statistics or metadata | **yes** — enforced by `scale_gate/features.py` and asserted by a test whose fake cache raises if a target-only key is read |
| not driven by one sequence or a few outlier clips | **yes** — 65.0 % of clips improve, median Δ +0.0028, and a symmetric 10-clip trim gives +0.0033 [+0.0015, +0.0051] |
| predictions positive, finite, calibrated | **yes** — 13.9–34.8, corr(ŝ, s\*) = 0.737, scale error median 0.131 → 0.074 |

**The learned model passes all five criteria.** `depth_mlp` (74 561 params) does marginally
better downstream (+0.0030) with a weaker signed-rank result (p = 0.043); `image_mlp`
(pre-GCT features alone) shows **no** effect — consistent with the previous study's finding
that LingBot tokens are not a reliable stand-alone signal, and the reason the plan forbade
making them the only input.

## 9. Failures and anomalies

1. **A one-sided robustness test initially misled me.** Dropping only the 10 *best* clips
   collapsed the gain to +0.0004 (CI including zero), which looked like outlier-driven
   success. That test removes positive outliers while keeping negative ones and is biased
   against the model. The symmetric trim and the signed-rank test — both reported above —
   show the improvement is broad-based. The one-sided number is recorded here because it is
   the kind of statistic that would have produced a wrong verdict.
2. **`oracle_depth` beats `oracle_joint`.** Expected — the downstream metric is built from
   depth, so a depth-fitted scalar optimises it. The joint target is nevertheless used for
   training to avoid circularity.
3. **The previous oracle (0.081 < 0.087) is explained**, not reproduced: it was a single
   Sim(3) trajectory scale per 500-frame chunk, and Sim(3) absorbs scale. See
   `oracle_scale_report.md` §4.
4. **Absolute IoU here (0.04–0.05) is below the previous 0.087** because this protocol
   accumulates 5 frames rather than 20. Not comparable; no claim made.
5. **Occ3D-nuScenes could not be scored** — nuScenes images and poses are present but all
   semantic/occupancy ground truth is absent. Requirements in `docs/data_setup_scale_gate.md`.

## 10. Comparison with the previous 0.087 / 0.068

Neither is contradicted. The 0.087 SemanticKITTI figure uses a 20-frame causal history at
stride 1 and a running depth-scale anchor; this gate uses 5-frame clips at stride 5. What
this study adds is the *decomposition*: of that pipeline's performance, essentially all of
it comes from having **some** metric scale (0.0000 → 0.0410), and at most 18 % more is
available from making the scale per-clip optimal.

## 11. Is scale a major downstream bottleneck?

**No — it is a prerequisite, not a bottleneck.** Getting scale roughly right is
non-negotiable: without it the reconstruction collapses to IoU 0. Getting it *exactly*
right is worth 8.3 points of ceiling, against the 59.1 points that lie beyond scale.
A fitted constant is free and already reaches 32.6 % of 40.9 %, so the remaining occupancy
failure is depth precision and unrecovered geometry — consistent with the separate finding
that predicted geometry recovers only 16 % of GT-occupied voxels against a LiDAR ceiling of
31.8 % on the 20-frame protocol.

## 12. Deliverable checklist

| plan item | status |
|---|---|
| tested SemanticKITTI preparation pipeline | done — 49 tests |
| validated frozen LingBot cache | done — 924/926 clips, hash-stamped, resumable, 27–29 ms/frame |
| depth / pose / joint oracle scale | done |
| correct application to depth **and** pose translations | done, unit-tested |
| raw / median / oracle downstream comparison | done |
| bootstrap confidence intervals | done, 10 000 resamples, seed 0 |
| protocol ceiling (GT depth + GT poses) | done — 0.1255, the denominator for every method |
| learned baselines A–C + combined | done |
| cross-dataset adapter/setup documentation | `docs/data_setup_scale_gate.md`, `tools/check_{ddad,vkitti2}_layout.py` |
| tests and reproducible commands | done |

Cross-dataset **training** (Phase 5 proper) was not run: DDAD, VKITTI2 and PandaSet are
not on this machine and the plan forbids downloading them. Adapters are documented and
layout checkers provided.

## 13. Next action

Do **not** invest further in scale. Adopt `global_median_train` (s = 27.35) as the default
and keep `combined_mlp` as an optional +2.1-point refinement. Attack **depth precision**
instead: at 0.2 m voxels the scaled reconstruction reaches precision 0.26–0.31 against the
ceiling's 0.94, and that gap holds 59.1 of the 67.4 points of ceiling still unclaimed.
Measure it as a fraction of the ceiling from now on — the absolute IoU on this benchmark
is dominated by observability, not by method quality.

**Per the plan, the next project stage was not started.**
