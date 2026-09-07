# Gate 2 — minimal depth refinement

## Recommendation: `DEPTH_REFINEMENT_PASSES`

All seven Gate-2 criteria are met. **But the mechanism is not what the gate was designed
to test:** the head recovers per-clip *metric scale* almost entirely, and adds very little
depth *shape* correction. The 33.1 % depth-shape gap that Gate 1 identified remains open.
That qualification is developed in §6 and must travel with this verdict.

> **These are in-domain SemanticKITTI results.** Training sequences 00–07, selection 09–10,
> validation 08 — one dataset, one sensor, one city. Nothing here is cross-dataset or
> demonstrated to generalise.

## 1. Data and splits

| role | sequences | frames | used for |
|---|---|---:|---|
| train | 00–07 | 3 170 | gradient updates and normalisation statistics |
| selection | 09, 10 | 550 | early stopping only |
| **validation** | **08** | **815** | reported metrics only |

Splits are by **complete sequence**. Sequence 08 was not used for training, early stopping,
architecture choice, thresholds or normalisation statistics — asserted at run start
(`assert "08" not in train|select sequences`). Semantic labels are never loaded.

Reused verbatim from Gate 0/1: repaired LingBot caches, projected metric LiDAR, calibration,
clip manifests, the validated `mode="crop"` resize, and the repaired `pred_pose_c2w`.
RGB was cached once at the processed 154×518 resolution through the same transform.

The model receives only `rgb`, `lingbot_depth`, `lingbot_confidence`, `valid_lingbot_mask`
and normalised image coordinates. It never sees sequence id, dataset id, filename, LiDAR
counts or any ground-truth statistic; a unit test mutates the LiDAR target and asserts the
network input is bit-identical.

## 2. Model and losses

```
D_base    = s_hat * D_lingbot          s_hat = 27.3665  (Gate-1 A3 deployable constant)
D_refined = D_base * exp(r_theta)      r = 0.7 * tanh(raw)
```

The tanh bound makes the refined depth finite and strictly positive by construction, and
the output head is zero-initialised so training starts exactly at the unrefined depth.

| run | arch | params | in-ch | best epoch | selection AbsRel | train s | peak GPU |
|---|---|---:|---:|---:|---:|---:|---:|
| `identity` | Baseline 0 | 0 | — | — | 0.2088 | 0 | — |
| **`depth_cnn`** | Baseline 1, residual CNN | **48 129** | 5 | 20 | **0.0855** | 501 | 1.37 GiB |
| `rgbd_unet` | Baseline 2, U-Net + RGB | 474 273 | 8 | 7 | 0.0906 | 276 | 2.12 GiB |
| `rgbd_unet_cw` | + confidence weighting | 474 273 | 8 | 7 | 0.0961 | 278 | 2.12 GiB |
| `rgbd_unet_rel` | + relative loss | 474 273 | 8 | 17 | 0.0850 | 467 | 2.12 GiB |

Primary loss, on valid projected-LiDAR pixels only:

```
L = SmoothL1(log D_refined[valid], log D_lidar[valid], beta=0.1) + 0.05 * mean|r|
```

Ablated separately: `L_relative = mean(|D_refined - D_lidar| / D_lidar)`; confidence
weighting. Baseline 3 (dense LingBot features) was **not** implemented — no spatially
aligned dense feature map is accessible without modifying LingBot, and Baselines 1–2 do not
justify adding one.

## 3. Depth metrics — sequence 08, 14.7 M valid LiDAR pixels

| run | AbsRel | log RMSE | RMSE (m) | δ1 | median abs err (m) |
|---|---:|---:|---:|---:|---:|
| base (identity) | 0.2046 | 0.2592 | 5.164 | 0.7055 | 1.570 |
| **`depth_cnn`** | **0.0898** | **0.1714** | **3.864** | **0.9158** | **0.485** |
| `rgbd_unet_rel` | 0.0883 | 0.1749 | 3.930 | 0.9151 | 0.497 |
| `rgbd_unet` | 0.0950 | 0.1773 | 3.962 | 0.9125 | 0.537 |
| `rgbd_unet_cw` | 0.1024 | 0.1820 | 4.117 | 0.9095 | 0.569 |

Paired clip-level bootstrap on ΔAbsRel (10 000 resamples, seed 0):

| run | ΔAbsRel | 95 % CI | clips improved |
|---|---:|---|---:|
| `depth_cnn` | −0.1151 | [−0.1360, −0.0953] \* | 95.1 % |
| `rgbd_unet_rel` | −0.1165 | [−0.1377, −0.0966] \* | 94.5 % |
| `rgbd_unet` | −0.1098 | [−0.1308, −0.0901] \* | 90.8 % |
| `rgbd_unet_cw` | −0.1024 | [−0.1226, −0.0832] \* | 87.1 % |

### By distance (`depth_cnn`)

| range | AbsRel base → refined | δ1 base → refined | valid pixels |
|---|---:|---:|---:|
| 0–10 m | 0.2220 → **0.0766** | 0.6954 → 0.9515 | 6 406 924 |
| 10–20 m | 0.1914 → **0.0821** | 0.7292 → 0.9314 | 5 411 933 |
| 20–40 m | 0.1847 → **0.1180** | 0.7175 → 0.8567 | 2 218 992 |
| 40–80 m | 0.2106 → **0.1801** | 0.5776 → 0.6562 | 700 413 |

All four ranges improve. The gain shrinks with distance, which is expected: a fixed log
residual is a proportionally larger metric error further away.

## 4. Occupancy — corrected Gate-1 fusion, unchanged settings

5-frame clips, stride 5, `pixel_stride = 1`, confidence 1.5, unchanged grid, voxeliser and
evaluation mask. None of these were tuned during Gate 2.

`depth_cnn`:

| row | depth | IoU | P | R | occupied | points | in-grid | out-grid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GT poses (diagnostic) | base | 0.0782 | 0.406 | 0.090 | 24 801 | 318 150 | 0.635 | 0.365 |
| | refined | 0.0781 | 0.377 | 0.092 | 27 856 | 318 174 | 0.635 | 0.365 |
| | **ΔIoU** | **−0.0000 [−0.0026, +0.0025]** | | | | | | |
| **deployable: LingBot poses + constant s** | base | 0.0573 | 0.318 | 0.066 | 23 508 | 318 183 | 0.590 | 0.410 |
| | refined | **0.0775** | 0.358 | 0.092 | 29 341 | 318 184 | 0.623 | 0.377 |
| | **ΔIoU** | **+0.0203 [+0.0150, +0.0256] \*** | | | | 76.1 % of clips | | |
| LingBot poses + oracle s (diagnostic) | base | 0.0769 | 0.406 | 0.088 | 24 249 | 318 150 | 0.637 | 0.363 |
| | refined | 0.0778 | 0.379 | 0.091 | 27 479 | 318 174 | 0.637 | 0.363 |
| | **ΔIoU** | **+0.0009 [−0.0017, +0.0035]** | | | | | | |

`rgbd_unet` reproduces the pattern: deployable +0.0196 [+0.0143, +0.0248] \*, GT-pose
+0.0012 (n.s.), oracle-scale +0.0019 (n.s.).

## 5. Gate-2 criteria

| # | criterion | result |
|---|---|---|
| 1 | seq-08 AbsRel improves, CI excludes zero | **pass** — −0.1151 [−0.1360, −0.0953] |
| 2 | deployable occupancy IoU improves, CI excludes zero | **pass** — +0.0203 [+0.0150, +0.0256] |
| 3 | not produced by deleting points / collapsing recall | **pass** — points 318 183 → 318 184, occupied 23 508 → 29 341, **recall 0.066 → 0.092** |
| 4 | at least two distance ranges improve | **pass** — all four |
| 5 | broad across clips, not a few outliers | **pass** — 95.1 % of clips on depth, 76.1 % on occupancy |
| 6 | no semantic annotations or validation-target information | **pass** — enforced in code, asserted by test |
| 7 | LingBot frozen | **pass** — never loaded during Gate 2; training reads the frozen cache |

# `DEPTH_REFINEMENT_PASSES`

## 6. The finding that qualifies the verdict

Refinement helps **only when the base carries the constant scale**. With an oracle scale
already applied, it adds nothing measurable. A direct test explains why:

| base scale | variant | AbsRel | log RMSE | δ1 |
|---|---|---:|---:|---:|
| constant 27.3665 | base | 0.2046 | 0.2592 | 0.7055 |
| constant 27.3665 | **refined** | **0.0898** | 0.1714 | 0.9158 |
| oracle per-clip | base | 0.0932 | 0.1790 | 0.9170 |
| oracle per-clip | refined | 0.0877 | 0.1718 | 0.9153 |

The refined-from-constant result (0.0898) lands essentially on the **oracle-scaled base**
(0.0932). Refining an already oracle-scaled base improves AbsRel by only 0.0055 and leaves
δ1 unchanged (0.9170 → 0.9153, marginally worse).

**So the head is functioning as a per-pixel scale estimator, not a shape corrector.** That
is a real and useful result — it recovers from image evidence what Gate 1 A3's explicit
scale models could not, and it does so deployably. But it does **not** address the depth
*shape* error that Gate 1 measured as the 33.1 %-of-ceiling factor, and the +0.0203 IoU it
delivers is close to the 15.7 % scale-estimation term Gate 1 already attributed to the gap
between constant and oracle scale.

## 7. Failures and anomalies

1. **RGB does not help.** `rgbd_unet` (474 273 params) is *worse* than `depth_cnn`
   (48 129 params) on both selection and validation. The signal is in depth and confidence
   statistics. This mirrors Gate 1 A3, where `image_mlp` also added nothing.
2. **Confidence weighting hurts** (0.0961 vs 0.0906 selection AbsRel; 0.1024 vs 0.0950 on
   seq 08). Reported as the required ablation; uniform weighting is used for the headline.
3. **Precision falls while recall rises** in every occupancy row (deployable 0.318 → 0.358
   is the exception; the GT-pose and oracle rows go 0.406 → ~0.378). Refinement moves more
   points into the grid, some of them wrong.
4. **My first overfit criterion was wrong.** I judged it on total loss, which includes the
   residual regulariser and so cannot approach zero once the head learns a real correction.
   Re-judged on AbsRel against identity: 0.0847 → 0.0470 on 16 frames, a pass. Exact
   residual recovery is proven separately by synthetic tests.
5. **A `pkill -f "depth_gate/train.py"` killed four waiting shells** whose own command lines
   matched the pattern. No training or data was affected; all five runs had already
   completed and their checkpoints are intact.

## 8. Commands

```bash
export PYTHONPATH=$PWD
PY=/home/minh/anaconda3/envs/cu128/bin/python

$PY tools/depth_gate/cache_rgb.py --config configs/depth_gate/refine.yaml
$PY -m pytest tests/depth_gate -q                                   # 18 passed
$PY tools/depth_gate/train.py --arch rgbd_unet --overfit 16 --epochs 60 --run-name overfit16
$PY tools/depth_gate/train.py --arch rgbd_unet --smoke --run-name smoke
$PY tools/depth_gate/train.py --arch identity  --run-name identity
$PY tools/depth_gate/train.py --arch depth_cnn --run-name depth_cnn
$PY tools/depth_gate/train.py --arch rgbd_unet --run-name rgbd_unet
$PY tools/depth_gate/train.py --arch rgbd_unet --confidence-weighted --run-name rgbd_unet_cw
$PY tools/depth_gate/train.py --arch rgbd_unet --primary relative   --run-name rgbd_unet_rel
for r in identity depth_cnn rgbd_unet rgbd_unet_cw rgbd_unet_rel; do
  $PY tools/depth_gate/eval_depth.py     --run $r
  $PY tools/depth_gate/eval_occupancy.py --run $r; done
```

## 9. Runtime

Training 276–501 s per model on one RTX PRO 6000, peak 1.37–2.12 GiB. RGB cache 926 clips.
Depth evaluation ~40 s per run; occupancy evaluation ~6 min per run (163 clips × 6 fusions).

## 10. Artifacts

| path | contents |
|---|---|
| `artifacts/depth_gate/runs/<run>/{best.pt,last.pt,train.json}` | checkpoints, resolved config, curve, provenance |
| `artifacts/depth_gate/depth_eval/{summary,per_clip}_<run>.{json,csv}` | depth metrics, distance bins, bootstrap |
| `artifacts/depth_gate/occupancy_eval/{summary,per_clip}_<run>.{json,csv}` | occupancy rows and ΔIoU CIs |
| `configs/depth_gate/refine.yaml`, `depth_gate/`, `tools/depth_gate/`, `tests/depth_gate/` | implementation |

Scene completion, semantic lifting and external-dataset training have **not** been started.
Stopping here for review.
