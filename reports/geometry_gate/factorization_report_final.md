# Gate 1 — geometry error factorization (FINAL, corrected)

Supersedes `factorization_report.md`, which is retained unchanged for provenance. Three
things changed: a pose-convention bug was found and repaired, a controlled-support
configuration was added, and the scale models were retrained on the repaired caches.

## Final diagnosis: `DEPTH_DOMINANT`

Depth *shape*, measured on an identical pixel set with ground-truth poses, costs
**0.0416 IoU (33.1 % of the five-frame visible-reconstruction ceiling)**. It is
**≈ 18.9× the pose effect** and **≈ 2.1× the deployable-scale effect**.

## 1. The pose-convention bug and what it touched

`pose_encoding_to_extri_intri` returns **camera-to-world**, despite `demo.py` inverting it
under a "convert w2c to c2w" comment. The Gate-0 cache applied that inversion, storing the
inverse of the intended transform. Detected because predicted relative translations matched
ground truth in magnitude but pointed exactly opposite (`cos = −1.000`). Verified by warping:

| convention | rotation error | translation direction | vector error |
|---|---:|---:|---:|
| **raw output used as c2w** | **0.57°** | **+1.000** | **0.31 m** |
| inverted (as originally cached) | 2.67° | −0.999 | 16.26 m |

| artifact | status |
|---|---|
| `pred_pose_c2w` in all 926 cached clips | **repaired** from stored `pose_enc`, no re-inference (`tools/scale_gate/repair_poses.py`) |
| `tools/scale_gate/cache_lingbot.py` | **fixed**; future caches are correct |
| scale targets | **regenerated**; barely moved (agreement median 0.0288 → 0.0277) because the pose estimator uses translation *magnitudes*, which the inversion approximately preserved |
| Gate-0 occupancy numbers | **superseded**; `oracle_depth` 0.0514 → 0.0526 and its precision 0.308 → 0.418 at stride 2 |
| learned scale checkpoints | **invalidated and preserved** under `artifacts/scale_gate/scale_models/INVALID_AFTER_POSE_CONVENTION_FIX/` with a README; retrained below |
| the 0.1255 ceiling | **unaffected** — it uses `gt_pose_c2w`, which was always correct |
| `reports/scale_gate/*.md` | left as written; they predate the fix and say so here |

## 2. Configurations — 163 clips, sequence 08, `pixel_stride = 1`

Same calibration, manifests, caches, voxeliser, grid, evaluation mask and confidence
threshold (1.5) as Gate 0. Only the factor under test changes between rows.

| | config | support | IoU | % ceiling | P | R | masked px | occupied | points | in-grid | out-grid | AbsRel | rot | trans |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** | GT LiDAR + GT poses | sparse LiDAR | **0.1255** | 100.0 % | 0.938 | 0.128 | 90 419 | 14 933 | 90 419 | 0.678 | 0.322 | — | 0.000° | 0.000 m |
| **A_common** | GT LiDAR + GT poses, **C's pixel set** | controlled common | 0.1033 | 82.3 % | 0.957 | 0.105 | **81 031** | 12 236 | 81 031 | 0.674 | 0.326 | — | 0.000° | 0.000 m |
| **B** | GT LiDAR + LingBot poses (s\*) | sparse LiDAR | 0.1233 | 98.3 % | 0.890 | 0.127 | 90 419 | 15 623 | 90 419 | 0.681 | 0.319 | — | 0.508° | 0.288 m |
| **C** | LingBot depth + GT poses | controlled common | 0.0618 | 49.2 % | 0.664 | 0.064 | **81 031** | 10 607 | 81 031 | 0.671 | 0.329 | 0.0935 | 0.000° | 0.000 m |
| **D** | LingBot depth, full + GT poses | full predicted | 0.0782 | 62.3 % | 0.406 | 0.090 | 318 150 | 24 801 | 318 150 | 0.635 | 0.365 | 0.0935 | 0.000° | 0.000 m |
| **E** | LingBot depth, full + LingBot poses (s\*) | full predicted | 0.0769 | 61.3 % | 0.406 | 0.088 | 318 150 | 24 249 | 318 150 | 0.637 | 0.363 | 0.0935 | 0.508° | 0.288 m |
| **F** | deployable, s = 27.3665 | full predicted | 0.0572 | 45.6 % | 0.318 | 0.066 | 318 184 | 23 513 | 318 184 | 0.590 | 0.410 | 0.2049 | 0.508° | 0.288 m |

`A_common` and `C` select **exactly** the same pixels — the intersection of valid projected
LiDAR, the confidence threshold and the metric depth range, evaluated on predicted depth in
canonical units with C's scale. The implementation asserts equality of the pre-fusion count
per clip and aborts with a diagnostic otherwise; that assertion fired during development and
caught a real construction error (A_common was applying the range filter at s = 1).

## 3. Corrected factorization

Clip-level bootstrap, 10 000 resamples, seed 0; \* = CI excludes zero.

| factor | contrast | mean ΔIoU | 95 % CI | % of ceiling |
|---|---|---:|---|---:|
| pose alone | A → B | −0.0022 | [−0.0039, −0.0007] \* | −1.7 % |
| support restriction | A → A_common | −0.0222 | [−0.0240, −0.0205] \* | −17.7 % |
| **depth shape (controlled)** | **A_common → C** | **−0.0416** | **[−0.0465, −0.0375] \*** | **−33.1 %** |
| depth + support (uncontrolled) | A → C | −0.0637 | [−0.0686, −0.0595] \* | −50.8 % |
| coverage beyond LiDAR support | C → D | +0.0164 | [+0.0152, +0.0175] \* | +13.1 % |
| pose given predicted depth | D → E | −0.0013 | [−0.0020, −0.0005] \* | −1.0 % |
| scale estimation (oracle → constant) | E → F | −0.0197 | [−0.0252, −0.0141] \* | −15.7 % |

**What A1 changed.** The preliminary report attributed 50.8 % of the ceiling to depth. That
number conflated two things: A had 90 419 fused points and C only 81 031, so 17.7 points
were the cost of *having fewer pixels*, not of depth being wrong. With the pixel set held
identical, the true depth-shape effect is **33.1 %**. A → C is no longer described as a pure
depth contrast anywhere.

### Corrected ratios

```
depth shape ≈ 0.0416
pose        ≈ 0.0022      ->  depth ≈ 18.9x pose
scale       ≈ 0.0197      ->  depth ≈  2.1x deployable-scale
```

The preliminary report's "an order of magnitude larger than scale" was wrong; the ratio was
3.2× then and is **2.1×** now that depth is measured on controlled support. Depth remains
the single largest factor, but it is only about twice the scale term, not ten times.

## 4. Corrected scale models

Retrained from the repaired caches, identical sequence-level splits: fit on 00–07
(634 reliable clips), select on 09–10 (110 clips), **sequence 08 never used** for training,
model selection or thresholds.

| model | params | select median \|log s err\| |
|---|---:|---:|
| `global_median` (constant, s = **27.3665**) | 0 | 0.131 on val |
| `depth_mlp` | 74 561 | 0.0546 |
| `image_mlp` | 594 689 | 0.0664 |
| `combined_mlp` | 608 621 | 0.0613 |

Downstream on sequence 08, against the matched-stride constant baseline:

| model | stride | IoU | baseline | ΔIoU [95 % CI] | win rate | Wilcoxon p |
|---|---:|---:|---:|---|---:|---:|
| `depth_mlp` | 1 | 0.0613 | 0.0572 | +0.0041 [−0.0018, +0.0097] | 54.6 % | 0.086 |
| `image_mlp` | 1 | 0.0586 | 0.0572 | +0.0013 [−0.0037, +0.0063] | 54.0 % | 0.093 |
| `combined_mlp` | 1 | 0.0620 | 0.0572 | +0.0048 [−0.0014, +0.0110] | 59.5 % | **0.0097** |
| `depth_mlp` | 2 | 0.0400 | 0.0372 | +0.0028 [−0.0013, +0.0067] | 50.9 % | 0.113 |
| `image_mlp` | 2 | 0.0377 | 0.0372 | +0.0005 [−0.0030, +0.0039] | 55.2 % | 0.078 |
| `combined_mlp` | 2 | 0.0405 | 0.0372 | +0.0033 [−0.0010, +0.0074] | 61.3 % | **0.0037** |

Oracle references: stride 1 — `oracle_depth` 0.0788, `oracle_joint` 0.0769; stride 2 —
`oracle_depth` 0.0526, `oracle_joint` 0.0507.

**The pose fix removed the learned models' significance.** Before the repair,
`combined_mlp` had a bootstrap CI of [+0.0000, +0.0053]; after it, every model's CI includes
zero at both strides. `combined_mlp` still passes a signed-rank test (p = 0.0097 / 0.0037)
because it wins on ~60 % of clips by a small consistent margin — but a few clips lose badly,
so the *mean* is not distinguishable from the constant. Since the deployed quantity is
aggregate IoU, the mean is the decision-relevant statistic.

### Deployable scale policy for Gate 2

**Constant `s = 27.3665`**, fitted only on source-training sequences 00–07. The learned
models do not reliably beat it. Oracle scale is retained as a diagnostic row only.

## 5. Why the diagnosis stands

* `POSE_DOMINANT` — excluded. Pose costs 1.7 % with GT depth and 1.0 % with predicted
  depth; two independent measurements agree.
* `COVERAGE_DOMINANT` — excluded. Extra predicted pixels *help* (+13.1 %, recall
  0.064 → 0.090). The method is not losing for lack of points.
* `COORDINATE_ERROR` — excluded **after** §1. The bug was real and is repaired; A → B is
  now 1.7 % and row A behaves as a ground-truth reconstruction should (precision 0.938).
* `MIXED_GEOMETRY` — considered seriously, because depth is only 2.1× the scale term rather
  than the 3.2× previously claimed. Rejected because depth is still the largest single
  factor, and because scale already has a chosen remedy (the constant) while depth has none.
* `DEPTH_DOMINANT` — **selected**. On identical pixels with perfect poses, predicted depth
  drops IoU 0.1033 → 0.0618 and precision 0.957 → 0.664.

# `DEPTH_DOMINANT`

## 6. Reproduction

```bash
export PYTHONPATH=$PWD
PY=/home/minh/anaconda3/envs/cu128/bin/python

$PY tools/scale_gate/repair_poses.py       --config configs/scale_gate/semantickitti.yaml
$PY tools/scale_gate/build_scale_targets.py --config configs/scale_gate/semantickitti.yaml
$PY tools/geometry_gate/factorize.py        --config configs/scale_gate/semantickitti.yaml --split val
for m in global_median depth_mlp image_mlp combined_mlp; do
  $PY tools/scale_gate/train_scale.py --config configs/scale_gate/semantickitti.yaml --model $m; done
for st in 1 2; do
  $PY tools/scale_gate/eval_oracle_scale.py --config <cfg_for_stride> --split val
  for m in depth_mlp image_mlp combined_mlp; do
    $PY tools/scale_gate/eval_scale.py --config <cfg_for_stride> \
       --checkpoint artifacts/scale_gate/scale_models/${m}_best.pt --pixel-stride $st; done; done
```

Artifacts: `artifacts/geometry_gate/factorization_v2_{per_clip.csv,summary.json}`,
`artifacts/scale_gate/oracle_metrics_per_clip_s{1,2}.csv`,
`artifacts/scale_gate/learned_{metrics,summary}_*_s{1,2}.*`.
