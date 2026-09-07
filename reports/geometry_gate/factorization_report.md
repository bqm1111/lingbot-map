# Gate 1 — geometry error factorization

## Diagnosis: `DEPTH_DOMINANT`

Predicted depth costs **0.0637 IoU** (50.8 % of the five-frame visible-reconstruction
ceiling). Predicted pose costs **0.0022** (1.7 %). Depth error is **29× the pose effect**,
and every pairwise contrast supporting that separation has a bootstrap CI excluding zero.

## 1. Protocol

Identical to Gate 0: SemanticKITTI **sequence 08**, 5-frame clips at stride 5, 163 clips,
the same calibration, manifests, LingBot cache, LiDAR projection, `SEMANTICKITTI_GRID`
(256×256×32 @ 0.2 m), floor-binning voxeliser, evaluation `valid` mask and confidence
threshold (1.5). Only the factor under test changes between rows.

**One deviation, applied uniformly:** every configuration here uses `pixel_stride = 1`,
where Gate 0's LingBot fusion used stride 2. Using all pixels keeps the sparse
common-support comparison meaningful. Because it is applied to *all six* configurations
the factorization is unaffected; the consequence is that row F (0.0572) is not directly
comparable to Gate 0's deployable number (0.0372 at stride 2).

`0.1255` is called the **five-frame visible-reconstruction ceiling** throughout — it is
what this protocol achieves with ground-truth depth *and* ground-truth poses. It is not a
scene-completion ceiling and no completion is performed anywhere in this gate.

### Support discipline

Rows A–C are **sparse common-support** evaluations: only pixels where LiDAR returned.
Rows D–F are **full predicted-depth** evaluations: every confidence-passing pixel. The two
are never averaged, and the distinction is carried as a column in
`artifacts/geometry_gate/factorization_per_clip.csv`.

## 2. A coordinate bug found and fixed before measuring

The first factorization run reported 16.3° of rotation error and 16.9 m of translation
error. Inspection showed the predicted relative translation matched ground truth in
**magnitude** but pointed in exactly the opposite direction — `cos = −1.000`. That is a
convention error, not model error.

`pose_encoding_to_extri_intri` returns **camera-to-world**, despite `demo.py` inverting it
under a "convert w2c to c2w" comment. The Gate-0 cache had applied that inversion. Tested
directly by warping between frames:

| convention | rotation error | translation direction | vector error |
|---|---:|---:|---:|
| **raw output used as c2w** | **0.57°** | **+1.000** | **0.31 m** |
| inverted (as originally cached) | 2.67° | −0.999 | 16.26 m |

`tools/scale_gate/repair_poses.py` recomputed `pred_pose_c2w` for all 926 cached clips from
the stored `pose_enc` — no re-inference needed — and `cache_lingbot.py` is fixed.

**Impact on Gate 0.** Scale targets are essentially unchanged (depth/pose agreement 0.0288
→ 0.0277) because the pose estimator uses translation *magnitudes* only, which the bug
preserved. The occupancy numbers moved: with correct poses `oracle_depth` rises 0.0514 →
0.0526 and its precision 0.308 → 0.418, while `global_median` moves 0.0410 → 0.0372. Every
Gate-0 method carried the same wrong transform, so the *ordering* and the qualitative
conclusion (scale essential, per-clip refinement small) stand; the absolute values in the
Gate-0 report predate the fix. **Learned-model numbers were not recomputed** — their
features include pose statistics, so a corrected evaluation would require retraining, which
this task explicitly forbids.

## 3. Results — 163 clips, sequence 08

| | config | support | IoU | % of ceiling | precision | recall |
|---|---|---|---:|---:|---:|---:|
| **A** | GT LiDAR depth + GT poses | sparse common | **0.1255** | 100.0 % | 0.938 | 0.128 |
| **B** | GT LiDAR depth + LingBot poses (s\*) | sparse common | 0.1233 | 98.3 % | 0.890 | 0.127 |
| **C** | LingBot depth + GT poses | sparse common | 0.0618 | 49.2 % | 0.664 | 0.064 |
| **D** | LingBot depth, full + GT poses | full predicted | 0.0782 | 62.3 % | 0.406 | 0.090 |
| **E** | LingBot depth, full + LingBot poses (s\*) | full predicted | 0.0769 | 61.3 % | 0.406 | 0.088 |
| **F** | LingBot depth, full + LingBot poses (s = 27.35) | full predicted | 0.0572 | 45.6 % | 0.318 | 0.066 |

| | occupied voxels | fused points | in-grid | out-of-grid | sparse AbsRel | pose rot | pose trans |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 14 933 | 90 419 | 0.678 | 0.322 | — | 0.000° | 0.000 m |
| B | 15 623 | 90 419 | 0.681 | 0.319 | — | 0.508° | 0.288 m |
| C | 10 607 | 81 031 | 0.671 | 0.329 | 0.0935 | 0.000° | 0.000 m |
| D | 24 801 | 318 150 | 0.635 | 0.365 | 0.0935 | 0.000° | 0.000 m |
| E | 24 249 | 318 150 | 0.637 | 0.363 | 0.0935 | 0.508° | 0.288 m |
| F | 23 513 | 318 184 | 0.590 | 0.410 | 0.2049 | 0.508° | 0.288 m |

## 4. Factorization

Clip-level bootstrap, 10 000 resamples, seed 0; \* = CI excludes zero.

| factor | contrast | mean ΔIoU | 95 % CI | as % of ceiling |
|---|---|---:|---|---:|
| **pose alone** | A → B | **−0.0022** | [−0.0039, −0.0007] \* | **−1.7 %** |
| **depth alone** | A → C | **−0.0637** | [−0.0686, −0.0595] \* | **−50.8 %** |
| coverage beyond LiDAR support | C → D | **+0.0164** | [+0.0152, +0.0175] \* | +13.1 % |
| pose given predicted depth | D → E | −0.0013 | [−0.0020, −0.0005] \* | −1.0 % |
| scale estimation (oracle → constant) | E → F | −0.0197 | [−0.0252, −0.0141] \* | −15.7 % |

All 15 pairwise contrasts are in `factorization_summary.json`.

### Reading it

1. **Pose is nearly free.** Substituting LingBot's poses for ground truth, with ground-truth
   depth held fixed, costs 1.7 % of the ceiling. Measured error is 0.508° of rotation and
   0.288 m of translation over a clip spanning up to ~23 m of motion. `POSE_DOMINANT` is
   excluded.
2. **Depth is the whole problem.** On matched support, with ground-truth poses, predicted
   depth halves the result: 0.1255 → 0.0618. Precision falls 0.938 → 0.664 even though the
   pixel set is identical. AbsRel is 0.0935 with an oracle per-clip scale — good monocular
   depth, but a 9.4 % error at 20 m is ~1.9 m, roughly nine 0.2 m voxels.
3. **Coverage is a benefit, not a deficit.** Predicting depth where LiDAR returned nothing
   *adds* 0.0164 IoU and lifts recall 0.064 → 0.090. `COVERAGE_DOMINANT` is excluded — the
   camera-only method is not losing because it sees too little, it is losing because what it
   sees is misplaced. Note the cost: precision drops 0.664 → 0.406, so the extra pixels are
   individually less reliable while still being worth having.
4. **Scale estimation is second-order but real.** Replacing the oracle scalar with the
   deployable constant costs 0.0197 (15.7 % of ceiling) and more than doubles AbsRel
   (0.0935 → 0.2049). Larger than the pose term, an order of magnitude smaller than depth.
5. **Coordinates are sound after the §2 fix** — A → B is 1.7 %, and the ceiling row behaves
   exactly as a ground-truth reconstruction should (precision 0.938). `COORDINATE_ERROR`
   does not apply to the corrected pipeline.
6. **`MIXED_GEOMETRY` is excluded** because the terms are not comparable: depth is 29× pose
   and 3.2× scale estimation.

### Where the ceiling itself goes

Even row A reaches recall 0.128 and leaves 32.2 % of its own fused points outside the grid.
That is the observability limit of a forward-facing camera over five causal frames against
ground truth accumulated from multi-pass LiDAR — it bounds every row equally and is why all
numbers are reported as a fraction of 0.1255.

## 5. Diagnosis

# `DEPTH_DOMINANT`

## 6. Recommended next trainable component

**A metric depth-refinement head on frozen LingBot features, supervised by projected LiDAR.**

The measurement supports this specifically:

* Depth accounts for 50.8 % of the ceiling gap; pose for 1.7 %. Any effort not spent on
  depth is spent on at most a fiftieth of the problem.
* The failure is *placement*, not coverage: on identical pixels with perfect poses,
  precision falls 0.938 → 0.664. Extra predicted pixels are already net-positive
  (C → D, +0.0164), so the head should improve existing depth rather than extend range.
* Supervision already exists and is validated — 17 000–20 000 LiDAR-projected metric pixels
  per frame, cached for all 926 clips, with calibration QA passed in Gate 0.
* It composes with the scale work: at 9.4 % AbsRel under an oracle scalar, the residual is
  depth *shape*, which a scalar cannot fix by construction.

Secondary, and much cheaper: the E → F gap shows 15.7 % of ceiling sitting in scale
estimation, so the existing `combined_mlp` refinement remains worth keeping — but Gate 0
already showed that headroom is small and it should not be reopened before depth is
addressed.

**Not recommended:** pose refinement (1.7 % available), coverage extension (already
positive), or any completion/occupancy-prediction network — none is implied by this
factorization.

Per instruction, implementation of the recommended component has **not** been started.

## 7. Artifacts and reproduction

```bash
export PYTHONPATH=$PWD
PY=/home/minh/anaconda3/envs/cu128/bin/python

$PY tools/scale_gate/repair_poses.py --config configs/scale_gate/semantickitti.yaml
$PY tools/scale_gate/build_scale_targets.py --config configs/scale_gate/semantickitti.yaml
$PY tools/geometry_gate/factorize.py --config configs/scale_gate/semantickitti.yaml --split val
```

| artifact | contents |
|---|---|
| `artifacts/geometry_gate/factorization_per_clip.csv` | 978 rows: 163 clips × 6 configs, every metric per clip |
| `artifacts/geometry_gate/factorization_summary.json` | aggregates, all 15 pairwise bootstrap CIs, provenance |
| `tools/geometry_gate/factorize.py` | the implementation |
| `tools/scale_gate/repair_poses.py` | the pose-convention repair |
