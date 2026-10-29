# Observed surface, remembered surface, or genuinely hidden geometry?

**Measured 2026-10-28.** Repository `/home/minh/workspace/lingbot-map_fork`, branch
`feature-completion-study`, HEAD `db465c2ce2a692817d9ce4a141df7c24c4fa2d13`. Geometry only.
No training, no new model predictions: the canonical seed-0 predictions and OccAny's released
outputs are rescored under a finer region partition by the same scorer that produced every
previous canonical number.

Generated tables: `artifacts/method_vs_data/observed_hidden/tables.md` and `metrics.csv`.
Machine-readable: `regional_semantickitti.json`, `regional_occ3d.json`. Figures:
`figures/phase_runtime_hidden/regional_*.png` and `qualitative_panel_raw.png`.

---

## 1. The question, and the partition that answers it

> Relative to OccAny, does our method improve visible-surface prediction, recover geometry seen
> by earlier causal frames, or infer geometry unseen throughout the causal input window?

Four mutually exclusive regions, built from the **ground-truth occupancy and the ground-truth
poses** of the five permitted causal frames — never from any method's own reconstruction, so
the same masks apply to every arm:

| region | definition |
|---|---|
| `anchor_observed` | a ray from the **anchor** camera reaches it, up to and including the first GT-occupied voxel on that ray |
| `earlier_only` | reached the same way from one of the four **earlier** permitted frames, but not from the anchor |
| `never_observed_in_frustum` | reached from **none** of the five, and inside the anchor camera's image rectangle |
| `never_observed_out_frustum` | reached from **none** of the five, and outside it |

The union `never_observed` is reported as its own row and is exactly the last two.

**Visibility construction.** A ray is cast through every second pixel of the input image and
marched in 0.1 m steps to 80 m. Every voxel a ray *enters* up to and including the first
GT-occupied voxel is observed: the free space the camera sees through **and** the surface the
ray terminates on. Everything behind that first surface is not. Free space is therefore inside
the observed region, which is what makes IoU well defined there — a false positive is counted
in the region where it occurs. Voxels the label marks `255` are outside `valid` and outside
every region. Coverage is complete in the sense that the four regions are an exact partition of
`valid`; it is incomplete in the sense that a 2-pixel ray stride can miss a voxel that a
denser cast would reach, which would move it from `anchor_observed`/`earlier_only` into
`never_observed`. That bias is identical for every arm because the masks are shared.

**Checks that ran on every anchor, not just at the end.** The four regions were asserted
pairwise disjoint and asserted to sum to `valid`; `never_observed` was asserted equal to the
union of its two legs; the anchor slot's camera-to-grid transform was asserted to reproduce the
transform stored in the canonical prediction; and for **every arm and both conventions** the
four regions' TP/FP/FN were asserted to sum to the global counts. Aggregate confirmation
(ours, KITTI-360, raw): TP 44,053,033 / FP 73,588,119 / FN 49,245,689 summed over the four
regions, identical to the `all` row, over 1,173,859,441 valid voxels.

**Three families that are not the same partition, and are not mixed.**

| family | members | what it is |
|---|---|---|
| strict GT visibility | the four above | could a permitted frame see it? |
| frustum | `in_frustum` / `out_frustum` | is it inside the anchor image rectangle? |
| hard-map evidence | `A0fixed_observed` / `A0fixed_completed` | did **our** map receive evidence there? |

In-frustum is not observed: 68.05 % of GT occupancy is in-frustum **and never observed**.
Out-of-frustum is not never-observed either, although on this benchmark it nearly is — but the
converse fails badly, and that is the direction that matters: only **23.6 %** of never-observed
GT-occupied voxels lie outside the frustum. Hard-map known/unknown is a property of our
reconstruction, not of the ground truth, and is reported only as context.

**Scope limit, load-bearing.** These masks describe visibility under **our five causal frames**.
OccAny's five-frame configuration reads `t … t+40` — 4.0 s of future. A voxel that is
`never_observed` here may be plainly visible to that arm. The masks bound our information, not
theirs. Only OccAny's **one-frame causal** arm is bounded by the same information these masks
describe.

## 2. Arms

| arm | what it is |
|---|---|
| `k360` | ours, the frozen KITTI-360 head — **the main method result** |
| `waymo` | ours, Waymo-trained — source-transfer evidence |
| `once` | ours, ONCE-trained — **exploratory**; its occupied-prevalence gate remains failed |
| `map_only` | the same pipeline with the completion head removed: the hard map's occupied channel read onto the evaluation grid |
| `occany_1f` | OccAny's released one-frame causal preset [reproduced] |
| `occany_5f_fwd` | OccAny's released five-frame **forward** preset, `t … t+40` [reproduced] |

All three of our arms are **seed 0**: `GEOMETRY_SEED = 0` is fixed in the prediction pipeline,
so the intervals below are evaluation-sample uncertainty and **not** training-seed spread.

`map_only` was recomputed from the frozen causal clips for this analysis. Two gates confirm it
is the same pipeline minus the head: the recomputed hard-map support is asserted bit-identical
to the `observed` mask stored in each canonical prediction, and its score reproduces the
previously published five-frame map-only floor to every digit — **8.786818096341399** raw and
**15.043313568062288** pooled.

## 3. What the benchmark actually asks for

| region | % of valid volume | % of GT-occupied voxels |
|---|---:|---:|
| `anchor_observed` | 32.69 % | **4.76 %** |
| `earlier_only` | 24.81 % | **6.18 %** |
| `never_observed_in_frustum` | 29.71 % | **68.05 %** |
| `never_observed_out_frustum` | 12.79 % | **21.01 %** |
| `never_observed` (union) | 42.50 % | **89.06 %** |

Each permitted frame alone sees 4.59–5.16 % of the GT-occupied surface and their union sees
10.94 %. **SemanticKITTI's completion benchmark is overwhelmingly a hidden-geometry task**:
89 % of the surface to be predicted was never visible to any of the five causal inputs, and
most of that is straight ahead, inside the camera's own image rectangle.

## 4. SemanticKITTI — the main result

801 canonical anchors. Paired block bootstrap, 2,000 resamples, seed 0, 33 blocks of 25
contiguous anchors; every arm resampled on the same anchors in the same pass; global IoU
recomputed from summed TP/FP/FN inside each resample. `*` = interval excludes zero.

### 4.1 SC IoU by region

| region | ours K360 | ours Waymo | ours ONCE | map only | OccAny 1f causal | OccAny 5f forward |
|---|---:|---:|---:|---:|---:|---:|
| **raw** | | | | | | |
| `all` | **26.40** | 23.55 | 18.20 | 8.79 | 20.32 | 25.21 |
| `anchor_observed` | 11.47 | 10.29 | 7.51 | 11.16 | **15.76** | 11.72 |
| `earlier_only` | **17.95** | 15.13 | 11.48 | 12.14 | 12.97 | 11.65 |
| `never_observed_in_frustum` | 28.56 | 26.50 | 19.61 | 7.67 | 25.45 | **34.30** |
| `never_observed_out_frustum` | 35.12 | **35.19** | 30.22 | 8.45 | 7.98 | 9.17 |
| `never_observed` | **30.07** | 28.55 | 22.17 | 7.86 | 21.53 | 29.02 |
| **pooled** | | | | | | |
| `all` | **26.34** | 23.37 | 20.26 | 15.04 | 24.06 | 25.93 |
| `anchor_observed` | 9.02 | 7.80 | 7.42 | 9.57 | **11.72** | 8.30 |
| `earlier_only` | **14.57** | 11.90 | 9.72 | 11.91 | 13.92 | 11.06 |
| `never_observed_in_frustum` | 31.39 | 30.20 | 25.83 | 17.63 | 31.64 | **37.63** |
| `never_observed_out_frustum` | 35.62 | **36.26** | 32.86 | 17.79 | 12.92 | 13.14 |
| `never_observed` | 32.37 | 31.64 | 27.56 | 17.67 | 27.60 | **32.80** |

### 4.2 Against the only causal comparator (OccAny, one causal frame)

| region | ours K360 − OccAny 1f, raw | pooled |
|---|---|---|
| `all` | +6.08 [+4.82, +7.33] \* | +2.28 [+1.23, +3.33] \* |
| `anchor_observed` | **−4.29 [−5.08, −3.60] \*** | **−2.70 [−3.20, −2.29] \*** |
| `earlier_only` | +4.98 [+3.57, +6.64] \* | +0.65 [−0.33, +1.74] ns |
| `never_observed_in_frustum` | +3.11 [+1.45, +4.78] \* | **−0.25 [−1.67, +1.26] ns** |
| `never_observed_out_frustum` | +27.14 [+25.00, +29.33] \* | +22.70 [+20.52, +25.03] \* |
| `never_observed` | +8.54 [+7.01, +10.06] \* | +4.77 [+3.25, +6.23] \* |

**The answer to the primary question, stated precisely.**

1. **Not visible surface.** On surfaces the anchor camera actually looks at we are **worse**,
   significantly, in both conventions (−4.29 / −2.70).
2. **Not mainly accumulation.** `earlier_only` is a raw win that **does not survive pooling**
   (+0.65, interval contains zero), and it is only 6.18 % of the GT surface.
3. **Hidden geometry — but the union figure hides where.** The `never_observed` win (+8.54 raw,
   +4.77 pooled) is real, and it is **not uniform across the region**. Cut by the frustum:
   - outside the anchor frustum (21.01 % of GT occupancy): **+27.14 raw, +22.70 pooled**, both
     overwhelming;
   - inside it (68.05 % of GT occupancy): **+3.11 raw but −0.25 pooled, no difference
     detected**.

   So under OccAny's own published pooled convention, **the never-observed advantage against
   their causal mode lives entirely in the fifth of hidden surface outside the camera's view
   volume.** Under raw scoring there is also a modest, significant in-frustum hidden win.

**And the out-of-frustum win is a coverage result, not an occlusion-reasoning result.** The
confusion counts say so plainly: out of frustum, OccAny's causal arm has *recall 8.38* at
*precision 62.89* — it barely predicts there at all — against our recall 52.52 at precision
51.46. It is precise where it predicts; it predicts almost nothing outside the image rectangle.
Our grid extends past the frustum and the map plus the completion head fill it. That is a
genuine capability difference and a genuine benchmark gain, but it **does not** by itself
establish better reasoning about occlusion, and it is not described here as if it did.

### 4.3 Against the forward comparator (OccAny, five forward frames, `t … t+40`)

| region | ours K360 − OccAny 5f fwd, raw | pooled |
|---|---|---|
| `all` | +1.19 [+0.35, +2.09] \* | +0.41 [−0.34, +1.20] ns |
| `anchor_observed` | −0.24 [−0.95, +0.49] ns | +0.72 [+0.26, +1.23] \* |
| `earlier_only` | +6.30 [+5.05, +7.74] \* | +3.51 [+2.64, +4.47] \* |
| `never_observed_in_frustum` | **−5.74 [−7.05, −4.40] \*** | **−6.24 [−7.51, −4.98] \*** |
| `never_observed_out_frustum` | +25.95 [+23.00, +29.05] \* | +22.48 [+19.72, +25.52] \* |
| `never_observed` | +1.05 [−0.14, +2.24] ns | −0.43 [−1.61, +0.72] ns |

This is exactly what the scope limit predicts. The in-frustum hidden surface is the surface a
forward-looking vehicle drives into, so 4.0 s of future frames reveal it — and there OccAny's
forward arm beats us by 5.7–6.2 points. Outside the frustum its future frames do not help and
we win by 22–26. The union is a **tie in both conventions**, which is a statement about no
detected difference, not about equivalence.

### 4.4 How much of this is the learned head? (ours vs map only)

| region | ours K360 − map only, raw | pooled |
|---|---|---|
| `all` | +17.61 [+16.55, +18.68] \* | +11.30 [+10.11, +12.53] \* |
| `anchor_observed` | **+0.32 [−0.33, +0.98] ns** | **−0.56 [−0.90, −0.23] \*** |
| `earlier_only` | +5.81 [+4.46, +7.36] \* | +2.66 [+1.74, +3.75] \* |
| `never_observed_in_frustum` | +20.89 [+19.31, +22.57] \* | +13.76 [+11.96, +15.68] \* |
| `never_observed_out_frustum` | +26.67 [+24.86, +28.36] \* | +17.83 [+15.59, +19.93] \* |
| `never_observed` | +22.21 [+20.71, +23.74] \* | +14.71 [+12.88, +16.59] \* |

**The 967,322-parameter head contributes nothing on observed surface and everything in hidden
space.** On `anchor_observed` it is a tie raw and a small significant *loss* pooled; in
`never_observed` it is worth +22.21 raw / +14.71 pooled. Accumulated reconstruction alone
(`map_only`, 8.79 raw) is **11.53 points below OccAny's causal arm**; the learned completion is
what closes that gap and overturns it.

It also matters where the head is *not* enough: `map_only` already beats OccAny's causal arm
out of frustum under pooling (+4.87 \*), so part of the out-of-frustum advantage is the
accumulated map's coverage, and the head roughly quadruples it.

### 4.5 The other source arms, in the same regions

| contrast | `never_observed` raw | pooled | `never_obs ∩ frustum` raw | pooled | `never_obs ∖ frustum` raw | pooled |
|---|---|---|---|---|---|---|
| Waymo − OccAny 1f | +7.02 \* | +4.04 \* | +1.06 ns | −1.45 \* | +27.20 \* | +23.34 \* |
| ONCE − OccAny 1f | +0.65 ns | −0.04 ns | −5.84 \* | −5.82 \* | +22.24 \* | +19.94 \* |
| Waymo − map only | +20.69 \* | +13.97 \* | +18.84 \* | +12.57 \* | +26.74 \* | +18.47 \* |
| ONCE − map only | +14.32 \* | +9.90 \* | +11.94 \* | +8.20 \* | +21.77 \* | +15.06 \* |
| K360 − Waymo | +1.52 \* | +0.73 \* | +2.05 \* | +1.19 \* | −0.07 ns | −0.64 ns |
| K360 − ONCE | +7.89 \* | +4.81 \* | +8.95 \* | +5.56 \* | +4.90 \* | +2.76 \* |

Two things stand out. **The out-of-frustum advantage is the most source-robust thing in this
table** — all three of our arms win it by 20–27 points raw, and KITTI-360 is not significantly
better than Waymo there. **The in-frustum hidden result is the source-sensitive one**: KITTI-360
beats Waymo by 2.05 and ONCE by 8.95 there, and only the KITTI-360 arm manages a positive
in-frustum hidden result against OccAny's causal mode at all.

### 4.6 Why the aggregates move the way they do — from the counts

Regional IoUs are not additive and valid-voxel fractions alone do not determine the aggregate,
so the explanation is in the confusion counts.

Globally, raw, ours against OccAny's causal arm: precision 37.45 against 55.64, recall 47.22
against 24.25. **Every one of our arms trades precision for recall**, and the trade pays
wherever the ground truth is dense and the comparator under-predicts. Pooling dilates the
prediction only; it adds proportionally far more false positives to OccAny's sparse output than
to our dense one, which is why their IoU gains more from pooling than ours does and why several
raw wins become pooled ties. The region composition explains *which* regions matter —
`never_observed_in_frustum` carries 68 % of the GT occupancy, so a 6-point loss there against
the forward arm is what drags the aggregate down to +1.19 — but the convention flip is a
property of where each method sits on the precision/recall trade, not of the regions.

## 5. Occ3D-nuScenes — the same construction, newly available

**The strict visibility split now exists on Occ3D.** The previous record
(`method_vs_data_reconciled_conclusions.md` Q2) states that "Occ3D has no strict never-observed
mask in this harness". That limitation is **lifted**: correctness can be established from data
already on disk, so the construction was implemented rather than approximated.

What it needs, and what exists: the ground-truth occupancy (OccAny's stored `voxel_label`, the
same volume both methods are scored against), a per-frame camera intrinsic (`Frame.K` from the
nuScenes calibration), and a ground-truth camera-to-grid transform for each of the five
permitted frames. The last is the only non-trivial one, and nuScenes supplies it directly:

```
T_cam(j)->grid = inv(T_ego_keyframe(anchor)->world) @ T_ego_cam(j)->world @ T_camera_to_ego_cam(j)
```

This is the same single-timestamp chain `core/datasets/nuscenes.Frame.T_camera_to_ego` already
applies, so at the anchor slot it must collapse to the transform stored in the canonical
prediction. **That identity is asserted on every one of the 4,219 anchors** (atol 1e-6), and so
are the partition and the TP/FP/FN reconstruction. No proxy was invented; the construction is
the same code with the SemanticKITTI pose chain swapped for the nuScenes one.

**There is no causal OccAny comparator on Occ3D.** No one-frame nuScenes OccAny run exists, so
every Occ3D contrast below is against their **forward** five-frame arm, which reads 4.0 s of
future. This is stated on every row rather than in a footnote.

### 5.1 Composition — a different benchmark, structurally

| region | % of valid | % of GT occupied | SemanticKITTI, for contrast |
|---|---:|---:|---:|
| `anchor_observed` | 37.05 % | **21.48 %** | 4.76 % |
| `earlier_only` | 20.43 % | 16.77 % | 6.18 % |
| `never_observed_in_frustum` | 11.33 % | 20.20 % | 68.05 % |
| `never_observed_out_frustum` | 31.20 % | **41.56 %** | 21.01 % |
| `never_observed` | 42.52 % | **61.75 %** | 89.06 % |

Occ3D's grid is a 80 × 80 × 6.4 m box around the ego, evaluated from **CAM_FRONT alone**, so
**54.5 % of its GT occupancy is outside the input camera's frustum** against 25.5 % on
SemanticKITTI. The benchmark asks a single forward camera to complete a surround volume. Any
aggregate Occ3D number is therefore dominated by out-of-frustum coverage, which is a property
of the benchmark, not of either method.

### 5.2 SC IoU by region

| region | ours K360 | ours Waymo | ours ONCE | map only | OccAny 5f forward |
|---|---:|---:|---:|---:|---:|
| **raw** | | | | | |
| `all` | **31.94** | 22.85 | 18.67 | 9.80 | 20.66 |
| `anchor_observed` | **33.55** | 20.22 | 15.01 | 13.19 | 29.42 |
| `earlier_only` | **36.47** | 28.40 | 21.25 | 16.65 | 17.37 |
| `never_observed_in_frustum` | **38.64** | 26.91 | 21.65 | 10.94 | 29.08 |
| `never_observed_out_frustum` | **25.65** | 19.66 | 18.11 | 4.26 | 9.07 |
| `never_observed` | **30.07** | 22.17 | 19.28 | 6.49 | 17.05 |
| **pooled** | | | | | |
| `all` | **41.56** | 38.05 | 30.52 | 22.16 | 23.55 |
| `anchor_observed` | **43.87** | 39.08 | 26.20 | 30.80 | 25.52 |
| `earlier_only` | **44.06** | 41.00 | 28.75 | 30.86 | 20.92 |
| `never_observed_in_frustum` | **48.40** | 43.47 | 40.34 | 24.02 | 37.43 |
| `never_observed_out_frustum` | **35.31** | 32.72 | 30.04 | 10.50 | 12.59 |
| `never_observed` | **39.84** | 36.54 | 33.44 | 15.16 | 22.96 |

### 5.3 Contrasts

| region | K360 − OccAny 5f fwd, raw | pooled | K360 − map only, raw | pooled |
|---|---|---|---|---|
| `all` | +11.28 [+9.25, +13.37] \* | +18.01 [+16.67, +19.34] \* | +22.14 [+20.60, +23.75] \* | +19.40 [+17.74, +21.10] \* |
| `anchor_observed` | +4.13 [+1.12, +7.09] \* | +18.36 [+16.93, +19.83] \* | +20.36 [+18.66, +22.15] \* | +13.08 [+10.88, +15.33] \* |
| `earlier_only` | +19.10 [+16.35, +21.87] \* | +23.14 [+21.52, +24.85] \* | +19.82 [+18.32, +21.43] \* | +13.20 [+11.75, +14.72] \* |
| `never_observed_in_frustum` | +9.56 [+6.07, +13.09] \* | +10.97 [+8.57, +13.47] \* | +27.70 [+25.23, +30.30] \* | +24.37 [+21.32, +27.44] \* |
| `never_observed_out_frustum` | +16.58 [+14.53, +18.50] \* | +22.72 [+20.80, +24.67] \* | +21.39 [+19.60, +23.17] \* | +24.82 [+22.38, +27.16] \* |
| `never_observed` | +13.02 [+10.87, +15.13] \* | +16.88 [+15.16, +18.57] \* | +23.58 [+21.82, +25.32] \* | +24.68 [+22.57, +26.73] \* |

**On Occ3D the KITTI-360 arm wins every region in both conventions, against a comparator that
reads 4.0 s of future.** Including `anchor_observed`, where it loses on SemanticKITTI. Two
things differ and neither is the method: the comparator is non-causal but has no causal
counterpart here, and the benchmark's region composition is completely different.

The source arms behave as they do on SemanticKITTI, only more so: Waymo is a tie on `all` raw
(+2.20 [−0.02, +4.35] ns) and a large pooled win (+14.51 \*), ONCE is a significant loss raw
(−1.99 \*) and a win pooled (+6.98 \*), and `map_only` alone loses to OccAny on `all` raw
(−10.86 \*) and is a tie pooled (−1.39 [−3.04, +0.20] ns).

### 5.4 Precision and recall tell the same story differently here

Raw, `all`: ours precision 77.9 / recall 35.1 against OccAny's 43.0 / 28.5. On Occ3D we are the
**precise** arm, not the high-recall one — the opposite of SemanticKITTI, where we run precision
37.5 / recall 47.2 against their 55.6 / 24.3. The two benchmarks place the same checkpoint on
opposite sides of the precision/recall trade, which is why a single "our method over-predicts"
summary would be wrong, and why pooling helps us a great deal on Occ3D (+9.6 on `all`) and not
at all on SemanticKITTI (−0.06).

## 6. What the qualitative panel shows

`figures/phase_runtime_hidden/qualitative_panel_raw.png`, five anchors chosen by a recorded
rule: all 801 anchors ranked by the per-anchor SC IoU difference (ours minus OccAny causal, raw,
whole valid volume, `panel_ranking.json`), one taken at each of the quantiles 0.04, 0.27, 0.50,
0.73, 0.96. The low quantiles are anchors where **OccAny wins**, so failures are in the figure
by construction, not by selection.

| quantile | anchor | map only | ours | OccAny 1f | OccAny 5f fwd |
|---|---|---:|---:|---:|---:|
| 0.04 | `08_003820` | 5.6 | **17.9** | **21.0** | 23.4 |
| 0.27 | `08_000700` | 5.9 | **22.1** | 19.4 | 18.6 |
| 0.50 | `08_002925` | 11.0 | **31.4** | 25.5 | 31.5 |
| 0.73 | `08_001550` | 6.4 | **36.3** | 25.5 | 32.6 |
| 0.96 | `08_002120` | 8.0 | **31.1** | 11.5 | 18.4 |

Column 1 colours the ground truth by region and makes the composition visible: the road and
structures ahead are `never_observed_in_frustum` (orange), the flanks are
`never_observed_out_frustum` (purple), and `anchor_observed` (green) and `earlier_only` (yellow)
are thin skins. Column 6 shows where our prediction and OccAny's causal prediction each recover
GT-occupied voxels: the lateral bands are almost entirely "only ours", which is the
out-of-frustum result drawn rather than tabulated.

## 7. Limitations

1. **Seed 0 only.** `GEOMETRY_SEED = 0` is fixed in the prediction pipeline, so all three of our
   arms have canonical predictions at one seed. Every interval here is evaluation-sample
   uncertainty. **No three-seed interval is reported, and none should be inferred.** Producing
   one would require regenerating model predictions.
2. **No causal comparator on Occ3D.** Every Occ3D contrast is against a non-causal arm.
3. **The masks bound our information, not OccAny's forward arm's.** §1 states this; §4.3 shows
   its consequence.
4. **Ray-stride bias.** A 2-pixel stride can leave a thin surface unreached and so classify it
   `never_observed`. The bias is shared by every arm but it is a bias.
5. **Dynamic objects.** The construction uses the GT occupancy of the anchor timestamp and the
   GT poses of five frames spanning ~2 s. A moving object is carved as if static, so a voxel a
   vehicle has since vacated can be marked observed or not on the strength of where that
   vehicle was. This affects all arms identically and is not corrected.
6. **Out-of-frustum gains are coverage, not demonstrated occlusion reasoning.** §4.2 gives the
   recall figures that make this concrete.
7. **A confidence interval containing zero means no difference was detected**, never that the
   two are equivalent. Every `ns` above should be read that way.
8. **ONCE is exploratory.** Its occupied-prevalence gate remains failed and is not revisited
   here, and its supervision window differs from the frozen recipe
   (`training_supervision_provenance.md`), so it is not a clean source substitution.

## 8. Reproduction

```bash
cd /home/minh/workspace/lingbot-map_fork
P=/home/minh/anaconda3/envs/cu128/bin/python

# the map-only arm (cached once per benchmark; asserts bit-identity with the stored hard map)
PYTHONPATH=$PWD $P tools/method_vs_data/map_only_predictions.py --dataset semantickitti --device cuda:3
PYTHONPATH=$PWD $P tools/method_vs_data/map_only_predictions.py --dataset occ3d --device cuda:2

# the regional scorer: both conventions in one pass, all arms on the same anchors
PYTHONPATH=$PWD $P tools/method_vs_data/observed_hidden.py --dataset semantickitti --device cuda:3
PYTHONPATH=$PWD $P tools/method_vs_data/observed_hidden.py --dataset occ3d --device cuda:2 \
    --arms k360 waymo once map_only occany_5f_fwd

# tables, CSV, figures
$P tools/method_vs_data/observed_hidden_tables.py
PYTHONPATH=$PWD $P tools/method_vs_data/phase_figures.py --which regional
PYTHONPATH=$PWD $P tools/method_vs_data/qualitative_panel.py --device cuda:3 --post raw
```

Outputs: `artifacts/method_vs_data/observed_hidden/` — `regional_semantickitti.json`,
`regional_occ3d.json`, `tables.md`, `metrics.csv` (220 rows), `map_only/`, `logs/`. Figures in
`artifacts/method_vs_data/figures/phase_runtime_hidden/`, with `panel_ranking.json` and
`qualitative_panel_raw.json` recording the anchor selection.

Runtime of the analysis itself: SemanticKITTI 1.43 s per anchor (20 min), Occ3D 0.90 s per
anchor (63 min), dominated by the ray casting.

## 9. What is NOT claimed

- **Not** that we are better at predicting visible surface. On SemanticKITTI we are
  significantly worse there in both conventions.
- **Not** that the advantage is temporal accumulation. `earlier_only` is 6.18 % of the GT
  surface on SemanticKITTI and its advantage does not survive pooling.
- **Not** that an out-of-frustum win is an occlusion-completion win. The two are measured
  separately here precisely so they are not conflated.
- **Not** that the `never_observed` union figure can be quoted alone. Under pooling, against
  the causal comparator, its two legs disagree: +22.70 outside the frustum, no difference
  detected inside it.
- **Not** that the Occ3D result is a causal comparison. No causal OccAny arm exists there.
- **Not** a seed-variation statement anywhere.
