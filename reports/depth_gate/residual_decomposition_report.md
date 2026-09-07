# Gate-2 diagnostic — what did the `depth_cnn` residual head actually learn?

## Diagnosis: `SCALE_CORRECTION_ONLY`

The head is a **clip-level metric-scale estimator wearing a dense-residual costume**. Its
per-clip median log residual alone reproduces **101.5 %** of the reported Gate-2 occupancy
gain. Once that scalar is removed, the remaining median-zero spatial residual produces no
significant occupancy improvement on a correctly scaled base — `O0→O1` **+0.0005
[−0.0012, +0.0022]** and `G0→G1` **−0.0005 [−0.0022, +0.0013]**, both straddling zero.

The result is *not* an inconsistent-geometry artefact. Correcting the depth/pose
coupling that the original implementation left open makes the result slightly **better**
(`C1→C5` +0.0015 [+0.0008, +0.0023] \*), not worse.

> **All results in this report are in-domain SemanticKITTI results**, sequence 08 only,
> one dataset, one sensor, one city. Nothing here establishes generalisable metric-scale
> recovery, and the 163 clips are not 163 independent environments — the confidence
> intervals describe variation *within* sequence 08.

No model was trained or retrained. Nothing in `reports/depth_gate/depth_refinement_report.md`,
`artifacts/depth_gate/{runs,depth_eval,occupancy_eval}` or any cache was modified.

---

## 1. Provenance

| item | value |
|---|---|
| refinement checkpoint | `artifacts/depth_gate/runs/depth_cnn/best.pt` |
| checkpoint SHA-256 | `2a91822ea5d1182a6c0e5394751e74c56200ee5c4cae3d98c59a8a519faafbe1` |
| arch / params | `depth_cnn`, 48 129 |
| LingBot checkpoint | `checkpoints/lingbot-map/204754b/lingbot-map.pt` |
| LingBot SHA-256 | `ee665103348e07e6b826d529b8e61de8f413d5432a4f2e84970d6c8fd2e1cd72` |
| LingBot cache (seq 08) | 163 files, dir-SHA-256 `d4273efb36438255170f37f65a74fdb1118c12d79ee8908a0105317e9532e7bd` |
| projected-LiDAR cache | 163 files, dir-SHA-256 `b52409398c7ba47e3b4d9b1192d6416e329082b1bf0a6f61410fca63ff68dae0` |
| RGB cache | 163 files, dir-SHA-256 `63d94b36a94e6a854f08eb7b92a22c5a59543f3ac2c457c51c200ac888088b20` |
| clip manifest | `artifacts/scale_gate/manifests/val.jsonl` — `16c8e6b5…f29ddfe0` |
| oracle scale table | `artifacts/scale_gate/scale_targets_val.csv` — `0382ea1b…a6525263` |
| run config | `configs/depth_gate/refine.yaml` — `b7692022…41e0dd92` |
| geometry config | `configs/scale_gate/semantickitti.yaml` — `472b4531…5f3e4ea086` |
| tool | `tools/depth_gate/decompose_residual.py` |
| outputs | `artifacts/depth_gate/residual_decomposition/` |

`s0` is read from the resolved run configuration (`scale.constant = 27.3665`) and asserted
equal to the checkpoint's stored `metric_scale`; no second constant is introduced. (The
Gate-1 `F_deployable` row used the marginally different 27.3498 fitted on source sequences
only; Gate 2 resolved 27.3665, and that is the value the head was trained against.)

Frozen protocol, unchanged: sequence 08, 163 five-frame clips, clip stride 5,
`pixel_stride = 1`, confidence threshold 1.5, `SEMANTICKITTI_GRID` (256×256×32 @ 0.2 m),
the same floor-binning voxeliser, the same valid evaluation mask, the same calibration and
the repaired `pred_pose_c2w` convention. Semantic labels are never loaded.

---

## 2. Decomposition equations

The residual is computed **once**, from the reported deployable input
(`base = s0 · D_lingbot`, confidence, valid mask, normalised image coordinates, and the
training-set normalisation statistics stored in the checkpoint):

```
r[t,p]      = depth_cnn(s0 · D_lingbot, conf, valid, x, y)        r = 0.7 · tanh(raw)

a_clip      = median over the clip's unchanged fusion support of r[t,p]
r_shape     = r − a_clip                                          median(r_shape) ≡ 0
s_learned   = s0 · exp(a_clip)                                    s0 = 27.3665
```

The support used for `a_clip` is the **C0 mask** — the mask the unrefined deployable
pipeline already fuses with:

```
support = (conf ≥ 1.5) ∧ finite(s0·D) ∧ (s0·D > 1 m) ∧ (s0·D < 60 m)
```

It is derived from LingBot outputs alone. No LiDAR, no ground-truth pose, no oracle scale
and no semantic label enters it. One scalar per clip is taken over all five frames jointly;
no per-frame scale is fitted.

Asserted per clip, and enforced as a hard failure:

* `|median(r_shape restricted to support)| < 1e-9` — exact by construction;
* `s_learned · D · exp(r_shape) == s0 · D · exp(r)` to `rtol 1e-9` — i.e. **C5 and C1 carry
  bit-comparable depth and differ only in pose-translation scale**.

*(Both assertions initially tripped at ~1e-7 because `s0` is a weak Python float and
`s0 * D_float32` silently stayed float32 under NEP 50. Depth is now cast to float64 once,
before any scaling.)*

---

## 3. Pose-scaling implementation and assertions

A metric scale is a property of the whole clip geometry, so it must move depth **and**
camera translation. Scaling is applied to camera centres relative to the clip's anchor
frame, in the clip's own canonical frame, **before** the fused points are mapped into the
SemanticKITTI velodyne/world frame:

```python
def scaled_relative_pose(pose_c2w, f, anchor, s):
    ca   = pose_c2w[anchor][:3, 3]
    Pf_s = pose_c2w[f].copy()
    Pf_s[:3, 3] = ca + s * (pose_c2w[f][:3, 3] - ca)     # rotations untouched
    return np.linalg.inv(pose_c2w[anchor]) @ Pf_s
```

The `cam_to_velo` transform is applied afterwards, to the already-fused point cloud, so no
pose is scaled after world alignment.

Asserted at runtime on real sequence-08 poses (first 5 clips, both `s_learned` and
`s_oracle`) and in `tests/depth_gate/test_residual_decomposition.py`:

| assertion | status |
|---|---|
| anchor is a fixed point — `scaled_relative_pose(P, a, a, s) == I` for every `s` | pass |
| rotation block is bit-identical across all `s` | pass |
| relative translation magnitude ratio `‖t(s)‖ / ‖t(1)‖ == s` to 1e-12 | pass |
| scale is not applied twice — `t(s) == s·t(1)` and `t(s) ≠ s²·t(1)` | pass |
| agrees with the frozen Gate-0/1 `relative_metric` convention to 1e-10 | pass |
| coupled scaling is a pure similarity: `fuse(s·D, s·t) == s · fuse(D, t)` to 1e-12 | pass |
| depth-only scaling is **not** a similarity (the C1 inconsistency, demonstrated) | pass |

17 new tests; the full `tests/depth_gate` + `tests/scale_gate` suite is **84 passed**.

---

## 4. Reproduction of the reported Gate-2 results

Required before any new configuration was interpreted:

| config | reported | reproduced | \|Δ\| | within 5e-4 |
|---|---:|---:|---:|:--:|
| C0 constant-scale deployable base | 0.0573 | **0.05725** | 4.9e-5 | ✔ |
| C1 existing full depth-only refinement | 0.0775 | **0.07754** | 3.8e-5 | ✔ |
| O0 oracle-scale LingBot-pose base | 0.0769 | **0.07691** | 1.2e-5 | ✔ |
| G0 oracle-scale GT-pose base | 0.0782 | **0.07817** | 3.1e-5 | ✔ |

The residuals are rounding of the published 4-decimal figures. Base depth metrics also
reproduce exactly: C0 AbsRel 0.2046, δ1 0.7055; C1 AbsRel 0.0898, δ1 0.9158.

---

## 5. Residual and scale statistics

### 5.1 How much of the residual is one scalar?

| statistic | value |
|---|---:|
| median `a_clip` | −0.0285 |
| `a_clip` 5–95 % range | −0.378 … +0.198 |
| `a_clip` min / max | −0.482 / +0.349 |
| median RMS of `r` over support | 0.1151 |
| median RMS of `r_shape` over support | 0.0701 |
| median \|`r_shape`\| over support | 0.0317 |
| residual **variance** explained by `a_clip`, `1 − RMS(r_shape)²/RMS(r)²` (median over clips) | **0.630** |
| same, 5–95 % range | 0.025 … 0.724 |
| `a_clip²/mean(r²)` (median over clips) | 0.897 |

Two thirds of the residual's energy is literally one number per clip. The reason the
remaining third does not buy any occupancy is §6.

The head is also **internally consistent across the five frames of a clip**, which is what
a scale estimator should be and a shape corrector need not be:

| statistic | median | 5–95 % |
|---|---:|---|
| std of the five per-frame residual medians about `a_clip` | 0.0188 | 0.008 … 0.042 |
| range (max − min) of the five per-frame medians | 0.0532 | 0.022 … 0.117 |

The within-clip spread of per-frame medians (0.019 in log units) is an order of magnitude
smaller than the between-clip spread of `a_clip` itself (5–95 % span of 0.576 log units).
The scalar is a clip property, not frame noise.

### 5.2 `s_learned` against the Gate-0 oracle scale

The oracle is `s_joint` from `artifacts/scale_gate/scale_targets_val.csv`, used verbatim
and **for evaluation only** — it never enters `a_clip`.

| statistic | learned `s_learned` | deployable constant `s0` |
|---|---:|---:|
| median \|log scale error\| vs oracle | **0.0295** | 0.1320 |
| median relative scale error | **2.9 %** | 12.7 % |
| Pearson r vs oracle | **0.981** | — (constant) |
| Spearman ρ vs oracle | **0.982** | — |
| Pearson r, log-space | 0.982 | — |

Distribution of `s_learned / s_oracle` over the 163 clips:

| min | p5 | p25 | median | p75 | p95 | max | mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.875 | 0.929 | 0.971 | **0.995** | 1.028 | 1.057 | 1.203 | 0.999 |

Median `s_learned` 26.60 against median `s_oracle` 26.27 (`s0` = 27.37). The head cut the
median scale error by **4.5×** relative to the constant, is essentially unbiased, and its
per-clip ranking of scale agrees with the oracle at ρ = 0.98. That is a scale estimator.

**This is an in-domain SemanticKITTI diagnostic. It does not establish generalisable scale
recovery** — sequence 08 shares camera height, intrinsics, sensor and city with the
training sequences, all of which are strong scale cues a network can memorise.

---

## 6. Depth and occupancy results

### 6.1 Configurations

| id | depth | pose | translation scale |
|---|---|---|---|
| C0 | `s0 · D` | LingBot | `s0` |
| C1 | `s0 · D · exp(r)` | LingBot | `s0` ← *existing Gate-2, uncoupled* |
| C2 | `s_learned · D` | LingBot | `s0` ← *uncoupled control* |
| C3 | `s_learned · D` | LingBot | `s_learned` |
| C4 | `s0 · D · exp(r_shape)` | LingBot | `s0` |
| C5 | `s_learned · D · exp(r_shape)` | LingBot | `s_learned` (depth ≡ C1) |
| O0 | `s_oracle · D` | LingBot | `s_oracle` |
| O1 | `s_oracle · D · exp(r_shape)` | LingBot | `s_oracle` |
| G0 | `s_oracle · D` | **GT** | — |
| G1 | `s_oracle · D · exp(r_shape)` | **GT** | — |

### 6.2 Occupancy — sequence 08, 163 clips

| id | IoU | P | R | occupied voxels | fused points | in-grid | out-of-grid |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.0573 | 0.318 | 0.066 | 23 508 | 318 183 | 0.590 | 0.410 |
| C1 | 0.0775 | 0.358 | 0.092 | 29 341 | 318 184 | 0.623 | 0.377 |
| C2 | 0.0768 | 0.389 | 0.089 | 25 910 | 318 184 | 0.622 | 0.378 |
| **C3** | **0.0778** | 0.408 | 0.090 | 24 717 | 318 184 | 0.640 | 0.360 |
| C4 | 0.0627 | 0.307 | 0.074 | 27 725 | 318 003 | 0.590 | 0.410 |
| C5 | 0.0791 | 0.364 | 0.094 | 29 247 | 318 184 | 0.639 | 0.361 |
| O0 | 0.0769 | 0.406 | 0.088 | 24 249 | 318 150 | 0.637 | 0.363 |
| O1 | 0.0774 | 0.359 | 0.092 | 28 688 | 318 166 | 0.634 | 0.366 |
| G0 | 0.0782 | 0.406 | 0.090 | 24 801 | 318 150 | 0.635 | 0.365 |
| G1 | 0.0777 | 0.359 | 0.092 | 28 963 | 318 166 | 0.632 | 0.368 |

Fused-point counts are flat (318.0–318.2 k) across every configuration: **no configuration
improves by deleting points.** Recall rises with every beneficial change (C0 0.066 →
C3 0.090), so nothing here is a recall collapse traded for precision.

### 6.3 Depth — 14 738 262 valid projected-LiDAR pixels

| id | AbsRel | log RMSE | RMSE (m) | δ1 | median abs err (m) |
|---|---:|---:|---:|---:|---:|
| C0 | 0.2046 | 0.2592 | 5.164 | 0.7055 | 1.570 |
| C1 / C5 | **0.0898** | 0.1714 | 3.864 | 0.9158 | 0.485 |
| C2 / C3 | 0.0982 | 0.1808 | 4.056 | 0.9157 | 0.536 |
| C4 | 0.1852 | 0.2434 | 4.926 | 0.7370 | 1.383 |
| O0 / G0 | 0.0932 | 0.1790 | 4.084 | 0.9170 | 0.461 |
| O1 / G1 | 0.0909 | 0.1717 | 3.862 | 0.9168 | 0.508 |

The single scalar `s_learned` (C2/C3, AbsRel 0.0982, δ1 0.9157) already **matches the
oracle scale** (O0, 0.0932, δ1 0.9170) and beats the constant (C0, 0.2046, δ1 0.7055).
Applying the entire dense residual (C1, 0.0898) buys 0.008 AbsRel beyond the pure scalar
and **zero** δ1 (0.9158 vs 0.9157).

### By distance

| range | C0 | C2/C3 (scalar) | C1 (full) | O0 (oracle) | O1 (oracle+shape) |
|---|---:|---:|---:|---:|---:|
| 0–10 m | 0.2220 | 0.0875 | 0.0766 | 0.0785 | 0.0776 |
| 10–20 m | 0.1914 | 0.0914 | 0.0821 | 0.0880 | 0.0839 |
| 20–40 m | 0.1847 | 0.1222 | 0.1180 | 0.1232 | 0.1195 |
| 40–80 m | 0.2106 | 0.1733 | 0.1801 | 0.1725 | 0.1758 |

At 40–80 m the full residual is **worse** than the pure scalar (0.1801 vs 0.1733) and the
shape residual is worse than the oracle base (0.1758 vs 0.1725) — the "shape" component
actively degrades the far field, which is where occupancy recall is hardest to earn.

### 6.4 Paired clip-level bootstrap (10 000 resamples, seed 0, 163 paired clips)

| contrast | meaning | ΔIoU | 95 % CI | clips improved | ΔAbsRel |
|---|---|---:|---|---:|---:|
| C0 → C1 | original reported improvement | **+0.0203** | [+0.0150, +0.0256] \* | 76.1 % | −0.1151 \* |
| C0 → C2 | scalar correction, depth only | **+0.0195** | [+0.0145, +0.0247] \* | 74.2 % | −0.1065 \* |
| **C0 → C3** | **scalar correction, correctly coupled** | **+0.0206** | **[+0.0155, +0.0258] \*** | 74.8 % | −0.1065 \* |
| C0 → C4 | shape only, deployable constant | +0.0055 | [+0.0038, +0.0070] \* | 76.1 % | −0.0195 \* |
| C3 → C5 | added shape under the learned scale | +0.0012 | [−0.0005, +0.0030] | 58.9 % | −0.0086 \* |
| C1 → C5 | correcting the pose-scale coupling | +0.0015 | [+0.0008, +0.0023] \* | 62.6 % | −0.0000 |
| **O0 → O1** | **pure shape, oracle scale** | **+0.0005** | **[−0.0012, +0.0022]** | 55.8 % | −0.0023 \* |
| **G0 → G1** | **pure shape, oracle scale + GT poses** | **−0.0005** | **[−0.0022, +0.0013]** | 52.8 % | −0.0023 \* |

\* CI excludes zero.

**Fraction of the original C0→C1 IoU gain reproduced:**

| by | fraction |
|---|---:|
| C2 — scalar alone, depth only | 96.2 % |
| **C3 — scalar alone, correctly coupled** | **101.5 %** |
| C4 — shape alone, under the constant | 26.9 % |

---

## 7. What the network actually learned

**It learned one number per clip: the metric scale.**

1. **The scalar is the whole result.** `C3` — one scalar per clip, applied consistently to
   depth and translation, dense residual entirely discarded — reproduces **101.5 %** of the
   reported gain (+0.0206 vs +0.0203). The dense field adds nothing beyond it: `C3→C5` is
   +0.0012 [−0.0005, +0.0030], not significant.

2. **The scalar is a real scale estimate, not a fitted constant.** Median relative error
   against the Gate-0 oracle is 2.9 % against the constant's 12.7 %, with Pearson/Spearman
   0.98 and a ratio distribution centred on 0.995 spanning 0.93–1.06 at 5–95 %. Its
   five per-frame medians agree within 0.019 log units, as a clip-level property should.

3. **The spatial part is not a shape corrector.** Applied where the scale is already
   correct, the *identical* `r_shape` yields +0.0005 (n.s.) with LingBot poses and −0.0005
   (n.s.) with GT poses. Its only measurable depth effect (ΔAbsRel −0.0023 \*, ~2 % of the
   scalar's −0.1065) leaves δ1 unchanged (0.9170 → 0.9168) and makes the 40–80 m band
   *worse*. Statistically detectable, geometrically irrelevant.

4. **C4's +0.0055 is leftover scale, not shape.** The same `r_shape` gains +0.0055 \* on a
   mis-scaled base and nothing on a correctly scaled one. Median-centring in log space
   removes the median but not the mean: `E[exp(r_shape)] > 1` by Jensen, so `r_shape`
   still carries a small isotropic expansion that partially compensates `s0`. Its
   signature is a *scale* signature — precision falls (0.318 → 0.307) while recall rises
   (0.066 → 0.074) as the cloud expands outward, with in-grid fraction pinned at 0.590,
   the value for the *unscaled* C0 — not a shape signature.

5. **The old result was mildly harmed, not caused, by the inconsistent coupling.**
   `C1→C5` = +0.0015 [+0.0008, +0.0023] \*: coupling the learned scale into pose
   translation as well as depth makes things slightly *better*. So the diagnosis is not
   `INCONSISTENT_SCALE_COMPENSATION` — the original number was real, just mislabelled. Its
   own in-grid fraction gives it away: C1 sits at 0.623 while the correctly coupled C5
   reaches 0.639, matching O0 (0.637) — C1 was fusing depth at one scale into a camera
   baseline at another, spraying a fraction of its points out of the grid.

6. **The Gate-1 depth-shape gap is untouched.** Gate 1 attributed −0.0416 IoU (−33.1 % of
   the 0.1255 five-frame visible-reconstruction ceiling) to depth *shape* under matched
   support and matched scale. Nothing in Gate 2 addressed it: the best shape contrast here
   is +0.0005, i.e. **1.2 % of that gap**, and not significant.

### Why the decision rules land here

| rule | requirement | observed |
|---|---|---|
| `SCALE_CORRECTION_ONLY` | C3 ≥ 80 % of C0→C1 | **101.5 %** ✔ |
| | O0→O1 and G0→G1 not significantly positive | +0.0005 n.s., −0.0005 n.s. ✔ |
| | C3 improves occupancy without collapsing recall | +0.0206 \*, recall 0.066 → 0.090 ✔ |
| `SCALE_AND_SHAPE_BOTH_USEFUL` | O1 > O0 **or** G1 > G0 significantly | neither ✘ |
| `SHAPE_CORRECTION_ONLY` | shape significant, scalar not | scalar is the entire effect ✘ |
| `INCONSISTENT_SCALE_COMPENSATION` | C1 gain disappears under correct coupling | C5 (0.0791) > C1 (0.0775) ✘ |

---

## 8. Recommended next project stage

Per `SCALE_CORRECTION_ONLY`:

1. **Retain an explicit clip-level scale component.** Replace the 48 129-parameter dense
   head with a scalar regressor predicting one `log s` per clip. It is what the current
   head does anyway, it makes the estimand auditable, and it removes 5 × 154 × 518 outputs
   per clip in favour of one. Couple it to depth *and* pose translation, as in C3/C5 —
   the coupling is worth a free +0.0015 \*.

2. **Discard the dense residual as a claimed shape refiner.** It should not be described
   as depth-shape correction in any write-up. The Gate-2 verdict
   `DEPTH_REFINEMENT_PASSES` stands numerically but its mechanism is now identified, and
   `reports/depth_gate/depth_refinement_report.md` §6's qualification is confirmed and
   sharpened by this report.

3. **Then proceed to 3D visible-voxel correction, and only afterwards to completion.** The
   remaining headroom is not in per-pixel depth: the correctly scaled configurations
   already sit at 0.078 against a 0.1255 visible-reconstruction ceiling, with 36 % of
   fused points falling outside the grid and recall at 0.090. Operating in voxel space
   attacks that directly, where a per-pixel log residual demonstrably cannot.

4. **Before trusting the scale component, test it out of domain.** Sequence 08 shares
   camera height, intrinsics, sensor and city with sequences 00–07. A scale estimator has
   every opportunity to memorise those. The 2.9 % median error is a ceiling on what this
   head has been shown to do, not evidence it transfers.

**All results in this report are in-domain SemanticKITTI results (sequence 08).** No model
was trained. The recommended next stage has not been started.

