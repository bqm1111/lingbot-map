# Gate 11 — uncertainty-aware observed geometry

**Decision: B — useful only at short range. Here that means 0–20 m, which is wider than the rule's
nominal ~10 m.**

This is a diagnostic only: no training, KITTI-360 only, GT relative poses, and no new depth
source. The depth is local-reset LB-M (`s_clip · z_LingBot`). Rays are Gate 9's LiDAR support
passing the frozen confidence gate (≥ 1.5). The code and uncertainty widths were frozen after the
0006 analysis and replayed unchanged on 0003, 0007 and 0010. Verification passes 15/15
(`verification.md`).

**Treating a depth as an interval recovers substantially more of the true surface than exact
endpoint occupancy, but it does so by claiming mostly wrong space.** On all four drives the
asymmetric interval (U4) reaches about 50 % coverage of LiDAR-occupied voxels. Hard LB-M geometry
covers only 7–17 %. U4 leaves 10–15 % of the evaluation mask uncertain and does not increase
false-free geometry. It passes the predeclared usefulness guide within 0–20 m on every drive, and
fails beyond 30 m on every drive.

Its containment efficiency is still only 9–16 %: 84–91 % of the voxels U4 claims as surface or
uncertain are LiDAR-measured free. The surface channel is the unchanged mapper, so its false
certainty is not reduced at all: 87–96 % of surface voxels contradict LiDAR.

## Result table (all voxels in the evaluation mask; counts pooled over clips)

| drive | condition | surface coverage (S) | band coverage (S or U) | uncertain / mask | containment efficiency | false certainty | false-free | IoU (S) |
|---|---|---|---|---|---|---|---|---|
| 0006 (590) | H | 0.165 | 0.165 | 0 | 0.132 | 0.264 | 0.028 | 0.079 |
| | U1 0.4 / 0.6 m | 0.166 | 0.205 / 0.244 | 0.013 / 0.025 | 0.138 / 0.142 | 0.264 / 0.266 | 0.028 / 0.026 | 0.079 |
| | U2 range | 0.166 | 0.405 | 0.103 | 0.129 | 0.283 | 0.013 | 0.079 |
| | U3 range + confidence | 0.166 | 0.465 | 0.139 | 0.122 | 0.295 | 0.009 | 0.079 |
| | **U4 asymmetric** | 0.166 | **0.505** | 0.101 | **0.163** | 0.267 | 0.024 | 0.079 |
| 0003 (65) | H / U2 / U3 / **U4** | 0.088 | 0.088 / 0.358 / 0.449 / **0.541** | 0 / 0.145 / 0.194 / 0.148 | 0.054 / 0.068 / 0.069 / **0.101** | 0.292 / 0.319 / 0.332 / 0.299 | 0.013 / 0.005 / 0.003 / 0.010 | 0.035 |
| 0007 (190) | H / U2 / U3 / **U4** | 0.075 | 0.075 / 0.292 / 0.380 / **0.494** | 0 / 0.142 / 0.192 / 0.152 | 0.051 / 0.061 / 0.064 / **0.099** | 0.297 / 0.325 / 0.339 / 0.303 | 0.009 / 0.003 / 0.002 / 0.006 | 0.031 |
| 0010 (199) | H / U2 / U3 / **U4** | 0.068 | 0.068 / 0.311 / 0.400 / **0.514** | 0 / 0.149 / 0.196 / 0.153 | 0.037 / 0.053 / 0.056 / **0.087** | 0.306 / 0.334 / 0.347 / 0.312 | 0.010 / 0.004 / 0.003 / 0.008 | 0.025 |

U1 on the replay drives: U1 0.4 m gives band coverage 0.118 / 0.101 / 0.092, and U1 0.6 m gives
0.149 / 0.127 / 0.119, with 1.6–3.1 % of the mask uncertain.

**Paired clip-block bootstrap CIs** (`results_table[_all].md`, `tables/paired_*.csv`):
- Band-coverage gain over H: U4 +0.34 / +0.45 / +0.42 / +0.45 and U2 +0.24 / +0.27 / +0.22 / +0.24;
  every CI excludes 0.
- Coverage is measured over all LiDAR-occupied voxels in the evaluation mask, while rays come only
  from Gate 9's LiDAR support passing the confidence gate. So "coverage" is coverage of LiDAR-measured occupied
  voxels, not of all scene surfaces.

### Representation and definitions (fixed before any occupancy number)

**Voxel states:**
- **S (surface):** the unchanged mapper's log-odds over the ±0.2 m band and carving samples is > 0.
- **F (definite free):** that log-odds is ≤ 0 and the voxel has free weight.
- **U (uncertain):** neither S nor F, but inside at least one ray's interval `[z − u_front, z + u_back]`.
- **Unknown:** no evidence.

**Carving:**
- **Conservative:** stops at `z − u_front`.
- **Hard (the `h` suffix):** stops at the current `z − 0.2 m`.
- H is the unchanged mapper. It equals Gate 9's gated LB-M map (PD_gate) on 1044/1044 clips.

**Binary conversions (secondary metrics only):** occupied := S; the "optimistic" conversion uses
S or U.

**Metrics, with R = LiDAR-occupied and Fr = LiDAR-free voxels in the mask E:**
- **Containment efficiency** = |(S ∪ U) ∩ R| / |(S ∪ U) ∩ E|.
- **False certainty** = (|S ∩ Fr| + |F ∩ R|) / (|S ∩ E| + |F ∩ E|).
- **Voxel range:** the anchor-camera z of the voxel centre.

**Predeclared guide (`g11.py`).** A representation is "useful" if all of the following hold:
- band coverage ≥ 0.50;
- containment efficiency ≥ H's precision;
- false-free rate ≤ H's;
- uncertain voxels ≤ 25 % of the mask.

Other guide criteria:
- **Substantial gain:** band coverage exceeds H's by ≥ 0.15.
- **Confidence benefit:** U3's uncertain volume ≤ 0.9 × U2's, losing ≤ 0.02 coverage.
- **Carving benefit:** conservative false-free rate ≤ 0.8 × hard.

### Width rules (`uncertainty_rules.json`)

These were derived from 0006 depth residuals only, before any occupancy run, and re-derivation
reproduces them exactly. The residual is `r = (z_LiDAR − z_LBM)·|ray|`.
- **U1:** a fixed half-width u ∈ {0.2, 0.4, 0.6} m.
- **U2:** `u2 = max(0.2, 0.066 + 0.127 z)`. This is a count-weighted least-squares fit of
  P68(|r|) per 2 m bin of predicted depth z.
- **U3:** `u3 = max(0.2, u2 · g(c))`, where g = 1.50 / 1.16 / 0.98 / 0.88 for confidence
  [1.5, 3) / [3, 6) / [6, 12) / ≥ 12. g is P68(|r| / u2) per bucket, and was already monotone.
- **U4:** `u_front = max(0.2, 0.308·u3)` and `u_back = max(0.2, 1.395·u3)`, from the 16th/84th
  percentiles of `r/u3`. Asymmetry ratio 4.5 ≥ the predeclared 1.5: 73 % of true surfaces lie
  behind the prediction (median r = +0.44 m).
- **Containment on the 0006 rows:** U1 0.17 / 0.34 / 0.48; U2 0.675; U3 0.680; U4 0.685.
  Mean interval widths are 2.28 / 2.53 / 2.17 m for U2 / U3 / U4.

## Required comparisons

**H vs U1.** A fixed interval of up to 0.6 m recovers little. U1 0.4 m adds 2.5–4.0 points of
coverage and U1 0.6 m adds 5.1–7.8 points, at 1.3–3.1 % of the mask uncertain. Neither reaches the
+15-point "substantial" bar. The typical residual is about 1 m (P68 of |r| = 1.02 m on 0006), so a
0.6 m band is simply too narrow.

**U1 vs U2.** Range dependence reaches coverage a fixed ≤ 0.6 m band cannot (+22–27 points over H),
but it is not clearly more efficient per claimed voxel.
- Containment efficiency, U2 − U1 0.6 m: −0.013 on 0006 [−0.018, −0.008], then +0.006 / +0.003 /
  +0.009 on the replay drives (CIs include 0 on 0003 and 0007).
- On 0006, U2 sits below the U1 efficiency trend extrapolated to its coverage (0.129 vs 0.164).

**U2 vs U3.** Confidence does not reduce uncertainty volume; it redistributes it toward
low-confidence rays and adds volume overall. U3 − U2:
- coverage +6.0 / +9.1 / +8.8 / +8.9 points;
- uncertain fraction +3.7 / +5.0 / +4.9 / +4.7 points;
- containment efficiency −0.007 to +0.003.

The confidence-benefit criterion fails on every drive. Per bucket, U3 contains 68 % of endpoints on
0006 (where it was fitted) but only 32–59 % on the replay drives, whose errors at equal confidence
are larger. At confidence ≥ 12 the median endpoint error is 0.40 m on 0006 and 0.69–0.86 m
elsewhere.

**Asymmetry, U3 vs U4 (the one refinement that helps).** U4 − U3:
- containment efficiency +0.030 to +0.041;
- coverage +4.0 to +11.4 points;
- uncertain fraction −3.8 to −4.6 points;

all consistent across drives.

**Hard vs conservative carving.** With the same intervals, stopping carving before the interval
removes false-free geometry: 53–78 % of it for U2/U3, but only 15–29 % for U4, whose `u_front` is
short.

| drive | U2 hard → conservative | U3 hard → conservative | U4 hard → conservative |
|---|---|---|---|
| 0006 | 0.028 → 0.013 | 0.028 → 0.009 | 0.028 → 0.024 |
| 0003 | 0.013 → 0.005 | 0.013 → 0.003 | 0.013 → 0.010 |
| 0007 | 0.009 → 0.003 | 0.009 → 0.002 | 0.009 → 0.006 |
| 0010 | 0.010 → 0.004 | 0.010 → 0.003 | 0.010 → 0.008 |

All reductions have CIs excluding 0. The cost is fewer correct free claims: U3's free-correct rate
falls from 0.171 to 0.144 on 0006. False certainty also rises slightly, because surface claims that
carving would have erased survive. The carving-benefit criterion holds for U2 and U3 on all drives,
and for U4 on the three replay drives.

## Range, edges and confidence

**By voxel range, U4.** Each cell is band coverage / containment efficiency / uncertain fraction.

| drive | 0–5 m | 5–10 m | 10–20 m | 20–30 m | > 30 m |
|---|---|---|---|---|---|
| 0006 | 0.585 / 0.252 / 0.089 | 0.637 / 0.188 / 0.099 | 0.627 / 0.139 / 0.138 | 0.409 / 0.103 / 0.127 | 0.097 / 0.104 / 0.035 |
| 0003 | 0.560 / 0.159 / 0.094 | 0.603 / 0.121 / 0.118 | 0.656 / 0.097 / 0.163 | 0.588 / 0.075 / 0.202 | 0.301 / 0.082 / 0.141 |
| 0007 | 0.515 / 0.153 / 0.116 | 0.555 / 0.122 / 0.125 | 0.590 / 0.090 / 0.174 | 0.512 / 0.071 / 0.201 | 0.251 / 0.072 / 0.121 |
| 0010 | 0.546 / 0.122 / 0.109 | 0.596 / 0.100 / 0.131 | 0.622 / 0.082 / 0.173 | 0.536 / 0.068 / 0.206 | 0.269 / 0.074 / 0.127 |

**The per-bin useful criteria:**
- **0–20 m:** met on every drive.
- **20–30 m:** met on 0003, 0007 and 0010, but not 0006 (coverage 0.41).
- **> 30 m:** met on none.

Beyond 20 m the 68 % intervals are 6–10 m long, and false certainty is 0.51–0.80.

**Edges (Gate 9 classes).** Uncertainty does not particularly help at depth boundaries; they remain
the worst rays.
- **Containment:** U4 contains 0.45–0.51 of endpoints at strong discontinuities, against 0.57–0.71
  on non-edge rays.
- **False-free at strong edges:** 2.2–6.0 % under U4 versus 0.7–3.1 % on non-edge rays. U3 lowers
  it by 31–48 % relative, but does not close the gap.

**Confidence.**
- **Width (U3/U4):** low-confidence rays ([1.5, 3)) receive 5.7–8.6 m intervals, and
  high-confidence rays (≥ 12) 1.1–1.4 m.
- **Contamination:** the fraction of predicted endpoints landing in LiDAR-free space falls from
  0.49–0.61 to 0.16–0.27 as confidence rises.
- **Summary:** confidence is informative about error size, but using it to scale widths buys
  coverage with volume, not efficiency.

**Accurate-depth control (LD, 0006).** The uncertainty logic does not damage good geometry, and
widening produces no artificial benefit:
- the surface channel and IoU(S) are unchanged (0.374);
- containment efficiency falls from 0.543 to 0.425 / 0.218 / 0.180 / 0.301 under U1 0.6 / U2 / U3 / U4;
- optimistic IoU falls from 0.374 to 0.17–0.33.

## Answers

1. **True-surface coverage of hard LB-M geometry:** 16.5 / 8.8 / 7.5 / 6.8 % of LiDAR-occupied
   voxels on 0006 / 0003 / 0007 / 0010, with only 3.4 % (0006) to 0.8–1.0 % of rays in the exact
   voxel. 87–96 % of its surface voxels are LiDAR-free, and false-free is 0.9–2.8 %.
2. **A fixed interval:** a 0.4 m / 0.6 m band adds 2.5–4.0 / 5.1–7.8 points of coverage. That is
   not substantial.
3. **Range-dependent uncertainty:** yes for reachable coverage (+22–27 points at 10–15 % of the mask
   uncertain), but no for efficiency. The trade-off improves only when asymmetry is added (U4).
4. **LingBot confidence:** no. It adds uncertain volume (+3.7–5.0 points) for its coverage gain,
   leaves efficiency unchanged, and its 0006 calibration undercovers on the replay drives.
5. **False-free avoided by conservative carving:** for the same intervals, 53–78 % of false-free
   geometry is removed for U2/U3 and 15–29 % for U4. The cost is
   fewer correct free claims.
6. **Uncertainty volume needed for 50 / 75 / 90 % coverage:**
   - **50 %** is reached at 17 / 21 / 22 % of the mask claimed (surface or uncertain), of which
     10 / 15 / 15 % is uncertain, on 0006 / 0003 / 0010. It is not reached on 0007 (maximum 49.4 %).
   - **75 % and 90 %** are reached by no rule on any drive.
   - **Per ray on 0006,** the half-width needed for 75 % containment is 0.74 / 1.09 / 2.06 / 3.69 /
     5.30 m across the five range bins, and for 90 % it is 1.21 / 1.85 / 3.36 / 5.82 / 8.09 m.
7. **Where it ceases to be useful:** beyond 20 m. It fails on 0006 at 20–30 m and on every drive
   beyond 30 m, where 6–10 m intervals still cover only 10–30 % of the surface.
8. **Is exact 0.2 m endpoint occupancy the wrong observation representation?** Yes, for this depth
   source. Exact endpoints miss 83–93 % of the LiDAR-occupied voxels an interval can place
   in-band.
   - An interval is the better observation.
   - But even the best interval is mostly empty space (9–16 % efficiency).
   - It does not correct the false certainty of the surface channel.
   - It is an input for a learned completion model, not a usable occupancy map on its own.
9. **Representation to freeze for the next completion experiment:** three separate channels from
   LB-M rays with confidence ≥ 1.5:
   - **Channels:** S (the unchanged ±0.2 m band), U (the U4 interval), and F (conservative carving
     to `z − u_front`).
   - **Widths:** `u_front = max(0.2, 0.308 u3)`, `u_back = max(0.2, 1.395 u3)`,
     `u3 = max(0.2, (0.066 + 0.127 z)·g(c))`, g = 1.50 / 1.16 / 0.98 / 0.88.
   - **Range limit:** rays with predicted depth ≥ 20 m contribute nothing; their voxels stay
     unknown.
   - **Untested parts:** the 20 m ray cut is a new rule set from the voxel-range results and has
     not been evaluated as its own map condition. Dense unsupported pixels were not tested.
10. **Next experiment:** retrain the minimal KITTI-360 geometry-completion model (0003 / 0007 /
    0010, validate on 0006) on this frozen range-limited S/U/F input, against the same model on
    the current hard H input.

## Limitations

- Rays are restricted to LiDAR-supported pixels, and all poses are GT. Deployable performance
  (estimated pose, dense pixels) is not established.
- The widths were fitted on 0006 depth residuals. The errors are larger on the replay drives, so
  they undercover there.
- The S channel was kept identical to H by design, so surface false certainty could not change. A
  representation that demotes low-confidence surface claims was not tested.
- These raw observed-map metrics say nothing about completion IoU. They neither reproduce nor
  invalidate the earlier Occ3D result.

## Figures

- `figures/1_2_width_rules.png`: width vs range and vs confidence.
- `figures/3_4_range_{0006,0003,0007,0010}.png`: coverage, false certainty and false-free vs range.
- `figures/5_volume_vs_coverage.png`: uncertainty volume vs surface coverage.
- `figures/6_side_view_rays.png`: H / U1 / U2 / U3 rays along one image column.
- `figures/maps/7_*.png`: good / median / bad clip map slices for all four drives.
- `figures/8_example_categories.png` and `figures/examples.json`: exact-depth failure, containment
  success, overly broad interval, and carving prevented, each at the anchor image column with the
  most such rays on the median 0006 clip.

The representative clips follow the predeclared Gate 9 PD_gate IoU percentile rule.

## Files

- `results_table.md` (0006 analysis, pre-freeze) and `results_table_all.md`.
- `analysis[_all].json`.
- `tables/*.csv`.
- `uncertainty_rules.json`.
- `verification.md` / `.json`.
- `code_freeze.json`, covering the code plus the rules, frozen at 21:47:26 before the replay.
- `logs/`.
- Per-clip data: `/media/SSD1/MINH_DATASETS/lingbot_gate11_uncertainty/clips/`.

## Reproduce

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH="$PWD" CUDA_HOME=/usr/local/cuda-12.8 HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python; T=tools/gate11_uncertainty; L=artifacts/gate11_uncertainty/logs; mkdir -p $L
$PY $T/derive_rules.py
for s in 0 1 2 3; do $PY $T/audit_u.py --drive 2013_05_28_drive_0006_sync --shard $s --shards 4 --device cuda:$s > $L/audit_0006_s$s.log 2>&1 & done; wait
$PY $T/analyze_u.py && $PY $T/figures_u.py                     # freeze here (code_freeze.json)
for d in 0003 0007 0010; do $PY $T/audit_u.py --drive 2013_05_28_drive_${d}_sync --device cuda:1 > $L/audit_${d}.log 2>&1; done
$PY $T/analyze_u.py --drives 2013_05_28_drive_0006_sync 2013_05_28_drive_0003_sync 2013_05_28_drive_0007_sync 2013_05_28_drive_0010_sync
for d in 0003 0007 0010; do $PY $T/figures_u.py --drive 2013_05_28_drive_${d}_sync; done
$PY $T/verify_u.py --device cuda:0
```
