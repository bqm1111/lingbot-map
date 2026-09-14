# Gate 9 scale correction and controlled depth-source comparison

KITTI-360 only, no training, GT relative poses, raw five-frame observed maps. 2026-09-13.

## Correction

**Gate 9 was already evaluating MoGe-scaled LingBot depth.** Its "PD" and "GF" conditions are
local-reset LingBot depth × the causal MoGe-2 clip scale, applied exactly once. There is **no
implementation error**. The error was in naming and interpretation: the Gate 9 report said
"LingBot depth", and the Gate 10 plan based on it proposed LingBot vs LingBot + MoGe-2 scale,
which duplicates PD. All Gate 9 outputs are preserved unchanged, and a dated correction note was
added to `artifacts/gate9_depth_occupancy/report.md`. The interrupted Gate 10 draft is marked
`artifacts/gate10_depth_source/SUPERSEDED.md`.

### Depth path, with code locations

| step | equation | code |
|---|---|---|
| LingBot output, cached raw | `z_LB` = `preds["depth"]`, canonical units | `gates/gate7b/replay.py:137` |
| per-frame MoGe log-ratio | `l_f = median_{v, MAD-clipped}(log z_MoGe − log z_LB)`; `v` = MoGe mask ∧ MoGe depth in (1, 60) m ∧ LingBot conf ≥ 1.5 | `gates/gate7b/scale.py:63` |
| clip scale, frames t−4..t | `s_clip = exp(median_{f ∈ t−4..t} l_f)`, stored separately as `scale` | `tools/local_registration_audit/run_local_lingbot.py:68,73` |
| Gate 9 PD / GF (= LB-M) | `z = s_clip · z_LB`; mapper scale forced to 1; GT metric translation × 1 | `tools/gate9_depth_occupancy/audit.py:122,129,144` |
| mapper | `d = s · depth_input`; `t_world = s · t_input` | `core/mapping/mapper.py:326,330` (`variant_mapper.py:31` identical) |
| deployable local-reset L | canonical poses and canonical depth with `force(s_clip)`, so depth and translation are both × `s_clip` | `tools/local_registration_audit/geometry_audit.py:79,115` |
| Gate 9 Gs | `z = r · s_clip · z_LB`, with `r = median_{anchor LiDAR px} z_LiDAR / (s_clip z_LB)` ≈ 1.07, a **residual** ratio on already-scaled dense gated depth | `tools/gate9_depth_occupancy/audit.py:124,136` |
| MoGe-2 depth | optical-axis `z` in metres (asserted `depth == points[...,2]`), calibrated fov_x 103.745° on the same LingBot-preprocessed 140×518 RGB | `moge_gauge/calibrated.py:38,42` |

### Numerical checks (`scale_audit.json`, 0006 clips 4 / 16 / 1549 / 1747)

- **Cached depth is raw.** Fresh local-reset re-inference reproduces the cache bit for bit. The
  median cached depth at LiDAR pixels is 0.59 / 0.35 / 0.32 / 0.36, against 10.9 / 5.9 / 7.6 /
  9.9 m of LiDAR depth.
- **The scale is causal.** `s_clip` recomputed from frames t−4..t alone equals the stored value
  exactly (15.52 / 20.63 / 21.43 / 25.40).
- **The scale is applied once.** Gate 9 stored depth = cached × `s_clip` (max relative difference
  6e-8, float32 rounding), and the mapper scale is 1.0.
- **Conventions agree.** The Gate 9 convention (scaled depth, metric GT translation, scale 1)
  reproduces the stored PD map exactly. The deployable convention (canonical depth, translation /
  `s`, scale `s`) differs from it by **0 voxels**. Applying the scale twice produces 0 occupied
  voxels, because all depths land beyond the 60 m range gate.
- **Pose translations use the same scale.** Clip translation t−4→t in canonical units is
  0.34 / 0.96 / 0.80 / 0.81. Multiplied by `s_clip` that is 5.3 / 19.8 / 17.2 / 20.5 m, against GT
  6.7 / 19.3 / 20.6 / 22.7 m. GT poses recomputed from `cam0_to_world` are identical and metric.
- **Gs is a residual.** The stored ratio (1.167 / 0.810 / 1.101 / 1.100) is recomputed exactly
  on the scaled depth.
- **The earlier Gs gap figure is not comparable.** Gs closed 20.6 % of the LD–PD gap (20.4 % from
  rounded IoUs), but Gs used dense gated pixels and PD used LiDAR support. It is not reused below.

## Controlled comparison

**Everything identical except the depth values:** Gate 9 clips, GT poses, lattice rays, grid,
evaluation masks and the frozen mapper (band 0.2 m, carving on).

**Primary mask S1:** Gate 9 LiDAR support ∧ MoGe-2 valid ∧ the frozen LingBot confidence gate
(≥ 1.5). It is built before any depth array, so it cannot depend on the compared depths. MoGe and
LingBot share the lattice, so no resampling is needed.
- S1 removes 6.6 / 7.7 / 5.9 / 7.0 % of Gate 9 support on 0006 / 0003 / 0007 / 0010.
- Almost all of that removal is the confidence gate; the MoGe mask removes ≤ 0.06 %.
- The mapper's 1–60 m range gate drops ≤ 0.05 % of S1 rays for any source.

**Conditions:**
- **LD:** LiDAR depth.
- **LB-M:** `s_clip · z_LB` (the existing baseline).
- **MG:** MoGe-2 z-depth, no LingBot scale.
- **LB-O:** `s* · z_LB`, with `s* = median z_LiDAR / z_LB` over the clip's S1 pixels (all five
  frames). This minimises Σ|log(s z_LB) − log z_LiDAR|. It is a depth-error fit, not an IoU optimum
  or an upper bound. Diagnostic only.

**How the metrics are computed:**
- IoU pools voxel counts over clips.
- Depth metrics pool Gate 9's deterministic 20 % pixel subsample inside S1.
- Bias = median of (z − z_LiDAR)/z_LiDAR.
- AbsRel is given as median / mean.
- Endpoint error = |z − z_LiDAR|·|ray|.

### Occupancy, S1

| drive | LD | LB-M | MG | LB-O | gap closed MG [95 % CI] | gap closed LB-O [95 % CI] | occupied in LiDAR-free: LD / LB-M / MG / LB-O |
|---|---|---|---|---|---|---|---|
| 0006 (590 clips) | 0.374 | 0.079 | 0.088 | 0.141 | +2.8 % [+1.6, +4.2] | +21.0 % [+18.4, +23.8] | 0.457 / 0.868 / 0.860 / 0.764 |
| 0003 (65) | 0.323 | 0.035 | 0.045 | 0.115 | +3.7 % [+1.9, +7.2] | +27.8 % [+24.0, +31.7] | 0.553 / 0.946 / 0.930 / 0.816 |
| 0007 (190) | 0.346 | 0.031 | 0.040 | 0.123 | +2.7 % [−0.1, +5.8] | +29.1 % [+25.1, +33.8] | 0.525 / 0.949 / 0.936 / 0.799 |
| 0010 (199) | 0.347 | 0.025 | 0.028 | 0.116 | +1.0 % [−0.4, +2.7] | +28.5 % [+26.2, +31.0] | 0.534 / 0.963 / 0.958 / 0.821 |

- **Gap closed** = (IoU_X − IoU_LB-M) / (IoU_LD − IoU_LB-M) on S1. CIs are a clip-block
  bootstrap (blocks of 20 clips, 1,000 resamples).
- **Precision / recall:** these are in `comparison_table[_all].md`. On 0006 they are
  LD 0.543 / 0.546, LB-M 0.132 / 0.165, MG 0.140 / 0.189 and LB-O 0.236 / 0.260.
- **Original Gate 9 numbers:** these were on unmasked LiDAR support without the confidence gate,
  and are reproduced exactly and kept separate: LD 0.438 / 0.369 / 0.378 / 0.393 and
  PD (= LB-M) 0.075 / 0.036 / 0.031 / 0.026.
- **Ungated shared mask S0:** the gap closed is MG +1.6 / +2.0 / +2.0 / +0.7 % and LB-O
  +14.1 / +21.7 / +24.2 / +21.8 %.

### Depth, S1 pixels

| drive | source | bias rel (m) | AbsRel med / mean | endpoint med / p90 (m) | within 0.2 m | within 0.6 m | exact 0.2 m voxel |
|---|---|---|---|---|---|---|---|
| 0006 | LB-M | −0.056 (−0.38) | 0.076 / 0.101 | 0.64 / 2.45 | 0.173 | 0.478 | 0.034 |
| 0006 | MG | −0.056 (−0.38) | 0.080 / 0.106 | 0.67 / 2.72 | 0.167 | 0.461 | 0.034 |
| 0006 | LB-O | 0.000 | 0.043 / 0.075 | 0.37 / 1.91 | 0.317 | 0.653 | 0.073 |
| 0003 | LB-M | −0.099 (−0.83) | 0.102 / 0.118 | 1.03 / 3.65 | 0.061 | 0.269 | 0.009 |
| 0003 | MG | −0.098 (−0.80) | 0.103 / 0.120 | 1.03 / 4.01 | 0.077 | 0.280 | 0.012 |
| 0003 | LB-O | 0.000 | 0.033 / 0.060 | 0.35 / 2.35 | 0.343 | 0.642 | 0.064 |
| 0007 | LB-M | −0.126 (−1.03) | 0.129 / 0.144 | 1.26 / 4.63 | 0.051 | 0.204 | 0.009 |
| 0007 | MG | −0.126 (−1.00) | 0.130 / 0.149 | 1.27 / 5.09 | 0.057 | 0.215 | 0.010 |
| 0007 | LB-O | 0.000 | 0.041 / 0.075 | 0.44 / 2.86 | 0.302 | 0.578 | 0.052 |
| 0010 | LB-M | −0.119 (−0.98) | 0.122 / 0.138 | 1.20 / 4.23 | 0.041 | 0.194 | 0.008 |
| 0010 | MG | −0.120 (−0.98) | 0.122 / 0.140 | 1.18 / 4.72 | 0.054 | 0.216 | 0.011 |
| 0010 | LB-O | 0.000 | 0.034 / 0.067 | 0.36 / 2.35 | 0.334 | 0.645 | 0.074 |

### Paired per-clip IoU differences, S1 (pooled difference [95 % CI]; share of clips improved)

| drive | MG − LB-M | LB-O − LB-M | LD − LB-O |
|---|---|---|---|
| 0006 | +0.008 [+0.005, +0.012]; 59 % | +0.062 [+0.053, +0.072]; 86 % | +0.233; 100 % |
| 0003 | +0.011 [+0.005, +0.021]; 69 % | +0.080 [+0.069, +0.092]; 100 % | +0.209; 100 % |
| 0007 | +0.008 [−0.000, +0.018]; 57 % | +0.092 [+0.073, +0.110]; 99 % | +0.223; 100 % |
| 0010 | +0.003 [−0.001, +0.008]; 51 % | +0.092 [+0.083, +0.101]; 100 % | +0.231; 100 % |

### Range, S1: within 0.2 m, LB-M / MG / LB-O

| range | 0006 | 0003 | 0007 | 0010 |
|---|---|---|---|---|
| 0–5 m | 0.295 / 0.290 / 0.450 | 0.130 / 0.186 / 0.605 | 0.149 / 0.167 / 0.569 | 0.101 / 0.166 / 0.585 |
| 5–10 m | 0.180 / 0.172 / 0.362 | 0.070 / 0.088 / 0.469 | 0.051 / 0.055 / 0.404 | 0.045 / 0.056 / 0.445 |
| 10–20 m | 0.085 / 0.079 / 0.170 | 0.041 / 0.049 / 0.195 | 0.033 / 0.038 / 0.164 | 0.026 / 0.027 / 0.194 |
| > 20 m | ≤ 0.043 for all three sources on every drive; LB-O ≤ 0.088 | | | |

Endpoint medians by range, and the within-0.6 m and bias columns, are in `tables/range_*.csv`.

### Supporting diagnostics

- **MG and LB-M are genuinely different depth maps.** Their median difference is 0.31–0.37 m, and
  they differ by more than 0.2 m on 64–69 % of S1 pixels.
- **Their errors are correlated (0.63–0.70) and they share the same metric bias.** `s_clip` is a
  median MoGe/LingBot ratio, so LB-M inherits MoGe's scale.
- **Both under-scale metric depth.** The LiDAR fit exceeds the MoGe clip scale by a median of
  6 / 10 / 15 / 14 %.

## Answers (tested configuration only)

1. **Was Gate 9 already evaluating MoGe-scaled LingBot depth?** Yes. PD and GF are
   `s_clip · z_LB` with the causal five-frame MoGe-2 scale applied once. Gs adds a residual LiDAR
   ratio on top of that.
2. **Implementation error, or naming/interpretation error?** Naming and interpretation only. The
   scale is causal, applied exactly once, and matches the deployable convention (0-voxel
   difference). GT translations are metric and unchanged. No result was superseded.
3. **Does MoGe-2's own depth improve observed occupancy over borrowing only its scale?** Barely.
   - IoU gain is +0.003 to +0.011, closing 1.0–3.7 % of the LD–LB-M gap. The CI includes 0 on 0007
     and 0010.
   - Occupied voxels in LiDAR-free space fall by only 0.5–1.6 points.
   - Exact 0.2 m voxel hits are unchanged at 0.8–3.4 %.
   - MG has the same metric bias as LB-M and a longer p90 endpoint tail.
4. **How much error remains after the LiDAR-fitted scalar?** Most of it.
   - LB-O reaches IoU 0.115–0.141, so 71–79 % of the LD–LB-M gap remains.
   - Endpoint error is still 0.35–0.44 m median (p90 1.9–2.9 m); only 30–34 % of endpoints are
     within 0.2 m and 5–7 % in the exact voxel.
   - 76–82 % of occupied voxels still fall in LiDAR-measured free space.
   - This is a depth-error fit, not an IoU bound. Even so, it gains roughly 7–30× more IoU than
     switching to MG (+0.06–0.09 vs +0.003–0.011).
5. **Does the evidence justify testing direct MoGe-2 depth next in the causal pipeline?** No, not
   as a fix for observed occupancy. Under GT pose on controlled support, MoGe-2's own depth is
   statistically indistinguishable from, or only marginally better than, the MoGe-scaled LingBot
   depth the pipeline already uses. Both leave 86–96 % of predicted occupied voxels in
   LiDAR-free space.
   - The larger measured lever is metric scale: both sources under-scale by 6–15 %.
   - Even a LiDAR-fitted scalar leaves most of the gap.
   - This does not test MoGe-2 on dense unsupported pixels, other MoGe confidence policies, other
     voxel sizes, or estimated pose.

GT-pose raw KITTI-360 map metrics do not establish deployable performance. They neither reproduce
nor invalidate the earlier 30.89 % Occ3D result.

## Records

- `comparison_table.md` (0006) and `comparison_table_all.md`.
- `analysis[_all].json` and `tables/*.csv`.
- `scale_audit.json`.
- `verification.md` / `.json`. Check 14's static token scan was amended after its first run
  flagged the prose disclaimer string in `analyze.py`; runtime file-access violations were 0 in
  every shard.
- `code_freeze.json`, frozen after the 0006 analysis and before the 0003/0007/0010 replay.
- `figures/` and `figures/maps/`.
- `logs/`.
- Per-clip data: `/media/SSD1/MINH_DATASETS/lingbot_depth_source_comparison/clips/`.

## Reproduce

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH="$PWD" CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python; T=tools/depth_source_comparison; L=artifacts/depth_source_comparison/logs; mkdir -p $L
$PY $T/scale_audit.py --device cuda:0 > $L/scale_audit.log 2>&1
for s in 0 1 2 3; do $PY $T/compare_sources.py --drive 2013_05_28_drive_0006_sync --shard $s --shards 4 --device cuda:$s > $L/compare_0006_s$s.log 2>&1 & done; wait
$PY $T/analyze.py                                   # code frozen after this step (code_freeze.json)
for d in 0003 0007 0010; do $PY $T/compare_sources.py --drive 2013_05_28_drive_${d}_sync --device cuda:1 > $L/compare_${d}.log 2>&1; done
$PY $T/analyze.py --drives 2013_05_28_drive_0006_sync 2013_05_28_drive_0003_sync 2013_05_28_drive_0007_sync 2013_05_28_drive_0010_sync
$PY $T/verify.py --device cuda:0
```
