# Gate 9 — depth-to-occupancy geometry audit

No training, no semantics. The 590 five-frame causal clips of KITTI-360 0006 used by the
local-registration audit (replayed after code freeze on 0003/0007/0010: 454 clips), GT
relative poses throughout, one lattice camera model, the anchor SSC grid, one evaluation mask
per clip (voxels measured by the same five LiDAR sweeps and inside a camera frustum). Raw
observed-map metrics only — not completion IoU. 14/14 verification checks
(`verification.md`); tables in `results_table.md` / `results_table_all.md` / `tables/*.csv`.

> **Correction (2026-09-13), naming and interpretation only — no number below changes.** A scale-path audit
> (`artifacts/depth_source_comparison/scale_audit.json`, correction note in
> `artifacts/depth_source_comparison/correction_note.md`) confirms the implementation: the cached local-reset depth is
> raw canonical LingBot output; the causal MoGe-2 clip scale (median of five per-frame MoGe/LingBot log-ratios over
> frames t-4..t) is applied exactly once (`audit.py:122`, mapper scale forced to 1, GT translations metric). Hence
> **"PD" and "GF" are MoGe-scaled LingBot depth (LB-M), not raw LingBot depth**, and wherever this report says
> "LingBot depth" it means LB-M. **"Gs" is a residual LiDAR median ratio (≈1.07) applied on top of the MoGe-scaled
> dense depth**, fitted on the anchor frame's LiDAR pixels; it is a depth-error fit, not an IoU-optimal or upper-bound
> scalar, and its 20.6 % "gap closed" vs PD mixes supports (dense gated vs LiDAR-supported) and is not comparable.
> Section 10's proposed "LingBot depth vs LingBot + causal MoGe-2 scale" comparison was therefore a duplicate; the
> corrected comparison is `artifacts/depth_source_comparison/`.

**Primary conclusion: B — LingBot depth is the main problem.** On matched pixels with GT
pose, replacing LiDAR depth by LingBot depth drops map IoU from 0.438 to 0.075 on 0006
(0.37–0.39 → 0.026–0.036 on the other drives), and no mapper setting recovers it.

| 0006 | V (all returns) | V (camera-visible) | LD | PD (= LB-M) | GF (= LB-M, dense) | Gs (LB-M dense x residual LiDAR ratio) |
|---|---|---|---|---|---|---|
| IoU | 1.000 | 0.852 | 0.438 | 0.075 | 0.083 | 0.149 |
| precision / recall | 1 / 1 | 1.000 / 0.852 | 0.532 / 0.711 | 0.108 / 0.196 | 0.130 / 0.185 | 0.234 / 0.292 |
| occupied on LiDAR-free | 0 | 0 | 0.468 | 0.892 | 0.870 | 0.766 |

## 1. What is the direct-LiDAR voxelization ceiling?

V over all returns is 1.000 by construction: the reference is the same five sweeps, so this
only verifies the transform and voxelization (direct and `rawtarget` voxels identical; the
camera route matches `velo_to_velo` to 8e-13 m). The meaningful ceiling for a camera-based map
is **camera-visible LiDAR: 0.852** (0.82–0.87 on all drives) — precision 1, but 15 % of measured
occupied voxels in the evaluated frusta are only seen by LiDAR returns outside the image.

## 2. How much is lost purely by passing accurate LiDAR depth through the mapper?

About half: 0.852 → **0.438** (0.37–0.44 across drives). Two mechanisms, neither carving: the
±0.2 m occupied band triples the occupied samples per ray and halves precision (0.532; the
endpoint-only band gives 0.747 precision, IoU 0.453), and the 140×518 lattice z-buffer keeps one
depth per pixel so recall of camera-visible voxels is 0.711 (0.535 endpoint-only). Carving is
harmless with correct depth (false-free 0.7 %; IoU with/without 0.438 / 0.434). The mapper is
sane but lossy.

## 3. How much additional loss comes purely from LingBot depth on matched pixels?

**0.438 → 0.075 (−83 %)**; −91 to −93 % on 0003/0007/0010. Identical pixels, rays, poses, grid
and mask; 89 % of LingBot-derived occupied voxels land in LiDAR-measured free space, and only
11.6 % of LiDAR-occupied voxels hold a LingBot endpoint (LD 64 %). With LingBot depth every
mapper variant stays at 0.063–0.079.

## 4. Are LingBot depth errors scale, local residual, edge, range, or a mixture?

A mixture with a strong systematic component. Median AbsRel 0.078 (0.10–0.13 on the other
drives), but median metric error 0.58 m and endpoint error 0.68 m (p90 3.2 m); 72 % of
predictions are in front of the surface (89–92 % elsewhere), median relative error −5 %.
- **Scale/bias:** the best per-clip scalar cuts AbsRel to 0.046 (0.036–0.044 elsewhere), the
  largest single component, yet only lifts within-0.2 m from 16 % to 30 %.
- **Range:** relative error is nearly constant, so metric error grows with range: 0.36 m at
  0–5 m, 1.25 m at 10–20 m, 5.4 m beyond 30 m.
- **Edges:** strong discontinuities double AbsRel (0.138 vs 0.071).
- **Confidence** is informative: endpoint error 4.3 m below the frozen gate, 0.40 m at conf ≥ 12;
  the gate already removes the worst pixels (GF ≈ PD).

## 5. How often is a predicted surface within 0.2 / 0.4 / 0.6 m but in the wrong voxel?

Within 0.2 / 0.4 / 0.6 m: 16.4 / 31.9 / 45.2 %; exact voxel 3.2 %; within 0.2 / 0.4 / 0.6 m
but wrong voxel 13.2 / 28.7 / 42.0 %. On the other drives within 0.6 m is only 18–25 %. Most
endpoints are not close, so 0.2 m brittleness (Decision D) is a secondary effect.

## 6. How often does hard free-space carving turn depth error into false-free geometry?

Rarely overall, severely when depth is too far. PD false-free rate 4.7 % (1.7–2.7 % elsewhere)
vs 0 without carving, and IoU is unchanged (0.075 vs 0.074). Conditional on LingBot depth
beyond the surface by 0.5–1 / 1–2 / > 2 m, the carve passes the true endpoint on 100 % of rays
and that voxel ends FREE in 19 / 28 / 33 % of cases; within ±0.2 m it is < 1 %. Carving amplifies
the minority of too-far errors; it is not the main amplifier (Decision C rejected).

## 7. Is dense unsupported LingBot depth significantly worse than matched supported pixels?

No: GF 0.083 vs PD 0.075 (0.026–0.037 vs 0.026–0.036 elsewhere). The confidence gate on dense
depth removes roughly what LiDAR support would; unsupported pixels add no large extra damage.

## 8. Which component is now the dominant geometry bottleneck?

LingBot depth. The mapper costs ~half of the camera-visible ceiling but remains usable with
correct depth; with LingBot depth the map collapses to 0.03–0.08 IoU regardless of band,
carving, margin or gate, and even a per-clip LiDAR scale oracle only reaches 0.12–0.15.

## 9. What representation should be frozen for the next learned-completion experiment?

Local-reset five-frame LingBot inference (L: depth, confidence, pose, clip MoGe-2 scale) through
the current mapper **unchanged** (band 0.2 m, carving, confidence gate). No mapper variant
improves LingBot-depth maps, so there is nothing to freeze in their favour; endpoint-only
banding helps only accurate depth.

## 10. What single experiment should follow?

[Corrected 2026-09-13: the list below duplicates PD, which already is LingBot x causal MoGe-2 scale; see the correction note above.] A depth-source sufficiency test under GT pose on the same matched pixels and mapper: frozen
LingBot depth vs the already-cached frozen MoGe-2 metric depth vs LingBot with a causal
per-clip scale from MoGe-2 (the current rule) vs the LiDAR scale oracle, reporting the LD–PD gap
closed by each. It decides whether calibration can make the existing depth usable at 0.2 m or
whether the depth source must be replaced or supplemented before completion is retrained.

## Reproduce

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH="$PWD"; PY=/home/minh/anaconda3/envs/cu128/bin/python; T=tools/gate9_depth_occupancy
# prerequisite: local-reset LingBot clips (tools/local_registration_audit/run_local_lingbot.py)
for s in 0 1 2; do $PY $T/audit.py --drive 2013_05_28_drive_0006_sync --shard $s --shards 3 --device cuda:$s; done
$PY $T/analyze.py && $PY $T/maps.py --device cuda:0 && $PY $T/verify.py --device cuda:0
for d in 0003 0007 0010; do $PY $T/audit.py --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
$PY $T/analyze.py --drives 2013_05_28_drive_0006_sync 2013_05_28_drive_0003_sync 2013_05_28_drive_0007_sync 2013_05_28_drive_0010_sync
```
Code hashes at freeze (before the robustness replay): `code_freeze.json`. Per-clip data:
`/media/SSD1/MINH_DATASETS/lingbot_gate9_depth_occupancy/clips/`.
