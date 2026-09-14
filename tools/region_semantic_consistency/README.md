# Region semantic consistency (2026-09-13) -- Decision C

Question: is a coherent 3D region a more view-consistent semantic unit than a voxel?
Geometry-only regions (block-bounded 26-connected components of the LiDAR-occupied grid at
0.6 / 1.0 / 1.8 / 3.0 m), per-frame future teacher evidence from
`tools/future_semantic_completion/target_consistency/`, interleaved frame-disjoint A/B/C.

Result: no scale raises independent cross-view repeatability (0006: voxel 0.561, regions
0.557 / 0.552 / 0.539 / 0.525); regions pool ~2.1-2.25 contributing frames whatever their
size, and the local shuffle reproduces the agreement (true - R2 = +0.01-0.03). Coverage
rises (0.9 % -> 22 % A&B), diversity unchanged. Nothing trained; 10/10 checks.

```bash
export PYTHONPATH="$PWD" G8C1_DATA_ROOT=/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky; PY=/home/minh/anaconda3/envs/cu128/bin/python
# optional second population: all future-observed occupied voxels
for d in 0003 0007 0010 0006; do $PY tools/future_semantic_completion/target_consistency/build_evidence.py --drive 2013_05_28_drive_${d}_sync --population all_future_observed --device cuda:0; done
for d in 0003 0007 0010 0006; do $PY tools/region_semantic_consistency/analyze.py --drive 2013_05_28_drive_${d}_sync; done
$PY tools/region_semantic_consistency/analyze.py --merge
$PY tools/region_semantic_consistency/verify.py
$PY tools/region_semantic_consistency/figures.py
$PY tools/region_semantic_consistency/summarize.py
```

Outputs: `artifacts/region_semantic_consistency/{config.json, analysis.json, analysis_allocc.json,
per_drive/, verification.json, results_table.md, report.md, figures/}`; report copy at
`reports/region_semantic_consistency.md`.
