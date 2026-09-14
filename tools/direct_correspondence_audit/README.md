# Direct semantic-correspondence audit (2026-09-13) -- Decision C, mapper dominant

Physical points from raw KITTI-360 LiDAR + GT poses, projected with the calibrated camera,
z-buffer visibility against each frame's own sweep, Trident-H dense feature sampled in
image space; controls C0-C3 (+ same-view local); mapper association replayed from its
frozen inputs for comparison. Nothing trained; 12/12 checks.

Key numbers (0006, 1.60 M strict pairs): same-point class agreement 0.729 (mapper-voxel
0.568), feature cos 0.942 vs local 0.932 (AUC 0.561; mapper 0.520) vs same-view local
0.974; agreement 0.81 -> 0.47 from < 5 m / < 10 deg to > 30 deg. Mapper replay: any B
observation in voxel(X) 5.5 %; pose reprojection error 93 px median (30 -> 156 px along
the stream), depth 11.5 %; same-voxel agreement 0.53 vs 0.69 direct on the same pairs.

```bash
export PYTHONPATH="$PWD" G8C1_DATA_ROOT=/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky; PY=/home/minh/anaconda3/envs/cu128/bin/python
for d in 0006 0003 0007 0010; do $PY tools/direct_correspondence_audit/build_pairs.py --drive 2013_05_28_drive_${d}_sync; $PY tools/direct_correspondence_audit/sample_features.py --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
$PY tools/direct_correspondence_audit/mapper_compare.py --device cuda:0 --every 4
$PY tools/direct_correspondence_audit/analyze.py; $PY tools/direct_correspondence_audit/figures.py; $PY tools/direct_correspondence_audit/panels.py
$PY tools/direct_correspondence_audit/verify.py; $PY tools/direct_correspondence_audit/summarize.py
```

Outputs: `artifacts/direct_semantic_correspondence_audit/{config.json, pairs/, sampling/, analysis.json,
mapper_comparison.json, verification.json, results_table.md, report.md, figures/, panels/}`; report copy
at `reports/direct_semantic_correspondence_audit.md`; pair/row caches under
`/media/SSD1/MINH_DATASETS/lingbot_direct_correspondence_audit/`.
