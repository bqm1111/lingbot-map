# Gate 11 — uncertainty-aware observed geometry (2026-09-13)

This is a diagnostic only: no training. It tests whether LB-M monocular depth becomes useful 0.2 m
geometry when each depth is treated as an interval along the ray instead of an exact surface.
Report: `artifacts/gate11_uncertainty/report.md`.

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH="$PWD" CUDA_HOME=/usr/local/cuda-12.8 HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python; T=tools/gate11_uncertainty; L=artifacts/gate11_uncertainty/logs; mkdir -p $L
# prerequisites (existing caches): local-reset clips, Gate 9 clip records
$PY $T/derive_rules.py                                    # width rules from 0006 depth residuals only
for s in 0 1 2 3; do $PY $T/audit_u.py --drive 2013_05_28_drive_0006_sync --shard $s --shards 4 --device cuda:$s > $L/audit_0006_s$s.log 2>&1 & done; wait
$PY $T/analyze_u.py && $PY $T/figures_u.py
# freeze (code_freeze.json: code + uncertainty_rules.json), then the unchanged replay
for d in 0003 0007 0010; do $PY $T/audit_u.py --drive 2013_05_28_drive_${d}_sync --device cuda:1 > $L/audit_${d}.log 2>&1; done
$PY $T/analyze_u.py --drives 2013_05_28_drive_0006_sync 2013_05_28_drive_0003_sync 2013_05_28_drive_0007_sync 2013_05_28_drive_0010_sync
for d in 0003 0007 0010; do $PY $T/figures_u.py --drive 2013_05_28_drive_${d}_sync; done
$PY $T/verify_u.py --device cuda:0
```
