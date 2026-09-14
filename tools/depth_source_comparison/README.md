# Controlled depth-source comparison (Gate 9 scale correction), 2026-09-13

Scale-path audit of Gate 9 + LD / LB-M / MG / LB-O on identical pixels, GT pose, frozen mapper.
No training. KITTI-360 only. Correction note and results: `artifacts/depth_source_comparison/`.

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH="$PWD" CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python; T=tools/depth_source_comparison; L=artifacts/depth_source_comparison/logs; mkdir -p $L
# prerequisites (existing caches): local-reset clips (tools/local_registration_audit/run_local_lingbot.py),
# Gate 9 clip records (tools/gate9_depth_occupancy/audit.py), MoGe-2 B caches (Gate 5.2 / Gate 8B)
$PY $T/scale_audit.py --device cuda:0 > $L/scale_audit.log 2>&1
for s in 0 1 2 3; do $PY $T/compare_sources.py --drive 2013_05_28_drive_0006_sync --shard $s --shards 4 --device cuda:$s > $L/compare_0006_s$s.log 2>&1 & done; wait
$PY $T/analyze.py
# code frozen here (code_freeze.json), then the replay
for d in 0003 0007 0010; do $PY $T/compare_sources.py --drive 2013_05_28_drive_${d}_sync --device cuda:1 > $L/compare_${d}.log 2>&1; done
$PY $T/analyze.py --drives 2013_05_28_drive_0006_sync 2013_05_28_drive_0003_sync 2013_05_28_drive_0007_sync 2013_05_28_drive_0010_sync
$PY $T/verify.py --device cuda:0
```
