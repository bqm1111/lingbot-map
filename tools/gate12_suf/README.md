# Gate 12 — H vs S/U/F learned completion (2026-09-13)

**Decision B:** S/U/F does not meaningfully improve completion (+0.0024 IoU on 0006, below the
predeclared +0.005). Keep H. Report: `artifacts/gate12_suf/report.md`.

Pipeline:
1. `gen_local_clips.py`: stride-1 local-reset L clips for the training drives.
2. `build_inputs.py`: paired H and S/U/F volumes; no LiDAR, no GT pose.
3. `build_regions.py`: 0006 evaluation regions.
4. `sanity_inputs.py`: pre-training checks.
5. `run_all.sh`: sanity gate → freeze → `train12.py` × 6 → `eval12.py` → `analyze12.py`.
6. `verify12.py`.

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH="$PWD" CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python; T=tools/gate12_suf
for d in 0003 0007 0010; do $PY $T/gen_local_clips.py --drive 2013_05_28_drive_${d}_sync --device cuda:1; done
for d in 0006 0003 0007 0010; do $PY $T/build_inputs.py --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
$PY $T/build_regions.py --device cuda:0
bash $T/run_all.sh
$PY $T/verify12.py --device cuda:0
```
