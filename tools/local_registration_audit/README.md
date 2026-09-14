# Local registration audit (2026-09-13) -- Decision C

Five-frame causal clips under S (stream cache), L (LingBot re-run per clip, fresh state),
Lp (L pose + S depth), G (GT pose + S depth), GLd (GT pose + L depth), Gs (GT pose + LiDAR-oracle
depth scale, diagnostic). Local reset fixes most pose drift (0006 reprojection 128 -> 47 px, flat
along the stream) but GT pose barely improves the raw map (IoU 0.054 -> 0.061): depth is the
geometry bottleneck. No training; 12/12 checks. Report: `artifacts/local_registration_audit/report.md`.

```bash
export PYTHONPATH="$PWD" CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python; T=tools/local_registration_audit
for d in 0006 0003 0007 0010; do $PY $T/run_local_lingbot.py --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
$PY $T/verify_reset.py --device cuda:0
for d in 0006 0003 0007 0010; do $PY $T/pose_audit.py --drive 2013_05_28_drive_${d}_sync; $PY $T/geometry_audit.py --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
$PY $T/semantic_check.py --device cuda:0
$PY $T/analyze.py; $PY $T/visualize.py; $PY $T/verify.py
```
Caches: `/media/SSD1/MINH_DATASETS/lingbot_local_registration_audit/{local_lingbot,pose_audit,geometry}`.
