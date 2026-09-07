#!/usr/bin/env bash
# Gate 8C-1: causal-input + rebuilt-target samples, one drive per GPU.
set -u
REPO=/home/minh/workspace/lingbot-map_fork
cd "$REPO"; source tools/gate8/gpus.sh
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python
mkdir -p artifacts/gate8c1/logs
i=0
for D in 2013_05_28_drive_0003_sync 2013_05_28_drive_0007_sync 2013_05_28_drive_0010_sync; do
  $PY tools/gate8c1/build_samples.py --drive "$D" --device "cuda:$(g8 $i)" \
      > "artifacts/gate8c1/logs/samples_${D:17:4}.log" 2>&1 &
  i=$((i+1))
done
wait
$PY tools/gate8c1/build_samples.py --drive 2013_05_28_drive_0006_sync --stride 1 \
    --device "cuda:$(g8 0)" > artifacts/gate8c1/logs/samples_0006.log 2>&1
echo "=== gate8c1 samples done $(date -Is) ==="
