#!/usr/bin/env bash
# Feature voxel targets for every drive, and the drive-0006 map-observability masks.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
export G8C1_DATA_ROOT=/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
PY=/home/minh/anaconda3/envs/cu128/bin/python
LOG=artifacts/feature_completion_study/logs
mkdir -p "$LOG"
for drive in "$@"; do
  for s in 0 1 2 3; do
    $PY tools/feature_study/build_feature_targets.py --drive "$drive" --device "cuda:$s" \
        --shard "$s" --shards 4 > "$LOG/ftgt_${drive:17:4}_s${s}.log" 2>&1 &
  done
  wait
  echo "=== $drive feature targets done ==="
done
