#!/usr/bin/env bash
# Cache SAM-H dense embeddings for every KITTI-360 stream frame, four shards in parallel.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:/home/minh/workspace/third_party/Trident"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python
mkdir -p artifacts/feature_completion_study/logs
for drive in 2013_05_28_drive_0003_sync 2013_05_28_drive_0007_sync \
             2013_05_28_drive_0010_sync 2013_05_28_drive_0006_sync; do
  for s in 0 1 2 3; do
    $PY tools/feature_study/cache_sam_features.py --drive "$drive" --device "cuda:$s" \
        --shard "$s" --shards 4 \
        > "artifacts/feature_completion_study/logs/cache_sam_${drive:17:4}_s${s}.log" 2>&1 &
  done
  wait
  echo "=== $drive done ==="
done
