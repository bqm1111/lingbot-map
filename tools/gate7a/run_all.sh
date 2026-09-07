#!/usr/bin/env bash
# Gate 7A reachability analysis: every dataset, four shards, one GPU each.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/minh/anaconda3/envs/cu128/bin/python
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
mkdir -p artifacts/gate7a/logs
for DS in semantickitti occ3d kitti360; do
  for S in 0 1 2 3; do
    $PY tools/gate7a/reachability.py --dataset "$DS" --shard "$S" --shards 4 \
        --device "cuda:$S" > "artifacts/gate7a/logs/${DS}_${S}.log" 2>&1 &
  done
  wait
  echo "=== $DS done ==="
done
