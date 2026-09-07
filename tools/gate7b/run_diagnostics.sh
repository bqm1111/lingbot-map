#!/usr/bin/env bash
# Gate 7B: map thickness / duplicate surfaces, and the forward-looking recoverability
# diagnostic. One dataset per GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/minh/anaconda3/envs/cu128/bin/python
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
mkdir -p artifacts/gate7b/logs
i=0
for DS in semantickitti occ3d kitti360; do
  ( $PY tools/gate7b/thickness.py --dataset "$DS" --device "cuda:$i" --stride 10 \
        > "artifacts/gate7b/logs/thickness_${DS}.log" 2>&1
    $PY tools/gate7b/recoverability.py --dataset "$DS" --device "cuda:$i" --stride 4 \
        > "artifacts/gate7b/logs/recover_${DS}.log" 2>&1 ) &
  i=$((i+1))
done
wait
echo "=== diagnostics done ==="
