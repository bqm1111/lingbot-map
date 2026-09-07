#!/usr/bin/env bash
# Gate 8C-1: seeds 0, 1, 2 concurrently, one per GPU of the pool.
set -u
REPO=/home/minh/workspace/lingbot-map_fork
cd "$REPO"; source tools/gate8/gpus.sh
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python
mkdir -p artifacts/gate8c1/logs
for s in 0 1 2; do
  $PY tools/gate8c1/train.py --config "configs/gate8c1/seed${s}.yaml" \
      --device "cuda:$(g8 $s)" > "artifacts/gate8c1/logs/train_seed${s}.log" 2>&1 &
done
wait
echo "=== gate8c1 training done $(date -Is) ==="
