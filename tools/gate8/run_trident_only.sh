#!/usr/bin/env bash
# Trident-H cache for one or more Gate-8 training sources, sharded across the GPU pool.
# Resumes: frames whose cache file already exists are skipped.
#     tools/gate8/run_trident_only.sh occ3d_train
#     GATE8_GPUS="2 3" tools/gate8/run_trident_only.sh sk_train occ3d_train
set -u
REPO=/home/minh/workspace/lingbot-map_fork
TRI=/home/minh/workspace/third_party/Trident
TPY=/home/minh/workspace/third_party/trident_env/bin/python
cd "$REPO"; source tools/gate8/gpus.sh
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
mkdir -p artifacts/gate8/logs
echo "GPU pool: ${G8[*]}"
cd "$TRI"; export PYTHONPATH=$REPO:$TRI
for SRC in "$@"; do
  for i in "${!G8[@]}"; do
    # CUDA_VISIBLE_DEVICES masks to a single card, so --device cuda:0 IS the pool GPU
    CUDA_VISIBLE_DEVICES="${G8[$i]}" $TPY "$REPO/tools/gate8/cache_trident.py" --source "$SRC" --shard "$i" --num-shards "${#G8[@]}" --device cuda:0 > "$REPO/artifacts/gate8/logs/trident_${SRC}_s${i}.log" 2>&1 &
  done
  wait
  echo "=== trident $SRC done $(date -Is) ==="
done
