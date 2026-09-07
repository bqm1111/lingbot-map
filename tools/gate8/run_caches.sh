#!/usr/bin/env bash
# Gate 8: frozen-model passes for the two training sources.
# GPUs come from the shared pool (default "1 2 3"); override with GATE8_GPUS.
set -u
REPO=/home/minh/workspace/lingbot-map_fork
TRI=/home/minh/workspace/third_party/Trident
PY=/home/minh/anaconda3/envs/cu128/bin/python
TPY=/home/minh/workspace/third_party/trident_env/bin/python
cd "$REPO"; source tools/gate8/gpus.sh
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
mkdir -p artifacts/gate8/logs
echo "GPU pool: ${G8[*]}"
$PY tools/gate8/stream_sources.py --source sk_train    --device "cuda:$(g8 0)" > artifacts/gate8/logs/stream_sk_train.log 2>&1 &
$PY tools/gate8/stream_sources.py --source occ3d_train --device "cuda:$(g8 1)" > artifacts/gate8/logs/stream_occ3d_train.log 2>&1 &
( cd "$TRI"; export PYTHONPATH=$REPO:$TRI
  for SRC in sk_train occ3d_train; do
    for i in "${!G8[@]}"; do
      # CUDA_VISIBLE_DEVICES masks to one card, so --device cuda:0 IS the pool GPU
      CUDA_VISIBLE_DEVICES="${G8[$i]}" $TPY "$REPO/tools/gate8/cache_trident.py" \
          --source $SRC --shard "$i" --num-shards "${#G8[@]}" --device cuda:0 \
          > "$REPO/artifacts/gate8/logs/trident_${SRC}_s${i}.log" 2>&1 &
    done; wait
    echo "=== trident $SRC done $(date -Is) ==="
  done ) &
wait
echo "=== all caches done $(date -Is) ==="
