#!/usr/bin/env bash
# Gate 8B caches for the new training source k360_train: LingBot stream + MoGe-B + scale
# candidates (cu128 env, 3 shards on the first pool GPU) and Trident-H (Trident env, one
# shard on each remaining pool GPU). Resumable: existing files are skipped.
set -u
REPO=/home/minh/workspace/lingbot-map_fork
TRI=/home/minh/workspace/third_party/Trident
TPY=/home/minh/workspace/third_party/trident_env/bin/python
PY=/home/minh/anaconda3/envs/cu128/bin/python
cd "$REPO"; source tools/gate8/gpus.sh
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
mkdir -p artifacts/gate8b/logs
echo "GPU pool: ${G8[*]}"
for i in 0 1 2; do
  $PY tools/gate8b/stream_source.py --shard $i --shards 3 --device "cuda:$(g8 0)" \
      > artifacts/gate8b/logs/stream_k360_train_s$i.log 2>&1 &
done
( cd "$TRI"; export PYTHONPATH=$REPO:$TRI
  for i in 0 1; do
    CUDA_VISIBLE_DEVICES="$(g8 $((i+1)))" $TPY "$REPO/tools/gate8b/cache_trident.py" \
        --shard $i --num-shards 2 --device cuda:0 \
        > "$REPO/artifacts/gate8b/logs/trident_k360_train_s$i.log" 2>&1 &
  done
  wait )
wait
echo "=== gate8b k360_train caches done $(date -Is) ==="
