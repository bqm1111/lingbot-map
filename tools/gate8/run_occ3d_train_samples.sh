#!/usr/bin/env bash
# Build occ3d_train samples once its Trident cache is complete. Pool-aware.
set -u; cd /home/minh/workspace/lingbot-map_fork; source tools/gate8/gpus.sh
PY=/home/minh/anaconda3/envs/cu128/bin/python
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
until [ "$(ls /media/SSD1/MINH_DATASETS/lingbot_gate8/semantics/occ3d_train 2>/dev/null | wc -l)" -ge 3973 ]; do sleep 60; done
echo "=== occ3d_train Trident complete $(date -Is); GPU pool ${G8[*]} ==="
N=${#G8[@]}
for i in "${!G8[@]}"; do
  $PY tools/gate8/build_samples.py --source occ3d_train --device "cuda:${G8[$i]}" \
      --shard "$i" --shards "$N" --anchor-stride 2 \
      > "artifacts/gate8/logs/samples_occ3d_train_s${i}.log" 2>&1 &
done; wait
echo "=== occ3d_train samples done $(date -Is) ==="
