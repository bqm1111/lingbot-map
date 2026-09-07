#!/usr/bin/env bash
# Build sk_train samples (resumes: existing sample files are skipped). Pool-aware.
set -u; cd /home/minh/workspace/lingbot-map_fork; source tools/gate8/gpus.sh
PY=/home/minh/anaconda3/envs/cu128/bin/python
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
N=${#G8[@]}
echo "GPU pool: ${G8[*]}"
for i in "${!G8[@]}"; do
  $PY tools/gate8/build_samples.py --source sk_train --device "cuda:${G8[$i]}" \
      --shard "$i" --shards "$N" > "artifacts/gate8/logs/samples_sk_train_s${i}.log" 2>&1 &
done; wait
echo "=== sk_train samples done $(date -Is) ==="
