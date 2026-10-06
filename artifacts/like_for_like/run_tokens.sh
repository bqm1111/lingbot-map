#!/bin/bash
# usage: run_tokens.sh <dataset> <shards>  -- extract the missing forward-window tokens, one shard per GPU
REPO=/home/minh/workspace/lingbot-map_fork; DS=$1; N=$2
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH HF_HUB_OFFLINE=1
cd /home/minh/workspace/third_party/Trident
for g in $(seq 0 $((N-1))); do
  LOG=$REPO/artifacts/like_for_like/logs/tokens_${DS}_s${g}of${N}.log
  (PYTHONPATH=$REPO:$PWD /usr/bin/time -v ../trident_env/bin/python $REPO/tools/like_for_like/extract_missing_tokens.py \
      --dataset $DS --shard $g --shards $N --device cuda:$g > $LOG 2>&1; echo "exit $?" >> $LOG) &
done
wait
