#!/bin/bash
# usage: run_predict.sh <dataset> <shards>  -- forward-protocol predictions with our frozen heads, one shard per GPU
REPO=/home/minh/workspace/lingbot-map_fork; DS=$1; N=$2
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH HF_HUB_OFFLINE=1 PYTHONPATH=$REPO
cd $REPO
for g in $(seq 0 $((N-1))); do
  LOG=$REPO/artifacts/like_for_like/logs/predict_${DS}_s${g}of${N}.log
  (/usr/bin/time -v /home/minh/anaconda3/envs/cu128/bin/python -c "
import json
from tools.like_for_like.predict import run_shard
print(json.dumps(run_shard('$DS', 'cuda:$g', $g, $N)))" > $LOG 2>&1; echo "exit $?" >> $LOG) &
done
wait
