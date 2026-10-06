#!/bin/bash
# usage: run_tokens.sh <gpu> <drive:shard:shards> [...]   -- Trident token extraction to the cold tier, sequentially on one GPU
REPO=/home/minh/workspace/lingbot-map_fork; GPU=$1; shift
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH HF_HUB_OFFLINE=1
cd /home/minh/workspace/third_party/Trident
for job in "$@"; do
  IFS=: read D S N <<< "$job"
  LOG=$REPO/artifacts/semantic_field_scaling/logs/tokens_${D:17:4}_s${S}of${N}.log
  PYTHONPATH=$REPO:$PWD /usr/bin/time -v ../trident_env/bin/python $REPO/tools/semantic_field/scaling/prepare_tokens.py --drive $D --device cuda:$GPU --shard $S --num-shards $N > $LOG 2>&1
  echo "exit $?" >> $LOG
done
