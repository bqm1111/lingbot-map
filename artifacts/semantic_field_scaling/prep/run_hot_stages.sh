#!/bin/bash
# usage: run_hot_stages.sh <gpu> <hot_root> <drive> [...]
# For each drive: link tokens, then evidence -> provenance -> delayed targets -> B3 train cache (hot tier), sequentially on one GPU.
# Stops at the first failing stage (its log ends with a non-zero exit line).
REPO=/home/minh/workspace/lingbot-map_fork; GPU=$1; HOT=$2; shift 2
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH HF_HUB_OFFLINE=1 PYTHONPATH=$REPO
cd $REPO
for D in "$@"; do
  for STAGE in link_tokens evidence provenance targets train_cache; do
    LOG=$REPO/artifacts/semantic_field_scaling/logs/${STAGE}_${D:17:4}.log
    /usr/bin/time -v /home/minh/anaconda3/envs/cu128/bin/python -m tools.semantic_field.scaling.prepare --stage $STAGE --drive $D --device cuda:$GPU --hot-root $HOT > $LOG 2>&1
    RC=$?; echo "exit $RC" >> $LOG; if [ $RC -ne 0 ]; then exit $RC; fi
  done
done
