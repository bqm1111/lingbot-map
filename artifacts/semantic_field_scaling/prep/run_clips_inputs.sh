#!/bin/bash
# usage: run_clips_inputs.sh <gpu> <drive> [...]   -- clips (cold) then hard-map inputs (hot), per drive, on one GPU
REPO=/home/minh/workspace/lingbot-map_fork; GPU=$1; shift
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1 PYTHONPATH=$REPO
cd $REPO
for D in "$@"; do
  for STAGE in clips inputs; do
    LOG=$REPO/artifacts/semantic_field_scaling/logs/${STAGE}_${D:17:4}.log
    /usr/bin/time -v /home/minh/anaconda3/envs/cu128/bin/python -m tools.semantic_field.scaling.prepare --stage $STAGE --drive $D --device cuda:$GPU > $LOG 2>&1
    RC=$?; echo "exit $RC" >> $LOG; if [ $RC -ne 0 ]; then exit $RC; fi
  done
done
