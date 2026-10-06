#!/bin/bash
# Launch (or resume: the same command continues from each run's resume.pt) the 12 declared runs.
#   GPU 0: B4 expanded seeds 0-2      GPU 1: B4 current seeds 0-2      GPU 2: B3 current + expanded seeds 0-2
# Each run logs to artifacts/semantic_field_scaling/logs/train_<method>_<pool>_seed<s>.log.
REPO=/home/minh/workspace/lingbot-map_fork; cd $REPO
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH HF_HUB_OFFLINE=1 PYTHONPATH=$REPO
PY=/home/minh/anaconda3/envs/cu128/bin/python
launch() {  # method pool seed gpu
  LOG=$REPO/artifacts/semantic_field_scaling/logs/train_$1_$2_seed$3.log
  ( /usr/bin/time -v $PY -m tools.semantic_field.scaling.cli train --method $1 --pool $2 --seed $3 --device cuda:$4 --host-cache-gib 0 >> $LOG 2>&1; echo "exit $?" >> $LOG ) &
}
for s in 0 1 2; do launch B4 expanded $s 0; launch B4 current $s 1; launch B3 current $s 2; launch B3 expanded $s 2; done
wait
