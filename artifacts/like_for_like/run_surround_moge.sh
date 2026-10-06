#!/usr/bin/env bash
# MoGe-2 metric-scale depth for the five non-front surround cameras (CAM_FRONT is already in the frozen cache).
set -u
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 PATH=/home/minh/anaconda3/envs/cu128/bin:$PATH
PY=/home/minh/anaconda3/envs/cu128/bin/python
for s in 0 1 2 3; do $PY -m tools.like_for_like.surround_moge --device cuda:$s --shard $s --shards 4 > artifacts/like_for_like/logs/moge_surround_s$s.log 2>&1 & done
wait
echo "moge surround done"
