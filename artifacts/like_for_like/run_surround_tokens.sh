#!/bin/bash
# usage: run_surround_tokens.sh <shards>   -- surround token extraction, one shard per GPU
#
# Needs ~330 GB: the cache is ~13.7 MB per frame and there are 24,095 non-front frames. The first attempt filled SSD1
# and every shard died with ENOSPC while the wrapper still exited 0, so this version checks space up front and
# propagates each shard's exit status.
set -u
REPO=/home/minh/workspace/lingbot-map_fork; N=${1:-4}
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/home/minh/anaconda3/envs/cu128/bin:/usr/local/cuda-12.8/bin:$PATH HF_HUB_OFFLINE=1

need_gb=340
free_gb=$(df -BG --output=avail /media/SSD1 | tail -1 | tr -dc '0-9')
if [ "$free_gb" -lt "$need_gb" ]; then
    echo "REFUSING TO START: /media/SSD1 has ${free_gb} GB free, the token cache needs ~${need_gb} GB." >&2
    exit 1
fi
echo "free on /media/SSD1: ${free_gb} GB"

cd /home/minh/workspace/third_party/Trident
pids=()
for g in $(seq 0 $((N-1))); do
  LOG=$REPO/artifacts/like_for_like/logs/tokens_surround_s${g}of${N}.log
  (PYTHONPATH=$REPO:$PWD /usr/bin/time -v ../trident_env/bin/python $REPO/tools/like_for_like/extract_surround_tokens.py \
      --dataset occ3d --shard $g --shards $N --device cuda:$g > $LOG 2>&1; rc=$?; echo "exit $rc" >> $LOG; exit $rc) &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ "$rc" -ne 0 ]; then
    echo "TOKEN EXTRACTION FAILED -- see $REPO/artifacts/like_for_like/logs/tokens_surround_s*of${N}.log" >&2
    grep -h -m1 "Error\|No space left" $REPO/artifacts/like_for_like/logs/tokens_surround_s*of${N}.log | head -4 >&2
    exit 1
fi
echo "token extraction ok"
