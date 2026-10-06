#!/usr/bin/env bash
# The surround (six-camera) Occ3D evaluation, end to end. Every stage skips work that already exists, so rerunning the
# script resumes. Stage 1 (Trident tokens, run_surround_tokens.sh) and stage 2 (MoGe-2, run_surround_moge.sh) come first.
#
# A stage that fails stops the script. This matters: the first run of these stages exhausted SSD1 and every shard died
# with ENOSPC, but a trailing `echo` made the script exit 0 and the failure looked like completion.
set -u
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 PATH=/home/minh/anaconda3/envs/cu128/bin:$PATH
PY=/home/minh/anaconda3/envs/cu128/bin/python
LOGS=artifacts/like_for_like/logs
mkdir -p "$LOGS"

# Clips and predictions land on SSD1; tokens are read from cold storage. Refuse to start without headroom.
need_gb=20
free_gb=$(df -BG --output=avail /media/SSD1 | tail -1 | tr -dc '0-9')
if [ "$free_gb" -lt "$need_gb" ]; then
    echo "REFUSING TO START: /media/SSD1 has ${free_gb} GB free, need ${need_gb} GB." >&2
    exit 1
fi

# Run one stage across the four GPUs and fail if any shard fails.
shards() {
    local name=$1; shift
    local pids=() rc=0
    for s in 0 1 2 3; do
        "$@" --device "cuda:$s" --shard "$s" --shards 4 > "$LOGS/${name}_s$s.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    if [ "$rc" -ne 0 ]; then
        echo "STAGE FAILED: $name -- see $LOGS/${name}_s*.log" >&2
        grep -h -m2 "Error\|error\|Traceback" "$LOGS/${name}"_s*.log | head -5 >&2
        return 1
    fi
    echo "  $name ok"
}

echo "== 3. six-view clips =="
shards surround_clips $PY -m tools.like_for_like.surround_clips || exit 1

echo "== 4. predictions =="
shards surround_predict $PY -m tools.like_for_like.surround_predict || exit 1

echo "== 5. scoring (raw and OccAny's majority pooling) =="
for p in none semantic; do
    $PY -m tools.like_for_like.evaluate --dataset occ3d --setting surround --pooling "$p" \
        > "$LOGS/eval_surround_$p.log" 2>&1 || { echo "SCORING FAILED ($p)" >&2; exit 1; }
done

echo "== 6. report =="
$PY -m tools.like_for_like.report
