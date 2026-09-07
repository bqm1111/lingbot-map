#!/usr/bin/env bash
# Gate 8C-1: rebuild every KITTI-360 occupancy target from raw LiDAR.
# Train drives at full anchor density; the source-validation drive at stride 3
# (590 of 1768) -- selection needs a representative sample, not every frame.
# Each drive is sharded across the GPU pool.
set -u
REPO=/home/minh/workspace/lingbot-map_fork
cd "$REPO"; source tools/gate8/gpus.sh
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python
L=artifacts/gate8c1/logs; mkdir -p "$L"
run_drive () {           # $1 = drive, $2 = stride
  local D=$1 S=$2 i
  for i in 0 1 2; do
    $PY tools/gate8c1/build_targets.py --drive "$D" --stride "$S" \
        --shard "$i" --shards 3 --device "cuda:$(g8 $i)" \
        > "$L/targets_${D:17:4}_s${i}.log" 2>&1 &
  done
  wait
  echo "=== targets ${D:17:4} done $(date -Is) ==="
}
for D in 2013_05_28_drive_0003_sync 2013_05_28_drive_0007_sync 2013_05_28_drive_0010_sync; do
  run_drive "$D" 1
done
run_drive 2013_05_28_drive_0006_sync 3
echo "=== gate8c1 raw targets complete $(date -Is) ==="
