#!/usr/bin/env bash
# Gate 8B pipeline after the k360_train caches: build the remaining training samples,
# then run both new folds concurrently on disjoint GPU pairs (fold SK: GPUs 1,2;
# fold Occ3D: GPUs 3,2 -- the shared GPU only hosts a second scoring pass).
set -u
REPO=/home/minh/workspace/lingbot-map_fork
cd "$REPO"
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python
L=artifacts/gate8b/logs; mkdir -p $L
until [ -f artifacts/gate8b/trident_k360_train_s0.json ] && [ -f artifacts/gate8b/trident_k360_train_s1.json ]; do sleep 30; done
echo "trident done $(date -Is)"
$PY tools/gate8b/build_samples.py --source k360_train --shard 1 --shards 3 --device cuda:1 > $L/samples_k360_train_s1.log 2>&1 &
$PY tools/gate8b/build_samples.py --source k360_train --shard 2 --shards 3 --device cuda:2 > $L/samples_k360_train_s2.log 2>&1 &
wait
until [ -f artifacts/gate8b/samples_kitti360_s0.json ]; do sleep 30; done
echo "samples done $(date -Is): k360_train $(ls /media/SSD1/MINH_DATASETS/lingbot_gate8b/samples/k360_train | wc -l), kitti360 $(ls /media/SSD1/MINH_DATASETS/lingbot_gate8b/samples/kitti360 | wc -l)"
GATE8_GPUS="1 2" tools/gate8b/run_fold.sh semantickitti > $L/run_fold_semantickitti.log 2>&1 &
GATE8_GPUS="3 2" tools/gate8b/run_fold.sh occ3d > $L/run_fold_occ3d.log 2>&1 &
wait
echo "=== both folds done $(date -Is) ==="
