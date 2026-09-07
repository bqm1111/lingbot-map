#!/usr/bin/env bash
# Gate 7B: the two frozen-model passes. One LingBot stream per dataset (one model per
# image geometry), and the calibrated-FOV MoGe cache for the two datasets whose Gate-5.1
# dense cache is the wrong variant.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/minh/anaconda3/envs/cu128/bin/python
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a"
export HF_HUB_OFFLINE=1
mkdir -p artifacts/gate7b/logs
$PY tools/gate7b/stream_lingbot.py --dataset semantickitti --device cuda:0 \
    > artifacts/gate7b/logs/stream_semantickitti.log 2>&1 &
$PY tools/gate7b/stream_lingbot.py --dataset kitti360 --device cuda:1 \
    > artifacts/gate7b/logs/stream_kitti360.log 2>&1 &
$PY tools/gate7b/stream_lingbot.py --dataset occ3d --device cuda:2 \
    > artifacts/gate7b/logs/stream_occ3d.log 2>&1 &
$PY tools/gate7b/cache_moge_b.py --dataset semantickitti --device cuda:3 \
    > artifacts/gate7b/logs/moge_semantickitti.log 2>&1 &
wait
$PY tools/gate7b/cache_moge_b.py --dataset occ3d --device cuda:3 \
    > artifacts/gate7b/logs/moge_occ3d.log 2>&1
echo "=== caches done ==="
