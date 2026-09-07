# How to rerun Gate 8B yourself

```bash
cd /home/minh/workspace/lingbot-map_fork
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1 GATE8_GPUS="1 2 3"
PY=/home/minh/anaconda3/envs/cu128/bin/python
```

| # | command | produces | cost (measured) |
|---|---|---|---|
| 0 | `$PY tools/gate8b/stage0.py` | `artifacts/gate8b/stage0_audit.json` | seconds |
| 1 | `$PY tools/gate8b/fetch_k360_labels.py --drives 0003 0007 0010` | SSCBench train-drive targets under `sscbench_kitti360/preprocess/labels/<drive>/` (byte-range, 201 MB) | 6 min |
| 2 | `tools/gate8b/run_caches.sh` | LingBot stream + MoGe-B + scale (3 shards, GPU 1) and Trident-H (2 shards, GPUs 2-3) for `k360_train` | ~25 min wall |
| 3 | `$PY tools/gate8b/build_samples.py --source kitti360 --anchor-stride 10` ; `--source k360_train --shard i --shards 3` | cached (causal input, privileged target) samples under `lingbot_gate8b/samples/` | 6 min + ~3 x 5 min |
| 4 | `tools/gate8b/run_fold.sh semantickitti` / `... occ3d` (or `tools/gate8b/run_all.sh` for 3 + 4 chained) | train → score {best,last} on the fold's two source-validation domains → `configs/gate8b/frozen_fold_<fold>.yaml` → locked target evaluation, both settings | ~26 min train + scoring + target |
| 5 | `$PY tools/gate8b/eval_target.py --fold kitti360 --setting clips` | the matched five-frame setting for the existing (Gate 8A) fold, unmodified model | ~10 min |
| 6 | `$PY tools/gate8b/teacher_accuracy.py --dataset <target>` | future-Trident target accuracy against semantic GT | minutes |
| 7 | `$PY tools/gate8b/aggregate.py` ; `$PY tools/gate8b/figures.py` ; `$PY tools/gate8b/report.py --write` | `gate8b_results.json`, `gate8b_manifest.json`, figures, the report | ~2 min |
| 8 | `$PY -m pytest tests/gate6 tests/gate7a tests/gate7b tests/gate8 tests/gate8a tests/gate8b -q` | 239 prior + new tests | ~4 min |

`eval_target.py` takes no checkpoint and no threshold: it reads the fold's frozen manifest and
refuses to run without it.

## Skeleton

```
gate8b/
  sources.py   k360_train (official SSCBench train drives 0003/0007/0010), the fold table,
               registration into gate8.sources
  clips.py     the matched five-frame setting: fresh map per official clip, pinned G51-B scale
  pooling.py   OccAny's separate pooling (dilation / vote / semantic), checked against the official code
tools/gate8b/
  fetch_k360_labels.py stream_source.py cache_trident.py build_samples.py train.py
  select_fold.py eval_target.py eval_clips.py teacher_accuracy.py aggregate.py figures.py report.py
```
