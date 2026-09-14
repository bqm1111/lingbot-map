# KITTI-360 data scaling, seven training drives (2026-09-13) -- Decision A (negligible magnitude)

Full (9,672 anchors, drives 0000/0002/0003/0004/0005/0007/0010) vs nested 25 % Small (2,483), two
seeds, 12,100 steps each, adopted Gate 8C-1 model and objective, GT-registered semantic targets,
unchanged causal inputs; development evaluation on 0006. Primary Δ semantic mIoU +0.0012 / +0.0007
(CIs > 0) on an absolute 0.005; occupancy change sign-inconsistent. Report:
`artifacts/kitti360_data_scaling_7drives/report.md`.

```bash
cd /home/minh/workspace/lingbot-map_fork
# frozen-model caches for the new drives (unchanged tools, registered through drives7.py)
PY=/home/minh/anaconda3/envs/cu128/bin/python; W=tools/kitti360_data_scaling_7drives/drives7.py
for s in 0 1 3 4; do $PY $W tools/gate8b/stream_source.py --shard $s --shards 7 --device cuda:0; done
(cd ../third_party/Trident && PYTHONPATH=$OLDPWD:$PWD ../trident_env/bin/python $OLDPWD/$W $OLDPWD/tools/gate8b/cache_trident.py --shard 0 --num-shards 1 --device cuda:0)
for d in 0000 0002 0003 0004 0005 0007 0010; do $PY $W core/run/build_targets.py --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
$PY $W core/run/build_targets.py --drive 2013_05_28_drive_0006_sync --stride 3 --device cuda:0
# everything after the caches: samples, semantic targets, eval masks, verification, manifests, 4 runs, evaluation, analysis
tools/kitti360_data_scaling_7drives/run_all.sh
$PY tools/kitti360_data_scaling_7drives/qualitative.py --device cuda:0
```
