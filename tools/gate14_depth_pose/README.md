# Gate 14 — stronger causal camera-only depth (MUSt3R-512) × pose (2026-09-14)

Stage 1 is a raw observed-geometry audit: no training.

**Design.**
- **Depth:** LB-M (LingBot × causal MoGe-2 clip scale) vs MUSt3R-512.
- **Pose:** local-reset LingBot pose vs KITTI-360 GT pose.
- **Supports:** shared and native.
- **Reference:** LiDAR camera depth + GT pose (LD-G).

**MUSt3R-512.**
- **Code:** official `naver/must3r`, commit `0c0d76b`, cloned to `/media/SSD1/MINH_DATASETS/lingbot_gate14_must3r/src/must3r`.
- **Checkpoint:** `MUSt3R_512.pth`, MD5 `8854f948a8674fb1740258c1872f80dc`.
- **Environment:** `occany_env` Python with the official source on `PYTHONPATH`; nothing installed.

**Stage 2** (completion training with MUSt3R + local-reset pose input) runs only if
`stage2_guard.require_promotion()` passes under the frozen `artifacts/gate14_depth_pose/promotion_rule.json`.

Report: `artifacts/gate14_depth_pose/report.md`.

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python; T=tools/gate14_depth_pose; L=artifacts/gate14_depth_pose/logs
R=/media/SSD1/MINH_DATASETS/lingbot_gate14_must3r; M=$R/src/must3r
git clone --recursive https://github.com/naver/must3r.git $M && git -C $M checkout 0c0d76b
# promotion_rule.json is written first (frozen before any MUSt3R occupancy result)
$PY $T/make_manifest.py --drives 2013_05_28_drive_0006_sync 2013_05_28_drive_0003_sync 2013_05_28_drive_0007_sync 2013_05_28_drive_0010_sync
for d in 0006 0003 0007 0010; do PYTHONPATH=$M:$M/dust3r:$M/dust3r/croco /home/minh/workspace/third_party/occany_env/bin/python $T/must3r_infer.py --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
for s in 0 1 2 3; do $PY $T/audit14.py --drive 2013_05_28_drive_0006_sync --shard $s --shards 4 --device cuda:$s > $L/audit_0006_s$s.log 2>&1 & done; wait
$PY $T/analyze14.py                                     # 0006 decision + promotion_result.json; code frozen here (code_freeze.json)
for d in 0003 0007 0010; do $PY $T/audit14.py --drive 2013_05_28_drive_${d}_sync --device cuda:1 > $L/audit_${d}.log 2>&1; done
$PY $T/analyze14.py --drives 2013_05_28_drive_0006_sync 2013_05_28_drive_0003_sync 2013_05_28_drive_0007_sync 2013_05_28_drive_0010_sync
$PY $T/verify14.py --device cuda:0
```
