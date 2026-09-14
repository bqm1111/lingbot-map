# Gate 13 — oracle observed-geometry completion ceiling (2026-09-13)

Paired comparison of the same completion model on two inputs:

- **M:** the monocular Gate 12 H input.
- **O:** oracle input from the five causal LiDAR sweeps with GT relative pose, through the same
  hard mapper (Gate 9 LD path).

Everything else is Gate 12's frozen recipe. No target dataset is accessed.
Report: `artifacts/gate13_oracle/report.md`.

Pipeline:
1. `build_oracle.py`: O inputs for Gate 12's anchors (sweeps t−4..t only).
2. `quality13.py`: input quality on 0006, and O == Gate 9 LD.
3. `run_all13.sh`: freeze → `train13.py` × 6 → `eval13.py` (final + stride-5 curves) → `analyze13.py` → `figures13.py`.
4. `verify13.py`.

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python; T=tools/gate13_oracle
# prerequisite: Gate 12 inputs, regions and initial weights (tools/gate12_suf/README.md)
for d in 0006 0003 0007 0010; do $PY $T/build_oracle.py --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
$PY $T/quality13.py --device cuda:0
bash $T/run_all13.sh
$PY $T/verify13.py --device cuda:0
```
