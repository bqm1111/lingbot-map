# How to rerun Gate 8A yourself

Gate 8A trains three small U-Nets and evaluates eight checkpoints; nothing upstream of the
completion head moves, so every cache from Gates 6-8 is reused as-is.

## 0. Environment (every command)

```bash
cd /home/minh/workspace/lingbot-map_fork
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a"     # sm_120 needs the suffixed arch
export HF_HUB_OFFLINE=1
export GATE8_GPUS="1 2 3"                    # GPU 0 is left for other users
PY=/home/minh/anaconda3/envs/cu128/bin/python
```

## 1. The pipeline

| # | command | produces | cost (measured) |
|---|---|---|---|
| 0 | `$PY tools/gate8a/stage0.py` | `artifacts/gate8a/stage0_audit.json` — hashes of everything Gate 8A must not change | seconds |
| 1 | `$PY tools/gate8a/sampler_stats.py --n 400` | `artifacts/gate8a/sampler_stats.json` — what each crop sampler actually shows the net | ~2 min |
| 2 | `tools/gate8a/run_ablation.sh` | the three new cells, one per GPU: `artifacts/gate8a/checkpoints/cell{B,C,D}_*.pt` | ~27 min wall |
| 3 | `tools/gate8a/run_source_eval.sh` | `scores_{semantickitti,occ3d}_source_*.npz`, then the frozen selection | SK ~4 min ‖ Occ3D ~45 min |
| 4 | `tools/gate8a/run_heldout.sh` | the single locked KITTI-360 read-out | ~13 min |
| 5 | `$PY tools/gate8a/aggregate.py` | `artifacts/gate8a/gate8a_results.json` + every paired interval | ~1 min |
| 6 | `$PY tools/gate8a/figures.py` | `artifacts/gate8a/fig_*.png` | seconds |
| 7 | `$PY tools/gate8a/report.py --write` | `reports/gate8a/gate8a_report.md` | seconds |
| 8 | `$PY -m pytest tests/gate6 tests/gate7a tests/gate7b tests/gate8 tests/gate8a -q` | 212 prior + 23 new tests | ~2.5 min |

Stage 3 writes `configs/gate8a/frozen_selection.yaml` **before** stage 4 opens KITTI-360.
Stage 4 reads the checkpoint and both thresholds out of that file and takes no other
arguments, which is the mechanism that keeps the held-out read-out honest.

## 2. The skeleton — read in this order

```
gate8a/
  regions.py    the two evaluation regions; the editable region IS apply_residual's gate
  scores.py     per-anchor score histograms; AP / PR / AUROC / Brier / ECE / any threshold
  baselines.py  all-occupied, editable-fill, matched-density random; protection invariants
  sampler.py    the uniform crop-origin sampler (sees only the lattice and the RNG)
  losses.py     the unweighted-BCE arm, masked to valid AND editable
  boot.py       paired scene-aware bootstrap over Gate 6's own units
tools/gate8a/
  stage0.py sampler_stats.py train.py evaluate.py selection.py aggregate.py figures.py report.py
```

## 3. What must not change

`tools/gate8a/stage0.py` hashes the frozen surface (LingBot/MoGe/Trident consumers, the
mapper, the completion architecture and inputs, the privileged targets, the Gate 6 metric
and grid code, and every Gate 8 artifact). Re-run it and it prints anything that drifted.
