# Gate 15 — sparse causal Trident-H evidence propagated into completed 3D space

**Question.** With occupancy frozen (the Current Geometry-Only Baseline, `tools/baseline_reconstruction`), can a small head on
the frozen completion representation carry language-aligned evidence observed on the five causal frames into occupied voxels
that no ray observed? *Where* is fixed by the geometry network; this gate only learns *what*.

**Evidence** (`evidence/`): a grid voxel is an evidence voxel iff an accepted causal LingBot ray (the frozen mapper's own
acceptance, clip scale applied once, anchor-identity local poses) ends in its world key **and** the frozen hard map marks it
occupied. Its feature is the normalised count-weighted mean of unit Trident-H token features sampled at the ray pixels. No labels,
LiDAR, GT poses, future frames or target data (strict file audit).

**Methods.** B0 = label of the nearest evidence voxel (fixed before training). B1 = `model/propagation_head.py` (290k params,
dilated residual 3D convs over [frozen trunk, projected evidence, mask], unit-norm 1024-d output). Readout = argmax cosine to
the 18 frozen text embeddings. Loss = 1 − cos on evidence voxels only, with evidence hidden from the input in random blocks.

**Evaluation** (`evaluation/`, 0006, human labels evaluation-only): primary region `completed_unobserved` = GT occupied ∧
predicted occupied ∧ no evidence; plus observed / oracle-occupancy / SSC regions, distance bins, collapse diagnostics, and a
label-free held-out-evidence cosine. The promotion rule (`rules.py` → `promotion_rule.json`) is frozen with the hash of the
measuring code before any trained head is evaluated; target datasets are opened only after Decision A.

```
config.py                 single source of truth (paths, grid, head/train config, firewall tokens, distance bins)
evidence/{trident,lift,build}.py   token sampling, ray-endpoint association, per-anchor cache
frozen_geometry.py        frozen CompletionUNet: occupancy + trunk from one no-grad forward
model/propagation_head.py losses/feature_distillation.py dataset.py train.py
evaluation/{baseline_nearest,regions,semantic_metrics,evaluate}.py
rules.py analysis.py reports/{report,figures,verification,final}.py cli/
tests/semantic_completion/
```

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-12.8 FLASHINFER_CUDA_ARCH_LIST="12.0a"
PY=/home/minh/anaconda3/envs/cu128/bin/python; T=tools/semantic_completion/cli
$PY $T/build_evidence.py --drive 2013_05_28_drive_0006_sync --shard 0 --shards 2 --device cuda:0   # and shard 1; 0003, 0007, 0010
$PY -c "from tools.semantic_completion import rules; rules.write_promotion_rule()"                 # before any trained head is evaluated
for s in 0 1 2; do $PY $T/train.py --seed $s --device cuda:$s & done; wait                          # ~1.6 h each
for s in 0 1 2 3; do $PY $T/evaluate_source.py --shard $s --shards 4 --device cuda:$s & done; wait
$PY $T/analyse.py --device cuda:0          # analysis.json, tables/, results_table.md, figures/, verification.md
$PY $T/report.py                           # report.md + code_freeze.json
$PY -m pytest tests/semantic_completion -q
```

Results: `artifacts/gate15_semantic_completion/` (start at `report.md`). Evidence cache: `/media/SSD1/MINH_DATASETS/lingbot_gate15_semantic/evidence` (~73 GB).
