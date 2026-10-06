# Semantic field (B4) — generative language-aligned 3D semantic completion, final semantic-method experiment

**Question.** Can a lightweight generative head predict dense Trident-H-space semantics in unobserved 3D space from frozen
completed geometry and sparse observed semantics? It is trained on KITTI-360 with masked current observations and privileged
future observations.

- **Frozen:** geometry, occupancy, the teacher, evidence lifting, delayed targets and B0 / B3.
- **Rules:** `artifacts/semantic_field/source_predeclaration.md`, hash-frozen in stages (`freeze.py`).
- **Targets:** stay locked unless `source_promotion.json` says SOURCE PASS and the `final` stage is frozen.

## Feature spaces

| quantity | definition |
|---|---|
| raw teacher feature | Trident-H CLIP ViT-H-14 token feature, 1024-d, unit L2 |
| centering | μ = mean raw unit evidence feature over training-drive evidence voxels (27.1 M voxels); `centre(f) = normalize(f − μ)` |
| B4 evidence input | Linear(1024 → 32)(centre(f)) at evidence voxels |
| B4 output | normalize(Linear(32 → 1024)(hidden)), raw teacher space |
| text logits | 40 · cos(f, text_k), frozen text bank (KITTI-360 18 classes for source; each target's own bank at target readout) |

## Modules

| module | role |
|---|---|
| `config/schema.py` | every predeclared parameter and path |
| `data/semantic_evidence.py` | `SparseSemanticEvidence`, `SemanticTarget`, cuboid masking, dense semantic channels |
| `data/feature_mean.py` | training-drive μ |
| `model/semantic_field_head.py` | B4: residual 3D UNet (32 / 48 / 64) + pointwise 1024-d head, 1,357,008 parameters |
| `losses/semantic_loss.py` | text KL distillation, centered feature cosine, masked and delayed losses |
| `training/trainer.py` | one seed; frozen full-grid geometry per step, masking, crop, losses, final checkpoint |
| `evaluation/predict.py` | B4 full-grid inference at query voxels |
| `evaluation/evaluate_source.py` | 0006: B0 / B3 / B4, regions, candidate support, distance, masked reconstruction |
| `evaluation/analysis.py` | metrics, bootstrap, S1–S6, `source_promotion.json` |
| `evaluation/report.py` | tables and figures |
| `freeze.py` | stage hashes and the target lock |

## Reproduce

```bash
cd /home/minh/workspace/lingbot-map_fork; export PYTHONPATH=$PWD HF_HUB_OFFLINE=1; PY=/home/minh/anaconda3/envs/cu128/bin/python
$PY -c "from tools.semantic_field import freeze; freeze.freeze('source_predeclaration')"
$PY -c "from tools.semantic_field.data.feature_mean import compute_feature_mean as c; print(c())"   # then freeze('centering', payload)
for s in 0 1 2; do $PY -c "from tools.semantic_field.training.trainer import train_seed; train_seed($s, 'cuda:$s')" & done; wait
$PY -c "from tools.semantic_field import freeze; freeze.freeze('source_code')"
for s in 0 1; do $PY -c "from tools.semantic_field.evaluation.evaluate_source import evaluate; evaluate('cuda:$s', $s, 4)" & done; wait
for s in 2 3; do $PY -c "from tools.semantic_field.evaluation.evaluate_source import evaluate; evaluate('cuda:$s', $s, 4)" & done; wait
$PY -c "from tools.semantic_field.evaluation.analysis import analyse; print(analyse()['decision'])"
$PY -c "from tools.semantic_field.evaluation.report import write_all; write_all()"
$PY -m pytest tests/semantic_field -q
```
