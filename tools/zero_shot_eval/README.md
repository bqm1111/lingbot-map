# Final zero-shot semantic completion evaluation

**Question.** Can the causal, single-camera, semantic-annotation-free system trained on KITTI-360 (frozen geometry + Trident-H
evidence + Gate 16 B3) perform useful zero-shot 3D semantic scene completion on SemanticKITTI and Occ3D-nuScenes?

- **Tracks:**
  - **Track P** (deployment-oriented, primary): the input camera's horizontal field of view within 40 m, with observed, completed and evidence-free space scored separately.
  - **Track O** (official): every valid voxel, OccAny's `SSCMetrics`.
- **Frozen before scoring:** the protocol, evaluators and analysis (`freeze.py`) — see `artifacts/final_zero_shot/practical_evaluation_predeclaration.md`.

Flow: **prediction → canonical artifact → masks → metrics → analysis**. One artifact per sample feeds both tracks.

| module | role |
|---|---|
| `config.py` | paths, frozen checkpoints (geometry seed 0, B3 seeds 0–2) |
| `samples.py` | evaluated samples (OccAny's scored samples with a causal window) and the history frames they read |
| `teacher/extract_tokens.py` | Trident-H token maps for target frames; runs in `third_party/trident_env` |
| `prediction/grid.py` | grid-parametrised voxel helpers (bit-identical to the KITTI-360 originals; tested) |
| `prediction/evidence.py` | Gate 15 evidence lifting + frame provenance on any grid |
| `prediction/semantics.py` | B0 nearest evidence; B3 association head on any grid |
| `prediction/pipeline.py` | sample → `CanonicalPrediction` (Occ3D: any-subvoxel occupancy, mean-feature semantic reduction) |
| `canonical/prediction.py` | the canonical prediction format (documented arrays, bit-packed volumes) |
| `canonical/metrics.py` | confusions, occupancy scores, Track P mIoU, OccAny SSC mIoU, superclasses |
| `practical/protocol.py` | every Track P / O parameter, class sets, statistics, paper criteria (hash-frozen) |
| `practical/masks.py` | valid, field of view, range, populations |
| `practical/evaluator.py` | Track P per-sample confusions and occupancy; OccAny geometry (Level A) |
| `official/evaluator.py` | Track O from the same predictions |
| `run_eval.py` | scores every prediction of a dataset |
| `analysis.py` | metrics, B3 − B0 block bootstrap, OccAny geometry deltas, paper decision |
| `report.py`, `verify.py`, `freeze.py` | tables and figures, verification, hash freezes |

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1; PY=/home/minh/anaconda3/envs/cu128/bin/python
$PY -m tools.zero_shot_eval.samples                                                     # frame lists
cd ../third_party/Trident && for g in 0 1 2 3; do (PYTHONPATH=$OLDPWD:$PWD ../trident_env/bin/python $OLDPWD/tools/zero_shot_eval/teacher/extract_tokens.py \
   --dataset semantickitti --shard $g --shards 4 --device cuda:$g && PYTHONPATH=$OLDPWD:$PWD ../trident_env/bin/python \
   $OLDPWD/tools/zero_shot_eval/teacher/extract_tokens.py --dataset occ3d --shard $g --shards 4 --device cuda:$g) & done; wait; cd -
$PY -c "from tools.zero_shot_eval import freeze; freeze.freeze('protocol'); freeze.freeze('evaluators')"      # before any score
for d in semantickitti occ3d; do for s in 0 1 2 3; do $PY -c "from tools.zero_shot_eval.prediction.pipeline import run_shard; run_shard('$d','cuda:$s',$s,4)" & done; wait; done
for d in semantickitti occ3d; do $PY -c "from tools.zero_shot_eval.run_eval import evaluate_dataset; evaluate_dataset('$d')"; done
$PY -c "from tools.zero_shot_eval import analysis; analysis.analyse()"
$PY -c "from tools.zero_shot_eval import report; report.write_all()"
$PY -c "from tools.zero_shot_eval import verify; verify.run('cuda:0')"
$PY -m pytest tests/zero_shot_eval -q
```

Data: `/media/SSD1/MINH_DATASETS/lingbot_zero_shot_eval` (tokens, text embeddings, canonical predictions). Results: `artifacts/final_zero_shot/`.
