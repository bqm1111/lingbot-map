# Feature-supervision study

**Question.** Does predicting a frozen segmentation teacher's *dense features* complete
map-unobserved space better than predicting its *class probabilities*?

Three conditions, identical in everything else:

| | objective | auxiliary head that receives gradient |
|---|---|---|
| **A** | `L_occ` | none |
| **B** | `L_occ + 0.5 * L_KL` | the 25-class softmax head |
| **C** | `L_occ + 0.5 * L_feature` | a new 64-channel feature head |

`L_occ` is the adopted masked focal + Dice objective. `L_KL` is the existing future-teacher
class-probability distillation. `L_feature` is mean cosine distance to a frozen SAM-H
embedding, projected to 64 dimensions by a PCA fitted on the training drives alone.

All three keep the same 32 input channels, **including the map's own semantic evidence**, so
A is "occupancy-*loss*-only", not "no semantics".

## What is modified, and what is not

The only edit outside this directory is `core/model/net.py`: `CompletionUNet` gained an
optional `n_feat` head and a `return_trunk` flag on `forward`. The head is constructed last
and only when asked for, so a network built with it and one built without it share their
trunk and their two original heads bit for bit — `verify.py::init_shared` asserts this, and
`n_feat=0` (every historical checkpoint) is unchanged.

Everything else is inherited: the mapper, the causal 32-channel export, the rebuilt
raw-LiDAR occupancy targets, the residual lock, the crop sampler, the optimizer and
schedule, and the `past5` evaluator.

**Data root.** `core.datasets.kitti360_splits` resolves `G8C1_DATA_ROOT` at import time and
its default is the *pre*-ceiling-fix root, whose targets predate the open-sky rule. Both
roots hold identical file names, so reading the wrong one fails silently. Every tool here
therefore imports `tools.feature_study.study` **before** anything under `core`, and calls
`ST.require_data_root()`, which raises if the wrong root was resolved.

## Reproduce

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH="$PWD:/home/minh/workspace/third_party/Trident"
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export G8C1_DATA_ROOT=/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky
PY=/home/minh/anaconda3/envs/cu128/bin/python

# 1. frozen SAM-H dense embeddings for every KITTI-360 stream frame (~6 min, 4 GPUs)
bash tools/feature_study/run_cache_sam.sh

# 2. the 64-d projection, fitted on drives 0003/0007/0010 only, then frozen
$PY tools/feature_study/fit_pca.py --device cuda:0

# 3. voxel feature targets over the privileged future window, and the drive-0006
#    map-observability partition (~12 min, 4 GPUs)
bash tools/feature_study/run_build_targets.sh \
    2013_05_28_drive_0003_sync 2013_05_28_drive_0007_sync \
    2013_05_28_drive_0010_sync 2013_05_28_drive_0006_sync
bash tools/feature_study/run_eval_masks.sh

# 4. nine runs: three conditions x seeds 0,1,2, 6 000 steps each (~15 min per run)
bash tools/feature_study/run_study.sh

# 5. source measurement, then freeze, then the single transfer evaluation per checkpoint
bash tools/feature_study/run_source.sh
$PY tools/feature_study/freeze_manifest.py
bash tools/feature_study/run_transfer.sh

# 6. checks, comparison table, prediction dump
$PY tools/feature_study/verify.py --device cuda:0
$PY tools/feature_study/compare.py
$PY tools/feature_study/dump_predictions.py --seed 0 --device cuda:0
```

## Result

`reports/feature_completion_study.md` answers the six questions the brief asked.
`artifacts/feature_completion_study/comparison_table.md` is the numbers behind it.

Short version: C beats B on the primary metric on every seed, by +0.4 pp, and **both lose to
occupancy loss alone in-domain**. The gain does not transfer — C − B changes sign across seeds
on both target datasets. C's feature head sits ~0.012 cosine above a constant mean-feature
predictor, and the two auxiliary objectives do not push the trunk equally hard (4.3x vs 2.0x
the occupancy gradient at the same nominal weight), so the C-vs-B difference is confounded
with optimisation rather than isolating the representation.

## Where things land

```
artifacts/feature_completion_study/
  pca_fit.json                 basis provenance: fit frames, explained variance, hashes
  feature_targets/, eval_masks/, cache_sam/    per-shard build records
  train_fcs_<cond>_seed<s>.json   config, curves, gradient probes, firewall summary
  trace_fcs_<cond>_seed<s>.npz    every (file, crop origin) drawn — the pairing evidence
  checkpoints/                 the nine final-step checkpoints
  source/                      drive-0006 partition metrics + per-clip counts
  frozen_manifest_<cond>.json  frozen before any target dataset is opened
  transfer/<cond>/             the gate evaluator's own output, one run per checkpoint
  verification.json, comparison.json, predictions/
```

Bulk caches live under `$FCS_DATA_ROOT` (default
`/media/SSD1/MINH_DATASETS/lingbot_feature_study`): `sam_vith/` (6.6 GB),
`feature_targets/`, `source_eval_masks/`.

## Reading the outputs honestly

* **SAM features are not text-aligned.** Their channels are not class names, and the PCA
  colours in `predictions/*.png` are a feature visualisation, nothing more.
* **A and C never train the 25-class head**, so it stays at initialisation and their
  semantic metrics are not measurements of those conditions. Only B has a semantic decoder.
* **Teacher agreement is agreement with a teacher**, not accuracy against a human label.
* **A lower feature loss is not evidence of better completion.** The primary number is SC
  IoU on map-unobserved/later-measured voxels, and it is reported with its precision and
  recall so that filling more space cannot be mistaken for completing better.
