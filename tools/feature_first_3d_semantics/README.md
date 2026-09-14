# Feature-first 3D semantics (2026-09-13) -- Decision C

Question: are Trident-H's pre-classification dense features more view-consistent than its
class probabilities, and does fusing features before classifying beat fusing decisions?

Representation: the CLIP ViT-H-14 token feature Trident projects onto its text embeddings
(1024-d, 34 x 128 tokens, language-aligned; reproduces the official coarse probabilities
to max |dp| 0.004). Cached for 3,162 frames (28 GB fp16) and lifted with the frozen mapper
at the exact voxels/rays/weights of the probability path.

Result: same-surface feature cosine 0.918 vs 0.915 for a local-shuffle neighbour (AUC
0.52) -- the class vector gives 0.762 vs 0.748 (AUC 0.51); feature-fuse-then-classify vs
classify-then-fuse on the held-out view 0.540 vs 0.542 (0006), same decision on 95 % of
voxels; more views add ~+0.02 at most. Text queries work on the fused feature. 12/12
verification checks. Report: `artifacts/feature_first_3d_semantics/report.md`
(copy at `reports/feature_first_3d_semantics.md`).

```bash
REPO=/home/minh/workspace/lingbot-map_fork; TRI=/home/minh/workspace/third_party/Trident
# Trident environment, from the Trident root
cd $TRI && PYTHONPATH=$REPO:$TRI HF_HUB_OFFLINE=1 ../trident_env/bin/python $REPO/tools/feature_first_3d_semantics/extract_features.py --drive 2013_05_28_drive_0006_sync --device cuda:0
cd $TRI && PYTHONPATH=$REPO:$TRI HF_HUB_OFFLINE=1 ../trident_env/bin/python $REPO/tools/feature_first_3d_semantics/encode_queries.py
cd $TRI && PYTHONPATH=$REPO:$TRI HF_HUB_OFFLINE=1 ../trident_env/bin/python $REPO/tools/feature_first_3d_semantics/verify_determinism.py
# cu128 environment
cd $REPO; export PYTHONPATH=$PWD G8C1_DATA_ROOT=/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky; PY=/home/minh/anaconda3/envs/cu128/bin/python
for pop in eligible all_future_observed; do for d in 0003 0007 0010 0006; do $PY tools/feature_first_3d_semantics/lift_features.py --drive 2013_05_28_drive_${d}_sync --population $pop; done; $PY tools/feature_first_3d_semantics/analyze.py --population $pop; done
$PY tools/feature_first_3d_semantics/openvocab.py; $PY tools/feature_first_3d_semantics/verify.py
$PY tools/feature_first_3d_semantics/figures.py --population eligible; $PY tools/feature_first_3d_semantics/summarize.py
```

Data: `/media/SSD1/MINH_DATASETS/lingbot_feature_first_3d/{trident_tokens, lifted, text_embeddings.npz, text_queries.npz}`.
