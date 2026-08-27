# Prompted LingbotMap — sensor-conditioned metric anchoring

> The repository's top-level `README.md` was deleted in commit `36441e2`, so this
> file carries the usage section rather than re-creating it.

Tests whether sparse metric depth or pose observations can anchor LingbotMap's
scale and reduce long-horizon drift on subsequent image-only frames, **without
retraining LingbotMap**. The foundation model is frozen and only ever run once,
into a cache; everything after that reads the cache.

Read `docs/prompted_lingbot_feasibility.md` first — it records the geometric
conventions, two of which contradict the upstream docstrings.

## Environment

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:/home/minh/anaconda3/envs/cu128/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a"          # sm_120 needs the explicit arch
export PYTHONPATH=/home/minh/workspace/lingbot-map_fork
PY=/home/minh/anaconda3/envs/cu128/bin/python
```

`ninja` must be on `PATH` (it is in the `cu128` env) or FlashInfer's JIT fails with
`FileNotFoundError: 'ninja'`.

## 1. Cache frozen predictions

Writes one compressed `.npz` per 500-frame chunk plus `manifest.json`. Resumable:
re-running skips chunks that already exist.

```bash
# TartanAir V2 (metric depth + metric poses) -- the primary corpus
$PY scripts/cache_lingbot_predictions.py \
    --dataset tartanair --dataset-root /media/minh/TartanAir/dataset \
    --checkpoint checkpoints/lingbot-map/204754b/lingbot-map.pt \
    --output-dir outputs/prompted_lingbot/cache --split tartanair_main

# KITTI odometry 09/10 (metric poses only) -- real-world hold-out
$PY scripts/cache_lingbot_predictions.py \
    --dataset kitti --dataset-root data/kitti/dataset --kitti-sequences 09 10 \
    --checkpoint checkpoints/lingbot-map/204754b/lingbot-map.pt \
    --output-dir outputs/prompted_lingbot/cache_kitti --split kitti_holdout
```

Useful flags: `--depth-stride` (default 2; caches depth at half resolution),
`--chunk-size`, `--limit N`, `--device cuda:2`, `--overwrite`, `--kitti-with-depth`
(opts into the densified KITTI depth — provenance unverified, see the audit §6).

`outputs/prompted_lingbot/cache` is a symlink to `/media/SSD1/prompted_lingbot/cache`
because the repository disk has ~28 GB free and the cache is ~4 GB.

## 2. Training-free baselines

```bash
$PY scripts/evaluate_prompted_lingbot.py \
    --cache-dir outputs/prompted_lingbot/cache \
    --output-dir outputs/prompted_lingbot/evaluation \
    --split-file configs/prompted_lingbot/splits.json --split test \
    --tag baselines --workers 8
```

Runs 58 prompt configurations × every anchor. Add `--no-point-cloud` to skip the
(much slower) point-cloud metrics, `--configs a b c` / `--anchors x y` to subset.

## 3. Train the learned corrector

```bash
$PY scripts/train_prompted_corrector.py \
    --config configs/prompted_lingbot/corrector.yaml \
    --output-dir outputs/prompted_lingbot/corrector
```

Everything tunable — seed, hidden size, prompt schedule, noise levels, loss weights,
optimiser, learning rate, early stopping — lives in the YAML. Prompt regimes are
resampled per sequence so one checkpoint covers many sensor rigs. The corrector is
~390 k parameters and trains on cached tensors; LingbotMap is never loaded.

`model.base: anchor` makes the corrector a residual on the strongest training-free
anchor (the sharp test); `model.base: identity` makes it standalone.

## 4. Evaluate the learned corrector against every baseline

```bash
$PY scripts/evaluate_prompted_lingbot.py \
    --cache-dir outputs/prompted_lingbot/cache outputs/prompted_lingbot/cache_kitti \
    --output-dir outputs/prompted_lingbot/evaluation \
    --split-file configs/prompted_lingbot/splits.json --split test \
    --learned-checkpoint outputs/prompted_lingbot/corrector/best.pt \
    --tag learned --workers 8
```

## 5. Figures

```bash
$PY scripts/plot_prompted_lingbot.py \
    --rows outputs/prompted_lingbot/evaluation/learned_rows.json \
    --output-dir outputs/prompted_lingbot/evaluation --tag fig
```

Produces: error vs frames since last prompt; ATE vs trajectory length; error vs
prompt interval; accuracy vs prompt noise; runtime vs method.

## 6. Tests

```bash
$PY -m pytest tests/prompted_lingbot/ -q
```

Covers Sim(3) recovery, robust scale under noise and outliers, identity behaviour,
depth-vs-translation scaling, rotation composition, degenerate correspondences,
missing prompts, dropout, reset between sequences, strict causality (no future
leakage, checked end to end through the runner for every causal anchor including
the learned one), deterministic prompt generation, and state serialisation.

## The anchors

| name | causal | what it does |
|---|---|---|
| `raw` | ✔ | no correction |
| `first_depth_scale` | ✔ | one robust scale from the first depth prompt, frozen |
| `running_depth_scale` | ✔ | inverse-variance log-scale filter, MAD-gated, forgetting |
| `causal_pose_sim3` | ✔ | weighted Sim(3) over all pose correspondences so far |
| `sliding_pose_sim3` | ✔ | same, last 10 correspondences only |
| `depth_scale_pose_se3` | ✔ | scale from depth prompts, rigid part from pose prompts |
| `learned_corrector` | ✔ | GRU residual on `depth_scale_pose_se3` |
| `offline_oracle_sim3` | ✘ | **diagnostic**: global Sim(3) fitted to all GT poses |
| `per_frame_oracle_scale` | ✘ | **diagnostic**: per-frame oracle depth scale |

The two oracles see the future. They bound what is achievable; they are never
methods and are drawn dashed in every figure.

## Interface

```python
class MetricAnchor:
    def reset(self) -> None: ...
    def update(self, prediction: Prediction, prompt: Prompt) -> None: ...
    def correct(self, prediction: Prediction) -> dict: ...
    @property
    def correction(self) -> Sim3: ...
```

The correction `C_t = (s_t, R_t, t_t)` is applied as
`x_metric = R_t (s_t x_pred) + t_t`, `c_metric = R_t (s_t c_pred) + t_t`,
`D_metric = s_t D_pred` — depth takes the scale alone, because a rigid motion of
the world does not change a camera-frame Z coordinate.
