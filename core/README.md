# core/ — the pipeline that is actually running

> Looking for the headline results and how to rerun them? Start at
> [`../REPRODUCE.md`](../REPRODUCE.md). This file explains the code.

The repository holds **602 Python files** across ~30 research "gates". **57 of them** are
reachable from the current line of work (Gate 8C-1). This directory is those 57 files,
copied out and reorganised by role, with imports rewritten and nothing else changed.

The gate packages themselves now live under [`gates/`](../gates/README.md) — that is the
research *history*, one frozen experiment per gate. This directory is the *pipeline*.

The original tree is untouched. Every `manifests/*.json` and `artifacts/*/config.json`
records **sha256 hashes of source files at their original paths**, and the audit
reproducibility depends on them, so the originals remain the authoritative copy. `core/`
is for reading and for running; if you need to prove a past result, use the original path.

Verified after extraction: all 57 modules import; the seed-0 checkpoint loads into
`core.model.net` with every parameter tensor identical; on eight real drive-0006 samples
the completion logits are **bit-identical** to the original code path and the resulting
TP/FP/FN/TN reproduce `artifacts/gate8c1_source_sanity/phase1_per_case.csv` exactly.

---

## What the project is trying to do

Predict dense 3D occupancy for a driving scene from **monocular images alone**, causally —
one frame in, a persistent map out, no future frames, no LiDAR at inference.

Three frozen models supply the per-frame evidence (they are never trained here, and their
outputs are cached to disk as `.npz` once):

| model | supplies |
|---|---|
| **LingBot-Map** | depth + camera pose per frame |
| **MoGe-2** | a second depth opinion, used to confirm geometry |
| **Trident-H** | open-vocabulary semantic probabilities per pixel |

Those go into an **incremental voxel map** (log-odds occupancy, free space, semantic
evidence). The map is sparse and full of holes, so a small **3D U-Net** completes it. The
network sees only the map and the current frame; it is *trained* against privileged
future observations. That is the whole idea: privileged supervision at training time,
causal input at inference time.

---

## The pipeline

```
  KITTI-360 raw images                    raw Velodyne sweeps
          │                                        │
          │  (cached once, frozen)                 │
          ▼                                        ▼
  LingBot-Map / MoGe-2 / Trident-H         run/build_targets.py
  → /media/SSD1/…/lingbot_gate7b/            supervision/rawtarget.py
        stream/ scale/ trident/              → occupied endpoints, free ray
          │                                     interiors, ignored elsewhere
          │                                        │
          ▼                                        │  run/validate_targets.py
  mapping/feed.py  CachedFeed                      │  (agreement vs oracle LiDAR)
          │  one FrameInput per frame              │
          ▼                                        │
  mapping/mapper.py  IncrementalMapper             │
    ├─ mapping/scale.py    metric gauge, frozen after 5 frames
    ├─ mapping/depth.py    depth conventions + the LingBot gate
    ├─ mapping/rays.py     free before the surface, occupied band at it
    └─ mapping/voxmap.py   log-odds + semantic accumulators
          │                                        │
          │  query at the anchor frame             │
          ▼                                        ▼
        run/build_samples.py ──────────────────────┘
        → /media/SSD1/…/lingbot_gate8c1/{samples,targets}/<drive>/*.npz
             32-channel causal input  +  rebuilt occupancy target
          │
          ▼
  run/train.py  ──►  model/net.py  CompletionUNet (986,114 params)
    model/losses.py  focal+dice, masked          artifacts/gate8c1/checkpoints/
    model/data.py    per-drive crop sampler
          │
          ▼
  run/eval_target.py  ──►  evaluation/{regions,scores,baselines,pooling}.py
```

**The residual lock.** The network never overwrites confident map voxels:

```python
apply_residual(base, res) = base + res * (base.abs() < LOCK_LOGODDS)   # LOCK_LOGODDS = 2.0
```

**The 32 input channels** (`supervision/targets.py:unpack_sample`):

| ch | content | | ch | content |
|---|---|---|---|---|
| 0 | `logodds / 4` | | 4 | `n_obs.clamp(50) / 50` |
| 1 | `w_free.clamp(20) / 20` | | 5 | `age` |
| 2 | `observed` | | 6 | `semw.clamp(20) / 20` |
| 3 | `1 - observed` | | 7.. | semantic class probabilities |

**The grid**: 256 × 256 × 32 at 0.2 m, origin (0, −25.6, −2), XYZ, `pad_z = 0`.
**The threshold**: τ = −0.125, selected on the source domain and then frozen.

---

## Layout

### `core/datasets/` — where the data is, and what frame it is in
| file | |
|---|---|
| `config.py` | config loading + run provenance (`REPO_ROOT`, `write_json`) |
| `sources.py` | every stream, train and validation alike, behind one interface |
| `kitti360_splits.py` | **the Gate 8C-1 firewall**: drives, anchors, `FORBIDDEN_PATH_TOKENS` |
| `audit.py` | monkeypatches `open`/`np.load`/`np.fromfile` to *prove* target data was never read |
| `kitti360.py` | SSCBench-KITTI-360 adapter (official validation drive 0006) |
| `kitti360_transforms.py` | the full KITTI-360 transform chain, written once and tested |
| `kitti360_folds.py` | KITTI-360 as a *training* source; fold definitions |
| `grids.py`, `occ3d_grid.py` | per-benchmark prediction grids |
| `frames.py`, `streams.py` | unique-frame index; chronological deduplicated stream |
| `vocab.py`, `union_vocab.py` | official per-benchmark classes; the shared union label space |
| `kitti_odometry.py`, `nuscenes.py`, `occ_datasets.py`, `occupancy.py` | other benchmark adapters |

### `core/mapping/` — frozen per-frame outputs → persistent causal voxel map
`depth.py` · `rays.py` · `scale.py` · `voxmap.py` · `mapper.py` · `feed.py`
Nothing here is trained. `mapper.py` is the heart: scale frozen from the first five
frames and never re-gauged, every frame integrated exactly once, dense grids are an
*export* (`query`) and never a replay.

### `core/supervision/` — what the network is trained against
| file | |
|---|---|
| `rawtarget.py` | occupancy rebuilt from **raw Velodyne**, never SSCBench's `_1_1.npy` |
| `targets.py` | privileged future-window targets; `unpack_sample` |
| `lifting.py` | frozen semantic lifting, teacher pixels → voxel probabilities |
| `oracle.py` | oracle geometry: GT LiDAR through the adapter, and how well it lands |

> Why `rawtarget.py` exists: Gate 8C-0 showed SSCBench-KITTI-360's own `_1_1.npy`
> completion label contradicts its LiDAR — a ground-truth sweep lands on a label-*free*
> voxel **71%** of the time (against 1.7% on SemanticKITTI). That label is never opened.

### `core/model/` — the completion network
`net.py` (3-level dense 3D U-Net, width 24, GroupNorm(8,C)+GELU, ten 3×3×3 convs) ·
`losses.py` · `data.py` · `sampler.py` · `losses_bce_ablation.py`

### `core/evaluation/` — scoring
`regions.py` (boundary = centre < 0.4 m from a grid face; interior = ≥ 2 m from every
face) · `scores.py` (threshold-free histogram accumulators) · `baselines.py` ·
`metrics.py` · `pooling.py` (OccAny's official 3×3×3 majority pooling) ·
`official_targets.py` · `voxels.py`

### `core/run/` — the executable stages
```
build_targets  →  validate_targets  →  build_samples  →  train  →  eval_target
```
A **leading underscore** marks a module inherited from an earlier gate. It still executes
— `train.py` takes `step_loss` from `_gate8a_train.py`, which takes its batching from
`_gate8_train.py` — but it is not this gate's entry point. That three-deep inheritance is
the single most confusing thing left in the pipeline, and it is the first thing to
flatten if you refactor.

---

## Running it

```bash
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python      # bare `python` is not on PATH

$PY core/run/build_targets.py  --drive 2013_05_28_drive_0003_sync --device cuda:0
$PY core/run/validate_targets.py --device cuda:0
$PY core/run/build_samples.py  --drive 2013_05_28_drive_0003_sync --device cuda:0
$PY core/run/train.py          --config configs/gate8c1/seed0.yaml --device cuda:0
$PY core/run/eval_target.py    --dataset semantickitti --mode stream --seed 0
```

Data roots (read-only; all writes go to `artifacts/`):

| path | holds |
|---|---|
| `/media/SSD1/MINH_DATASETS/lingbot_gate7b/{stream,scale}/` | cached LingBot depth/pose + metric scale |
| `/media/SSD1/MINH_DATASETS/lingbot_gate8/` | cached Trident-H and MoGe-2 per frame |
| `/media/SSD1/MINH_DATASETS/lingbot_gate8c1/{samples,targets}/` | the training pairs |

Drives: `0003` (193) · `0007` (569) · `0010` (596) train; `0006` (590) validation.
Training: AdamW lr 1e-3, wd 0.01, cosine, batch 4, crop 128×128×32, 6000 steps,
focal+dice with `w_unknown` 2.0, plus a 0.5-weighted semantic KL. See
`configs/gate8c1/seed0.yaml`.

---

## Where the investigation stands

Read `artifacts/gate8c1_source_sanity/report.md` first — it supersedes the earlier ones.

| result | value |
|---|---|
| Completion, all 590 drive-0006 IDs, τ = −0.125 | **10.70%** SC IoU |
| Incremental mapper alone | 3.24% SC IoU, **11.66% precision** |
| Mapper + 0.4 m dilation | 6.23% SC IoU |
| SemanticKITTI (target domain) | 12.74% — *higher* than the source domain |
| Fresh head, 2000 steps on 16 fixed clips | loss −10.0% (needed −90%), train IoU 12.25% (needed 80%) |

A fresh head **cannot memorise sixteen clips**, so the fault is not the architecture and
not cross-dataset transfer. The sharpest clue is upstream: the incremental mapper is built
from the same poses, depths and scale that feed the network's 32 input channels, and it
has **11.66% precision** against the raw-LiDAR target — nearly nine of ten voxels it calls
occupied are not occupied in the supervision. Input geometry and supervision do not agree.

Two earlier probes, both rejected as primary explanations, kept for the record:
`artifacts/gate8c1_padding_source_probe/` (zero-padding boundary ceiling) and
`artifacts/gate8c1_groupnorm_stat_probe/` (GroupNorm-statistics propagation).

**Next experiment, recommended but not started:** a source-only geometric alignment audit
— nearest-neighbour distance from each mapper-occupied voxel to the nearest target-occupied
voxel, plus SC IoU under a small fixed grid of rigid translations and scale factors applied
to the target — to decide whether the cached map and the raw-LiDAR supervision share a
metric frame.

---

## What was left behind, and why

545 files. None are imported by the pipeline above.

| | |
|---|---|
| **superseded gates** | `gates/gate7a`, `gates/gate7c`, `gates/gate8d`, `gates/depth_gate`, `gates/voxel_gate_validation`, `moge_gauge`, `occ3d_zeroshot/pipeline` — each answered its question; the answer is in `artifacts/` and in [`gates/README.md`](../gates/README.md) |
| **one-off diagnostics** | 36 scripts in `tools/gate8c1/` alone (`diag_*.py`, `figure_*.py`, `padding_*.py`, `slab_diagnosis.py`, …) — each written for a single question already answered |
| **unrelated subsystems** | `lingbot_map/` (the GCT model itself — *not needed here*, the pipeline runs entirely off cached outputs), `demo_render/`, `semantic/`, `semantic_sidecar/`, `benchmark/`, `research/` |
| **tests** | `tests/gate8c1/` still has two known failures from the earlier ceiling/padding work |

To see the reachable set for yourself:

```bash
$PY -c "
import ast; from pathlib import Path
R=Path('.'); L={p.name for p in R.iterdir() if (p/'__init__.py').exists()}|{'tools'}
def imp(f):
    o=set()
    for n in ast.walk(ast.parse(Path(f).read_text())):
        if isinstance(n,ast.Import): o|={a.name for a in n.names}
        elif isinstance(n,ast.ImportFrom) and n.level==0 and n.module:
            o|={n.module}|{n.module+'.'+a.name for a in n.names}
    return {m for m in o if m.split('.')[0] in L}
def path(m):
    for p in (R/(m.replace('.','/')+'.py'), R/m.replace('.','/')/'__init__.py'):
        if p.exists(): return str(p)
seen=set(); st=[f'tools/gate8c1/{x}.py' for x in ('build_samples','build_targets','train','eval_target')]
while st:
    f=st.pop()
    if f in seen: continue
    seen.add(f); st+=[p for m in imp(f) if (p:=path(m))]
print(len(seen))"
```
