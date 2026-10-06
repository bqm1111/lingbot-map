# B4 source-data scaling: code guide

**What this package is.** `tools/semantic_field/scaling/` trains the **unchanged** B4 and B3 methods from disk instead of from GPU memory. This lets a pool of ≈ 9,000 anchors fit.

**What it changes.** Only when anchor bytes are read, and whether a run can resume.

**What it does not change.**
- It does not touch model code, losses, masking, targets or the optimiser.
- It does not modify the frozen files `tools/semantic_field/training/trainer.py`, `tools/semantic_field/config/schema.py`, `tools/delayed_semantic/train.py` or Gate 16.
- The only update to the research settings is the stretched schedule length.

## Files

| file | responsibility |
|---|---|
| `config.py` | operational settings only: update-budget formula, host-cache bound, resume interval, resource caps, output paths |
| `source_pool.py` | which anchors a pool contains (`current_pool`, `expanded_pool`, the latter only from the frozen manifest); refuses 0006 / 0009 / non-training drives |
| `loaders.py` | the original per-anchor loaders (`trainer.load_training_anchor` for B4, pilot `train.load_anchor` for B3) called on the CPU, plus the files each one reads |
| `anchor_store.py` | `DiskAnchorSequence`: `anchors[i]` loads anchor i on demand (bounded LRU host cache) and returns a device copy; load errors name the anchor and files |
| `b4_step.py` | one B4 update, the loop body of `trainer.train_seed` lifted verbatim |
| `b3_step.py` | one B3 update: pilot `anchor_step` (imported) aggregated as pilot `fit` |
| `resumable.py` | the shared AdamW + cosine + clip loop with complete resume state |
| `run.py` | builds one run (method × pool × seed), writes the final checkpoint in the method's original format and `run.json` |
| `cli.py` | launch / resume |
| `verify_loader.py` | equivalence with the original trainers (values, draws, tensors, losses, parameters, resume) |
| `benchmark.py` | throughput on real files |
| `geography.py` | exact grid-footprint overlap check against the 0006 development grids |
| `registration.py` | registers the added drives 0000 / 0002 / 0004 / 0005 for the unchanged cache tools (in-process; refuses 0006 / 0009) |
| `storage.py` | hot (NVMe) vs cold (USB MINH1) placement of the added drives' caches: per-drive symlinks at the fixed tool paths, token links |
| `prepare.py`, `prepare_tokens.py` | one preparation stage per call with the unchanged producers (tokens run in the Trident env) |
| `reliability_check.py` | the frozen pilot reliability gate on the pooled 7 training drives |
| `study_freeze.py` | hash freezes: `source` before training, `evaluation` before scoring |
| `evaluate.py` | the frozen 0006 evaluator pointed at one pool's checkpoints |
| `analysis.py` | the frozen analysis / S1–S6 per pool, plus the cross-pool contrasts with the frozen bootstrap |
| `resources.py` | GPU device time (union of busy intervals per GPU) and new storage, from logs and disk usage |
| `cache_trim.py` | releases the page cache of the study's own files every minute, because the session memory guard reacts to low *free* memory |
| `freeze_source.py` | refuses unless every added-drive stage finished and the reliability gate passed; then writes the final manifest, hashes every cache file both methods read, and freezes stage `source` |

**Shell launchers** live in `artifacts/semantic_field_scaling/prep/`:
- `run_tokens.sh` and `run_clips_inputs.sh` build the cold data;
- `run_hot_stages.sh` builds the hot data;
- `move_cold.sh` relocates rarely used directories to MINH1 with checksum verification, leaving a symlink;
- `run_training.sh` starts the training runs.

## Storage layout for the added drives

**Why links.** The frozen tools read fixed paths under `/media/SSD1/MINH_DATASETS/...`. For an added drive, the per-drive directory at that path is a symlink:

| data | where it lives | how it is linked |
|---|---|---|
| inputs, evidence, provenance, delayed targets, B3 train cache | NVMe (`…/lingbot_b4_scaling/hot`) | per-drive directory symlink |
| local-reset clips | MINH1 (`/media/minh/MINH1/lingbot_b4_scaling/cold`) | per-drive directory symlink |
| Trident tokens | MINH1 | one symlink per file in `trident_tokens/k360_train` |

Every link is recorded in `artifacts/semantic_field_scaling/prep/storage_layout.json`. Existing drives are never relinked.

**Moved directories.** Rarely used directories moved off NVMe are listed with checksum results in `prep/cold_moves.jsonl`. Each keeps a symlink at its old path.

**Raw data path.** `/media/welf/MINH` is a symlink to `/media/minh/MINH1`, so the hard-coded `KITTI360_ROOT` still resolves.

## One B4 anchor, from disk to a parameter update

1. **Pool and index.** `source_pool.current_pool()` lists `(drive, t)` in drive order, then `t` ascending. Call the list index `i`; the sampler draws `i`.
2. **Sampling.** `b4_step.b4_step` calls `torch.randint(len(anchors), (4,), generator=gen)`. `gen` is seeded with the run seed; the delayed subsample uses `gen_d`, seeded with seed + 1.
3. **Disk.**
   - `anchors[i]` → `DiskAnchorSequence.__getitem__` → `loaders.host_loader("B4")`.
   - That calls `trainer.load_training_anchor(drive, t, "cpu", tiers)`, which reads:
     - `…/lingbot_gate15_semantic/evidence/<drive>/<t>.npz`: `flat_index` [N] int32 → int64; `feature` [N, 1024] **fp16**, unit L2; `count` [N] int32;
     - `…/lingbot_gate12_suf/inputs/<drive>/<t>.npz`: `S_H`, `F_H` packbits [32,768] uint8;
     - `…/lingbot_delayed_semantic/delayed_targets/<drive>/<t>.npz`: non-evidence tier A/B voxels, `D_flat` [M] int64, `D_prob` [M, 18] fp16. These are **training-only future observations**.
   - Tensors are moved to the GPU unchanged: same dtype, same bytes.
4. **Frozen geometry.**
   - `[S, 0, F]` on the full 256×256×32 grid (velodyne frame of t, 0.2 m) goes through the CompletionUNet under `no_grad` in bf16.
   - It returns occupancy O [4, N] and trunk d1 [4, 24, 256, 256, 32].
5. **Crop and masks.**
   - Crop origin: 3 × `randint` on `gen`, giving a 128×128×32 crop.
   - `mask_evidence` draws one `rand` over 16×16×8 cuboids on `gen`. Hidden evidence becomes the masked target; visible evidence becomes input.
   - `to_crop` keeps voxels inside the crop.
6. **Dense input.**
   - [61, 128, 128, 32] = trunk (24), S, F, O (3), `Linear(1024→32)` of centred visible evidence (32), evidence mask (1), log-count (1).
   - The projection is trainable and computed fresh every step; nothing trainable is cached.
7. **Targets.**
   - Up to 8,192 hidden evidence voxels (`randperm` on `gen`).
   - Up to 4,096 delayed voxels (`randperm` on `gen_d`).
8. **Loss.** `SemanticFieldHead.hidden_field` (bf16) → `semantic_features` at target voxels → `masked_reconstruction_loss` + `delayed_loss` (`tools/semantic_field/losses/semantic_loss.py`).
9. **Update** (`resumable.train_resumable`): `zero_grad`, `backward`, clip to 1.0, AdamW step, cosine scheduler step. The step's draw records and log row are appended.

**B3.**
- **Loader:** `train.load_anchor` reads the evidence plus:
  - `provenance/<drive>/<t>.npz` (frame bits);
  - `train_cache/student/<drive>/<t>.npz` (packed F/S/O, `trunk_E` [N, 24] fp16);
  - `delayed_targets`;
  - `train_cache/query/<drive>/<t>.npz` (`trunk_D` [M, 24] fp16).
- **Step:** 8 anchors per update; `anchor_step` picks hidden and delayed queries and their 16 nearest candidates.

## Determinism and resume

**Draws are exactly reproducible.** Loading consumes no random numbers. The draw sequence depends only on `gen` / `gen_d` and on the anchors' stored flat indices.

**What a resume checkpoint (`runs/<method>_<pool>_seed<s>/resume.pt`) stores:**
- model, optimizer and scheduler state;
- both generator states;
- the draw records;
- the log.

**Resume behaviour.** Resuming draws exactly what an uninterrupted run draws.

**B3 is bit-identical;** it has no nondeterministic kernels.

**B4 matches to floating-point tolerance only.** Its backward pass uses CUDA kernels without a deterministic implementation (`upsample_trilinear3d_backward`), so B4 losses match to about 1e-3 after a few steps, not bit-for-bit.

**The original B4 trainer varies by the same amount.** Two runs of it differ by that much too (`verification/loader_equivalence.json`, `original_vs_original`).

## Commands

```bash
cd /home/minh/workspace/lingbot-map_fork; export PYTHONPATH=$PWD HF_HUB_OFFLINE=1; PY=/home/minh/anaconda3/envs/cu128/bin/python
# checks
$PY -m pytest tests/semantic_field/test_scaling.py -q                          # CPU: cache bound, errors, resume, membership, budgets
B4_SCALING_GPU_TESTS=1 $PY -m pytest tests/semantic_field/test_scaling.py -q   # + GPU equivalence with the original B4 trainer
$PY -m tools.semantic_field.scaling.verify_loader --device cuda:0             # full equivalence record
# launch; running the same command again resumes from resume.pt
$PY -m tools.semantic_field.scaling.cli train --method B4 --pool current --seed 0 --device cuda:0
$PY -m tools.semantic_field.scaling.cli train --method B4 --pool current --seed 0 --device cuda:0 --stop-after-step 10000   # stop cleanly
```

The expanded pool can only be launched after `artifacts/semantic_field_scaling/manifests/source_pool_expanded.json` is frozen.

## Inspecting one sample

```python
from tools.semantic_field.scaling import loaders
from tools.semantic_field.scaling.anchor_store import DiskAnchorSequence
from tools.semantic_field.scaling.source_pool import current_pool
pool = current_pool(); tiers = loaders.delayed_tiers()
seq = DiskAnchorSequence(pool.anchors, loaders.host_loader("B4", tiers), loaders.files_of("B4"), "cpu", 0)
a = seq[0]; print(pool.anchors[0], {k: (tuple(v.shape), v.dtype) for k, v in a.items() if hasattr(v, "shape")})
```

## Diagnosing a loader error

`AnchorLoadError: loading anchor drive=… t=… failed while reading [files]: KeyError: 'feature'` means that anchor's file lacks the expected array.

1. Open the named file with `numpy.load` and compare its keys to the schema above.
2. Check its producer's log (evidence: `artifacts/gate15_semantic_completion/logs`; targets: `artifacts/delayed_semantic_pilot/logs`).
3. Rebuild only that anchor with the producer.

Training never skips or replaces an anchor: a load error stops the run, and the last `resume.pt` is intact.
