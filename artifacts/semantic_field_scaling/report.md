# B4 source-data scaling study: BLOCKED (required reliability gate failed)

**Status** (2026-09-15, ~21:25).
- Cache preparation is complete for all 7,708 added anchors.
- The declared, required delayed-target reliability gate **failed on G1 (registration)**, so the study stops before the manifest freeze.
- No study training step ran and no new semantic score exists; SemanticKITTI and Occ3D were not accessed.
- This is not SOURCE FAIL. B4's earlier SOURCE FAIL (8.22 vs B0 8.80 / B3 9.38) is unchanged.

**Nothing is running.** The page-cache trimmer and all preparation jobs are stopped.

## The blocker

**Declared rule** (`source_declaration.md`): the frozen pilot gate (`tools/delayed_semantic/rules.py::RELIABILITY_GATE`) is applied with its frozen thresholds and its frozen reading. G1 is pooled over the training drives, now 7, plus 0006 development statistics. It must pass with delayed-loss tiers [1, 2]; otherwise the study is BLOCKED.

**Result** (`prep/reliability_expanded.json`, `tools/semantic_field/scaling/reliability_check.py`):

| check | requirement | expanded (7 drives) | original pilot (3 drives) |
|---|---|---|---|
| G1 registration @ +1 frame | median ≤ 0.10 m, p90 ≤ 0.20 m | 0.024 / 0.067 | 0.027 / 0.073 |
| G1 registration @ +5 frames | same | 0.060 / 0.148 | 0.055 / 0.137 |
| **G1 registration @ +10 frames** | same | 0.081 / **0.204 → fail** | 0.069 / 0.184 → pass |
| G2 coverage | mean non-evidence targets ≥ 2,000 per anchor, ≥ 8 classes with ≥ 10k voxels | 4,280; 18 classes → pass | pass |
| G3 repeatability, tier A | train ≥ 0.65 with margin ≥ 0.10, dev ≥ 0.60 with margin ≥ 0.10 | train 0.767 (shuffled 0.342), dev 0.759 (0.359) → pass | pass |
| G3 repeatability, tier B | same | 0.726 (0.310) → pass | pass |
| G4 re-observation | ≥ 0.65 | 0.679 → pass | pass |

**Gate outcome:** `pass = False`, `delayed_loss_tiers = []` (the frozen stop rule).

**Per-drive G1 at +10 frames (median / p90, m).** Every drive passes at +1 and +5.

| drive | pool | +10 median | +10 p90 |
|---|---|---:|---:|
| 0003 | current | 0.086 | 0.180 |
| 0007 | current | 0.045 | 0.137 |
| 0010 | current | 0.076 | **0.207** |
| 0000 | added | 0.070 | 0.199 |
| 0002 | added | 0.087 | **0.212** |
| 0004 | added | 0.086 | **0.214** |
| 0005 | added | 0.084 | 0.190 |
| 0006 | development | 0.090 | **0.217** |

**Reading the numbers.**
- The pooled +10-frame p90 rises from 0.184 m to 0.204 m because 0002 and 0004 contribute most of the added samples.
- Current-pool drive 0010 and the development drive 0006 already exceed 0.20 m on their own; the original pooled reading passed because 0003 and 0007 were lower.
- The +10-frame offset (~5 s) is the largest offset with data.
- The perturbation control (+0.2 m) still separates clearly (`registration.*.perturbed`).

**Why I stopped.** Continuing would need a protocol change that is yours to make, for example:
- keep the original 3-drive gate record as the delayed-target reliability authorisation;
- change the pooling or reading;
- drop drives or anchors, which changes the pool and recomputes the budgets.

I did not change the rule, thresholds, reading or pool.

## What is complete

**Code** (`tools/semantic_field/scaling/`, guide `GUIDE.md`, tests `tests/semantic_field/test_scaling.py`: 35 passed, 7 skipped in the semantic suites):
- disk-backed bounded loading with resumable training (verified against the original trainers);
- throughput benchmarks;
- geography check;
- drive registration and hot/cold storage placement;
- per-stage preparation;
- page-cache trimmer;
- reliability wrapper;
- freeze;
- evaluator and analysis adapters;
- resource accounting.

**Loader equivalence** (`verification/loader_equivalence.json`).
- **B3:** bit-identical to the original trainer, including resume.
- **B4:** identical draws, loaded values, inputs and targets. Later-step values differ only by the original trainer's own run-to-run CUDA nondeterminism.

**Source pool** (`manifests/source_pool_expanded.DRAFT.json`, **not frozen**).
- 9,066 anchors (1,358 current + 7,708 added: 0000 2,094; 0002 2,769; 0004 2,034; 0005 811).
- Budgets: B4 40,056, B3 20,028.
- Exact footprint check: 0 kept anchors overlap a 0006 development grid.

**Preparation.** Every stage exited 0 for every kept anchor, with no firewall violations and no dropped anchors:
- Trident-H tokens: 8,350 frames;
- local-reset clips and hard maps;
- Gate 15 evidence;
- provenance: every drive's re-lift checks pass (max |Δfeature| ≤ 2.7e-4, class agreement ≥ 0.9996);
- 10 s delayed targets;
- B3 train caches.

All added anchors load through the real B4 and B3 loaders (checked on the first, middle and last anchor of each finished drive).

**Storage.**
- Hot caches: SSD2 240 GB (0005 / 0000 / 0002), SSD1 83 GB (0004).
- Cold (tokens, clips): 82 GB on `/media/minh/MINH1`.
- Existing data relocated: 110.8 GB of rarely used directories moved to MINH1 (checksum-verified, with symlinks at their old paths; `prep/cold_moves.jsonl`).
- `/media/welf/MINH` is a symlink to `/media/minh/MINH1`, so `KITTI360_ROOT` resolves.

**Resources** (`resources/ledger.jsonl`, `resources/estimate.md`).
- **GPU:** 6.32 of 48 GPU-h used. Remaining estimate if unblocked ≈ 25–30 GPU-h (12 runs ≈ 24 + source evaluation 0.6), so the whole study ≈ 31–37.
- **Storage:** 405 GB of 750 new.

**Operational incident.** At 19:50 the session memory guard stopped the background jobs while free memory was low from page cache. All writers are atomic; the newest outputs were re-validated and the jobs resumed. `cache_trim.py` must run during any later training.

## If unblocked: exact resume steps

```bash
cd /home/minh/workspace/lingbot-map_fork; export PYTHONPATH=$PWD HF_HUB_OFFLINE=1; PY=/home/minh/anaconda3/envs/cu128/bin/python
$PY -m tools.semantic_field.scaling.cache_trim --every 60 --with-read-roots &           # keep running throughout
$PY -m tools.semantic_field.scaling.freeze_source                                       # refuses unless the gate record says ok
bash artifacts/semantic_field_scaling/prep/run_training.sh                              # 12 runs; rerun the same command to resume
for p in current expanded; do for s in 0 1 2 3; do $PY -m tools.semantic_field.scaling.evaluate --pool $p --device cuda:$s --shard $s --shards 4 & done; wait; done
$PY -c "from tools.semantic_field.scaling.analysis import run; run()"
```

**Caveat.** Any decision that accepts the current gate result must be recorded in `source_declaration.md` before `freeze_source`. `freeze_source` itself reads `prep/reliability_expanded.json`, so a decision that changes the required gate record needs that file, or the script's check, updated explicitly and visibly.
