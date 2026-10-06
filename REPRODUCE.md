# Zero-shot semantic scene completion — the results, and how to rebuild them

**The system.** Causal, single-camera, semantic-annotation-free 3D semantic scene completion,
trained on KITTI-360 only and evaluated on SemanticKITTI 08 and Occ3D-nuScenes val.

- **Frozen models** (never fine-tuned): LingBot-Map (depth, pose), MoGe-2 (metric clip scale),
  Trident-H (open-vocabulary features).
- **Geometry:** a hard mapper over five causal frames t-4..t (local-reset LingBot), completed by
  the frozen geometry baseline `CompletionUNet` (seed 0).
- **Semantics:** Trident-H features lifted to voxels at accepted ray endpoints (Gate 15 evidence).
  - **B0** names a voxel after its nearest evidence voxel.
  - **B3** (Gate 16, seeds 0/1/2) picks among the 16 nearest evidence candidates with a head trained on
    delayed (future) observations.

This is target-domain-free transfer of the trained parts, not whole-system zero-shot.
LingBot-Map's published training mixture includes KITTI-360.

The previous version of this file described the Gate 8C-1 pipeline and its pre-ceiling-fix numbers.
Older reports cite it, and it is kept at `reports/gate8c1/REPRODUCE_gate8c1.md`.

---

## 1. The results

Read from `artifacts/final_zero_shot/results_numbers.md`. Interpretation is in
`final_zero_shot_results.md` and `paper_decision.md` in the same folder.

**Samples.** Every SemanticKITTI 08 and Occ3D-nuScenes val sample that OccAny scored and that
has a causal window: 801 and 4,219.

**Track P** is the protocol frozen before scoring: the input camera's horizontal field of view,
within 40 m, on the official grids.

| dataset | SC IoU | SSC mIoU B0 / B3 | completed no-evidence mIoU B0 / B3 | Δ B3−B0 central [95 % CI] |
|---|---:|---:|---:|---:|
| SemanticKITTI 08 | 32.65 | 5.45 / 5.64 | 12.33 / 12.58 | +0.25 [−0.08, +0.57] |
| Occ3D-nuScenes val | 36.87 | 4.36 / 4.54 | 8.18 / 8.72 | +0.54 [+0.29, +0.86] |

**Official** (Track O, OccAny's `SSCMetrics`):

| dataset | SC IoU | SSC mIoU B0 / B3 |
|---|---:|---:|
| SemanticKITTI 08 | 26.40 | 4.32 / 4.43 |
| Occ3D-nuScenes val | 31.94 | 3.71 / 3.80 |

**Geometry against OccAny.** Scored with the same evaluator in Track P, ours beats OccAny's
released geometry by +1.4 (SemanticKITTI) and +6.8 (Occ3D).

**Semantics are teacher-capped.** OccAny's published SSC mIoU (7.28 / 6.66) is not like-for-like:
on Occ3D it uses 6 surround cameras, and on SemanticKITTI it uses future frames.

**Outcome:** PAPER C — geometry-centric, with semantics as a limitation.

---

## 2. Three levels of rerun

### A. Re-score and re-report from the stored predictions (no training, about 1 GPU-hour)

See `tools/zero_shot_eval/README.md`.

- The predictions are bit-reproducible from the frozen checkpoints and caches. Verified on 2026-10-06:
  5 samples per dataset, rebuilt into a scratch `ZERO_SHOT_EVAL_DATA_ROOT` (step 11 below), are
  byte-identical to the stored files.
- For a quick example, run `run_shard(..., limit=5)` and then `evaluate_dataset(..., limit=5)`.
  Both write `_limit5` files and don't touch the full results.

### B. Rebuild the target side from the frozen checkpoints

Steps 9–11 below. Steps 9–10 already ran for the OccAny comparison, and their outputs are cached.

### C. Full rebuild from raw data

Steps 0–11 below, in order. Each step reads only what earlier steps wrote.

- **These are traced, not tested.** The commands come from each package's README and code. Only
  step 11 has been rerun end-to-end since the project closed.
- **Expect small numeric drift.** The training steps (5, 8) are not bit-reproducible, so a
  retrained system will not reproduce the numbers in § 1 exactly.

---

## 3. Environment

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0a
PY=/home/minh/anaconda3/envs/cu128/bin/python                    # bare `python` is not on PATH
TRI=/home/minh/workspace/third_party/Trident; TPY=/home/minh/workspace/third_party/trident_env/bin/python
```

| what | where |
|---|---|
| main env | conda `cu128`, torch 2.7.1+cu128, 4× RTX PRO 6000 Blackwell (sm_120); `pip install -e . --no-deps` so `lingbot_map` imports from anywhere |
| teacher env | `../third_party/Trident` + `../third_party/trident_env` (CLIP ViT-H-14 laion2b + SAM ViT-H `../third_party/checkpoints/sam_vit_h_4b8939.pth`); always run from `$TRI` with `PYTHONPATH=$REPO:$TRI` |
| OccAny | `../third_party/OccAny` + `occany_env`. Only its released 5-frame geometry is used. Its semantics (GroundingDINO deformable attention) do not build on sm_120 |
| LingBot-Map | `checkpoints/lingbot-map/204754b/lingbot-map.pt` |
| MoGe-2 | wrapped by the in-repo `moge_gauge/` (calibrated horizontal FOV) |
| KITTI-360 raw | `/media/welf/MINH/datasets/kitti360/KITTI-360/` (images, Velodyne, GT poses) |
| SSCBench-KITTI-360 | `/media/SSD1/MINH_DATASETS/sscbench_kitti360/` (labels used only to evaluate drive 0006) |
| SemanticKITTI | `data/kitti/dataset/` (sequence 08) |
| nuScenes + Occ3D | `/media/SSD1/MINH_DATASETS/nuscenes/` (`occ3d_gt/Occupancy3D-nuScenes-trainval`) |

**Splits.** KITTI-360 training drives are 0003, 0007 and 0010. Development evaluation uses 0006.
SemanticKITTI and Occ3D are opened only in steps 9–11.

---

## 4. The pipeline

Every cache root below is under `/media/SSD1/MINH_DATASETS/`.

### Source side — KITTI-360 only

**Step 1. Frozen per-frame caches.**

- What it builds: LingBot stream, MoGe-B depth, scale candidates and Trident-H probabilities.
- Commands: the cache block at the top of `tools/kitti360_data_scaling_7drives/README.md`, which runs
  `tools/gate8b/stream_source.py` and `tools/gate8b/cache_trident.py` through `drives7.py`.
  `tools/gate8b/run_caches.sh` is the older 3-drive runner.
- Output: `lingbot_gate8*/{stream,moge_b,scale,semantics}`.

**Step 2. Raw-LiDAR occupancy targets and evaluation masks.**

```bash
for d in 0003 0007 0010; do $PY tools/kitti360_data_scaling_7drives/drives7.py core/run/build_targets.py --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
$PY tools/kitti360_data_scaling_7drives/drives7.py core/run/build_targets.py --drive 2013_05_28_drive_0006_sync --stride 3 --device cuda:0
```

Then run `build_eval_masks.py` from `tools/kitti360_data_scaling_7drives/run_all.sh`.
Output: `lingbot_k360_7drives/{targets,eval_masks}`.

**Step 3. Local-reset five-frame LingBot clips, each with its own causal MoGe clip scale.**

```bash
for d in 0003 0007 0010; do $PY tools/gate12_suf/gen_local_clips.py --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
$PY tools/local_registration_audit/run_local_lingbot.py --drive 2013_05_28_drive_0006_sync --device cuda:0
```

Output: `lingbot_gate12_suf/local_lingbot` (training) and
`lingbot_local_registration_audit/local_lingbot` (0006).

**Step 4. Hard-mapper inputs (S, F) and 0006 regions.**

```bash
for d in 0006 0003 0007 0010; do $PY tools/gate12_suf/build_inputs.py --drive 2013_05_28_drive_${d}_sync --device cuda:0; done
$PY tools/gate12_suf/build_regions.py --device cuda:0
```

Output: `lingbot_gate12_suf/{inputs,regions}`.

**Step 5. Geometry baseline: `CompletionUNet` with 3 channels, seeds 0/1/2.**

```bash
T=tools/baseline_reconstruction/cli
for s in 0 1 2; do $PY $T/train_current.py --seed $s --device cuda:$s & done; wait    # ~9 min per seed
$PY $T/eval_source.py --device cuda:0
$PY $T/freeze_source.py      # writes code_freeze.json; refuses on a FAIL verdict
```

Output: `artifacts/baseline_reconstruction/checkpoints/geo_seed*_step06000.pt` and `code_freeze.json`.
Every later step loads geometry seed 0 and checks its sha256 against this freeze.

**Step 6. Trident-H token maps and Gate 15 evidence.**

First, the token maps (Trident env), for each of 0003, 0006, 0007 and 0010:

```bash
cd $TRI && PYTHONPATH=$OLDPWD:$TRI $TPY $OLDPWD/tools/feature_first_3d_semantics/extract_features.py --drive 2013_05_28_drive_0006_sync --device cuda:0
cd $TRI && PYTHONPATH=$OLDPWD:$TRI $TPY $OLDPWD/tools/feature_first_3d_semantics/encode_queries.py
```

Then the evidence cache (cu128 env), for each drive and shard:

```bash
$PY tools/semantic_completion/cli/build_evidence.py --drive 2013_05_28_drive_0006_sync --shard 0 --shards 2 --device cuda:0
```

Output: `lingbot_feature_first_3d/trident_tokens`, `text_embeddings.npz`, and
`lingbot_gate15_semantic/evidence` (~73 GB).

Gate 15's own B1 training (`tools/semantic_completion/README.md`) is not needed for B3.

**Step 7. Delayed-supervision caches: provenance, delayed targets, reliability gate, train cache.**

- Commands: the "Reproduce" block of `tools/delayed_semantic/README.md`. Run the `rules.freeze(...)`
  steps in the order given there.
- The reliability gate (`run.py reliability`) must pass before training.
- The pilot's B1/B2/B2s heads are controls, not inputs to B3.
- Output: `lingbot_delayed_semantic/{provenance,delayed_targets,train_cache,...}`.

**Step 8. Gate 16 B3 heads, seeds 0/1/2.**

```bash
X=tools/delayed_semantic/gate16/cli.py
$PY $X freeze --stage predeclaration
$PY $X ceiling --device cuda:0 && $PY $X freeze --stage ceiling
$PY $X train --variant B3 --device cuda:1
$PY $X freeze --stage code
# optional source evaluation (two shards at a time, host-memory guard), then analyse / report / verify
```

Output: `artifacts/gate16_candidate_selection/checkpoints/B3_seed{0,1,2}_step03000.pt`.

### Target side — SemanticKITTI 08 and Occ3D-nuScenes val

**Step 9. OccAny released geometry.**

- Run in `../third_party/OccAny` with its released 5-frame configuration
  (`OccAny_5frames_{kitti,nuscenes}512_rot60_vpi10_fwd3_sTrans2`; thresholds 2.5 / 1.1).
- Output goes to `occany_out/ssc_voxel_pred/`.
- This step is needed for two reasons:
  - **The sample list.** The 805 / 4,819 scored samples are defined by what OccAny scored
    (`tools/occany_comparison/samples.py`).
  - **The labels and the geometry comparison.** Its stored labels were verified identical to the
    official files on every sample.
- The configuration is recorded in `reports/gate8c1_original_audit.md` and
  `artifacts/final_zero_shot/occany_comparison_audit.md`.

**Step 10. Causal target clips.**

```bash
$PY tools/occany_comparison/cli/build_moge.py --dataset occ3d --device cuda:0
for k in 0 1 2 3; do $PY tools/occany_comparison/cli/build_clips.py --dataset occ3d --device cuda:$k --shard $k --shards 4 & done; wait
$PY tools/occany_comparison/cli/build_clips.py --dataset semantickitti --device cuda:0
```

- The full OccAny comparison (`cli/evaluate.py`, `cli/report.py`) is optional here.
- Output: `lingbot_occany_comparison/{clips,moge_b}`. The final evaluation reads `clips/<dataset>/causal`.

**Step 11. Final zero-shot evaluation.** The complete block is in `tools/zero_shot_eval/README.md`.

```bash
$PY -m tools.zero_shot_eval.samples                                   # frame lists
# Trident-H tokens of the target frames (Trident env, one shard per GPU), both datasets
cd $TRI && for g in 0 1 2 3; do PYTHONPATH=$OLDPWD:$TRI $TPY $OLDPWD/tools/zero_shot_eval/teacher/extract_tokens.py \
    --dataset semantickitti --shard $g --shards 4 --device cuda:$g & done; wait; cd -
$PY -c "from tools.zero_shot_eval import freeze; freeze.freeze('protocol'); freeze.freeze('evaluators')"   # once, before any score
for d in semantickitti occ3d; do for s in 0 1 2 3; do
  $PY -c "from tools.zero_shot_eval.prediction.pipeline import run_shard; run_shard('$d','cuda:$s',$s,4)" & done; wait; done
for d in semantickitti occ3d; do $PY -c "from tools.zero_shot_eval.run_eval import evaluate_dataset; evaluate_dataset('$d')"; done
$PY -c "from tools.zero_shot_eval import analysis; analysis.analyse()"
$PY -c "from tools.zero_shot_eval import report; report.write_all()"
$PY -c "from tools.zero_shot_eval import verify; verify.run('cuda:0')"
$PY -m pytest tests/zero_shot_eval -q
```

- **Cost:** about 2 s per sample on SemanticKITTI and 4 s on Occ3D, so about 1 h on 4 GPUs.
- **Caches:** `lingbot_zero_shot_eval/{trident_tokens,predictions}`.
- **Results:** `artifacts/final_zero_shot/`.

---

## 5. Read before rebuilding

- **Freezes are guards, and re-running one overwrites the record.**
  - The packages hash-freeze their protocol and code before scoring:
    - geometry: `code_freeze.json`
    - Gate 16: `frozen_*.json`
    - final evaluation: `frozen_protocol.json` and `frozen_evaluators.json`
  - Every later entry point re-checks those hashes and refuses to run on a mismatch.
  - Re-running `freeze(...)` in place replaces the original pre-registration record.
  - **To keep the originals, rebuild into fresh roots.** Most packages honour a data-root variable:
    - `BASELINE_RECON_DATA_ROOT`
    - `GATE15_DATA_ROOT`
    - `DELAYED_SEM_DATA_ROOT`
    - `OCCANY_COMPARISON_DATA_ROOT`
    - `ZERO_SHOT_EVAL_DATA_ROOT`

  Artifact folders under `artifacts/` are fixed paths, so use a separate clone for a full rebuild.
- **Prediction resumes, it does not overwrite.** `run_shard` and `extract_tokens.py` skip files that
  already exist. Pointing them at the existing roots re-scores the stored predictions instead of
  rebuilding them.
- **Rebuilding step 2 is not a no-op.** `core/supervision/rawtarget.py` gained an open-sky rule after
  some older targets were cached, and a bug in that rule (marking a measured return as free) was fixed
  in `647bcf3`. Rebuilt KITTI-360 targets move source-domain numbers. Target-domain scores use the
  official labels and are unaffected.
- **Some Gate 8C-1 sources are unrecoverable.** Three of them drifted from their manifest hashes before
  the tree was tracked. This affects only the historical baseline (SemanticKITTI 22.71 / Occ3D 33.88),
  not this pipeline.
- **The frozen-model caches are the expensive part.** Steps 1, 6 and 11's token extraction dominate;
  training is cheap:

  | training | cost |
  |---|---|
  | geometry | about 9 min per seed |
  | Gate 16 | minutes per head once caches are loaded |

---

## 6. Where everything is

```
lingbot_map/       the frozen LingBot-Map package
moge_gauge/        MoGe-2 with calibrated FOV
core/              mapper, targets, CompletionUNet, datasets, grids (core/README.md)
gates/             earlier research gates, frozen
tools/<package>/   one package per study; each has a README with its run order
artifacts/<pkg>/   results (bulk .npz/.pt are gitignored)
reports/           written reports; reports/METHOD.md explains the method end to end
tests/<pkg>/       tests
```

| step | package | results |
|---|---|---|
| 5 | `tools/baseline_reconstruction` | `artifacts/baseline_reconstruction/report.md` |
| 6 | `tools/semantic_completion` (Gate 15) | `artifacts/gate15_semantic_completion/report.md` |
| 7 | `tools/delayed_semantic` | `artifacts/delayed_semantic_pilot/report.md` |
| 8 | `tools/delayed_semantic/gate16` | `artifacts/gate16_candidate_selection/report.md` |
| 9–10 | `tools/occany_comparison`, `tools/like_for_like` | `artifacts/occany_comparison/report.md`, `artifacts/like_for_like/report.md` |
| 11 | `tools/zero_shot_eval` | `artifacts/final_zero_shot/final_zero_shot_results.md` |
