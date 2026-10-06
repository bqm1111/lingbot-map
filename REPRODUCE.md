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

## 2. How to run it

The whole pipeline is one script, **[`reproduce.sh`](reproduce.sh)**. Read it top to bottom: its header
draws the data flow, and each of its eleven stages says what it makes, what it reads and where it writes.

```bash
bash reproduce.sh --dry-run        # print every command, run nothing
bash reproduce.sh                  # everything, raw data to final numbers
bash reproduce.sh --from 11        # only the final evaluation (re-scores the cached predictions)
bash reproduce.sh --from 5 --to 8  # retrain geometry and the semantic head
```

- **Tested:** stage 11's prediction step was rerun on 2026-10-06; its predictions were byte-identical
  to the stored ones.
- **Not tested end to end:** stages 1–10 come from each package's README and argument parser.
- **Not bit-reproducible:** training (stages 5 and 8), so a retrained system drifts slightly from § 1.
- **Per-stage details:** `tools/<package>/README.md` (table in § 4).

Environment: conda `cu128` (torch 2.7.1+cu128) on sm_120 GPUs, plus the Trident and OccAny envs under
`../third_party/`. The script sets the CUDA 12.8 / FlashInfer variables; dataset paths are in § 4.

---

## 3. Read before rebuilding

- **Freezes are guards.** Stages 5, 7, 8 and 11 hash-freeze their protocol or code before scoring,
  and every later entry point re-checks those hashes and refuses to run on a mismatch.
  - `reproduce.sh` never overwrites an existing freeze record (`freeze_once`).
  - Consequence: retraining in place against the old records fails the hash checks, which is intended.
    For a full rebuild, use a separate clone and empty data roots. Most packages read a variable for
    their root: `BASELINE_RECON_DATA_ROOT`, `GATE15_DATA_ROOT`, `DELAYED_SEM_DATA_ROOT`,
    `OCCANY_COMPARISON_DATA_ROOT`, `ZERO_SHOT_EVAL_DATA_ROOT`. Folders under `artifacts/` are fixed paths.
- **Prediction resumes, it does not overwrite.** `run_shard` and `extract_tokens.py` skip files that
  already exist. Pointing them at the existing roots re-scores the stored predictions instead of
  rebuilding them.
- **Rebuilding stage 2 is not a no-op.** `core/supervision/rawtarget.py` gained an open-sky rule after
  some older targets were cached, and a bug in that rule (marking a measured return as free) was fixed
  in `647bcf3`. Rebuilt KITTI-360 targets move source-domain numbers. Target-domain scores use the
  official labels and are unaffected.
- **Some Gate 8C-1 sources are unrecoverable.** Three of them drifted from their manifest hashes before
  the tree was tracked. This affects only the historical baseline (SemanticKITTI 22.71 / Occ3D 33.88),
  not this pipeline.
- **The frozen-model caches are the expensive part.** Stages 1 and 6 and the token extraction in stage 11 dominate;
  training is cheap:

  | training | cost |
  |---|---|
  | geometry | about 9 min per seed |
  | Gate 16 | minutes per head once caches are loaded |

---

## 4. Where everything is

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

| stage | package | results |
|---|---|---|
| 5 | `tools/baseline_reconstruction` | `artifacts/baseline_reconstruction/report.md` |
| 6 | `tools/semantic_completion` (Gate 15) | `artifacts/gate15_semantic_completion/report.md` |
| 7 | `tools/delayed_semantic` | `artifacts/delayed_semantic_pilot/report.md` |
| 8 | `tools/delayed_semantic/gate16` | `artifacts/gate16_candidate_selection/report.md` |
| 9–10 | `tools/occany_comparison`, `tools/like_for_like` | `artifacts/occany_comparison/report.md`, `artifacts/like_for_like/report.md` |
| 11 | `tools/zero_shot_eval` | `artifacts/final_zero_shot/final_zero_shot_results.md` |

Data and third-party code:

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
SemanticKITTI and Occ3D are opened only in stages 9–11.
