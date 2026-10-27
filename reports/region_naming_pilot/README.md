# Region naming for semantic completion — an isolated pilot

One question: **with completed occupancy fixed, can a vision-language model name objects and coherent
regions more accurately than the deployed Trident evidence, and can those names be carried to occupied
voxels that have no causal semantic evidence?**

This directory is the whole experiment. It is deliberately disjoint from both repositories it reads
about:

| identity | path | how it is treated |
|---|---|---|
| `EXPERIMENT_ROOT` | `/media/SSD1/region_naming_pilot` | the only place anything is written |
| `LEGACY_REPO` | `/home/minh/workspace/lingbot-map_fork` | imported read-only, bytecode writes disabled, no script executed |
| `REPRO_REPO` | `/home/minh/workspace/StreamingSSC` | **never read, imported or written** |

`code/paths.py` asserts the disjointness, that `PYTHONDONTWRITEBYTECODE` is set and that `HF_HOME`
points inside this directory, at import time. `code/verify_final.py` re-checks both protected trees
afterwards and stores the before/after snapshots in `artifacts/`.

## Environment

An independent venv, created from the machine's `cu128` interpreter with `--system-site-packages` so
that `torch 2.7.1+cu128` (the only build that supports these sm_120 GPUs) is reused without installing
anything into the conda environment. `transformers`, `accelerate` and `qwen-vl-utils` were installed
into the venv overlay only; `cu128` still has no `transformers`.

```bash
EXP=/media/SSD1/region_naming_pilot
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=$EXP/cache/hf TORCH_HOME=$EXP/cache/torch TRITON_CACHE_DIR=$EXP/cache/triton
export PYTHONPATH=$EXP/code
export CUDA_VISIBLE_DEVICES=1          # the designated device; see "Compute" below
PY=$EXP/venv/bin/python
```

## Rerun the pilot

Every stage is resumable and idempotent: it skips anchors whose output file already exists.

```bash
$PY $EXP/code/manifest.py                  # 0  anchor selection + experiment_manifest.json
$PY $EXP/code/verify_b0.py      cuda:0     # 1  adapter alignment: B0 over all 590 anchors of the drive
$PY $EXP/code/run_proposals.py  cuda:0     # 2  frozen SAM region proposals          (~2 min / anchor)
$PY $EXP/code/calibrate.py      cuda:0     # 3  choose max_hops on calibration anchors only
$PY $EXP/code/run_qwen.py       cuda:0     # 4  Qwen-VL names the queried regions    (~3 min / anchor)
$PY $EXP/code/run_predict.py    cuda:0     # 5  the five label sets per anchor
$PY $EXP/code/evaluate.py       cuda:0     # 6  scoring with the legacy scorer -> artifacts/analysis.json
$PY $EXP/code/figures.py        cuda:0     # 7  figures/
$PY $EXP/code/report.py                    # 8  results.csv + report.md
$PY $EXP/code/verify_final.py   cuda:0     # 9  the concrete risk checks -> artifacts/verification.json
```

Stage 1 is the gate: it must print `ALL MATCH`. It recomputes the deployed baseline through this
experiment's adapter and compares it with the three populations the legacy Gate 17 report published, to
twelve decimal places. If it does not match, nothing downstream is comparable to anything.

## Compute

No GPU on this machine is free: all four carry other users' long-running IsaacLab jobs. None of them
belongs to the reproduction effort, which holds no GPU. The pilot therefore runs on **one** designated
device, `CUDA_VISIBLE_DEVICES=1` (about 69 GiB free of 98 GiB), with a single 8B checkpoint, no
multi-GPU, and no simultaneous large download. Nothing else was killed, restarted or reprioritised.

## What is where

```
code/        paths, the legacy adapter, proposals, recognition, extension, prediction, scoring, report
cache/hf     the Qwen checkpoint (nothing is written to the user's shared HF cache)
cache/proposals, cache/qwen, cache/crops, cache/predictions, cache/membership
artifacts/   experiment_manifest.json, calibration.json, verify_b0.json, analysis.json,
             verification.json, scores/confusions.npz, protected_state_{before,after}.txt
figures/     cases.png, recognition.png, bev_*.png
results.csv  matched metrics, support, coverage
report.md    the conclusion, the evidence, the limitations and one next action
```

## Design, in one paragraph

Region proposals come from frozen SAM automatic mask generation on each of the five causal frames; a
mask pixel becomes 3D support only if it is a ray the frozen mapper already accepts, so support is the
frame's own visible surface and the lifted region set is **exactly** the frozen evidence set (asserted
per anchor). Masks are linked across frames by 3D overlap with an appearance guard. Each region is then
named twice — once by pooling its own Trident features through the frozen text readout (P), once by
Qwen-VL from crops and a highlighted context frame (Q) — and the names are propagated two ways: the
original nearest-evidence rule (`_N`) and one bounded geodesic growth over the fixed occupancy (`_R`).
All four variants share one occupancy volume, one evidence set, one region decomposition and one
membership map; only the name differs. Human labels are opened by the evaluation and figure stages
only, and the prediction stages run inside a file audit that raises if one is touched.
