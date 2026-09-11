# LingBot-MAP + scene completion: getting started

This fork studies **how to turn images from one camera into a completed 3D scene**. Start by understanding one cached input and its saved predictions. You do not need to train a model, rebuild a dataset, or read every experiment to get started.

There are two parts:

| Part | Input → output | Where to look |
|---|---|---|
| **Reconstruction** | An image sequence → estimated depth, camera motion and visible 3D geometry | `lingbot_map/`, `demo.py` |
| **Scene completion** | Accumulated geometry and semantic evidence → occupied/free predictions, including unobserved regions | `core/`, `gates/gate8/`, original Gate 8C-1 artifacts |

The eventual research objective is a generalizable scene-completion baseline that beats OccAny on SemanticKITTI and Occ3D-nuScenes, followed by semantic prediction. **That objective has not been established.** The original model has reproducible measurements, but evaluation differences and poor source-domain behaviour still need careful explanation.

**Your first session:** follow steps 1–4 below. They explain the system and inspect existing evidence without running inference. Steps 5–6 are optional GPU examples. Training and full benchmark evaluation are separate research decisions.

## 1. Open the project and check the environment

The commands below target the existing research machine. Paste this block into a Bash terminal; keep that terminal open for the later commands.

```bash
cd /home/minh/workspace/lingbot-map_fork

export LINGBOT_PY=/home/minh/anaconda3/envs/cu128/bin/python
export PYTHONPATH="$PWD:$PWD/gates${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/home/minh/anaconda3/envs/cu128/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.8
export FLASHINFER_CUDA_ARCH_LIST="12.0a"
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR=/tmp/lingbot_map_mpl

"$LINGBOT_PY" -c 'import sys, torch; print("Python:", sys.version.split()[0]); print("PyTorch:", torch.__version__); print("CUDA runtime:", torch.version.cuda); print("GPU accessible:", torch.cuda.is_available())'
"$LINGBOT_PY" -c 'import lingbot_map, core.model.net; print("Project imports OK")'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

The installed environment uses Python 3.10 and PyTorch **2.7.1+cu128**. This machine has four RTX PRO 6000 Blackwell GPUs. Use `nvidia-smi` to choose an available GPU; the optional examples below use physical GPU **2**.

Why these settings exist:

- `LINGBOT_PY` chooses the existing interpreter without needing `conda activate`.
- `PYTHONPATH` makes this checkout and the relocated `gates/` packages importable. Some historical code still imports names such as `gate8`.
- `CUDA_HOME`, the CUDA compiler on `PATH`, and the FlashInfer architecture setting match this machine's Blackwell setup. The Python environment's `bin/` also contains `ninja`.
- `HF_HUB_OFFLINE=1` uses downloaded weights. It does not download missing models.
- `MPLCONFIGDIR` gives plotting code a writable cache directory.

If `GPU accessible` is false, the CPU inspection commands still work. GPU inference needs a terminal/session with GPU access. Do not interpret a sandbox visibility problem as a broken checkpoint.

On another machine, these absolute paths must be replaced. [pyproject.toml](pyproject.toml) describes the base package and visualization extras; it does **not** provision the complete research environment, datasets, checkpoints or prediction caches. A fresh clone alone cannot reproduce the local experiments. Start with the existing environment when it is available.

## 2. Learn the five ideas behind the pipeline

Imagine watching a street through one camera. Reconstruction estimates the geometry you can see. Completion tries to infer occupancy where observations are incomplete.

```text
RGB images, ordered in time
       |
       +-- frozen LingBot-MAP --> depth and camera poses
       +-- frozen MoGe-2 -----> metric-scale reference
       +-- frozen Trident-H --> semantic evidence
       |
       v
Incremental mapper: occupied evidence, free-space evidence, unknown space
       |
       v
Export a voxel volume with 32 input channels
       |
       v
Original completion U-Net: predict an occupancy residual
       |
       v
Add residual where editing is allowed; apply the fixed threshold
       |
       v
Score against occupied/free labels on valid voxels only
```

**A voxel is a small 3D cell.** For the KITTI-360 source volumes, cells are 0.2 m wide. The array is 256×256×32 in XYZ order, with origin `(0, -25.6, -2)` metres in the anchor's Velodyne frame. It covers 51.2×51.2×6.4 m. Here x points forward, y left and z up.

**The mapper accumulates evidence.** Depth and estimated camera motion place observations into a shared coordinate frame. Rays provide free-space evidence before their observed surfaces. Missing evidence remains unknown. MoGe supplies the metric gauge: the audited pipeline fixes one scale from the first five frames and applies it consistently to depth and pose translation. The cached inference path does not generally run a second MoGe geometry-confirmation pass; `CachedFeed` defaults to `with_moge=False`.

**The completion network sees a map representation.** Its 32 channels include log-odds occupancy, free-space weight, observed/unobserved indicators, observation count, age, semantic weight and 25 semantic probabilities. Frozen semantic evidence remains part of the input even when the experiment scores binary occupancy only. A uniform unknown input has the unobserved channel equal to one; an all-zero tensor is a different input.

**The network predicts a correction.** The original model has **986,114 parameters**. Its editing rule is:

```python
final_logit = base_logodds + residual * (base_logodds.abs() < 2.0)
occupied = final_logit >= threshold
```

Confident map values stay locked. Weak values can change in either direction, so completion can remove true occupancy as well as add it. Original seed 0 uses threshold **−0.125**. Other original seeds have their own source-selected thresholds; never substitute a target-selected threshold.

**Supervision is different from input.** Cached samples store both the causal input and training targets. Intended Gate 8C-1 occupancy targets were built from raw LiDAR, including future sweeps: occupied endpoints, free ray interiors, ignored unobserved/conflicting regions. These labels are for training/scoring, not input channels. The rejected SSCBench `_1_1.npy` labels are not the intended targets. Current builders have changed since the historical caches were created; rebuilding is not a neutral setup step.

### Words you will encounter

| Term | Meaning in this project |
|---|---|
| **SC** | Scene completion: binary occupied versus free |
| **SSC** | Semantic scene completion: occupancy plus the occupied cell's class |
| **Anchor** | The frame whose coordinate system and labels define an evaluated scene |
| **Causal** | Uses observations available by the anchor, subject to documented initialization |
| **Checkpoint** | Saved network weights; identify the exact file by its SHA-256 hash |
| **Cache** | Saved intermediate output, so expensive frozen models need not run again |
| **Source domain** | KITTI-360, used for completion training and validation |
| **Target domain** | SemanticKITTI or Occ3D-nuScenes, held out from completion training/selection |
| **Gate** | A named research experiment with its own question and artifacts; not a software release |
| **Unknown / ignored** | Unknown describes missing input evidence; ignored describes labels excluded from scoring. They are not interchangeable. |

Dataset boundary: **KITTI-360 drives 0003/0007/0010 train; drive 0006 validates/selects.** SemanticKITTI and Occ3D results must not select thresholds, checkpoints or model variants. Frozen foundation-model training provenance is a separate question from completion-module training provenance.

## 3. Inspect one source sample — CPU, no inference

First check the local assets:

```bash
"$LINGBOT_PY" - <<'PY'
from pathlib import Path

paths = [
    Path("checkpoints/lingbot-map/204754b/lingbot-map.pt"),
    Path("artifacts/gate8c1/checkpoints/seed0_last.pt"),
    Path("artifacts/gate8c1/frozen_manifest.json"),
    Path("artifacts/gate8c1_original_audit/code_used/gate8/net.py"),
    Path("/media/SSD1/MINH_DATASETS/lingbot_gate8c1/samples/2013_05_28_drive_0006_sync"),
    Path("/media/SSD1/MINH_DATASETS/lingbot_gate8c1/targets/2013_05_28_drive_0006_sync"),
]
for path in paths:
    print("OK     " if path.exists() else "MISSING", path)
PY
```

Now open source sample `00004`, the first case used in the padding probe:

```bash
"$LINGBOT_PY" - <<'PY'
from pathlib import Path
import numpy as np

root = Path("/media/SSD1/MINH_DATASETS/lingbot_gate8c1")
drive = "2013_05_28_drive_0006_sync"
with np.load(root / "samples" / drive / "00004.npz", allow_pickle=False) as sample:
    dims = tuple(int(x) for x in sample["dims"])
    n = int(np.prod(dims))
    occupied = np.unpackbits(sample["gt_occ"], count=n).astype(bool)
    valid = np.unpackbits(sample["gt_valid"], count=n).astype(bool)
    print("Grid:", dims, "Stream index:", int(sample["t"]),
          "Native image frame:", int(sample["native_frame"]))
    print("Input frames:", sample["input_frames"].tolist())
    print("Future target frames:", sample["target_frames"].tolist())
    print("Valid:", int(valid.sum()), "Occupied:", int((occupied & valid).sum()),
          "Free:", int((~occupied & valid).sum()), "Ignored:", int((~valid).sum()))
    with np.load(root / "targets" / drive / "00004.npz", allow_pickle=False) as target:
        assert np.array_equal(sample["gt_occ"], target["occ_packed"])
        assert np.array_equal(sample["gt_valid"], target["valid_packed"])
        print("Embedded masks match the intended current raw-LiDAR target.")
PY
```

For the current file, expect **249,672 valid cells: 19,375 occupied and 230,297 free**. The stream index is 4; the native image frame is 137. Matching masks establish current consistency, not recovery of historical target bytes.

The `.npz` input is a sparse stored map plus labels. `unpack_sample` expands it into a dense 32-channel tensor. Read [the preserved unpacker](artifacts/gate8c1_original_audit/code_used/gate8/targets.py) after running the command; focus on `unpack_sample`, not the target-building functions.

## 4. Read the results before changing the model

Read in this order:

1. [Original Gate 8C-1 audit](reports/gate8c1_original_audit.md): what reproduced, scoring defects, causal inputs and OccAny compatibility.
2. [Padding probe](artifacts/gate8c1_padding_source_probe/report.md): a controlled change can suppress a ceiling and still hurt useful geometry.
3. [GroupNorm probe](artifacts/gate8c1_groupnorm_stat_probe/report.md): a further recorded source-only diagnostic.
4. [Source sanity report](artifacts/gate8c1_source_sanity/report.md): results on 590 validation samples and a recorded 16-clip fitting attempt.

These are saved experiments. Reading their reports does not require rerunning them. Some older prose, including parts of [REPRODUCE.md](REPRODUCE.md) and [core/README.md](core/README.md), makes stronger claims than the measurements establish. Use the original audit for historical comparison conditions; use the count files for numerical claims.

### How to read an occupancy score

Count only valid voxels. A true positive (TP) is correctly predicted occupied; a false positive (FP) is predicted occupied where the label is free; a false negative (FN) misses occupied ground truth. A true negative (TN) is correctly free.

```text
SC IoU    = TP / (TP + FP + FN)
Precision = TP / (TP + FP)       How much predicted occupancy is correct?
Recall    = TP / (TP + FN)       How much labelled occupancy was recovered?
FPR       = FP / (FP + TN)       How much labelled free space was filled incorrectly?
```

Sum confusion counts across samples **before** computing pooled IoU. An average of frame IoUs is a different statistic. A zero denominator is undefined in the padding probe; it is not silently converted to zero.

The original causal scores reproduce as **seed medians ± sample standard deviation**:

| Dataset | Original anchors | Median SC IoU % | Mean SC IoU % |
|---|---:|---:|---:|
| SemanticKITTI sequence 08 | 163 | 12.74 ± 1.03 | 13.11 |
| Occ3D-nuScenes validation | 1,182 | 30.89 ± 0.29 | 30.95 |

`past5` builds a fresh map from five observations ending at the anchor, while the cached LingBot geometry carries earlier stream context. It is not an exclusive five-image budget. `stream` accumulates an all-past map. `occany_fwd` uses images after the anchor; its 161/882-anchor results must be kept separate.

The **published** OccAny sequence references are 25.91% and 23.55%. They are forward-looking and post-processed, so placing them next to causal raw scores does not establish superiority. The audit also corrects an old claim: **OccAny was not trained on these two target datasets**; its reported training mixture is Waymo, DDAD, PandaSet, VKITTI2 and ONCE. See the [original audit's comparison](reports/gate8c1_original_audit.md) and [OccAny paper](https://arxiv.org/html/2603.23502v2).

Print the saved audit summaries without changing anything:

```bash
"$LINGBOT_PY" - <<'PY'
import json
from pathlib import Path

results = json.loads(Path("artifacts/gate8c1_original_audit/results.json").read_text())
for dataset in ("semantickitti", "occ3d"):
    scores = results[dataset]["past5"]["historical"]["seed_aggregation"]["iou"]
    print(f"{dataset}: median={100*scores['median']:.4f}%, "
          f"mean={100*scores['mean']:.4f}%, sample SD={100*scores['sample_sd']:.4f} pp")

source = json.loads(Path("artifacts/gate8c1_source_sanity/phase1_pooled.json").read_text())
for method in ("mapper", "mapper_dilate", "completion"):
    row = source[method]["all"]
    print(f"source {method}: IoU={100*row['sc_iou']:.2f}%, "
          f"precision={100*row['precision']:.2f}%, recall={100*row['recall']:.2f}%")
PY
```

### What is currently unresolved?

The recorded padding probe changed only ten convolutions from zero to replicated padding. Uniform boundary occupancy fell 50.29%, but real TP retention was only 55.56%; pooled source IoU fell **14.15% → 10.61%**. Suppressing a visible artifact is not sufficient evidence of a better completion model. The subsequent GroupNorm-statistics intervention also failed the useful-geometry screen.

The later source sanity results record **10.70% completion IoU on all 590 drive-0006 cases**, mapper precision of **11.66%**, and failure of a particular 16-clip fitting attempt to meet its criteria. These motivate checking geometry/supervision alignment and the training implementation. They do not prove a calibration bug, rule out architectural effects, or rule out domain shift. Comparing source and target IoUs alone cannot do that because their scored voxel distributions differ.

For learning the project, your next step is to understand sample `00004`, the residual rule, and these counts. The next research experiment needs its own frozen protocol and review; the README does not launch one.

### View the existing figure

Open [the padding z-profile](artifacts/gate8c1_padding_source_probe/z_profiles.png) directly in your editor. Solid lines show unmasked occupancy; dashed lines show occupancy among scored voxels. For browser viewing, run this in a separate terminal:

```bash
cd /home/minh/workspace/lingbot-map_fork
/home/minh/anaconda3/envs/cu128/bin/python -m http.server 8000 \
  --bind 127.0.0.1 --directory artifacts/gate8c1_padding_source_probe
```

Visit `http://127.0.0.1:8000/z_profiles.png`. Stop the server with Ctrl+C. If working remotely, forward port 8000 through your editor or SSH connection.

## 5. Optional: run one source completion example — GPU, no training

This executes **original zero-padding inference on one existing KITTI-360 input** and prints counts. It does not run the RGB foundation models, write predictions or modify historical artifacts. Use it to see where the saved input enters the network. It is not a new benchmark evaluation or a padding comparison.

The command exposes physical GPU 2 as logical `cuda:0`. Keep both settings together; choose another physical GPU only after checking availability.

```bash
CUDA_VISIBLE_DEVICES=2 "$LINGBOT_PY" - <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

root = Path.cwd()
reference = root / "artifacts/gate8c1_original_audit/code_used"
sys.path.insert(0, str(reference))
from gate8.net import load_checkpoint, apply_residual
from gate8.targets import unpack_sample

assert torch.cuda.is_available(), "Run in a session with GPU access."
torch.set_num_threads(4)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0")
manifest = json.loads((root / "artifacts/gate8c1/frozen_manifest.json").read_text())
seed = manifest["seeds"]["0"]
checkpoint = root / seed["checkpoint"]
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == seed["checkpoint_sha256"]
assert seed["occupancy_threshold"] == -0.125
model = load_checkpoint(str(checkpoint), device)
sample_path = Path("/media/SSD1/MINH_DATASETS/lingbot_gate8c1/samples/2013_05_28_drive_0006_sync/00004.npz")
with np.load(sample_path, allow_pickle=False) as packed:
    sample = unpack_sample(packed, device)

with torch.inference_mode():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        residual, _ = model.net(sample["input"][None])
    final = apply_residual(sample["base_logodds"], residual[0, 0].float())
    pred = final >= -0.125
    gt, valid = sample["gt_occ"].bool(), sample["gt_valid"].bool()
    tp = int((pred & gt & valid).sum())
    fp = int((pred & ~gt & valid).sum())
    fn = int((~pred & gt & valid).sum())
    print("Input shape:", tuple(sample["input"].shape), "Parameters:", model.net.n_params())
    print("TP:", tp, "FP:", fp, "FN:", fn)
    denominator = tp + fp + fn
    print("SC IoU:", f"{100*tp/denominator:.2f}%" if denominator else "undefined")
PY
```

The padding probe's A condition recorded **TP=10,037, FP=51,833, FN=9,338; IoU=14.10%** for this case. Small numerical differences can occur across runtime/kernel versions. The model call uses only `sample["input"]`; the base log-odds implement residual locking, and labels/masks are used afterwards. No z padding is introduced.

The snapshot takes import priority intentionally. The historical Git commit alone does not recover the original dirty working tree; replacing snapshot imports with live `core/` code changes the provenance of a reproduction.

## 6. Optional: see RGB reconstruction — a separate GPU demo

This shows what LingBot-MAP contributes before the completion network. It produces reconstructed geometry, **not SC benchmark scores**. Demo defaults also differ from the audited completion-cache protocol.

### Interactive viewer

```bash
CUDA_VISIBLE_DEVICES=2 "$LINGBOT_PY" demo.py \
  --model_path checkpoints/lingbot-map/204754b/lingbot-map.pt \
  --image_folder example/courthouse \
  --first_k 20 \
  --port 8080
```

Open `http://127.0.0.1:8080` after inference starts the viewer. Forward port 8080 if connecting remotely. Stop with Ctrl+C. The first invocation may compile CUDA kernels and take longer than subsequent runs. This example uses no sky filtering.

### Save reconstructed predictions without opening a viewer

```bash
mkdir -p output
export LINGBOT_DEMO_OUT
LINGBOT_DEMO_OUT=$(mktemp -d "$PWD/output/getting_started.XXXXXX")

CUDA_VISIBLE_DEVICES=2 "$LINGBOT_PY" demo_render/batch_demo.py \
  --input_folder example/courthouse \
  --output_folder "$LINGBOT_DEMO_OUT" \
  --model_path checkpoints/lingbot-map/204754b/lingbot-map.pt \
  --config demo_render/config/default.yaml \
  --first_k 20 \
  --save_predictions \
  --no_render

find "$LINGBOT_DEMO_OUT" -maxdepth 3 -type f
```

This creates a fresh output directory, saved NPZ predictions and `batch_results.json`. Read that JSON for per-scene success/failure; file creation alone does not prove successful inference. To also render video, omit `--no_render` in a separate invocation with a fresh output directory. Headless video rendering needs the installed renderer/CUDA extensions as well as the model environment.

All supported flags are discoverable without inference:

```bash
"$LINGBOT_PY" demo.py --help
"$LINGBOT_PY" demo_render/batch_demo.py --help
```

For long personal image sequences, inspect the windowed options (`--mode windowed --window_size 128`). Streaming has a finite frame-position table; do not assume unlimited sequence length. Windowed reconstruction is a different inference protocol from the original audited cache.

## 7. Navigate the code without getting lost

Read these files in sequence, following a sample rather than every gate:

| Order | File | Question it answers |
|---|---|---|
| 1 | [core/datasets/kitti360_splits.py](core/datasets/kitti360_splits.py) | Which drives may be used for training and validation? |
| 2 | [core/mapping/feed.py](core/mapping/feed.py) | How are cached per-frame estimates loaded? |
| 3 | [core/mapping/mapper.py](core/mapping/mapper.py) | How are observations accumulated and queried? |
| 4 | [core/supervision/targets.py](core/supervision/targets.py) | How does a stored sample become 32 input channels? |
| 5 | [core/model/net.py](core/model/net.py) | What does the completion network predict? |
| 6 | [core/run/eval_target.py](core/run/eval_target.py) | How do inference protocols and scoring connect? Read-only orientation here. |

`core/` is a reorganized implementation suitable for understanding the pipeline. `gates/` holds the relocated research packages. Historical tools live in `tools/<gate>/`; their names are not instructions to execute them. Some audit scripts deliberately import the preserved snapshot instead of either live tree.

| Location | Contents |
|---|---|
| `artifacts/gate8c1/frozen_manifest.json` | Original checkpoints, hashes, thresholds and recorded configuration |
| `artifacts/gate8c1_original_audit/` | Counts, frozen IDs, provenance, saved predictions and execution snapshot |
| `artifacts/gate8c1_padding_source_probe/` | Three-case A/B padding experiment, figure and invariants |
| `artifacts/gate8c1_groupnorm_stat_probe/` | Subsequent normalization-statistics diagnostic |
| `artifacts/gate8c1_source_sanity/` | Source validation and recorded small fitting experiment |
| `configs/gate8c1/` | Original source training configurations, not a command to retrain |
| `reports/`, `artifacts/*/report.md` | Reports; dates and explicit artifact identities matter |
| `tests/` | Checks spanning many historical tasks |
| `semantic/`, `semantic_sidecar/` | Additional semantic subsystems; not required for this binary-occupancy introduction |
| `benchmark/` | Base reconstruction model's benchmark tooling; see its [README](benchmark/README.md) |

Important bulk-data locations on this machine:

| Path | Purpose |
|---|---|
| `/media/SSD1/MINH_DATASETS/lingbot_gate8c1/samples/<drive>/` | Cached source map inputs with embedded labels |
| `/media/SSD1/MINH_DATASETS/lingbot_gate8c1/targets/<drive>/` | Intended raw-LiDAR source targets |
| `/media/SSD1/MINH_DATASETS/lingbot_gate7b/{stream,scale}/` | Streaming geometry and scale caches used by several datasets |
| `/media/SSD1/MINH_DATASETS/lingbot_gate8/` | Additional source-specific caches; resolve exact paths in `core/datasets/sources.py` |
| `/media/SSD1/MINH_DATASETS/lingbot_gate6/semantics/` | Frozen semantic evidence for the audited target paths |
| `data/` | Local dataset links; target data is not needed for steps 1–6 |

Treat data/checkpoint/history directories as read-only. Write new runs to a distinct output directory and record code, input and checkpoint hashes. Do not overwrite a frozen manifest to make a provenance check pass.

## 8. Checks and troubleshooting

Run four focused CPU checks for map keys, fixed scale and the residual rule:

```bash
"$LINGBOT_PY" -m pytest -q tests/gate8/test_gate8.py \
  -k 'key_pack_roundtrip or scale_fixed_from_the_first_five_frames_and_then_ignored or residual_never_changes_a_locked_voxel or completion_disabled_is_the_identity'
```

These check basic behaviour; they do not certify a benchmark score, historical provenance or a training procedure. Avoid treating an old full-suite failure count as an expected definition of correctness: the repository and environment have changed.

| Symptom | First thing to check |
|---|---|
| `python: command not found` | Use `"$LINGBOT_PY"` after step 1. |
| `No module named lingbot_map` or `gate8` | Run from the repository root with the step-1 `PYTHONPATH`. |
| CUDA unavailable | Check `nvidia-smi` and whether this session has GPU access. CPU artifact inspection is still available. |
| `ninja` missing / FlashInfer architecture error | Check the step-1 `PATH`, `CUDA_HOME`, and architecture setting before installing anything. |
| Hugging Face cache miss while offline | The needed weights are absent locally. Resolve the specific missing asset; the completion example needs only its local checkpoint and cached sample. |
| Missing source `.npz` | Check the mounted SSD and exact sample/target directories. Do not rebuild supervision as a substitute. |
| Matplotlib cache permissions | Use the writable `MPLCONFIGDIR` from step 1. |
| Padding probe says the experiment already started | This protects frozen results. Read its saved report/counts; do not delete its guard files to rerun. |
| Historical hash mismatch | Record the differing file and use the preserved reference sources. Relocation and later edits do not recreate the original bytes. |
| Renderer failed | Read `batch_results.json` and the traceback. Use `--no_render` if you only need reconstruction predictions. |

The onboarding commands were checked on **2026-09-08**: CPU imports, checkpoint loading/hash, sample inspection, saved-result readers, CLI help and the four focused tests. GPU inference examples are derived from the existing entry points and preserved inference path; they were not rerun as part of this documentation update.

The [original audit](reports/gate8c1_original_audit.md) is the starting point for scientific claims. [core/README.md](core/README.md) and [gates/README.md](gates/README.md) give more implementation/history detail. Retain the distinctions between reproducible numbers, compatible comparisons, sound training and established causes when deciding what to investigate next.

See [LICENSE.txt](LICENSE.txt) for the repository license; external datasets and model weights have their own terms.
