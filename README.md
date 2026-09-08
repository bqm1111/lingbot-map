# LingBot-MAP + scene completion

This repository is **two projects stacked on top of each other**. Knowing which one
you are looking at is most of the battle.

**1. LingBot-MAP** — a feed-forward 3D foundation model for streaming reconstruction
from image sequences or video. Images in, camera poses and dense depth out, one frame
at a time. This is upstream code plus a fork; it is *frozen* for everything below.

**2. A scene-completion research track** — built on top of that model, asking one
question:

> Can we predict dense 3D occupancy for a driving scene from monocular images alone,
> causally, and have it transfer to datasets it was never trained on?

The answer so far is *partly*, and the interesting part is why not. Section 6.

If you are returning to this repo cold, read this file top to bottom once. Every step
is runnable and takes under a minute unless marked otherwise.

---

## 1. The 30-second orientation

```
lingbot_map/     the reconstruction model      demo.py, demo_render/, semantic/
     │                (frozen)
     ▼
core/            the completion pipeline       core/README.md
gates/           the research history          gates/README.md
     │
     ▼
artifacts/       results          REPRODUCE.md ← the headline numbers and how to rerun
```

Three doors, depending on what you want:

| I want to… | go to |
|---|---|
| see 3D reconstruction working, with pictures | § 3 below |
| understand the completion pipeline | [`core/README.md`](core/README.md) |
| see the results and rerun them | [`REPRODUCE.md`](REPRODUCE.md) |

---

## 2. Environment

This machine has a specific, non-obvious setup. Use it exactly.

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a"
export HF_HUB_OFFLINE=1

PY=/home/minh/anaconda3/envs/cu128/bin/python     # bare `python` is NOT on PATH
```

conda env `cu128`, torch 2.7.1+cu128, 4× RTX PRO 6000 Blackwell (96 GB each, sm_120).

**Why each variable matters:**

- `/usr/local/cuda` points at CUDA 12.6, which cannot compile sm_120. You must name 12.8.
- FlashInfer's arch auto-detection refuses SM 12.x unless CUDA ≥ 12.9, then fails with a
  **misleading** `"FlashInfer requires GPUs with sm75 or higher"`. The explicit `12.0a`
  suffix bypasses that check and works with nvcc 12.8.
- `HF_HUB_OFFLINE=1` stops the frozen models from phoning home mid-run.

### Verify in three commands

```bash
$PY -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
$PY -c "import lingbot_map; print('model package OK')"
$PY -m pytest tests/ -q                       # ~3.5 min
```

Expected test result: **861 passed, 6 failed**. The six failures are known and
documented in [`REPRODUCE.md`](REPRODUCE.md) § 5 — one manifest-hash test and five
`semantic_sidecar` tests that need a `ninja` binary that isn't installed. If you see
those six and nothing else, your environment is correct.

---

## 3. Step one: watch the reconstruction model run

Start here even if you only care about the research track — it makes the rest concrete.

```bash
$PY demo.py --model_path checkpoints/lingbot-map/204754b/lingbot-map.pt \
    --image_folder example/courthouse --mask_sky
```

Opens an interactive [viser](https://github.com/nerfstudio-project/viser) viewer at
`http://localhost:8080`. Other bundled scenes: `example/university`, `example/loop`.

Headless render to MP4 instead:

```bash
PYTHONPATH=$PWD $PY demo_render/batch_demo.py \
  --input_folder example/courthouse --output_folder output/courthouse_render \
  --model_path checkpoints/lingbot-map/204754b/lingbot-map.pt \
  --config demo_render/config/default.yaml \
  --mask_sky --sky_mask_dir example/courthouse_sky_masks --save_predictions
```

> **`PYTHONPATH=$PWD` is required here and not optional.** The editable install points
> at a path that no longer exists, so `demo_render/batch_demo.py` fails with
> `ModuleNotFoundError: No module named 'lingbot_map'` without it. `demo.py` works
> without it only because the current directory happens to be on `sys.path`.
> Permanent fix: `pip install -e . --no-deps` from the repo root.

**Long sequences (>~1000 frames) must use windowed mode.** The 3D RoPE table covers
`max_frame_num` (default 1024) global frame indices. Past that the table slice silently
returns the wrong last dimension and inference crashes with
`The size of tensor a (32) must match the size of tensor b (22)`. Use
`--mode windowed --window_size 128`.

The full upstream documentation — every flag, camera-path YAML, worked examples — is
610 lines that were deleted when the research track began. Recover it any time:

```bash
git show 36441e2^:README.md | less
```

---

## 4. Step two: the idea behind the completion pipeline

Now the research half. Four ideas; everything else is plumbing.

**(a) Three frozen models supply per-frame evidence.** They are never trained here, and
their outputs are cached to disk as `.npz` once:

| model | supplies |
|---|---|
| LingBot-Map | depth + camera pose per frame |
| MoGe-2 | a second depth opinion, used to confirm geometry |
| Trident-H | open-vocabulary semantic probabilities per pixel |

**(b) An incremental voxel map accumulates them.** One frame in, a persistent map out.
Log-odds occupancy, free space, semantic evidence, on a 256×256×32 grid at 0.2 m. The
metric scale is fixed from the first five frames and **never re-gauged**. Every frame is
integrated exactly once; nothing is replayed.

**(c) The map is full of holes, so a small network completes it.** A 986,114-parameter
dense 3D U-Net reads 32 channels and predicts a residual.

**(d) The trick: privileged supervision.** At *training* time the target is built from
**future** LiDAR sweeps the network never sees. At *inference* time it gets only past
frames and the current one. That asymmetry is the whole design.

One safety rule runs through it — the network may not overwrite confident map voxels:

```python
apply_residual(base, res) = base + res * (base.abs() < 2.0)
```

Decision threshold τ = −0.125, chosen on the source domain and then frozen.

Read [`core/README.md`](core/README.md) next. It has the full data-flow diagram, the
32-channel table, and one line on what each of the 57 live files does.

---

## 5. Step three: the results

```bash
less artifacts/gate8c1/report.md          # the full gate report
less REPRODUCE.md                         # headline numbers + how to rerun
```

Trained on KITTI-360 only (drives 0003/0007/0010), evaluated on datasets it never saw.
Causal `past5` protocol, median of 3 seeds:

| target | ours SC IoU % | OccAny (published) | notes |
|---|---|---|---|
| SemanticKITTI 08 | **12.74 ± 1.03** | 25.91 | 49 % of OccAny |
| Occ3D-nuScenes val | **30.89 ± 0.29** | 23.55 | 131 % of OccAny |

OccAny trains *on both targets*, runs non-causally, and has ≈1.5 B parameters against
0.99 M here — so it is a deliberately unforgiving reference, not a peer.

**The recorded verdict is FAIL** against the pre-registered bar. The reason worth
knowing: on SemanticKITTI, taking the raw incremental map and applying a single 0.4 m
dilation scores **15.41**, beating the trained network's 12.74. On that target the
learned residual is not yet doing anything a morphological operation doesn't do better.
On Occ3D it beats every non-learned baseline by ~8 points.

To rerun: § 2A of `REPRODUCE.md` rebuilds the report in seconds; the full target
evaluation is 67 GPU-min; a complete rebuild from KITTI-360 is 2.2 GPU-h.

---

## 6. Step four: where the project actually stands

The open question is *not* cross-dataset transfer. SemanticKITTI (12.74 %) actually
scores **higher** than the source domain KITTI-360 does (10.70 %). The problem is
upstream of the network:

- A fresh completion head given 2,000 steps on **16 fixed clips** cannot memorise them.
  Loss fell 10 % (the pre-registered bar was 90 %); training SC IoU reached 12.25 %
  (bar: 80 %). Both criteria failed.
- The incremental mapper — built from the *same* poses, depths and scale that feed the
  network's 32 input channels — has **11.66 % precision** against the raw-LiDAR target.
  Nearly nine of every ten voxels it calls occupied are not occupied in the supervision.

So input geometry and supervision do not agree, and no amount of architecture work fixes
that. Full write-up: [`artifacts/gate8c1_source_sanity/report.md`](artifacts/gate8c1_source_sanity/report.md).

**The recommended next experiment** (stated, not started): a source-only geometric
alignment audit — nearest-neighbour distance from each mapper-occupied voxel to the
nearest target-occupied voxel, plus SC IoU under a small grid of rigid translations and
scale factors applied to the target — to decide whether the cached map and the raw-LiDAR
supervision even share a metric frame.

---

## 7. Map of the repository

603 Python files in the research tree, of which **57 are load-bearing**. `core/`
holds exactly those 57, extracted and reorganised by role — it is a copy, so the
originals stay where the audit records expect them. The rest is history and one-off
diagnostics.

| path | what it is |
|---|---|
| `core/` | the live pipeline, reorganised by role — **start here** |
| `gates/` | 14 frozen research gates, one per question asked |
| `tools/<gate>/` | each gate's scripts and shell runners |
| `configs/<gate>/` | frozen configuration |
| `artifacts/<gate>/` | results (bulk `.npz`/`.pt` are gitignored) |
| `reports/<gate>/` | written reports |
| `tests/<gate>/` | tests |
| `lingbot_map/` | the reconstruction model — **not used by the completion pipeline**, which runs entirely off cached outputs |
| `demo.py`, `demo_render/` | interactive and offline rendering |
| `semantic/` | open-vocabulary extension (see `semantic/README.md`) |
| `benchmark/` | evaluation harness for the base model |

Every gate keys by name across five directories: `gates/gate8c1/`, `tools/gate8c1/`,
`configs/gate8c1/`, `artifacts/gate8c1/`, `tests/gate8c1/`.

### Reading order for a newcomer

1. this file
2. [`REPRODUCE.md`](REPRODUCE.md) — the claim, the numbers, how to rerun
3. [`core/README.md`](core/README.md) — the pipeline in detail
4. [`gates/README.md`](gates/README.md) — why each earlier gate existed and what it found
5. [`artifacts/gate8c1/report.md`](artifacts/gate8c1/report.md) — the full gate report
6. [`artifacts/gate8c1_source_sanity/report.md`](artifacts/gate8c1_source_sanity/report.md) — the open question

---

## 8. Traps

Things that will cost you an afternoon if you don't know them.

**Dataset directories are read-only.** Write to `output/` or `artifacts/`.

**Bare `python` is not on PATH.** Always the full interpreter path.

**Rebuilding the KITTI-360 targets is not a no-op.** `gates/gate8c1/rawtarget.py`
gained an open-sky rule after the cached targets were written. Rebuilding drops the
*source-domain* SC IoU from 9.29 % to 4.67 % — the rule working as designed, not a
regression. Target-domain results are unaffected. Read `REPRODUCE.md` § 5 first.

**Eval JSONs are not bit-reproducible; the numbers are.** GPU float non-associativity
gives ~1e-5 relative wobble. Headline SC IoU reproduces to 5 decimals.

**The 3D figures cannot be regenerated.** `figure_3d_compare.py` and
`figure_3d_gallery.py` are non-deterministic, and the stored versions were made with
`--n 8` by code that has since changed. They are gitignored, so they exist only on
disk — **do not delete them**.

**`eval_target.py` writes four npz families** (`scores_*`, `bincounts_*`, `counts_*`,
`newly_*`) alongside each JSON, and `aggregate.py` recomputes from them. Back them up
together or the aggregate will mix runs.

**Two files still differ from the frozen manifest** and their originals are
unrecoverable (the tree was untracked before commit `f7b7552`). `REPRODUCE.md` § 5.

---

## 9. What this is not

The headline claim is **target-domain-free transfer of the completion module** — the
completion network never sees SemanticKITTI or Occ3D. It is *not* whole-system
zero-shot: the frozen models beneath it have not had their training provenance audited,
and LingBot-Map's published mixture includes KITTI-360. Keep that distinction in any
writeup; it is the difference between an honest claim and an overclaim.
