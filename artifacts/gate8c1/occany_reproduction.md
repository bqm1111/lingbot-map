# Gate 8C-1 — OccAny reproduction

> **Status update (2026-09-04).** Everything below the line marked *ORIGINAL GATE RECORD*
> was written while the released checkpoint was still unavailable on this machine. It has
> since been **downloaded and run here over all 4 819 official Occ3D-nuScenes validation
> samples**, and their published number reproduced to 0.01. The verified section comes
> first; the original record is kept unedited for provenance.

---

## VERIFIED REPRODUCTION (released checkpoint, run on this machine)

Run: `OccAny_5frames_nuscenes512_rot60_vpi10_fwd3_sTrans2`, EXP_ID 2 recipe, geometry-only,
`--recon_threshold 1.1`, all 4 819 samples their loader emits. Scored by **their own**
`occany/scripts/compute_metrics_from_saved_voxels.py`.

| | precision | recall | **SC IoU** |
|---|---|---|---|
| with majority pooling (`USE_MAJORITY_POOLING=1`, their default) | 36.11 | 40.39 | **23.56** |
| **published (paper / README)** | 36.09 | 40.39 | **23.55** |
| raw, no pooling | 42.67 | 28.61 | **20.67** |

**Two conclusions.** The reproduction is faithful (Δ = 0.01 IoU), and the published
23.55 % is definitively the **pooled** figure — OccAny's raw geometry number is **20.67 %**.

### Cross-checks

* Our own scoring of their saved voxels over all 4 819 returns **42.67 / 28.61 / 20.67** —
  identical to their script, so the two evaluators agree exactly.
* Their stored `voxel_label` is **byte-identical** to the target our evaluator builds, on
  every matched sample. Same GT, same mask, same grid.

### Head-to-head, 882 samples both methods evaluate

Both sides scored by OccAny's evaluator, both using **OccAny's own pool-then-mask order**.

| method | SC IoU | precision | recall | pred/GT |
|---|---|---|---|---|
| OccAny, raw | 20.64 | 42.55 | 28.61 | 0.67 |
| OccAny, pooled *(their published protocol)* | 23.48 | 35.96 | 40.36 | 1.12 |
| ours, **matched forward input**, raw | **27.56** | 63.83 | 32.66 | 0.51 |
| ours, **matched forward input**, pooled | **40.47** | 55.75 | 59.62 | 1.07 |
| ours, causal 5 past frames, raw | 30.43 | 65.16 | 36.34 | 0.56 |
| ours, causal 5 past frames, pooled | 41.13 | 55.48 | 61.39 | 1.11 |

OccAny scores 20.64 / 23.48 here against 20.67 / 23.56 on the full set, so this subset is
representative to 0.08 IoU.

Matched-input deltas (same five frames on both sides): **+6.92 raw**, **+16.99 pooled**.
Causal-input deltas (we additionally give up all future observation): **+9.79**, **+17.65**.

### A correction to our own earlier pooled number

`gate8c1_results.json` reports our pooled Occ3D result as **33.43 %**. That applied the
valid mask *before* pooling. OccAny pools the full prediction and lets the evaluator mask
afterwards (`eval_target.py:141` computes `comp = (s_fin >= tau) & valid` and pools after).
Measured on the same 250 samples:

| order | SC IoU |
|---|---|
| pool-then-mask (**OccAny's convention**) | 42.04 |
| mask-then-pool (what 33.43 used) | 33.26 |

So our previously published pooled figure was *stricter on us* than OccAny's own
convention. The head-to-head table uses their order for both sides. **Raw numbers are
unaffected** by this.

### Remaining asymmetries — all favour OccAny

1. They **trained on nuScenes**; our module never saw it.
2. They **tune the confidence threshold per dataset** (1.1 nuScenes vs 2.5 KITTI); ours was
   frozen on KITTI-360 before Occ3D was opened.
3. In the causal rows our input is **strictly harder** — no future observation at all.

Machine-readable record: `occany_headtohead_verified.json`.
Figures: `fig_occany_2d.png`, `fig_occany_3d.png`.

### What it took to run it here

Six missing dependencies (`torch_scatter` prebuilt wheel for torch-2.7+cu128, `roma`,
`h5py`, `xformers==0.0.31 --no-deps`), and three upstream source patches, all backed up
before editing:

| file | issue |
|---|---|
| `third_party/dust3r/.../batched_sampler.py:290` | `np.c_[a, *idxs]` — starred expression illegal in a subscript on this Python |
| `third_party/dust3r/.../easy_dataset.py:274,301` | same construct |
| `occany/model/model_sam2.py:521` | `Union[np.ndarray, Image]` — annotates the PIL *module* |

The 24 GB checkpoint set lives at `/media/SSD1/MINH_DATASETS/occany_checkpoints`
(`OccAny/checkpoints` is a symlink to it; `/` had only 9.8 GB free).

---

# ORIGINAL GATE RECORD (superseded above; kept unedited)

## Gate 8C-1 — OccAny reproduction: what was attempted and what is possible here

## Summary

**The released OccAny checkpoint could not be run on this machine within this gate.** Its
official *evaluator and post-processing* import and run correctly, and are used on our
predictions. The comparison in `report.md` is therefore:

| component | status |
|---|---|
| OccAny **published** five-frame SC IoU (SemanticKITTI 25.91 %, Occ3D 23.55 %) | quoted, with a full protocol audit |
| OccAny **official evaluator** (`occany/metrics/ssc.py::SSCMetrics.get_score_completion`) | **run on our predictions** |
| OccAny **official pooling** (`occany/utils/helpers.py::apply_majority_pooling`, separate mode) | **run on our predictions**, verified bit-for-bit in Gate 8B |
| OccAny **released model re-evaluated on our causal inputs** | **not run** — see below |

Consequently every OccAny number in the report is labelled `published`, and every number of
ours is labelled `ours (raw)` or `ours (+ OccAny pooling)`. A raw number of ours is never
placed against a post-processed number of theirs without both being shown.

## What blocks running the released model

The checkout at `/home/minh/workspace/OccAny` (commit `04d4616`) is complete, and
`/home/minh/workspace/third_party/occany_env` exists, but the model stack does not import:

```
OK    occany.metrics.ssc            <- the official evaluator, used
OK    occany.utils.helpers  (pooling only; its module-level import of depth_anything_3 fails)
OK    groundingdino
OK    sam2
FAIL  croco                 ModuleNotFoundError
FAIL  dust3r                ModuleNotFoundError
FAIL  mast3r                ModuleNotFoundError
FAIL  depth_anything_3      ModuleNotFoundError
FAIL  diffusers             ModuleNotFoundError
FAIL  torchsparse           ModuleNotFoundError
```

`occany/datasets/eval_helper.py` — the entry point that builds the official evaluation
sequences — fails at `import croco`, so even the dataset-construction half of the official
pipeline cannot be exercised.

Weights are published at `anhquancao/OccAny` and are reachable; the geometry-only 5-frame
setting (`EXP_ID 0`) needs

| file | size |
|---|---|
| `checkpoints/occany.pth` | 3.75 GB |
| `checkpoints/MUSt3R_512.pth` | 1.69 GB |
| `checkpoints/groundingdino_swinb_cogcoor.pth` | 0.94 GB |
| **total** | **6.38 GB** |

(the full checkpoint set is 24.24 GB).

Running it end to end would additionally require installing six source-only packages, two
of which (`torchsparse`, `croco`) build CUDA extensions, then extracting GroundingDINO
boxes for every evaluation clip (`--boxes_folder resized_1216_box5_text5_DINOB` is a
*required* argument of the published 5-frame recipe: 163 SemanticKITTI clips + 1 182 Occ3D
clips × 5 frames), and finally running a generative diffusion model
(`--gen --batch_gen_view 12`) over all of them. That is an infrastructure task of several
GPU-hours plus install risk, outside a confirmation gate whose instruction is to stop after
reporting.

**This was not attempted rather than attempted-and-failed**, and the report says so. It is
a real limitation of the comparison, not a claim about OccAny.

## What is used instead

1. **OccAny's official SC metric.** `SSCMetrics.get_score_completion` is imported from the
   checkout and run on our voxel predictions, so the geometry numbers are computed by
   their code, not by a reimplementation of it.
2. **OccAny's official post-processing.** `apply_majority_pooling` in `separate` mode.
   Gate 8B verified our reproduction against the official function bit-for-bit in the
   OccAny environment (dilation, vote and semantic modes all identical), and Gate 8C-1
   re-runs that check. Note the documented trap: in geometry-only mode the official default
   is `use_dilation=True`, which is a 3×3×3 **max-pool** (a one-voxel dilation), not a
   majority vote. Both are reported as separate columns.
3. **A protocol audit** enumerating every point at which our causal protocol differs from
   OccAny's published one, so the reader can see exactly what the published number is a
   number *of*.

## Protocol difference that matters most

OccAny's published five-frame setting is **target-first and forward-looking**:
`occany/datasets/eval_helper.py` builds a 10-frame video at `frame_interval=5` and takes
`recon_view_idx = [0, 2, 4, 6, 8]` for SemanticKITTI — the target frame is view 0 and the
other four are **after** it. For nuScenes it takes `video_length=5, frame_interval=2` from
the target sample forward. Our Protocol A is the mirror image: five frames **ending** at
the target, no future observation at any point. Protocol B reproduces OccAny's forward
sampling for comparison only and is labelled non-causal throughout.

---

## Comparability verification (added after the report, from the OccAny checkout)

Asked directly: *is our Occ3D number faithfully comparable to OccAny's published 23.55 %?*
Machine-readable record: `occany_comparability.json`.

### The measurement is faithful

OccAny's own `SSCMetrics.get_score_completion` was run on **193 of our real Occ3D
predictions** and compared against our counting: **byte-identical TP/FP/FN on every clip**
(IoU 34.23 % from both, on that 25-scene subset). Everything the metric depends on matches:

| item | OccAny | ours | same |
|---|---|---|---|
| grid | 200×200×16 @ 0.4 m, origin (−40,−40,−1), ego of target | identical | yes |
| camera mask | `voxel_label[mask_camera==0] = 255` | identical | yes |
| lidar mask | not applied | not applied | yes |
| single-camera cut | `voxel_label[:100,:,:] = 255` (rear half) | identical | yes |
| occupied / free | class 17 free, 255 ignored from TP/FP/FN | identical | yes |
| split / camera | Occ3D-nuScenes official val, 150 scenes, CAM_FRONT | identical | yes |
| aggregation | pooled TP/FP/FN over all samples | identical | yes |

**Sample subset.** We score 1 182 anchors (8 per scene, a uniform 1-in-5 stride); OccAny
scores every frame that has eight more ahead of it (≈ 4 800 of the 6 019 val frames that
carry a GT file). Checked rather than assumed: pooled occupancy prevalence under the
identical mask is **0.22953 on our 1 182** against **0.22700 on all 6 019** — a 1.1 %
relative difference. The subsample is representative, so the pooled IoU is comparable in
expectation.

### Three asymmetries, all of which favour OccAny

1. **The published 23.55 % is post-processed; our 30.89 % is raw.**
   `sh/compute_metric.sh` sets `USE_MAJORITY_POOLING=1` **by default**, and
   `sh/exp_lists/metric_occany.sh` EXP_ID 2 (nuScenes 5-frame geometry) passes
   `--geometry_only $MAJORITY_POOLING_ARG`. In geometry-only mode
   `apply_majority_pooling` defaults to `use_dilation=True` — a 3×3×3 **max-pool**, i.e. a
   one-voxel dilation. Our number with the same operation applied is **33.43 %**.
2. **Their five frames look forward from the target; ours look backward into it.** OccAny's
   target is the *first* frame and the other four follow it at interval 2 — roughly four
   seconds of *future* observation of the very region being completed. Our primary protocol
   has none. Mirroring their sampling gives us **27.56 %** raw, **30.51 %** pooled.
3. **They trained on nuScenes and tuned the confidence threshold per dataset**
   (`--recon_threshold 1.1` for nuScenes against `2.5` for KITTI). Our module never saw
   nuScenes, and its threshold was frozen on KITTI-360 before Occ3D was opened.

### How to state the result

The strictest like-for-like — matched temporal sampling, matched post-processing, their
evaluator, identical grid, masks and split — is **30.51 % (ours) vs 23.55 % (OccAny)**, and
even that still hands them the target-domain training and the tuned threshold. The number
in `report.md` (30.89 raw vs 23.55 pooled) is *conservative* for us on the post-processing
axis and *favourable* on the temporal axis; the matched pair above is the one to quote.

None of this changes SemanticKITTI: 12.74 % raw / 12.77 % matched-and-pooled against
25.91 % — still about half, still the finding that matters.
