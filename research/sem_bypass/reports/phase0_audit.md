# Phase 0 — audit of the existing semantic-sidecar implementation

Every number below was read from the committed artifacts on this machine, not from the
prose of the previous report. Where the two disagree, the artifact wins and the
discrepancy is flagged.

**Audit outcome: not `BLOCKED`.** All required code, checkpoints and data for Gates 1
and 2 are present. Gate 3 has a **data blocker** documented in §14 — it is recorded now
so it is not discovered late, but per the protocol the gates still run in order.

## 1. Environment and state

| | |
|---|---|
| repo | `/home/minh/workspace/lingbot-map_fork`, branch `dev`, HEAD `4071d1e` |
| working tree | clean except the untracked `research/` directory from the previous study |
| python / torch | 3.10.19, 2.7.1+cu128, CUDA 12.8 |
| GPUs | 4 × RTX PRO 6000 Blackwell Max-Q, 97 887 MiB each |
| `open_clip` | **3.3.0, importable** |
| CLIP weights | `~/.cache/clip/ViT-B-16.pt` present (offline load works) |
| LingBot checkpoint | `checkpoints/lingbot-map/204754b/lingbot-map.pt`, SHA-256 `ee665103348e07e6b826d529b8e61de8f413d5432a4f2e84970d6c8fd2e1cd72` |
| LingBot parameters | **1 157 943 540**, 0 trainable when frozen (confirmed in the previous study's tests) |

## 2. Which representation produced 10.04 — **the single most important finding**

**`matched_miou = 10.043686628341675`** comes from
`output/semantic_sidecar/indomain/maps/mlp_full_d025/eval.json`, produced by
`configs/semantic_sidecar/indomain_diagnostic.yaml`, whose LingBot block reads:

```yaml
lingbot:
  layers: [11, 23]        # <-- mid-GCT block 11 AND final-GCT block 23
model:
  fuse: concat
```

The cache manifest confirms it: `output/semantic_sidecar/cache/kitti08_c000/manifest.json`
has `"layers": [11, 23]`, `"layer_slots": [1, 3]`, `"token_channels": 2048`, and the
cached tensor is `tokens [32, 2, 407, 2048]`.

**So the 10.04 result was produced from mid-GCT + final-GCT tokens, not from the pre-GCT
encoder.** Three consequences for this project:

1. It directly contradicts this project's own scope rule *"Never use final-GCT tokens as
   the primary semantic representation"* — the headline prior result **is** a final-GCT
   result.
2. 10.04 is therefore **not a baseline SemBypass can be expected to match**, because
   SemBypass is mandated to use pre-GCT `x_norm_patchtokens`. This is a genuinely
   different experiment, not a reproduction.
3. The previous phase-1 study found the pre-GCT encoder retains semantics *best*
   (fidelity ratio 1.000 by construction) and final-GCT worst (0.842). Whether that
   ordering survives a MaskCLIP teacher and a mIoU metric is exactly what Gate 1 tests
   — and it is genuinely uncertain, in both directions.

## 3. What `mlp_full` actually is — **second major finding**

From `tools/run_feasibility_experiment.py:61-62`:

```python
("mlp_full", "mlp",
 {"lambda_pixel": 0.0, "lambda_consensus": 1.0, "lambda_mv": 0.25, "lambda_rel": 0.1}, []),
```

`mlp_full` has **`lambda_pixel = 0.0`**. Its supervision is *entirely* 3D consensus,
plus a multi-view term and a relational term.

This project's Gate 1 requires the primary run to use *"feature-distillation losses
already used by the successful experiment"* **and** *"no 3D consensus in the primary
configuration"*. Those two requirements are jointly unsatisfiable for `mlp_full`:
removing consensus from it removes 100 % of its supervision.

**Resolution adopted (documented, not silent):** the primary `sem_bypass` configuration
uses `lambda_pixel = 1.0` with all other loss terms at zero — i.e. the `mlp_pixel`
recipe, which *is* a feature-distillation loss used by the same study and is the only
consensus-free recipe it contains. The correct consensus-free reproduction target is
therefore **`mlp_pixel_d025` = 9.0239**, not 10.04.

Relevant in-domain numbers, all from `output/semantic_sidecar/indomain/maps/*/eval.json`:

| run | loss | matched mIoU | coverage |
|---|---|---:|---:|
| `mlp_full_d025` | consensus + mv + rel | **10.0437** | 87.0046 |
| `mlp_pixel_d025` | pixel only | **9.0239** | 87.0046 |
| `mlp_pixel_d100` | pixel only | 8.9829 | 87.0046 |
| `mlp_full_d100` | consensus + mv + rel | 8.8252 | 87.0046 |
| `mlp_consensus_d100` | consensus only | 7.0722 | 87.0046 |

The report's quoted "8.98 near-domain" is `mlp_pixel_d100` — confirmed exactly.

## 4. Sidecar architecture and exact parameter count

`semantic_sidecar/models/semantic_sidecar.py::SemanticSidecar` — per-layer
`LayerNorm → Linear(in_ch, 768)`, concat fusion, `Linear(1536, 768)`, 2 pre-norm
residual MLP blocks (`mlp_ratio 2.0`), `LayerNorm`, `Linear(768, 64)`, L2-normalise.

Recorded in `inference_summary.parameter_report` of the in-domain eval:
`{"sidecar_total": 9112896, "sidecar_trainable": 9112896}` — **9 112 896**, matching the
reported ~9.1 M, for `num_layers=2, in_channels=2048`.

**With the mandated pre-GCT encoder input (`num_layers=1, in_channels=1024`) the same
unmodified architecture has 6 156 864 parameters** (verified by construction in
`research/sem_bypass/tests/`). That is **0.532 %** of LingBot's 1 157 943 540 — still
comfortably inside the paper claim's "sub-1 %-parameter" wording, and *smaller* than the
prior sidecar. No architectural redesign is involved; only the input width changes.

## 5. MaskCLIP teacher

| | |
|---|---|
| implementation | `semantic/dense_clip.py::DenseCLIP`, `variant="maskclip"` (final block replaced by the value path) |
| model / weights | `open_clip` `ViT-B-16-quickgelu`, pretrained `openai` (local `~/.cache/clip/ViT-B-16.pt`) |
| patch size | **16** (LingBot uses 14) |
| output dim | **512** |
| normalization | `F.normalize(..., dim=-1)` inside `encode_dense`; text likewise |
| input scale | `input_scale: 2.0` — the teacher runs at 2× the LingBot input resolution |
| cached resolution | resampled to LingBot's patch grid **[11, 37] = 407 tokens**; cached tensor `features [64, 407, 512]` |
| cache manifest | `"teacher": "maskclip:ViT-B-16-quickgelu/openai/maskclip"`, `"embed_dim": "512"` |

**The sidecar does not predict 512-d.** It predicts a **64-d PCA subspace**
(`output/semantic_sidecar/cache/pca.safetensors`: `components [64, 512]`, `mean [512]`).
The basis was verified **orthonormal** here (`B Bᵀ = I` to 1e-4), and the pipeline uses
it *uncentered*, so cosine similarity is exactly preserved within the retained subspace
and text prompts are projected through the same basis. This is a faithful description of
"MaskCLIP space" only with that qualifier attached; the final report must state it.

## 6. Text encoder and prompts

`DenseCLIP.encode_text` — same `ViT-B-16-quickgelu/openai` text tower, averaged over a
**9-template ensemble** (`dense_clip.DEFAULT_TEMPLATES`: `"a photo of a {}."`,
`"a cropped photo of a {}."`, …), each L2-normalised, mean-pooled, re-normalised.

Class prompts are `semantic/eval_semantickitti.py::CLASS_PROMPTS`, 19 entries
(`"a car"`, `"a bicycle"`, … `"a traffic sign"`), recorded in every `eval.json` as
`prompt_source`. Softmax temperature `0.05`. They were **not** tuned against results.

## 7. The 25 % teacher-frame selection

`semantic_sidecar/teacher_features.py::sample_teacher_frames` — **frame-level uniform
random sampling without replacement**, `np.random.default_rng(seed)`,
`keep = max(1, round(num_frames * density))`, seeded by `train.teacher_density_seed`
(default **1234**). Applied in `tools/build_semantic_tracks.py:117`, so the density is
baked into the tracks directory (`tracks/density025/`) and recorded per scene as
`teacher_frames`. It is *not* a uniform stride.

## 8. Seeds — 10.04 is a single-seed result

`output/semantic_sidecar/runs/` contains **20 run directories, none seed-suffixed**;
`output/semantic_sidecar/indomain/runs/` contains 5. Every config carries `seed: 0`,
`train.seed: 0`, `teacher_density_seed: 1234`. There is no second seed anywhere for the
in-domain diagnostic.

**Confirmed: 10.04 is one seed, one run.** The previous report says as much
(§6 "Single seed"), and this project is right to demand three.

## 9. Semantic labels never entered a loss

`semantic_sidecar/losses.py` contains `cosine_loss`, `centered_cosine_loss`,
`multiview_consistency_loss`, `relational_loss` — a grep for `label`, `LEARNING_MAP` or
`semantic_label` across `losses.py` and `tools/train_semantic_sidecar.py` returns
**nothing**. Labels appear only in `semantic/eval_semantickitti.py` and the evaluator.
Training reads a frozen token cache and never loads LingBot, so no gradient can reach
geometry either (`tools/train_semantic_sidecar.py` docstring, verified).

## 10. Matched mIoU, coverage, and support identity

`tools/evaluate_semantic_reconstruction.py`:

* GT LiDAR points are projected into their own camera with the dataset's `P2`/`Tr`
  (`alignment: "calibrated-projection (per-frame P2/Tr); no Sim(3) fit is used for
  semantics"`), never using semantics in the correspondence.
* `end_to_end_miou` counts GT points landing in **no** populated voxel as wrong.
* `matched_miou` (`conf_matched`) is restricted to **covered** points only, isolating
  semantic quality from geometric coverage.
* `coverage = 100 * covered / total` = **87.0046 %**, and it is *identical to 10 decimal
  places* for every model in both studies, including `teacher_direct`.

## 11. Does the 10.50 teacher use the same support?

`output/semantic_sidecar/maps/teacher_direct/eval.json`: `matched_miou = 10.5027`,
`coverage = 87.00460420824152`, `frames_scored = 200` — identical coverage and frame
count to every student.

**One caveat the previous report overstates.** It claims a shared
`geometry_fingerprint` proves identical geometry. The fingerprints actually **differ**
between the primary study (`956dcd6494…`) and the in-domain run that produced 10.04
(`69685d4cc0…`). Inspecting `inference_summary` shows why: the fingerprint hashes the
*whole scene list of the map build*, and the two builds included different qualitative /
train scenes (`example_loop` vs `kitti08` chunks). It is therefore **not** a valid
cross-study identity proof. What *does* hold is that coverage is bit-identical
(87.00460420824152) and the frozen geometry is deterministic, so the 10.04-vs-10.50
comparison is on the same eval support. The conclusion survives; the stated evidence for
it was wrong.

## 12. Can the sidecar run without MaskCLIP?

Yes. Every sidecar `eval.json` records `needs_teacher_at_inference: false`;
`teacher_direct` records `true`. Training and inference read the frozen token cache and
the 64-d PCA basis; the MaskCLIP **image** tower is never needed after caching.

Text queries need the CLIP **text** tower, but only once: 19 prompts → `[19, 512]` →
projected through the stored PCA basis → `[19, 64]`, which can be precomputed and
shipped. So "no MaskCLIP at inference" is accurate for the image path and requires the
qualifier that text embeddings are precomputed.

## 13. Data available for Gates 1–2

| asset | status |
|---|---|
| SemanticKITTI seq 08 (eval, labelled) | present, 4 071 frames; 1 000 cached as `kitti08_c000..c003` |
| KITTI odometry 13/14/15/16 (train, **no labels exist**) | present; **2 982 frames** cached across 14 chunks |
| cached MaskCLIP teacher features | present for all the above, `[F, 407, 512]` |
| PCA basis | `output/semantic_sidecar/cache/pca.safetensors`, orthonormal, 64/512 |
| existing tracks | `indomain/tracks/density025` and `density100` (seed 1234 only) |
| existing LingBot token cache | **layers [11, 23] only — pre-GCT tokens are NOT cached** |

**The one thing that must be recomputed** is a LingBot feature cache holding the pre-GCT
`aggregator.patch_embed` → `x_norm_patchtokens` output (`[F, 1, 407, 1024]`). Geometry,
teacher features and the PCA basis are all reusable unchanged. At the previously
measured 44.1 ms/frame this is ≈ 3 minutes of inference for all 3 982 frames.

## 14. Gate 3 data blocker (recorded early, not yet triggered)

nuScenes exists at `/media/welf/MINH/datasets/nuscenes`:

* `v1.0-mini/` extracted (samples, sweeps, maps, 10 scenes of metadata);
* `v1.0-trainval01_blobs/` extracted (1 of 10 image blob archives);
* `v1.0-trainval_meta/`, `v1.0-test_meta/` extracted.

**Missing: every source of nuScenes semantic ground truth.**

* No `lidarseg/` directory and no `lidarseg.json` in any metadata folder.
* No Occ3D-nuScenes `gts/<scene>/<sample_token>/labels.npz` anywhere under `/media/welf`.
* `outputs/prompted_lingbot/cache_occ3d/` holds LingBot *predictions* for 40-frame
  scenes with `"has_gt_depth": false` — geometry only, no semantics.

Gate 3's required metrics (matched visible-surface mIoU, per-class IoU, rare-class
performance) all need semantic GT. Without it the gate cannot be scored, only
qualitatively demonstrated. **Exactly what is needed** — either one suffices:

1. **nuScenes-lidarseg** (~2 GB): `nuscenes/lidarseg/v1.0-trainval/*_lidarseg.bin`,
   plus `v1.0-trainval/lidarseg.json` and the updated `category.json` carrying the
   32-class index, laid out beside the existing `v1.0-trainval_meta/v1.0-trainval/`.
2. **Occ3D-nuScenes** (~30 GB): `gts/<scene_name>/<sample_token>/labels.npz`
   (`semantics`, `mask_lidar`, `mask_camera`) plus `annotations.json`, in an
   `Occ3D-nuScenes/` root.

Additionally, only `trainval01_blobs` is extracted, so camera coverage is partial;
`v1.0-mini` (10 scenes, all sensors) is complete and is the natural 300–500-frame subset
if labels arrive.

## 15. Discrepancies between reported numbers and the implementation

| claim | verdict | evidence |
|---|---|---|
| "10.04 with 25 % MaskCLIP supervision and 3D consensus" | **accurate**, and it is a *final-GCT* result | `layers: [11, 23]` |
| "8.98 in a near-domain configuration" | **accurate** — `mlp_pixel_d100` = 8.9829 | in-domain `eval.json` |
| "10.50 for the direct MaskCLIP teacher" | **accurate**, same coverage | `teacher_direct/eval.json` |
| "~9.1 M sidecar" | **accurate** for `[11,23]`; becomes 6 156 864 for pre-GCT | `parameter_report` |
| "single-seed" | **confirmed** | 20 runs, no seed variants |
| "geometry identical, proven by fingerprint" | **overstated** — fingerprints differ across studies; identity rests on coverage instead | §11 |
| "3D consensus improvement unstable" | **consistent with artifacts** — consensus wins at d025 (+1.02) and loses at d100 (−0.15), one seed each | §3 table |
| LingBot 44.1 ms/frame, probe 0.52 ms/frame | from the *previous* study, DINO teacher, 407-token grid | to be re-measured in Gate 2 with MaskCLIP |

## 16. Reproducing this audit

```bash
cd /home/minh/workspace/lingbot-map_fork
export PYTHONPATH=$PWD
PY=/home/minh/anaconda3/envs/cu128/bin/python

$PY -c "import json;print(json.load(open('output/semantic_sidecar/indomain/maps/mlp_full_d025/eval.json'))['matched_miou'])"
$PY -c "import json;print(json.load(open('output/semantic_sidecar/maps/teacher_direct/eval.json'))['matched_miou'])"
sed -n '148,151p' configs/semantic_sidecar/indomain_diagnostic.yaml     # layers: [11, 23]
sed -n '61,62p'   tools/run_feasibility_experiment.py                   # mlp_full lambda_pixel=0.0
$PY -c "import json;print(json.load(open('output/semantic_sidecar/cache/kitti08_c000/manifest.json'))['layers'])"
grep -rn "label" semantic_sidecar/losses.py                             # (no output)
```

## 17. Decisions carried into Phase 1

1. Primary `sem_bypass` = **pre-GCT `x_norm_patchtokens`**, `lambda_pixel = 1.0`, no
   consensus. Reproduction target **9.02** (`mlp_pixel_d025`), not 10.04.
2. Architecture unchanged; input width 1024 → **6 156 864** trainable parameters.
3. Teacher, PCA basis, prompts, evaluation protocol, class mapping and coverage
   definition are reused **byte-for-byte** from the existing implementation.
4. Three seeds vary `seed`, `train.seed` **and** `teacher_density_seed` together, so the
   teacher-frame draw is genuinely independent per seed.
5. Gate 3 is expected to return `BLOCKED` on data unless semantic GT is supplied (§14).
