# SemBypass — final decision

## Verdict: `FAIL_REPRODUCIBILITY`

Gate 1 failed on three of four criteria. Gates 2 and 3 were not run, and Phase 4 was not
built.

---

## 1. Hypothesis

> LingBot-Map's pre-GCT DINO encoder features contain useful semantic information. A
> small sidecar can decode those already-computed features into MaskCLIP's
> language-aligned dense feature space, producing open-vocabulary semantic point maps
> without running MaskCLIP during inference, fine-tuning LingBot-Map, degrading LingBot
> geometry, or using semantic annotations during training.

The claimed contribution was *"preserve semantics through a lightweight bypass around
geometric processing, instead of altering a large pretrained geometry model."*

## 2. Existing evidence audited

Full audit in [`phase0_audit.md`](phase0_audit.md). The three findings that changed the
experiment:

1. **The 10.04 headline used mid-GCT + final-GCT tokens `[11, 23]`**, verified in
   `configs/semantic_sidecar/indomain_diagnostic.yaml` and the cache manifest. It is a
   *final-GCT* result, which this project's own scope rule forbids as a primary
   representation. It is therefore not a target SemBypass could inherit.
2. **`mlp_full` has `lambda_pixel = 0.0`** (`tools/run_feasibility_experiment.py:61`) —
   its supervision is *entirely* 3D consensus plus multiview and relational terms. The
   brief's requirement to use "losses already used by the successful experiment" while
   running "no 3D consensus in the primary configuration" is jointly unsatisfiable for
   it. The only consensus-free recipe in that study is `mlp_pixel`, so the correct
   reproduction target is **9.0239** (`mlp_pixel_d025`), not 10.04.
3. **10.04 is a single seed** (20 run directories, none seed-suffixed).

Also confirmed: no semantic label enters any loss; `teacher_direct` (10.5027) shares the
students' support exactly; the sidecar runs with `needs_teacher_at_inference: false`. One
claim in the prior report was **overstated** — the shared `geometry_fingerprint` does not
prove cross-study geometry identity (the fingerprints differ; identity rests on coverage
matching to 10 decimals instead).

## 3. Dataset splits

| | |
|---|---|
| training | KITTI odometry **13, 14, 15, 16**, stride 2, **2 982 frames**. These sequences have **no released semantic labels at all** — leakage is impossible by construction. |
| evaluation | SemanticKITTI **seq 08**, frames 0–999 in 4 chunks of 250, **200 scored** at `frame_stride 5`, 19 classes. Labels used **only** in the evaluator. |
| disjointness | asserted in `tests/test_sem_bypass.py::test_gate1_train_and_eval_scenes_are_disjoint_and_unlabelled_for_training` |

## 4. Checkpoints and hashes

| | |
|---|---|
| LingBot-Map | `checkpoints/lingbot-map/204754b/lingbot-map.pt`, SHA-256 `ee665103348e07e6b826d529b8e61de8f413d5432a4f2e84970d6c8fd2e1cd72`, 1 157 943 540 params |
| MaskCLIP | `open_clip` `ViT-B-16-quickgelu` / `openai`, local `~/.cache/clip/ViT-B-16.pt`, `variant="maskclip"`, 512-d, `input_scale 2.0` |
| PCA basis | `output/semantic_sidecar/cache/pca.safetensors`, `components [64, 512]`, verified orthonormal (`B Bᵀ = I`, atol 1e-4), applied uncentered |
| prompts | `semantic/eval_semantickitti.py::CLASS_PROMPTS` (19), 9-template ensemble, temperature 0.05, untuned |
| sidecar checkpoints | `research/sem_bypass/outputs/runs/{sem_bypass,sem_bypass_consensus}_seed{0,1,2}/best.pt` |

## 5. Trainable parameter count

**6 156 864** — the unmodified `SemanticSidecar` with `num_layers=1, in_channels=1024`
(the prior 9 112 896 came from `num_layers=2, in_channels=2048`). No architectural
redesign; only the input width changed.

**0.5289 %** of LingBot, printed by the trainer and asserted in the tests.

## 6. Freezing confirmation

* LingBot: `lingbot_total 1 157 943 540`, `lingbot_trainable 0`, logged every run.
* MaskCLIP: never loaded during training or sidecar inference; used only to build the
  cached teacher features, then frozen and unused.
* Only sidecar parameters receive gradients (`test_only_sidecar_parameters_receive_gradients`).
* Training reads a frozen token cache and never constructs LingBot, so no gradient path
  to geometry exists.

## 7. Three-seed SemanticKITTI results

Coverage identical at **87.004604 %** for every row.

| model | teacher at inference | seed 0 | seed 1 | seed 2 | mean | σ | retention |
|---|---|---:|---:|---:|---:|---:|---:|
| `maskclip_direct` (ceiling) | yes | — | — | — | **10.5027** | — | 1.000 |
| **`sem_bypass`** | no | 8.7735 | 7.5195 | 7.7050 | **7.9994** | 0.553 | **0.762** |
| `sem_bypass_consensus` | no | 9.9883 | 8.3084 | 8.6702 | 8.9890 | 0.722 | 0.856 |
| control `[11, 23]` (diagnostic) | no | 8.7636 | 6.8439 | 7.7464 | 7.7847 | 0.784 | 0.741 |

| gate criterion | required | observed | |
|---|---|---:|:--:|
| retention | ≥ 0.90 | 0.762 | ✗ |
| absolute gap | ≤ 1.5 | 2.503 | ✗ |
| seed σ | ≤ 0.5 | 0.553 | ✗ |
| no coverage advantage | identical | identical | ✓ |

**Two diagnostics explain the failure:**

* **The representation is not the cause.** With matched seeds, reverting to the audited
  `[11, 23]` tokens gives 7.78 — *worse* than pre-GCT's 8.00. Both fail.
* **The prior 9.02 was a favourable teacher draw.** An exact replay at
  `teacher_density_seed = 1234` returns **9.0683** vs the published **9.0239**
  (delta +0.044) — the prior work reproduces at its own seed. But three independent draws
  of the same 25 % budget give 6.84 / 7.75 / 8.76, and 9.07 lies *above the maximum* of
  that range. The published number is real but under-sampled; the 25 % teacher draw
  contributes ~±1 mIoU, larger than any architectural effect in the previous study.

## 8. Consensus stability

Positive on **all three seeds** with low variance: +1.215 / +0.789 / +0.965,
mean **+0.990**, σ 0.175. This *contradicts* the prior report's "unstable" characterisation
and would meet the adoption rule — but it is **not adopted**, because the primary gate
failed and the protocol forbids promoting a secondary diagnostic to rescue it. Note it
raises dominant-class mIoU (24.2 → 28.0) while *lowering* rare-class mIoU (0.49 → 0.27).

## 9. MaskCLIP-versus-sidecar latency

**Not measured.** Gate 2 runs only if Gate 1 passes. The harness is implemented and
smoke-ready at `research/sem_bypass/benchmark.py` (5 pipelines, batch 1, ≥ 50 warm-up,
≥ 300 measured frames, `torch.cuda.synchronize()` around every timed region, all frames
preloaded to GPU so no I/O is inside the timed section). No latency number is claimed.

The only measured costs here are **not** inference latency and must not be quoted as such:
pre-GCT caching ran at 49.5 ms/frame including disk writes, and sidecar training took
105–191 s per 20 000-step run at 0.26 GB peak VRAM.

## 10. Zero-shot nuScenes/Occ3D results

**Not run** (Gate 3 requires Gates 1 and 2 to pass). Independently, it would have
returned `BLOCKED`: nuScenes exists at `/media/welf/MINH/datasets/nuscenes` with
`v1.0-mini`, `v1.0-trainval01_blobs` and metadata extracted, but **no semantic ground
truth of any kind** — no `lidarseg/`, no `lidarseg.json`, no Occ3D `gts/*/labels.npz`.
Exact requirement in [`phase0_audit.md`](phase0_audit.md) §14.

## 11. Per-class and rare-class analysis

`sem_bypass` **exceeds** the teacher on the two dominant classes (car 61.3 vs 54.7,
road 44.1 vs 37.6) and scores **exactly 0.00** on parking, trunk, pole and other-ground.
Dominant-class mean IoU 24.2 vs the teacher's 30.6; rare-class mean **0.49** vs 1.07,
with only **3.3 of 14** rare classes above 0.1 IoU. The student's score is carried almost
entirely by car / road / building / vegetation — the failure mode the gate's
"not only dominant classes" rule exists to catch.

## 12. Geometry invariance test

The SemBypass cache came from an **independent inference run** with a different forward
hook. Against the legacy `[11, 23]` cache, `depth`, `depth_conf`, `extrinsic`,
`intrinsic`, `pose_enc`, `frame_index` and `image_patch_rgb` are **bit-identical**
(`torch.equal`), asserted in the test suite. `teacher_direct` re-evaluated here returns
**10.5027** at coverage **87.004604** — byte-matching the previous study's stored value,
confirming identical evaluation support.

**The geometry-preservation part of the hypothesis is fully supported.** It is the only
part that is.

## 13. Failures and anomalies

1. **Gate 1 failed on 3 of 4 criteria** — the study's result.
2. **Seed variance exceeds the gate's own tolerance** (σ 0.553 > 0.5) even for the
   primary, and 0.784 for the control. The 25 % teacher draw dominates.
3. **The student beats its teacher on the two easiest classes** while collapsing to zero
   on four rare ones — regression to dominant modes, not superior semantics.
4. **Consensus reverses sign** versus the previous study's conclusion.
5. **Seed 0 is the best seed for every configuration**, which with n = 3 cannot be
   separated from chance but should stop anyone reporting seed 0 alone.
6. **Absolute mIoU is low for every model including the teacher** (10.50); ±0.5 mIoU
   differences between near-floor models carry little signal.
7. **The prior report's fingerprint-based geometry-identity claim is unsound** (audit §11).

## 14. Exact commands executed

See [`gate1_reproducibility.md`](gate1_reproducibility.md) §8. Summary:

```bash
$PY -m pytest research/sem_bypass/tests -q
$PY -m research.sem_bypass.cache_encoder --config research/sem_bypass/configs/gate1.yaml --roles train eval --device cuda:0
$PY -m research.sem_bypass.run_gate1 --seeds 0 1 2 --variants sem_bypass sem_bypass_consensus --stages tracks train teacher infer eval --device cuda:0 --overwrite
$PY -m research.sem_bypass.run_gate1 --config research/sem_bypass/configs/control_gct.yaml --seeds 0 1 2 --variants sem_bypass --stages tracks train teacher infer eval --device cuda:0 --overwrite
$PY -m research.sem_bypass.report_gate1
```

## 15. Final verdict

# `FAIL_REPRODUCIBILITY`

## 16. Is the paper claim supported?

> *"A sub-1 %-parameter semantic bypass recovers open-vocabulary MaskCLIP features from a
> frozen streaming geometry foundation model, preserves its reconstruction exactly, and
> transfers to a real unseen dataset without semantic annotations or a separate semantic
> encoder at inference."*

**No. The claim is not supported.** Clause by clause:

| clause | status |
|---|---|
| "sub-1 %-parameter" | **supported** — 6 156 864 params, 0.529 % of LingBot |
| "preserves its reconstruction exactly" | **supported** — bit-identical geometry, verified tensor-by-tensor |
| "without semantic annotations" | **supported** — no label enters any loss; training sequences have no labels at all |
| "no separate semantic encoder at inference" | **supported** — `needs_teacher_at_inference: false`; text embeddings precompute to `[19, 64]` |
| **"recovers open-vocabulary MaskCLIP features"** | **not supported** — 76.2 % retention, 2.50 mIoU below the teacher, rare classes at 0.49 mean IoU |
| **"transfers to a real unseen dataset"** | **not tested** — Gate 3 never ran, and would be blocked for lack of nuScenes semantic GT |

Four of six clauses hold. The two that carry the scientific claim do not. Writing the
sentence as stated would be unsupported by this evidence.

## 17. Single next action justified by the evidence

**Fix the sampling, not the architecture.** The largest measured effect in this study is
the **25 % teacher-frame draw (~±1 mIoU across seeds)**, which is bigger than the
pre-GCT-versus-GCT representation difference (0.215) and comparable to the consensus
effect (0.99). Everything the previous study concluded from single-seed contrasts of
1–2 mIoU sits inside that noise band.

So: **re-run the previous study's key contrasts at 5+ seeds before drawing any
architectural conclusion from them.** That is cheap — 105–191 s per run on cached
features — and it determines whether *any* of the sidecar findings survive. Only if a
configuration clears 90 % retention with σ well under 0.5 does the efficiency question
(Gate 2) become worth measuring.

Do **not** pursue: a larger sidecar, new losses, semantic keyframe selection, or Gate 3
data acquisition. None is justified by a primary method sitting at 76 % retention with
seed noise above the gate tolerance.
