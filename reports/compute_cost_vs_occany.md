# Computational cost: ours vs OccAny

Written 2026-10-20. Nothing was re-run: every number is read from logs of the runs that produced the paper numbers,
or counted from checkpoint files on the CPU. Same machine and GPU model for both (RTX PRO 6000 Blackwell Max-Q).

## Parameters at inference

| | ours | OccAny (Must3R + SAM2, the compared variant) |
|---|---:|---:|
| reconstruction | LingBot-Map 1,157.9 M | OccAny model 651.1 M (Must3R enc/dec + raymap enc + gen decoder; from its own load log) |
| metric scale | MoGe-2 ViT-L 326.2 M | — |
| semantics | Trident-H image side 1,359.0 M (CLIP ViT-H-14 visual 632.1 + SAM ViT-H 641.1 + DINO ViT-B/16 85.8) | SAM2.1-L 224.4 M + GroundingDINO-B 234.9 M |
| completion / association | CompletionUNet 0.97 M + B3 head 0.03 M | (inside the 651 M) |
| **total** | **≈ 2.84 B** | **≈ 1.11 B** |
| trained by the authors | **0.99 M** | 651 M |

The CLIP text tower (353 M) runs once offline to embed the class names and is excluded from both inference totals.

## Time per evaluated sample, SemanticKITTI 08 (5-frame window, semantic setting)

GPU-seconds per sample = wall-clock time / samples per GPU. Ours ran one worker per GPU. OccAny ran two concurrent
workers on one GPU, so its GPU-seconds are half its per-worker wall time.

| stage | ours | source |
|---|---:|---|
| LingBot-Map 5-frame clip | 0.19 | `artifacts/occany_comparison/logs/clips_semantickitti.log` (1,606 clips / 5.1 min) |
| MoGe-2 (≈1 new frame per sample, cached) | 0.06 | `artifacts/gate7b/logs/moge_semantickitti.log` (795 frames / 0.8 min) |
| Trident-H tokens (≈1 new frame per sample) | 1.69 | `artifacts/final_zero_shot/logs/tokens_sk_s*.log` |
| map + UNet + lift + B0 + B3 ×3 seeds + write | 2.49 | `artifacts/final_zero_shot/logs/predict_sk_s*.log` |
| **total, as evaluated** | **≈ 4.4** | |
| one B3 seed (deployable) | ≈ 3.4 | single-sample profile `artifacts/final_zero_shot/logs/efficiency.json` |
| geometry only (clip + MoGe + map + UNet) | ≈ 0.35 | |

| OccAny run | per-worker wall | GPU-s / sample | source (`/media/SSD1/occany_causal/logs/`) |
|---|---:|---:|---|
| KITTI 5-frame semantic (published 7.28 setting) | 24.7 s | 12.4 | `eval_e4_w{0,1}.log` (805 samples, 2 h 46 m) |
| + GroundingDINO boxes, precomputed | 1.73 s/frame (4 workers) | 0.43 | `gdino_w*.log` |
| **total** | | **≈ 12.8** | |
| KITTI causal 5-frame semantic | 24.4 s | 12.2 | `eval_causalA_e4_w*.log` |
| KITTI 1-frame semantic | 23.1 s | 11.6 | `k1frame_sem_w*.log` |
| KITTI 1-frame geometry | 3.4 s | 1.7 | `k1frame_w*.log` |

**Ratios (SemanticKITTI):** full semantic system **≈ 2.9× cheaper** than OccAny as evaluated (≈ 3.8× with one B3
seed). Geometry: ours from 5 frames ≈ 0.35 s vs OccAny from 1 frame 1.7 s, **≈ 5× cheaper**.

**Occ3D-nuScenes, for reference only (not like-for-like inputs):** ours single camera ≈ 3.7 GPU-s
(clip 0.29 + MoGe 0.07 + Trident 2.09 + predict 1.23); OccAny surround (6 cameras, 1 timestep) 16.96 s/sample with one
worker plus ≈ 2.5 s of GroundingDINO for 6 images, ≈ 19.5 GPU-s.

## Why the costs differ

- **OccAny's cost is dominated by semantics on rendered views.** Its 1-frame geometry run takes 3.4 s and its
  1-frame semantic run takes 23.1 s, so semantics (SAM2 distillation and video propagation over the generated novel views)
  are about 85 % of the total. Every sample is also recomputed from scratch, with nothing reused between overlapping windows.
- **Ours amortises per frame.** The large models (LingBot, MoGe, Trident) run once per incoming frame. Per sample, only
  the hard map, the 1 M-parameter UNet and the association head run. Trident-H is our largest single cost (≈ 40 %).
  `efficiency.json` notes that its logged time includes SAM refinement, which the token feature does not need. That has
  not been verified here.
- **We use 2.6× more parameters but ~3× less GPU time.** Parameter count is a poor proxy here: OccAny runs its
  651 M model many times per sample over ~50 generated views, while ours runs each frozen model once per frame.

## Caveats (state these if the numbers go in the paper)

1. **Batch-job wall-clock, not a controlled benchmark.** The GPUs are shared with other users' jobs whose load at
   the time is unknown, for both methods. These are throughputs, not single-sample latency. OccAny's single-worker
   latency lies somewhere between 12.4 and 24.7 s.
2. **OccAny ran with the slow PyTorch RoPE2D fallback** ("cannot find cuda-compiled version of RoPE2D" in every log).
   This inflates its time by an unmeasured amount.
3. Peak GPU memory is known only for our per-sample stage (5.5 GB, `efficiency.json`). It was not measured for OccAny.
4. FLOPs were not measured.

A controlled benchmark would fix 1–4: N samples, serial, `torch.cuda.synchronize`, an idle GPU, and OccAny with curope
compiled. It needs a GPU that no other job is using.
