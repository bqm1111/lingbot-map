We tested whether replacing zero convolution padding with replicated padding suppresses the original Gate 8C-1 boundary response while preserving useful source-domain occupancy.

Seed-0 weights matched the original manifest; threshold remained −0.125. We froze drive-0006 cases **00004, 00889, 01771** (first/middle/last of 590 eligible IDs) before inference. Their full 256×256×32, 0.2 m volumes retained all 32 channels, including frozen semantic evidence. Only ten padded Conv3d layers changed from A=zeros to B=replicate. Parameter tensors, strides, normalization, evaluation mode, bfloat16 autocast, residual locking, `pad_z=0` and `final >= −0.125` remained fixed. Exactly eight forward passes were executed: A/B for three real cases and one correctly encoded uniform-unknown control.

Existing raw-LiDAR targets matched embedded sample masks. Their current hashes are recorded; historical target bytes remain unverified. The three cases contain 877,810 valid voxels: 71,754 occupied and 806,056 free. Masks enter counting only; there is no post-processing.

Each table entry shows **A → B**. Boundary means centre-to-face distance <0.4 m; interior means ≥2 m from every face. IoU is pooled from confusion counts.

| Case | TP | FP | FN | IoU % | Boundary FPR % | Interior FPR % | TP retention % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00004 | 10,037→7,142 | 51,833→24,839 | 9,338→12,233 | 14.10→16.15 | 71.74→53.71 | 3.62→4.30 | 71.16 |
| 00889 | 8,363→4,876 | 21,409→17,965 | 27,597→31,084 | 14.58→9.04 | 33.69→24.71 | 1.90→2.85 | 58.30 |
| 01771 | 5,077→1,025 | 20,904→8,327 | 11,342→15,394 | 13.60→4.14 | 39.99→10.56 | 0.96→1.88 | 20.19 |
| Pooled | 23,477→13,043 | 94,146→51,131 | 48,277→58,711 | 14.15→10.61 | 55.58→37.41 | 1.93→2.83 | 55.56 |

Uniform boundary occupancy fell **82.92%→41.22%**, a **50.29% relative reduction: pass**, narrowly. Uniform interior occupancy remained **0%→0%**; no GT accuracy is assigned. [Layer fractions](z_profiles.csv) and the [z-profile figure](z_profiles.png) distinguish unmasked predictions from predictions among valid voxels.

Practical screening **failed**: boundary FPR reduction **32.68% passes**; TP retention **55.56% fails ≥95%**; pooled IoU **fails non-decrease**. Boundary support is sufficient under the recorded pre-inference rule: 33,520 scored free voxels and 18,630 baseline FP. No requested real rate is undefined; undefined per-layer scored fractions are explicitly null/blank.

This supports boundary sensitivity, but replicated padding sacrifices useful geometry and changes interior behaviour. It does not establish padding as the cause of every target-domain ceiling error or demonstrate target-domain benefit.

Recommend **one source-only normalization-control probe on these same cases**, holding A's GroupNorm statistics fixed during B, to test whether normalization contributes to TP loss and increased interior errors. This recommendation has **not** been executed.

[Configuration and commands](config.json), [selection and hashes](selection.json), [counts](counts.csv), [results](results.json), [invariants](invariants.json). Stopped for review.
