# GroupNorm-statistics counterfactual (source-only, drive 0006)

**1. Hypothesis.** GroupNorm-statistics propagation explains why replicated padding suppresses the artificial boundary ceiling but destroys useful occupancy.

**2. Controlled change.** Seed-0 checkpoint, τ = −0.125, frozen cases **00004 / 00889 / 01771**, same inputs, masks, 256×256×32 grid, channels, semantic evidence, residual lock, `pad_z=0` and precision. GroupNorm keeps no running statistics, so **C** is an explicit inference-only counterfactual: replicated padding on the same ten Conv3d layers, each GroupNorm normalised with the per-sample, per-group mean and variance captured from **A**, then A's own `num_groups`, `eps`, weight and bias. No weight changed; parameter hashes match across A, B, C; masks enter only at counting.

**3. Results.** A reproduces the previous probe **bit-exactly** (0 disagreeing voxels, ΔIoU 0.000 pp). Frozen statistics fed A's *own* values reproduce A to ≤0.031 logit and ≥99.997 % of decisions, validating the implementation.

| Pooled | TP | FP | FN | IoU % | Bnd FPR % | Int FPR % | TP ret % |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 23,477 | 94,146 | 48,277 | 14.15 | 55.58 | 1.93 | 100.00 |
| B | 13,043 | 51,131 | 58,711 | 10.61 | 37.41 | 2.83 | 55.56 |
| C | 7,655 | 27,989 | 64,099 | 7.67 | 6.48 | 1.91 | 32.61 |

Uniform-unknown boundary occupancy: 82.92 → 41.22 → **0.00 %**; interior 0 % throughout, unscored. A→B statistic shifts stay ≤0.05 σ in the encoder, peaking at `enc3.4` (median 0.18 σ) and `dec2` (0.15 σ); std ratios 0.68–1.25.

**4. Criteria.** Boundary-FPR reduction ≥25 %: **passed** (88.3 %). TP retention ≥95 %: **failed** (32.61 %). IoU non-decrease: **failed**. Not all satisfied: mechanism unsupported.

**5. Demonstrated / uncertain.** C loses TP faster than B, so GroupNorm-statistics propagation is **rejected as the primary explanation** of the lost occupancy. It does explain B's interior regression: freezing A's statistics returns interior FPR to 1.91 % against A's 1.93 %. A's statistics act as near-global suppression, not boundary-specific repair, so the boundary response is produced locally by zero padding rather than carried by normalisation. Other boundary mechanisms remain open; target-domain behaviour was untested.

**6. Next experiment.** One source-only probe, same three cases: keep zero padding, but pad the input volume with the correct unobserved encoding (`Completer.PAD_UNOBSERVED_CHANNEL`) and crop back, testing whether the boundary ceiling falls while TP retention stays ≥95 %.

[config](config.json), [results](results.json), [invariants](invariants.json), [counts](counts.csv), [statistics](groupnorm_statistics.csv), [figure](comparison.png). Stopped for review.
