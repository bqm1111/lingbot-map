# Gate 3.1 — clean validation of bounded local occupancy correction and infill

## 1. Validity diagnosis: `CLEAN_LOCAL_INFILL_PASSES`
## 2. Mechanism diagnosis: `FULL_FEATURES_CONTRIBUTE`
## 3. Scale-contribution diagnosis: `CLIP_SCALE_REMAINS_NECESSARY`

Supporting sub-diagnoses:

```
old-region provenance   OLD_GATE3_MASK_LEAK_CONFIRMED
visible/unobserved      LOCAL_OCCUPANCY_INFILL
```

## 4. Plain-language conclusion

Gate 3 leaked. The correction region was `dilate(C3, 3) AND valid`, where `valid` is the
SemanticKITTI per-sample `.invalid` mask — label-side information — and that region was
fed to the network as input channel 5 *and* used as the output gate. It removed about
75 700 voxels per clip, a quarter of the band. Gate 3's **0.2903 is non-authoritative and
is withdrawn.** Rebuilt on a region that is a pure radius-3 dilation of C3 occupancy, the
same architecture with the same frozen hyperparameters reaches **0.2154 ± 0.0008** over
three seeds. The leak was worth **+0.075 IoU, about 35 % of the reported effect.**

What survives is still a real result, and it survives cleanly. All three seeds improve on
C3 (0.0778) by ≈ +0.138 with every one of the 163 clips improving; the strongest
deterministic control is beaten by **+0.0395 [+0.0350, +0.0441]**; and the gain is not
volume inflation — the learned model uses **53 % fewer** occupied voxels than that control
and beats a volume-matched control by +0.0626. Two ablations sharpen the picture. Feeding
the network *only* occupancy and the region — zeroing point count, frame count, confidence
and depth — still reaches 0.2052, i.e. **92.6 % of the gain**, so the module is
predominantly a **learned anisotropic 3D occupancy prior**; the evidence channels add a
small but significant and reproducible +0.0102. And a corrector trained on constant-scale
C0 geometry reaches only 0.1819, so the clip-scale component is **more** valuable after
voxel correction than before it (a 0.0335 gap, up from 0.0205 uncorrected). Finally, 89.8 %
of what the model adds lies outside the five-frame visible set: this is **bounded local
occupancy correction and infill**, and the infill half is local completion inside 0.6 m,
not visible-voxel correction.

> **All results are in-domain SemanticKITTI and cannot establish generalization.** One
> dataset, one sensor rig, one city, fixed camera height and intrinsics; sequence 08 is
> held out temporally, not environmentally. The 163 clips are not independent environments.

Preserved untouched: every Gate 0–3 report, checkpoint, cache and artifact, plus the two
pre-existing modified files in the working tree. Gate 3's code and outputs are left exactly
as they were, as the record of the defect. The original report was **not** silently patched.

---

## 5. Old-region provenance audit (Part B)

### Finding: `OLD_GATE3_MASK_LEAK_CONFIRMED` — a real mask leak, not merely wording

Traced from raw cache construction to inference:

| stage | file:line | verdict |
|---|---|---|
| `c3_flat/count/n_frames/sum_conf/sum_depth` built from `c3_points` → `sparse_voxel_features` | `tools/voxel_gate/cache_c3.py` | **clean** — depends on LingBot depth/confidence/intrinsics/pose, the frozen head's clip scalar, and `calib.txt`. No LiDAR, no target, no oracle |
| clip scalar `a_clip` | `voxel_gate/c3.py:clip_scale` | **clean** — median of the frozen head's residual over the LingBot-only C0 support |
| region construction | **`voxel_gate/data.py:81`** — `region = correction_region(occupied, radius) & keep` | **LEAKED** — `keep = unpack(d["valid_bits"])`, i.e. `(target != 255) & (invalid == 0)` |
| region → input channel 5 | `voxel_gate/data.py:82` → `build_input(feat5, region, norm)` | **LEAKED into the input tensor** |
| region → output gate | `tools/voxel_gate/eval_voxel.py` → `apply_region(p >= tau, occ, R)` | **LEAKED into inference gating** |
| region → deterministic controls | `control(occ, R, **kw)` | **LEAKED into the controls too** |
| region → radius selection | `tools/voxel_gate/select_region.py:60` | **LEAKED into a hyperparameter choice** |

Checked and **not** implicated: binary LiDAR occupancy (`gt_flat`), visible-reconstruction
targets (`vc_flat`), semantic voxel labels (never loaded anywhere in Gates 0–3.1), oracle
scale (`scale_targets_*.csv` is never opened by any Gate-3 tool), oracle depth or pose.
The `c3_*` keys were audited by construction, not assumed clean because of their prefix.

Magnitude, measured on 40 sequence-08 clips before any rebuild: the intersection retained
**75.0 %** of the band and deleted **75 672 voxels per clip** from the inference region.

The Gate-3 test suite did not catch this because its leakage test mutated the targets and
then re-derived the region — but the region was *already* a function of the target, so
both the reference and the mutated version changed consistently, and `torch.equal` still
passed on the input built from the mutated file. Gate 3.1 tests provenance from raw
construction instead (§11, K5).

### Structural fix

`voxel_gate_validation/data.py:input_view` copies out **only** the five `c3_*` evidence
keys, so the inference path is handed a dict that physically does not contain
`valid_bits`, `gt_flat` or `vc_flat`. A later edit cannot reintroduce the dependency
without raising `KeyError`. `voxel_gate_validation/geometry.py:geometry_evidence` — the raw
builder — has a signature that admits no target-side argument at all, and a test asserts
both the signature and that its body contains no target token.

---

## 6. Clean protocol (Part C, D)

```
R_infer = binary_dilation(C3_occupied, radius = 3 voxels = 0.6 m)     inference region + output gate
S_train = R_infer AND valid                                           loss restriction only
S_eval  = valid                                                       metric restriction only
```

`R_infer` is **not** intersected with the valid mask. It legitimately contains
**62 906 evaluation-invalid voxels per clip** (24.8 % of the band); a test asserts this is
non-zero, because a zero would mean the leak had returned. Dilation is extensive, so
`R_infer` contains every C3-occupied voxel and "preserve C3 outside `R_infer`" and "force
empty outside `R_infer`" coincide. Measured violations outside the region: **0** for every
learned configuration and every deterministic control.

Everything else is frozen from Gate 3 and was **not** retuned after sequence 08 was rerun:
sequences 00–07 / 09–10 / 08; 256×256×32 at 0.2 m; radius 3; three 3×3×3 convolutions at
width 16 with GroupNorm(4) and GELU; zero-initialised 1×1×1 residual head; ±4 C3 prior
logits; weighted BCE + 0.5 soft Dice; AdamW lr 1e-3, weight decay 1e-4; batch 1; bf16;
≤20 epochs; patience 4; **τ = 0.45**. The full grid fits in 1.24 GiB, so no tiling,
cropping or downsampling is used. Every correction and ablation in this report was
specified before the new sequence-08 evaluation ran, and nothing changed after it began.

Caches were rebuilt from raw for both geometries: `artifacts/voxel_gate_validation/cache_c3`
(926 files, `6d94c244…5ed2b79a`) and `cache_c0` (926 files, `b3c8b085…9fefab79`). No cached
region was reused.

**Baseline reproduction, asserted before any learned number was accepted:**

| geometry | target | reproduced | \|Δ\| |
|---|---:|---:|---:|
| C0 (constant scale `s0` = 27.3665) | 0.0573 | **0.05725** | 4.9e-5 |
| C3 (learned clip scale) | 0.0778 | **0.07784** | 3.9e-5 |

### Provenance

| item | value |
|---|---|
| LingBot checkpoint | `ee665103…6d8f2e1cd72` (frozen, 1 157 943 540 params) |
| Gate-2 depth head | `2a91822e…19faafbe1` (frozen; only `a_clip` consumed, `r_shape` discarded) |
| config | `configs/voxel_gate_validation/clean_infill.yaml` — `a548ed08…1ce9d752` |
| `full_s0` / `full_s1` / `full_s2` | `05730ac7…` / `60c2ccc8…` / `2c7f4dd3…` |
| `occ_only_s0` / `c0_corrector_s0` | `b90a6e4e…` / `65900493…` |
| tools | `tools/voxel_gate_validation/{cache_geometry,train_clean,select_controls,eval_clean,figures}.py` |
| library | `voxel_gate_validation/{geometry,data}.py` |

**Seed provenance.** The original Gate-3 seed is **0**, read from
`configs/voxel_gate/visible_correction.yaml: experiment.seed`. It was **not** stored in the
Gate-3 checkpoint or `train.json`; Gate 3.1 stores `seed` in both.

---

## 7. Architecture

Full details, extracted from source and checkpoints, are in
**`reports/architecture_ledger.md`**, which covers all eight required components. Summary of
the trained module:

| | |
|---|---|
| input / output | `[B, 6, 256, 256, 32]` → `[B, 1, 256, 256, 32]` logits |
| channels | 0 `occupied`, 1 `log1p_count`, 2 `n_frames`, 3 `mean_confidence`, 4 `mean_point_depth_m`, 5 `R_infer` (clean) |
| body | `Conv3d(6→16,k3,p1)+GN(4)+GELU` → `Conv3d(16→16,k3,p1)+GN(4)+GELU` → `Conv3d(16→16,k3,p1)+GN(4)+GELU`, all stride 1 |
| head | `Conv3d(16→1,k1)`, weight and bias **zero-initialised** |
| output | `z = ±4·(2·occ−1) + head(body(x))`; `pred = (σ(z) ≥ 0.45)` inside `R_infer`, C3 preserved outside |
| receptive field | **7×7×7 voxels = 1.4 m cube** |
| parameters | **16 577** |
| normalisation | channels 1–4 standardised on **source-training clips only**, masked by channel 0; no positional encoding of any kind |

One point the ledger records explicitly: the frozen Gate-2 depth head takes depth,
confidence, validity and normalised image coordinates — **`use_rgb: False`, `in_ch: 5`,
confirmed from the checkpoint**. It never saw RGB or LingBot visual features, so Gate 2's
`SCALE_CORRECTION_ONLY` supports only the claim that *that geometry-only head* learned
scale rather than shape.

---

## 8. Results — sequence 08, 163 clips, frozen protocol

| id | IoU | P | R | occupied | TP | FP | FN | added | removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 constant scale | 0.0573 | 0.318 | 0.066 | 23 508 | 7 558 | 15 950 | 107 552 | 0 | 0 |
| C3 frozen input | 0.0778 | 0.408 | 0.090 | 24 717 | 9 995 | 14 722 | 105 115 | 0 | 0 |
| V1_clean `dilate_r2` | 0.1759 | 0.284 | 0.334 | 137 863 | 38 044 | 99 819 | 77 066 | 113 146 | 0 |
| V2_clean `fill_r1_k3` | 0.1528 | 0.371 | 0.215 | 66 984 | 24 236 | 42 748 | 90 874 | 42 267 | 0 |
| **V3 seed 0** | **0.2154** | 0.494 | 0.284 | 65 223 | 32 028 | 33 195 | 83 082 | 50 217 | 9 711 |
| **V3 seed 1** | **0.2161** | 0.446 | 0.306 | 78 795 | 34 606 | 44 189 | 80 504 | 64 045 | 9 967 |
| **V3 seed 2** | **0.2145** | 0.516 | 0.277 | 61 599 | 31 324 | 30 274 | 83 786 | 48 130 | 11 248 |
| A_occ_only | 0.2052 | 0.422 | 0.293 | 78 914 | 33 058 | 45 856 | 82 052 | 59 953 | 5 756 |
| A_c0_corrector | 0.1819 | 0.393 | 0.259 | 74 814 | 29 412 | 45 402 | 85 698 | 51 306 | 0 |
| VC visible set *(diagnostic)* | 0.1255 | 0.938 | 0.128 | 14 933 | 13 972 | 961 | 101 138 | — | — |
| VR in-band oracle *(diagnostic)* | 0.3873 | 1.000 | 0.387 | 44 190 | 44 190 | 0 | 70 920 | — | — |

Mean `|R_infer|` = 253 826 voxels (12.1 % of the grid), of which 62 906 are
evaluation-invalid. Voxels predicted outside `R_infer`: **0** for every method.

### Seed study (Part E)

| seed | run | best epoch | source-select IoU | seq-08 IoU | P | R | occupied | added | removed | peak GPU | train s | inference/clip |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (reference) | `full_s0` | 16 | 0.2028 | 0.2154 | 0.494 | 0.284 | 65 223 | 50 217 | 9 711 | 1.24 GiB | 627 s | 1.7 ms |
| 1 | `full_s1` | 19 | 0.2034 | 0.2161 | 0.446 | 0.306 | 78 795 | 64 045 | 9 967 | 1.24 GiB | 627 s | 0.3 ms |
| 2 | `full_s2` | 18 | 0.2032 | 0.2145 | 0.516 | 0.277 | 61 599 | 48 130 | 11 248 | 1.24 GiB | 626 s | 0.3 ms |
| **mean** | | | **0.2031** | **0.21536** | 0.485 | 0.289 | 68 539 | 54 131 | 10 309 | | | |
| **sd** | | | 0.0003 | **0.00081** | 0.035 | 0.015 | 8 995 | 8 616 | 826 | | | |
| min / max | | | | 0.2145 / 0.2161 | | | | | | | | |

**The three-seed mean, 0.2154, is the headline result.** Seed 0 is identified separately
as the reference for the paired diagnostic tables below. IoU is remarkably stable
(sd 0.0008) even though the seeds trade precision against recall over a wide range
(0.446–0.516 and 0.277–0.306) and differ by 28 % in occupied volume.

### Per-clip IoU quartiles

| id | p25 | median | p75 |
|---|---:|---:|---:|
| C3 | 0.0588 | 0.0758 | 0.0951 |
| V1_clean | 0.1533 | 0.1731 | 0.1979 |
| V2_clean | 0.1311 | 0.1499 | 0.1804 |
| **V3 seed 0** | **0.1830** | **0.2098** | **0.2473** |
| A_occ_only | 0.1748 | 0.1996 | 0.2329 |
| A_c0_corrector | 0.1592 | 0.1826 | 0.2131 |
| VR | 0.3150 | 0.3815 | 0.4486 |

### By distance — IoU / precision / recall

| id | 0–10 m | 10–20 m | 20–40 m | 40–80 m |
|---|---|---|---|---|
| C3 | 0.270 / 0.59 / 0.33 | 0.130 / 0.40 / 0.16 | 0.039 / 0.36 / 0.04 | 0.004 / 0.49 / 0.00 |
| V1_clean | 0.282 / 0.31 / 0.77 | 0.209 / 0.27 / 0.52 | 0.160 / 0.31 / 0.28 | 0.050 / 0.40 / 0.06 |
| V2_clean | 0.355 / 0.47 / 0.62 | 0.220 / 0.36 / 0.37 | 0.112 / 0.38 / 0.15 | 0.014 / 0.49 / 0.02 |
| **V3 seed 0** | **0.428** / 0.54 / 0.69 | **0.286** / 0.48 / 0.42 | **0.184** / 0.51 / 0.24 | **0.059** / 0.50 / 0.07 |
| A_occ_only | 0.384 / 0.47 / 0.70 | 0.265 / 0.39 / 0.46 | 0.178 / 0.46 / 0.24 | 0.049 / 0.55 / 0.06 |
| A_c0_corrector | 0.356 / 0.49 / 0.61 | 0.236 / 0.38 / 0.40 | 0.157 / 0.40 / 0.22 | 0.057 / 0.50 / 0.07 |
| VR *(oracle)* | 0.834 / 1.00 / 0.83 | 0.598 / 1.00 / 0.60 | 0.334 / 1.00 / 0.33 | 0.088 / 0.90 / 0.09 |

V3 leads every band. Beyond 40 m it reaches 0.059 against an in-band oracle of 0.088 — at
that range the band, not the model, is the binding constraint.

---

## 9. Deterministic controls (Part F)

Both selected on sequences 09–10 only; **no sequence-08 statistic was used.**

| control | rule | source-select IoU | source-select occupied |
|---|---|---:|---:|
| **V1_clean = `dilate_r2`** | best mean IoU on 09–10 over 44 candidates | 0.1653 | 133 161 |
| **V2_clean = `fill_r1_k3`** | occupied count on 09–10 closest to the reference-seed V3's 67 098 | 0.1471 | 66 028 (1.6 % match) |

Reference for scale: C3 on 09–10 is 0.0757 with 24 378 occupied; the clean reference-seed
V3 there is 0.2028 with 67 098.

> The Gate-3 V2 (`fill_r2_k22`, matched to V3's **sequence-08** count, IoU 0.1232) is
> retained here **only as a non-clean historical diagnostic**. It is superseded: the clean
> source-matched control is stronger (0.1528 vs 0.1232), which makes the comparison against
> V3 more conservative, not less.

Deterministic dilation still nearly doubles IoU from 0.0778 to 0.1759 by inflating volume
5.6×, at a precision cost (0.408 → 0.284). This remains the single most important
calibration fact in the project: **any SemanticKITTI occupancy number reported without a
volume-matched morphology control is uninterpretable.**

---

## 10. Statistical comparisons (Part J)

Paired clip-level bootstrap, **10 000 resamples, seed 0, 163 paired clips**.

| contrast | ΔIoU | 95 % CI | clips improved |
|---|---:|---|---:|
| C3 → V1_clean | +0.0981 | [+0.0927, +0.1034] \* | 162 / 163 |
| C3 → V2_clean | +0.0750 | [+0.0720, +0.0780] \* | 162 / 163 |
| **C3 → V3 seed 0** | **+0.1376** | **[+0.1319, +0.1433] \*** | **163 / 163** |
| **C3 → V3 seed 1** | **+0.1383** | **[+0.1326, +0.1440] \*** | **163 / 163** |
| **C3 → V3 seed 2** | **+0.1367** | **[+0.1309, +0.1427] \*** | **163 / 163** |
| **V1_clean → V3 seed 0** | **+0.0395** | **[+0.0350, +0.0441] \*** | 155 / 163 |
| **V2_clean → V3 seed 0** | **+0.0626** | **[+0.0587, +0.0665] \*** | 163 / 163 |
| **A_occ_only → V3 seed 0** | **+0.0102** | **[+0.0084, +0.0120] \*** | 131 / 163 |
| **A_c0_corrector → V3 seed 0** | **+0.0335** | **[+0.0276, +0.0395] \*** | 138 / 163 |
| C0 → A_c0_corrector | +0.1247 | [+0.1195, +0.1299] \* | 163 / 163 |
| C0 → C3 (uncorrected scale gap) | +0.0206 | [+0.0155, +0.0258] \* | 122 / 163 |

\* CI excludes zero.

Per-clip spread: C3 → V3 seed 0 has min **+0.0250**, p5 +0.0890, median +0.1320, max
+0.2290 — **no clip is degraded**. V1_clean → V3 seed 0 has min −0.0484, median +0.0357,
max +0.1369 (8 clips degrade). A_occ_only → V3 seed 0 has min −0.0237, median +0.0101, max
+0.0517.

### Fraction of the radius-3 in-band oracle headroom recovered

Gap from C3 (0.0778) to the in-band oracle VR (0.3873) = 0.3095.

| V1_clean | V2_clean | V3 s0 | V3 s1 | V3 s2 | A_occ_only |
|---:|---:|---:|---:|---:|---:|
| 31.7 % | 24.2 % | **44.5 %** | **44.7 %** | **44.2 %** | 41.1 % |

The old five-frame visible score of 0.1255 is **not** used as an upper bound; it is
retained as a diagnostic reference only. Plain dilation exceeds it, because the SSC target
marks occluded near-surface voxels occupied.

### Volume analysis — the gain is not densification

| | occupied | ×C3 | IoU | precision |
|---|---:|---:|---:|---:|
| C3 | 24 717 | 1.00 | 0.0778 | 0.408 |
| V1_clean | 137 863 | 5.58 | 0.1759 | 0.284 |
| V2_clean | 66 984 | 2.71 | 0.1528 | 0.371 |
| **V3 seed 0** | **65 223** | **2.64** | **0.2154** | **0.494** |
| VR oracle | 44 190 | 1.79 | 0.3873 | 1.000 |

Three independent facts: V3 uses **53 % fewer** occupied voxels than V1_clean and still
beats it by +0.0395; the volume-matched V2_clean (66 984 vs 65 223, a 2.7 % match) loses by
+0.0626; and V3's precision **rises** to 0.494 while both controls lose precision. V3 is
also the only configuration that **removes** voxels — 9 711 per clip against the oracle's
14 722 — which morphological dilation and filling are structurally incapable of.

---

## 11. Ablation 1 — learned morphology (Part G)

`A_occ_only`: identical `[6, 256, 256, 32]` tensor and identical architecture, with
channels 1–4 permanently zeroed. Same splits, loss, radius, epoch policy, τ = 0.45, gate
and evaluation. Reference seed 0.

| | seq-08 IoU | P | R | occupied | removed | source-select IoU | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_occ_only | 0.2052 | 0.422 | 0.293 | 78 914 | 5 756 | 0.1931 | 4 |
| V3 seed 0 (full) | **0.2154** | 0.494 | 0.284 | 65 223 | 9 711 | 0.2028 | 16 |

`A_occ_only → V3 seed 0` = **+0.0102 [+0.0084, +0.0120]**, CI excludes zero, 131/163 clips
improve. The full model significantly exceeds the occupancy-only model, so the diagnosis is
**`FULL_FEATURES_CONTRIBUTE`**.

**But the honest reading of the same numbers is that mechanism is dominated by learned
morphology.** `A_occ_only` alone captures **92.6 %** of the C3 → V3 gain
(0.1274 of 0.1376). What the evidence channels buy is a real but modest sharpening:
precision 0.422 → 0.494 at 17 % less volume, and 69 % more deletions. Point count, frame
count, confidence and metric depth tell the model *which* observed voxels to trust and
delete; they are not what lets it infill.

`A_occ_only` is **not** "simple morphology": it is a supervised learned 3D prior with a
1.4 m receptive field that can delete voxels, and it beats the best deterministic control
by 0.0293. The distinction matters — deterministic dilation reaches 0.1759, the learned
occupancy-only prior 0.2052, the full model 0.2154.

---

## 12. Ablation 2 — does C3 scale still matter? (Part H)

`A_c0_corrector`: same architecture, loss, policy and threshold, trained from scratch on
frozen **C0** constant-scale geometry, with its region defined as a radius-3 dilation of
**C0** occupancy (no valid mask). It is given neither the C3 clip scale nor the oracle
scale. C0 reproduced at 0.05725 before training.

| | IoU |
|---|---:|
| C0 (uncorrected) | 0.0573 |
| C3 (uncorrected) | 0.0778 |
| A_c0_corrector | 0.1819 |
| V3 seed 0 (C3-based) | **0.2154** |

| contrast | ΔIoU | 95 % CI | improved |
|---|---:|---|---:|
| C0 → C3 (scale gap, uncorrected) | +0.0206 | [+0.0155, +0.0258] \* | 122/163 |
| C0 → A_c0_corrector | +0.1247 | [+0.1195, +0.1299] \* | 163/163 |
| **A_c0_corrector → V3 seed 0** | **+0.0335** | **[+0.0276, +0.0395] \*** | 138/163 |

Diagnosis: **`CLIP_SCALE_REMAINS_NECESSARY`**. The C3-based corrector significantly
outperforms the separately trained C0 corrector, and the notable finding is that the value
of correct scale **grows** after voxel correction: the uncorrected gap is 0.0205 and the
corrected gap is 0.0335, 1.6× larger. The voxel corrector does not absorb scale error; it
amplifies the benefit of getting scale right, because a correctly placed surface gives the
local prior something correct to build on. Per the brief, the scale module is **not**
removed; this is recorded for the next design decision.

---

## 13. Visible versus unobserved (Part I)

The Gate-1 visible reconstruction is used **only** as a diagnostic partition — never in
training, never in inference, never in gating.

| config | added inside VC | added outside VC | % outside | visible IoU / P / R | non-visible IoU / P / R |
|---|---:|---:|---:|---|---|
| V1_clean | 6 501 | 106 644 | 94.3 % | 0.720 / 0.958 / 0.743 | 0.214 / 0.224 / 0.835 |
| V2_clean | 4 222 | 38 045 | 90.0 % | 0.578 / 0.966 / 0.590 | 0.214 / 0.281 / 0.489 |
| **V3 seed 0** | 5 146 | 45 070 | **89.8 %** | 0.628 / 0.970 / 0.641 | **0.349** / 0.415 / 0.714 |
| V3 seed 1 | 5 558 | 58 487 | 91.3 % | 0.651 / 0.968 / 0.666 | 0.332 / 0.372 / 0.777 |
| V3 seed 2 | 5 001 | 43 129 | 89.6 % | 0.610 / 0.971 / 0.622 | 0.360 / 0.437 / 0.694 |
| A_occ_only | 5 320 | 54 633 | 91.1 % | 0.646 / 0.968 / 0.661 | 0.304 / 0.347 / 0.743 |
| A_c0_corrector | 5 063 | 46 243 | 90.1 % | 0.524 / 0.968 / 0.533 | 0.284 / 0.329 / 0.703 |

Diagnosis: **`LOCAL_OCCUPANCY_INFILL`**. Nearly 90 % of what the model adds lies outside
what a perfect five-frame visible reconstruction would produce. The correct name for the
module is **bounded local occupancy correction and infill**, and the infill component *is*
local completion of unobserved occupancy — bounded at 0.6 m, but completion nonetheless.
This report does not claim the module has not performed completion.

Where V3 separates from the controls is precisely the non-visible partition: 0.349 against
0.214 for both controls and 0.304 for `A_occ_only`. On the visible partition V1_clean
actually scores *higher* (0.720 vs 0.628) — flooding the scene is a good strategy on a
small dense partition. The learned advantage is in inferring unobserved near-surface
occupancy, not in cleaning up observed voxels.

---

## 14. Visualisations

`artifacts/voxel_gate_validation/eval/clips_clean.png` — bird's-eye views (z collapsed) of
the most improved, median and least improved clip by C3 → V3 seed 0, across six
configurations, coloured true positive / false positive / missed.

| clip | role | ΔIoU | C3 | V1_clean | V2_clean | V3 s0 | A_occ_only | VC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `08_000300_000320_s5` | most improved | +0.229 | 0.085 | 0.224 | 0.202 | **0.314** | 0.303 | 0.127 |
| `08_002475_002495_s5` | median | +0.132 | 0.132 | 0.198 | 0.206 | **0.264** | 0.252 | 0.197 |
| `08_004050_004070_s5` | least improved | +0.025 | 0.065 | 0.040 | 0.064 | **0.090** | 0.074 | 0.420 |

No clip is degraded, so the third row is "least improved". It is also the clip where the
ground truth disagrees with every reconstruction — C3's precision there is 0.08 and even
the visible set reaches only 0.57 — which points to a registration or dynamic-object
problem in that clip rather than a model failure. The images show V1_clean and V2_clean
flooding the scene with red false positives while V3 places its additions on the road
surface and building façades; V3 and `A_occ_only` look visually similar, consistent with
§11.

---

## 15. Tests (Part K)

`tests/voxel_gate_validation/test_clean_infill.py` — 24 tests covering every mandated item:

| # | requirement | test |
|---|---|---|
| 1 | `R_infer` == pure radius-3 dilation | `test_r_infer_is_exactly_radius3_dilation_of_occupancy` (also asserts it differs from the leaky region, so the test can discriminate) |
| 2 | `R_infer` bit-identical under independent randomisation of `valid`, `gt`, `vc` | `test_r_infer_bit_identical_under_independent_target_randomisation` (parametrised, 3 trials each) |
| 3 | input tensor bit-identical under the same mutations | `test_input_tensor_bit_identical_under_target_mutation` |
| 4 | input builder raises on any target key | `test_input_builder_raises_if_it_touches_a_target_key` |
| 5 | **provenance from raw construction** | `test_geometry_evidence_signature_admits_no_target_side_argument` + `test_raw_evidence_rebuild_is_invariant_to_any_valid_mask` — rebuilds evidence from synthetic raw geometry, so no previously derived region is involved |
| 6 | channel 5 can be 1 in invalid voxels | `test_r_infer_contains_evaluation_invalid_voxels` (fails if zero) |
| 7 | loss moves only inside `R_infer ∧ valid` | `test_loss_moves_only_for_targets_inside_r_infer_and_valid` |
| 8 | evaluation ignores invalid without changing predictions | `test_evaluation_ignores_invalid_voxels_without_changing_predictions` |
| 9 | gating depends only on `R_infer` | `test_output_gating_depends_only_on_r_infer` |
| 10 | controls use the same clean region | `test_controls_use_the_same_clean_region` |
| 11 | source / selection / seq-08 clip ids disjoint | `test_source_selection_and_sequence08_clip_ids_are_disjoint` (both geometries) |
| 12 | no seq-08 statistic selects anything | `test_no_sequence08_statistic_selects_controls`, `…_checkpoints_or_hyperparameters` |
| 13 | seed control deterministic | `test_seed_control_is_deterministic`, `test_bootstrap_is_paired_and_deterministic` |
| 14 | C0 and C3 reproduce | `test_baselines_reproduce_within_tolerance` (parametrised), `test_c0_and_c3_geometries_actually_differ` |
| 15 | Gates 0–2 still pass | full-repo run below |

Plus frozen-protocol assertions (grid, radius, τ, architecture, epochs, seeds),
zero-init identity, and the `A_occ_only` channel-zeroing check.

**Exact totals — whole repository:**

```
387 passed, 2 skipped, 1 warning
```

The 2 skips are pre-existing and unrelated (`tests/prompted_lingbot/test_anchors.py:100`,
"oracles are fitted from ground truth, not from prompts").

| suite | result |
|---|---|
| `tests/scale_gate` | 49 passed |
| `tests/depth_gate` | 35 passed |
| `tests/voxel_gate` (Gate 3, unmodified) | 29 passed |
| `tests/voxel_gate_validation` (new) | 24 passed |
| remainder of `tests/` | 250 passed, 2 skipped |

---

## 16. Decision rules

### Validity — `CLEAN_LOCAL_INFILL_PASSES`

| requirement | observed | |
|---|---|:--:|
| inference features and region entirely target-independent | `input_view` excludes target keys structurally; `geometry_evidence` signature admits none; 5 leakage tests including raw-construction provenance | ✔ |
| three-seed mean significantly exceeds C3 | 0.2154 vs 0.0778; per-seed +0.1376 / +0.1383 / +0.1367, all CIs exclude zero | ✔ |
| three-seed mean significantly exceeds V1_clean | 0.2154 vs 0.1759; V1_clean → V3 s0 +0.0395 [+0.0350, +0.0441] | ✔ |
| all three seeds improve over C3 | 163/163 clips for every seed; sd 0.0008 | ✔ |
| no precision or recall collapse | precision 0.408 → 0.485 (mean), recall 0.090 → 0.289 | ✔ |
| not explained by volume inflation | 53 % fewer voxels than V1_clean; volume-matched V2_clean loses by 0.0626; only method that deletes | ✔ |

`TARGET_DERIVED_INFERENCE_MASK` does not apply to the clean rerun (it *did* apply to
Gate 3, which is why that result is withdrawn). `UNSTABLE_ACROSS_SEEDS` is excluded
(sd 0.0008 on a 0.1376 effect). `MORPHOLOGY_CONTROL_EXPLAINS_GAIN` is excluded — but note
that a *learned* occupancy-only prior does explain 92.6 % of it, which is §11's finding,
not this rule's.

### Mechanism — `FULL_FEATURES_CONTRIBUTE`

The full model significantly exceeds `A_occ_only` (+0.0102, CI excludes zero, 131/163
clips, above the project's own 0.005 practical threshold). Reported alongside, and equally
important: `A_occ_only` captures 92.6 % of the gain, so the module is **predominantly a
learned anisotropic 3D occupancy prior**, with the evidence channels contributing a small,
real, precision-oriented increment.

### Scale — `CLIP_SCALE_REMAINS_NECESSARY`

`A_c0_corrector → V3 seed 0` = +0.0335 [+0.0276, +0.0395], and the corrected scale gap
(0.0335) is 1.6× the uncorrected one (0.0205).

---

## 17. Limitations

1. **In-domain only.** Sequences 00–10 are one city, one sensor rig, one camera height, one
   intrinsics set. Sequence 08 is held out temporally, not environmentally. **These results
   cannot establish generalization.**
2. **Gate 3's 0.2903 is withdrawn.** It is superseded by 0.2154 ± 0.0008. The Gate-3 report
   and artifacts are preserved unmodified as the record of the defect; anyone quoting them
   must read this report first.
3. **This is local completion, not visible-voxel correction.** 89.8 % of additions are
   outside the five-frame visible set. The bound is geometric (0.6 m), not semantic.
4. **Mechanism is mostly learned morphology.** 92.6 % of the gain survives with only
   occupancy and the region as input. The rich evidence channels are worth +0.0102.
5. **The band radius controls the result** and was frozen at 3, not reselected. In-band
   oracle IoU rises monotonically with radius, so a wider band would raise both the ceiling
   and any method's score.
6. **SemanticKITTI's SSC target is aggregated over future frames**, so voxels occluded at
   the anchor instant are labelled occupied. Some apparent inference is the benchmark
   rewarding a plausible surface prior — the reason plain dilation reaches 0.1759.
7. **Three seeds, one architecture, one loss.** No architecture or loss variance is
   characterised, by design.
8. **τ = 0.45 was originally chosen on 09–10 under the leaky region** and was frozen rather
   than reselected, to avoid post-hoc tuning after sequence 08 had been observed. It is
   therefore mildly suboptimal for the clean protocol, which makes the clean result
   conservative.
9. **Sequence 08 was observed in Gate 3** before this rerun. Every correction and ablation
   here was specified in advance and nothing changed after the evaluation began, but this
   is a rerun on a previously seen test set, not a virgin evaluation.
10. **Deletions are modest.** V3 removes 9 711 of the 14 722 voxels the oracle would remove;
    C3's 14 722 false positives are only partly cleaned.

---

## 18. Recommendation for the next experiment

**Frozen cross-dataset evaluation.** The clean result is strong and stable, and the
remaining doubt is no longer about leakage or morphology — it is about domain. Three
components in the stack have now each produced large in-domain gains from modules that
could plausibly have memorised KITTI structure: the Gate-2 clip-scale estimator (fixed
camera height and intrinsics), the Gate-3.1 occupancy prior (flat road, consistent street
geometry, 1.4 m receptive field), and the fusion geometry itself. §12 shows these interact
— correct scale is worth *more* after voxel correction — so they must be tested together.

Concretely, for the next task: freeze the entire stack (LingBot → `a_clip` → C3 fusion →
`R_infer` → V3 seed 0), change **nothing**, and evaluate zero-shot on a second driving
dataset with LiDAR occupancy. Report C0, C3, V1_clean, V2_clean and V3 there under the same
volume-matched control discipline and the same visible/non-visible split. If V3's margin
over V1_clean survives out of domain, the local prior is real; if it collapses, it was
memorised road geometry, and that is worth knowing before any wider-band or
privileged-frame completion work.

Deferred until after that: widening the band (the in-band oracle at radius 3 is 0.3873 and
V3 reaches 44.5 % of the available headroom, so there is room, but widening moves further
from correction toward completion and should not be done on in-domain evidence alone), and
privileged-frame completion.

**Cross-dataset evaluation, wider-band completion, privileged-frame completion and
semantics have not been started.** This task stops at the report.
