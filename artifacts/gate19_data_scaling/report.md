# Gate 19 — training-data scaling: OUTCOME C (not primarily data limited)

Going from 1,358 to 9,066 KITTI-360 anchors — a **6.7x** increase — buys very little, and more than half of what looked like a data gain is really an optimisation gain.

| effect on geometry full IoU | value |
|---|---:|
| **more compute only** (D_small, 6k → 40k updates, same 1,358 anchors) | **+0.61** |
| **more data only** (D_small → D_large at equal 40k updates, 4.4x data) | **+0.39** |
| the two confounded together (the naive fixed-epochs reading) | +1.04 |

The naive reading would have reported +1.04 and called geometry data-limited. Isolating the variables shows **+0.61 of it is simply training longer on the data we already had**.

## 1. Available-data audit

Every drive with **both** a hard-map input and an occupancy target. Usable = the intersection; 0004 and 0005 have more targets than inputs.

| drive | usable anchors | | drive | usable anchors |
|---|---:|---|---|---:|
| 0003 | 193 | | 0000 | 2,094 |
| 0007 | 569 | | 0002 | 2,769 |
| 0010 | 596 | | 0004 | 2,034 |
| | | | 0005 | 811 |

**9,066 usable training anchors**; 0006 (590) stays frozen for evaluation and is never trained on. Two provenance checks decided whether the added drives were admissible: all eight drives share a single `rules_sha256`, so the inputs were built by the identical hard mapper; and the targets postdate the open-sky fix. Without both, the added data would not have been comparable.

This pool is the one the B4 scaling study prepared. That study is BLOCKED on **delayed semantic target** reliability — a constraint that does not apply here, because geometry trains on raw-LiDAR occupancy targets and never reads a delayed target. Its blocked status is unchanged by this gate.

## 2. Nested subsets

| pool | anchors | of which added | epochs at 6k steps | steps for fixed epochs |
|---|---:|---:|---:|---:|
| D_small | 1,358 | 0 | 17.67 | 6,000 |
| D_medium | 3,000 | 1,642 | 8.00 | 13,255 |
| D_large | 6,000 | 4,642 | 4.00 | 26,510 |
| D_all | 9,066 | 7,708 | 2.65 | 40,056 |

`D_small` is **exactly** the frozen pool, so the left end of every curve is the published baseline rather than a re-sample. Nesting is by prefix of one fixed permutation per drive and is asserted in code; additions are proportional to drive size and permuted rather than contiguous, so a small subset is not one stretch of road. Exact anchor IDs are in `pools/*.json`.

## 3. Three regimes, because two of them answer different questions

| regime | every scale gets | isolates |
|---|---|---|
| `fixed_steps` | 6,000 updates | data, at a small compute budget |
| `fixed_compute` | 40,056 updates | data, at a large compute budget — **the control** |
| `fixed_epochs` | 17.67 passes | nothing cleanly: bigger pools also get more updates |

### fixed_steps

| pool | anchors | steps | epochs | geo full | geo unobs-later | geo no-evidence | sem completed-unobs | sem SSC full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D_small | 1,358 | 6,000 | 17.67 | 13.51 | 13.79 | 12.32 | 8.83 | 1.56 |
| D_medium | 3,000 | 6,000 | 8.00 | 13.81 | 14.80 | 12.78 | 9.33 | 1.53 |
| D_large | 6,000 | 6,000 | 4.00 | 13.89 | 14.86 | 12.88 | 9.39 | 1.57 |
| D_all | 9,066 | 6,000 | 2.65 | 13.99 | 14.85 | 13.00 | 9.35 | 1.54 |

### fixed_compute

| pool | anchors | steps | epochs | geo full | geo unobs-later | geo no-evidence | sem completed-unobs | sem SSC full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D_small | 1,358 | 40,056 | 117.99 | 14.12 | 13.55 | 12.82 | 8.92 | 1.65 |
| D_large | 6,000 | 40,056 | 26.70 | 14.52 | 15.14 | 13.37 | 9.29 | 1.61 |

### fixed_epochs

| pool | anchors | steps | epochs | geo full | geo unobs-later | geo no-evidence | sem completed-unobs | sem SSC full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D_small | 1,358 | 6,000 | 17.67 | 13.51 | 13.79 | 12.32 | 8.83 | 1.56 |
| D_medium | 3,000 | 13,255 | 17.67 | 14.09 | 14.93 | 13.01 | 9.28 | 1.57 |
| D_large | 6,000 | 26,510 | 17.67 | 14.44 | 15.03 | 13.30 | 9.26 | 1.60 |
| D_all | 9,066 | 40,056 | 17.67 | 14.55 | 15.11 | 13.39 | 9.33 | 1.62 |

Seed spread is small — geometry sd **0.080**, semantic sd **0.122** over three seeds — so these movements are real, not noise.

## 4. Geometry learning curve

At equal compute, 4.4x the data is worth **+0.39 pp** of full IoU (14.12 → 14.52). Going from D_large (6,000) to D_all (9,066) adds +0.03 — the curve is already flat at the top end.

The gain also exceeds the seed noise, so it is a real effect — just a small one, well under the 1.0 pp bar recorded before the runs. This is a *small positive*, not *no effect*, and it should be quoted that way.

One diagnostic worth noting: `D_small` at 40,056 updates is **118 epochs** on 1,358 anchors, and while its full IoU rises to 14.12, its completed-unobserved IoU *falls* (13.79 → 13.55). Training the small pool far longer starts to cost exactly the region we care about — the signature of overfitting, and the reason a fixed-epochs comparison alone would have misled.

## 5. Semantic SSC learning curve

Flat everywhere. Completed-unobserved mIoU moves +0.37 for 4.4x the data at equal compute (8.92 → 9.29), and full SSC mIoU does not move at all across the whole experiment (1.53–1.65).

This is the decomposition the brief asked for, and it lands on its middle case sharpened: **geometry improves slightly with data; semantics does not follow.** Better occupancy does not produce better semantics, because the label placed on each newly-correct voxel still comes from a teacher that Gate 18 measured at 62.57% top-1 and 16.91 mIoU on its own evidence.

## 6. Per-class scaling

Completed-unobserved IoU per class, `fixed_steps`, D_small → D_all:

| class | D_small | D_all | delta |
|---|---:|---:|---:|
| 13 | 52.73 | 55.28 | +2.55 |
| 11 | 34.33 | 36.19 | +1.86 |
| 7 | 17.28 | 16.74 | -0.54 |
| 1 | 10.27 | 12.36 | +2.09 |
| 17 | 8.88 | 8.81 | -0.07 |
| 9 | 8.12 | 7.69 | -0.43 |
| 4 | 7.72 | 10.82 | +3.10 |
| 12 | 4.46 | 4.91 | +0.45 |
| 8 | 3.91 | 3.72 | -0.19 |
| 3 | 3.76 | 3.37 | -0.39 |
| 10 | 3.14 | 2.91 | -0.24 |
| 18 | 1.67 | 2.08 | +0.41 |
| 5 | 1.10 | 1.27 | +0.18 |
| 6 | 0.96 | 1.41 | +0.45 |
| 14 | 0.25 | 0.24 | -0.02 |
| 2 | 0.19 | 0.17 | -0.02 |
| 16 | 0.06 | 0.25 | +0.19 |
| 15 | 0.03 | 0.04 | +0.02 |

Mean IoU hides the shape: the movement is concentrated in the few classes the teacher can already read, and the classes it cannot read stay at zero no matter how much geometry training data is added.

## 7. Convergence

| regime | pool | train loss | val loss | gap |
|---|---|---:|---:|---:|
| fixed_steps | D_small | 0.9575 | 1.1184 | +0.1609 |
| fixed_steps | D_medium | 0.9921 | 1.0376 | +0.0455 |
| fixed_steps | D_large | 1.0018 | 1.0319 | +0.0301 |
| fixed_steps | D_all | 1.0009 | 1.0292 | +0.0283 |
| fixed_compute | D_small | 0.9150 | 1.1723 | +0.2573 |
| fixed_compute | D_large | 0.9846 | 1.0143 | +0.0297 |

At 6,000 steps the small pool is not overfitting — it is **underfitting relative to its own capacity**, which is why extra updates help it more than extra data does. At 40,056 steps the small pool's completed-unobserved region degrades, which is where overfitting shows up first.

## 8. Cost

27 runs, 12.6 GPU-hours: 12 at fixed steps, 9 at fixed epochs, 6 for the fixed-compute control. Everything else reused frozen caches.

**Validation.** `D_small` under `fixed_steps` reproduces the frozen baseline's `draw_digest`, `target_digest`, `init_state_digest`, `n_train`, `steps` and `config_sha256` **exactly**, so the trainer is a re-execution of the frozen recipe rather than a variant. The evaluator reproduces the frozen full IoU to 0.0016, inside the declared 0.005 tolerance.

## 9. Outcome

**OUTCOME C — the current formulation is not primarily data limited.** Both curves flatten early: at equal compute, 4.4x the data gives +0.39 pp geometry and +0.37 pp semantics, against bars of 1.0 and 0.5 pp set before the runs. Both exceed seed noise, so the honest statement is *small but real*, not *nothing*.

Per the brief: **do not scale training blindly, and do not respond by adding more datasets.** The case for SemanticKITTI / Occ3D / Waymo training was conditional on this curve still rising; it is not.

## 10. Recommended next experiment (exactly one)

**The same one Gate 18 recommended, now with a second independent reason to run it: measure whether the teacher can be improved at all.** Four gates have now converged on the same wall — instance association (Gate 17, +0.09), the feature-space objective (B5, −2.03), temporal accumulation (Gate 18, +0.19 controlled), and training-data volume (Gate 19, +0.37 controlled). Each ruled out a different mechanism, and each ended at a teacher that is 62.57% correct on what it directly observes.

Score alternative dense open-vocabulary encoders, and a prompt/vocabulary variation of Trident-H, on the frozen evidence voxels of 0006 against the same GT. No training, no new geometry, one forward pass and the existing scorer. If a different teacher moves 62.57% / 16.91 materially, the whole stack inherits it. If none does, the open-vocabulary readout itself is the thing to replace — and that, not more data or another propagation head, is where the next real effort belongs.

## 11. Reproduce

```bash
python -m tools.data_scaling.subsets
bash artifacts/gate19_data_scaling/run_train.sh
bash artifacts/gate19_data_scaling/run_control.sh
bash artifacts/gate19_data_scaling/run_eval.sh
python -m tools.data_scaling.analysis
python -m tools.data_scaling.report
```

Code: `tools/data_scaling/{config,subsets,train,evaluate,analysis,report}.py`. Pools, per-run records (with digests, steps, epochs, GPU hours), per-checkpoint evaluations and `analysis.json` are all under `artifacts/gate19_data_scaling/`. Deterministic, resumable, seeds [0, 1, 2]; commit in `git_commit.txt`.
