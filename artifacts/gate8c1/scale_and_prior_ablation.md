# Gate 8C-1 — what MoGe-2's scale buys, and what the completion is actually doing

Added 2026-09-04, in answer to two questions: *how does LingBot-Map perform without the
MoGe-2 scale?* and, following from that, *how much of the score comes from the observations
at all?* The second question was not planned; it fell out of the first and materially
changes how the Occ3D number should be read.

Scripts: `tools/gate8c1/scale_ablation.py`, `prior_ablation.py`, `lock_analysis.py`.
Machine-readable: `scale_ablation_{occ3d,semantickitti}.json`,
`prior_ablation_{occ3d,semantickitti}.json`.

---

## 1. The scale is not a small correction

LingBot-Map reconstructs up to one unknown global factor. Measured over every segment:

| split | segments | min | median | max | spread |
|---|---|---|---|---|---|
| Occ3D-nuScenes val | 150 | 9.37 | **24.48** | 55.28 | **5.9×** |
| SemanticKITTI 08 | 1 | 26.98 | 26.98 | 26.98 | — |

One LingBot canonical unit is roughly **1/25 of a metre**, and the factor **varies 5.9×
across scenes**. Two consequences:

* `s = 1` is not a degraded system, it is a different unit system — the whole reconstruction
  collapses into a ~2 m box at the origin.
* No hard-coded constant can replace MoGe. The scale is a property of the *scene*, not of
  the checkpoint.

## 2. Three scale conditions (Occ3D, 30 anchors; SemanticKITTI, 24)

`map` is the frozen stack's own occupancy; `completion` is the trained module on top. The
module was trained under per-scene MoGe scale, so its rows under the other two conditions
are out of distribution and are marked accordingly.

**Occ3D-nuScenes**

| condition | map IoU | map P | map R | completion IoU |
|---|---|---|---|---|
| `s = 1` — LingBot-Map alone | **0.00** | 0.00 | 0.00 | 24.39 † |
| `s = 24.48` — one constant for every scene | 9.24 | 55.86 | 9.96 | 27.33 † |
| `s` per scene from MoGe-2 — **deployed** | **10.43** | 66.87 | 11.00 | **29.77** |

**SemanticKITTI 08** (one segment, so "constant" and "per scene" are the same number)

| condition | map IoU | map P | map R | completion IoU |
|---|---|---|---|---|
| `s = 1` | **0.00** | 0.08 | 0.00 | 12.87 † |
| `s = 26.98` = deployed | 9.34 | 35.37 | 11.26 | 12.26 |

† out of distribution for the network.

**Without MoGe there is no metric map at all** — 0.00 IoU, not a reduced score. A single
global constant recovers most of it (9.24 of 10.43 on Occ3D) but still loses 1.2 IoU on the
map and 2.4 on the completion, because the per-scene spread is real.

## 3. The finding this turned up: most of the Occ3D score is a scene-independent prior

The `s = 1` row is strange — the map scores 0.00 yet the completion scores 24.39. That is
only possible if the network, told nothing was observed, emits something learned. It does.

Feeding the network a map in which **every cell is unobserved** (`prior_ablation.py`):

| Occ3D, 40 anchors | IoU | precision | recall |
|---|---|---|---|
| **blind prior — no observations at all** | **44.43** | 84.45 | 48.39 |
| deployed (real map) | 31.41 | 68.37 | 36.75 |
| fill every valid voxel | 23.78 | 23.78 | 100.00 |

**Verified, not inferred.** The blank-input prediction is byte-identical on **40/40**
anchors — it is literally one constant volume. Its content:

| layer | filled |
|---|---|
| z = −0.8 m | 100.0 % |
| z = −0.4 m | 100.0 % |
| z = +0.0 m | 98.0 % |
| z = +4.8 m | 3.9 % |
| z = +5.2 m | 99.5 % |

The module learned "there is ground under you" (and, at the top, the same mid-air slab that
wrecks SemanticKITTI — §8.2.1). Because 53 % of Occ3D ground truth inside the camera mask
*is* the road band, filling three ground layers blindly scores 44.4.

**So on Occ3D the observations make the prediction worse, by ~13 IoU.**

## 4. Why — partly the residual lock, mostly not (`lock_analysis.py`, 20 anchors)

| | share of evaluated cells |
|---|---|
| locked OCCUPIED by the map (`|logodds| ≥ 2`, residual cannot act) | 2.90 % |
| locked FREE by the map | 10.51 % |
| open to the residual | 87.20 % |

Of the cells the map locks **free**, the ground truth says **occupied for 26.6 %** — free
carving is wrong a quarter of the time and the residual is forbidden from repairing it.

But the lock is not the main cost:

| Occ3D, 20 anchors | IoU |
|---|---|
| prior everywhere | 43.97 |
| prior wherever the map is silent, map's answer where locked | 39.69 |
| deployed | 30.95 |

The lock accounts for ~4.3 IoU. The remaining **~8.7** is the network's own response to
observations in the *open* region: conditioning on the KITTI-360-style map evidence moves it
away from a prior that happened to be right for Occ3D.

## 5. How this changes the reading of the gate

It does not touch the frozen numbers, the FAIL verdict, or the OccAny reproduction. It
changes the *attribution*:

* The Occ3D result (30.9) should **not** be presented as "the completion module transfers".
  A scene-independent ground-plane prior from the same checkpoint scores 44.4 on the same
  masked grid, and that prior beats the deployed system, OccAny's 20.7, and every baseline.
* What genuinely transfers is a **ground-plane prior**, which is cheap and not the claim.
* The honest open question is why conditioning on observations *hurts* on Occ3D while the
  same conditioning is at worst neutral on SemanticKITTI (12.26 deployed vs 12.63 blind).
* Any future Occ3D comparison must report the blind-prior row as a baseline. Without it, a
  reader cannot tell how much of any number is scene understanding and how much is "there
  is ground under the car".
