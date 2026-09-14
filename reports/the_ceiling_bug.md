# The ceiling bug

*How a completion model came to predict a solid roof over every street, why one benchmark
screamed about it and the other did not, and what actually fixed it.*

Written 2026-09-08. Every number below is either measured in this document (commands given
in §10) or cited to the artifact that measured it. Where the two disagree, this file says so.

---

## 1. The symptom

The completion module — 986,114 parameters, trained on KITTI-360 alone — scored **12.74 %**
SC IoU on SemanticKITTI 08 and **30.89 %** on Occ3D-nuScenes. Two things about the
SemanticKITTI number were wrong beyond "the model is weak":

* it lost to baselines that involve no learning at all: a plain 0.4 m dilation of the raw
  incremental map scored **15.41**, and a frozen 5-frame map with dilation **16.00**;
* the 3D galleries showed a red sheet floating over the scene in every frame. The model was
  predicting occupancy in open sky.

A model that is merely undertrained does not do that. Sky is the easiest negative in the
dataset. Predicting it as solid is not a capability failure, it is a sign that something in
the pipeline never told the model sky was empty.

## 2. The wrong first hypothesis

The natural reading is calibration: the threshold τ is too low, the model floods the volume,
raise τ and the roof goes away. That was tested by sweeping τ **on SemanticKITTI itself** —
target-domain tuning the protocol forbids, done only to size the gap:

| τ | IoU | precision | recall |
|---|---|---|---|
| −0.500 | 23.54 | 29.64 | 53.32 |
| **−0.375 (oracle)** | **24.31** | 31.78 | 50.85 |
| −0.125 (source-selected) | 23.69 | 33.57 | 45.95 |
| +0.500 | 19.19 | 41.02 | 26.51 |

The source-selected threshold sits within 0.6 points of the best threshold the target dataset
could possibly hand it. Calibration was never the problem. *(Table from
`artifacts/gate8c1/ceiling_fix.md`; measured on the fixed model.)*

## 3. Diagnosis — the model was never told

### 3.1 The supervision gap

Supervision for the completion module is built from *future* LiDAR sweeps. `rawtarget.py`
assigned each voxel one of three states — `UNKNOWN=0`, `FREE=1`, `OCCUPIED=2` — and marked
anything no sweep had measured as `UNKNOWN`.

A Velodyne ray never climbs above its topmost ring. So every voxel high over the road is
unmeasured, and every one of them was `UNKNOWN`.

That matters because of one line in `core/model/losses.py:37`:

```python
def balanced_weights(gt_occ, gt_valid, observed, w_unknown: float = 2.0):
    """Per-voxel weights: valid supervision only; unknown/frontier voxels up-weighted."""
    w = gt_valid.float()
    ...
```

The weight is **exactly zero** on invalid voxels, in both `focal_bce` and `soft_dice`. The
region above the roofline contributed no gradient at all. An unsupervised region is one the
completion can fill for free.

Measuring it — height profile of the validity mask, 40 random training targets against 40
random SemanticKITTI 08 anchors, same 256×256×32 @ 0.2 m grid:

| height | KITTI-360 **train** % valid | SemanticKITTI **eval** % valid | eval % occupied |
|---|---|---|---|
| +4.3 m | **0.98** | **98.99** | 0.00 |
| +3.5 m | 3.27 | 97.29 | 0.05 |
| +2.9 m | 7.49 | 93.88 | 0.26 |
| +2.1 m | 20.15 | 82.84 | 1.46 |
| +0.1 m | 64.73 | 55.92 | 14.64 |

Near-mirror images. Training says "I have no opinion about the sky." SemanticKITTI says
"the sky is scored, and it is empty." Every voxel the model lit up there was a pure false
positive, and nothing in training had ever discouraged it.

The distinction the original code missed: *unmeasured* is not *occluded*. Nothing stands
between the sky and the sensor, and the column beneath it was measured all the way down to a
surface. Calling it UNKNOWN throws away a fact the geometry hands you for free.

### 3.2 A second defect: zero padding at the array boundary

Independently, input channels 2 and 3 of the network are `observed` and `1 - observed`. They
sum to 1 in every voxel that can occur. `nn.Conv3d(padding=1)` pads with **zeros**, so at the
array boundary it manufactures `observed = 0` *and* `unobserved = 0` — a state absent from
the training data. The response to it propagates inward, and the grid is only 32 voxels deep
in z, so the leak reaches a large fraction of the ceiling.

Signature: median final log-odds rose monotonically toward the boundary, **−1.06 at +1.9 m →
+0.13 at +4.3 m**, exactly where ground-truth occupancy falls to 0.01 %.
*(From `ceiling_fix.md`.)*

## 4. The fixes

### 4.1 Open sky is FREE, not UNKNOWN — `core/supervision/rawtarget.py:172`

In a column whose LiDAR returns reached a surface, voxels more than `SKY_MARGIN_VOX` above
the topmost return are marked free. A column with no return of its own borrows the highest
roof in a 15×15 neighbourhood:

```python
measured = (n_occ > 0).view(X, Y, Z)
top = torch.where(measured, zi.expand(X, Y, Z), neg).amax(dim=2)      # [X, Y]
top_local = F.max_pool2d(top.float()[None, None], kernel_size=2*SKY_NEIGHBOURHOOD_VOX+1,
                         stride=1, padding=SKY_NEIGHBOURHOOD_VOX)[0, 0].to(top.dtype)
reachable = top_local >= 0                       # some return within the window
sky = ((zi > (top_local + SKY_MARGIN_VOX).unsqueeze(2))
       & reachable.unsqueeze(2) & (sv == UNKNOWN) & ~measured)
sv[sky] = FREE
```

`reachable` is what keeps the rule honest: a window with no return anywhere frees nothing, so
it never invents free space where the sensor said nothing. The rule is derived from KITTI-360
LiDAR geometry alone; no target dataset is consulted. Top-layer supervision went
**1.12 % → 21.9 %**.

Targets were rebuilt under a redirected data root (`/media/SSD1/MINH_DATASETS/lingbot_gate8c1_sky`)
so the frozen originals survived, three seeds retrained from scratch, and the checkpoint and
threshold reselected by the gate's own rule on KITTI-360 drive 0006 only. Firewall
violations: 0 at every stage.

### 4.2 `padding_mode="replicate"` — `core/model/net.py:32`

`"replicate"` extends the volume with a copy of the edge voxel: the truthful statement
"outside the grid looks like its edge". It is stored in the checkpoint and defaults to
`"zeros"` so every earlier gate stays bit-identical.

An **inference-only** version of this was tried and rejected: the model had been *trained*
with the defect, so patching at test time is a train/test mismatch and cost source IoU
(11.33 → 8.99). The fix requires the retrain it got.

## 5. The ablation — only one fix did the work

`frozen_manifest_padfix.json` is the padding fix alone, on the *old* targets, so it isolates
§4.2 exactly. All three seeds, SemanticKITTI `past5`, 163 anchors:

| variant | seeds | median SC IoU |
|---|---|---|
| baseline (zeros padding, old targets) | 12.32 / 12.74 / 14.28 | **12.74** |
| padding fix only | 10.63 / 11.60 / 11.93 | **11.60** |
| padding fix + sky rule | 22.47 / 23.62 / 22.71 | **22.71** |

The padding fix alone is *worse*, consistently across all three seeds. **The entire
12.74 → 22.71 gain is the supervision fix.**

This refines what `ceiling_fix.md` says — it presents both changes together as "the fix"
without splitting them. The padding change remains defensible on its own terms (an impossible
input state is a real defect, and it lifted source AUROC), but it is not what moved the
target number. Caveat: no (zeros padding + new targets) variant exists on disk, so an
interaction cannot be ruled out; the comparison on *identical* targets is what is
unambiguous.

## 6. Verification

Predictions were dumped for the median seed and the score recomputed from the files on disk,
independently of the run that produced them: **163 anchors, SC IoU 22.71, P 34.49, R 39.95**,
with TP/FP/FN matching the eval run exactly.

The ceiling, measured from those files:

| height | % predicted occupied | % GT occupied |
|---|---|---|
| +2.1 m | 2.54 | 1.82 |
| +3.5 m | 1.10 | 0.06 |
| +4.3 m | **0.73** | 0.004 |

Against the pre-fix model (`ceiling_fix.md`): top layer **95.1 % → 0.73 %**, above 2 m
**22.24 % → 1.69 %**. False positives above 2 m fell from **71 % to 14.2 %** of the total.
AUROC rose **0.651 → 0.801**.

The error moved where the diagnosis predicted: **63.9 % of false positives now sit at or
below −1.0 m**, the road band — and those layers carry **82.9 % of all true positives**, so
they cannot simply be suppressed. That is surface-thickness error, not a boundary artefact.

Head-to-head against the released OccAny checkpoint, 161 shared target frames, scored by
OccAny's own `SSCMetrics`:

| | SC IoU | precision | recall | pred/GT |
|---|---|---|---|---|
| OccAny raw | **25.28** | 45.49 | 36.26 | 0.80 |
| OccAny pooled | 25.92 | 36.68 | 46.93 | 1.28 |
| ours (sky fix, seed 2) | **22.83** | 34.87 | 39.80 | 1.14 |
| ours pooled | 22.03 | 27.95 | 51.00 | 1.82 |

Ours wins **55 of 161** frames (34 %) — a consistent deficit, not a few bad scenes. Note that
pooling *helps* OccAny (+0.64) and *hurts* us (−0.80): a 3×3×3 max-pool on an already
over-thick prediction only adds false positives.

The 3D comparison figure is `fig_occany_kitti_3d.png` (six scenes, six panels each: ground
truth · what OccAny recovered · what we recovered · each method's false positives · a
who-recovers-what panel). The red sheet is gone; the remaining red sits on the ground plane.

## 7. Why Occ3D-nuScenes never showed the bug

The same checkpoint with the same defect scored 30.89 on Occ3D. The obvious explanation —
"Occ3D doesn't evaluate the sky" — is **wrong**. Both benchmarks devote about half their
scored voxels to the region above +2.0 m:

| | share of scored voxels above +2 m | GT occupancy there | top layer |
|---|---|---|---|
| SemanticKITTI 08 | 50.3 % | **0.40 %** | **0.004 %** (+4.3 m) |
| Occ3D-nuScenes | 48.2 % | **12.25 %** | **37.70 %** (+5.2 m) |

Both look overhead equally hard. SemanticKITTI finds empty sky; Occ3D finds *structure*.

The masks mean different things. SemanticKITTI's is `.invalid == 0` — LiDAR traversal — so
open sky above the road is marked valid **and free**, and the benchmark explicitly scores
emptiness. Occ3D's is `mask_camera` — camera visibility. Toggling it off is decisive:

| Occ3D | occupancy above +2 m | top layer |
|---|---|---|
| camera mask **on** (the protocol) | 12.25 % | 37.70 % |
| camera mask **off** | 2.90 % | 2.52 % |

Without the mask, Occ3D's sky is as empty as SemanticKITTI's. The camera mask selects *for*
structure overhead: high voxels that are camera-visible are near-field facades and canopy,
while open sky is either beyond a visible surface (occluded → ignored) or outside the
vertical FOV. So a voxel hallucinated overhead lands on real geometry a large fraction of the
time and scores as a **true positive**.

Prevalence multiplies the effect: Occ3D **22.61 %** against SemanticKITTI **7.82 %**. The same
absolute false-positive count costs roughly three times as much IoU when the positive class is
three times rarer.

Together, from the *same checkpoint with the same bug*:

| | precision | FP per TP | SC IoU |
|---|---|---|---|
| SemanticKITTI | 15.59 % | **5.41** | 12.32 |
| Occ3D | 67.14 % | **0.49** | 30.89 |

An 11× difference in over-prediction burden. And the fix moves each benchmark by exactly what
that predicts: SemanticKITTI 12.74 → 22.71 (**+78 %**), Occ3D 30.89 → 33.28 (**+7.7 %**).

**The implication is uncomfortable and should be carried forward.** Occ3D was not unaffected
because the geometry was sound there; it was unaffected because a camera-visibility mask plus
high prevalence forgives precisely the error SemanticKITTI punishes hardest. The strong Occ3D
number was never evidence of quality, and it coexisted with a catastrophic artefact. A related
warning points the same way: a blind prior from the same checkpoint scores 44.4 on Occ3D
against the model's 31.4 (`METHOD.md` §8.1.1b). SemanticKITTI is where this pipeline's
defects actually surface.

## 8. What was tried and rejected

**The neighbourhood extension (`sky2`).** Two thirds of the residual high false positives sit
*below* their own column's roof, so the column rule cannot reach them. An extension took the
maximum column top over a 15×15 window, letting an empty column borrow a roof from its
neighbours. Source quality improved markedly — AUROC 0.68 → 0.74, AP/prevalence 2.04 → 2.78 —
but the target results did not follow: SemanticKITTI a wash (22.83 → 22.93 median) and Occ3D a
**5.2-point regression** (33.28 → 28.12). The rule over-carves: recall falls (38.2 → 31.0)
faster than precision rises (72.3 → 75.0). **Not adopted.**

That result is worth keeping for its own sake: source AUROC rose while target IoU fell, so
source-domain ranking quality is not a reliable proxy for target transfer here.

## 9. What is still open

1. **The shipped headline numbers are the pre-fix model.** `artifacts/gate8c1/frozen_manifest.json`
   — the default `eval_target.py` reads — names `seed{0,1,2}_last.pt`, which carry no
   `padding_mode` key (so `"zeros"`). `README.md` and `REPRODUCE.md` therefore report
   **12.74 / 30.89**, not 22.71 / 33.28. Repointing them means re-running the full
   2 targets × 3 protocols × 3 seeds grid (~67 GPU-min).
2. **The sky model was trained on targets built with a *later*-corrected rule.** The `_sky`
   targets were built 2026-09-04; a bug in the sky rule itself — it carved through voxels a
   sweep had measured — was not fixed until commit `647bcf3` on 2026-09-07. Rebuilding and
   retraining could shift these numbers again.
3. **The road band.** 63.9 % of remaining false positives are at or below −1.0 m, in layers
   carrying 82.9 % of true positives. Closing that needs capacity or supervision density, not
   a bug fix. It is the highest-value remaining target.
4. **SemanticKITTI still loses** (22.83 vs OccAny's 25.28) and the oracle threshold only
   reaches 24.31, so the gap is capability rather than calibration.
5. On SemanticKITTI, OccAny's stored ground truth matches the volume our evaluator builds on
   only **1 of 161** frames (on Occ3D it was byte-identical). Both sides in §6 are scored
   against OccAny's stored label so the comparison is internally fair, but the discrepancy is
   unexplained.

## 10. Reproducing everything here

```bash
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
export G8C1_MANIFEST=$PWD/artifacts/gate8c1/frozen_manifest_sky.json
P=/home/minh/anaconda3/envs/cu128/bin/python

# the fixed model's score (~4 GPU-min per seed); swap the manifest for the ablation rows
$P -c "import os,sys; sys.path.insert(0,'.'); from core.run import eval_target as E; \
       E.run('semantickitti','past5',2,'cuda:1',None,'output/skyfix/eval')"

# dump per-anchor predictions and re-verify the score from the files
$P tools/skyfix/predict_semantickitti.py --seed 2 --out output/skyfix/pred_seed2 --device cuda:1

# head-to-head + the 3D comparison figure. The wrapper redirects the underlying tool's
# output directory, which otherwise overwrites the gate's frozen records in artifacts/gate8c1.
$P tools/skyfix/compare_occany_kitti_skyfix.py --out output/skyfix/figures \
     --seed 2 --n 6 --device cuda:1
```

The four model variants, all selected through `G8C1_MANIFEST`:

| manifest | change | SemanticKITTI `past5` median |
|---|---|---|
| `frozen_manifest.json` | none — the shipped headline | 12.74 |
| `frozen_manifest_padfix.json` | replicate padding only | 11.60 |
| `frozen_manifest_sky.json` | + open-sky-is-FREE — **adopted** | **22.71** |
| `frozen_manifest_sky2.json` | + neighbourhood roof — rejected | 22.93 |

## 11. The short version

The model was not broken and this was not a generalization failure. It was never told the sky
was empty, so it filled it. Telling it was the fix, and it was worth +78 % on SemanticKITTI.
The second benchmark stayed quiet for a structural reason worth remembering: its evaluation
mask scores visible structure rather than measured emptiness, so it barely charges for
hallucinating into open space.

---

### Sources

* `artifacts/gate8c1/ceiling_fix.md` — the original root-cause note and the rejected variants
* `artifacts/gate8c1/frozen_manifest{,_padfix,_sky,_sky2}.json` — the four variants
* `output/skyfix/` — this document's eval logs, prediction dump and figures
* `core/supervision/rawtarget.py`, `core/model/net.py`, `core/model/losses.py` — the code
* `reports/METHOD.md` §6, §8.2.1 — protocol and the prior false-positive analysis
