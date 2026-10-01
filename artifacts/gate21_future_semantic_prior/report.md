# Gate 21 — does privileged future supervision learn a semantic completion prior?

**OUTCOME C -- no transfer.**

## 1. The table that decides the experiment

KITTI-360 drive 0006, 590 anchors, mean ± s.d. over three head seeds, geometry seed 0 fixed.

| Population | B0 mIoU | C1 prior | C2 geometry | C3 full | C4 semantics | Δ full−B0 |
|---|---|---|---|---|---|---|
| E1 revealed | 6.39 | 1.22 ± 0.51 | 4.35 ± 0.02 | 6.44 ± 0.22 | 6.34 ± 0.11 | +0.05 |
| **E2 NEVER revealed** | 10.18 | **1.09 ± 0.82** | **4.98 ± 0.04** | **6.19 ± 0.25** | **6.35 ± 0.07** | **-3.99** |
| E0 all P0 | 8.80 | 1.13 ± 0.72 | 4.78 ± 0.03 | 6.28 ± 0.23 | 6.37 ± 0.07 | -2.52 |
| E1 P1 | 0.00 | 1.15 ± 0.27 | 3.33 ± 0.05 | 4.32 ± 0.06 | 4.17 ± 0.06 | +4.32 |
| E2 P1 | 0.00 | 0.76 ± 0.09 | 3.23 ± 0.08 | 4.05 ± 0.13 | 4.15 ± 0.07 | +4.05 |

`E2` is **1,866,958** voxels that are correctly completed, GT-occupied, never causally observed, and that **never become visible again** inside the frozen 10 s privileged window. No training target exists for any of them, in any drive, at any horizon. It is also the half where B0 is *strongest* (10.18 against 6.39 on `E1`), because never-revealed voxels sit closer to causal evidence.

## 2. The pre-registered decision

```
Δ E2 (mean of 3 seeds)   -3.989 pp
per seed                 -3.779, -4.341, -3.847
block bootstrap 95 % CI  [-5.028, -3.130]  (P(Δ>0) = 0.000)
A1  Δ ≥ +1.00 pp         not met
A2  all seeds positive   not met
A3  CI lower > 0         not met
C1  Δ ≤ +0.25 pp         FIRED
C2  a seed negative      FIRED
outcome                  C
```

The rule and these thresholds were written and hashed to `82c42383a456ce39…` before the first training run; `validate.py` check 1 recomputes the hash and check 2 requires the code's thresholds to appear verbatim in the document.

## 3. Interpretation

> **Privileged future supervision does not generalise from future-revealed hidden voxels to never-revealed hidden voxels under the simple completion model.**

The pre-registered statement is the one above, and the measured numbers say something more specific than it does.

The head *does* learn the pseudo-target: training top-1 agreement reaches 67 % on `U^R`, and on `E1_P1` and `E2_P1` — the voxels where B0 is wrong by construction and scores 0.00 — it recovers 4.32 and 4.05 mIoU, at 29.7 % and 22.8 % accuracy. It is producing real semantics on hidden space, on both halves, at almost the same rate.

What it never learns is **when to defer**. On `E1`, where its supervision lives, it breaks 157,230 of B0's correct answers and fixes 157,620 of its wrong ones — a net +390 voxels out of 876,254, an almost exact wash (+0.05 pp mIoU). On `E2` the same trade at the same rates loses, 366,911 broken against 216,849 fixed, because `E2` is where B0 is *right more often* (49.1 % against 39.5 % on `E1`). The -3.99 pp is not a failure to produce semantics for never-revealed space; it is a failure of the model's errors to be complementary to the baseline's.

The distance profile says the same thing from the other side: the loss is concentrated at 0-1 m (-6.29 pp over 851,732 voxels, 46 % of `E2`), where nearest-evidence copying is at its best, and vanishes at 2-4 m (-0.43 pp) where copying has degraded. A learned scene prior reaches parity with copying exactly where copying stops working, and never passes it.

This also is not an artefact of the metric. A class-collapse story predicts accuracy up and mIoU down; here the model loses top-1 as well (-8.04 pp against -3.99 pp of mIoU), so it is giving a worse answer rather than a differently-shaped one. Its concentration against B0 is mixed either way (§6b): fewer classes predicted (15 of 18 against 18) and lower entropy, but a smaller dominant share. The one place a metric effect is real is things: 3 of the 6 thing classes fall below 0.5 IoU where B0 did not (motorcycle 4.67 → 0.00, truck 11.95 → 0.05, other-vehicle 1.06 → 0.46), which B0 occasionally hits by copying, and that costs the thing-restricted mIoU -15.68 pp on 1.3 % of the voxels.

## 4. The five questions of section 21

**Q1 — can the model fit/use the future pseudo-supervision on E1?** Yes. Training top-1 agreement with the pseudo-target reaches 71.9 % and `E1` mIoU moves +0.05 pp (6.39 → 6.44). On `E1_P1`, where B0 scores 0.00 by construction, it reaches 4.32.

**Q2 — does that transfer to E2, which has no future supervision?** No: -3.99 pp, against the pre-registered +1.00 pp bar, with 95 % interval [-5.03, -3.13].

**Q3 — does geometry carry semantic-completion information beyond the causal evidence?** On `E2`: B0 10.18, C1 prior 1.09 / C2 geometry 4.98 / C3 full 6.19 / C4 semantics 6.35.

Two separate answers, and they point in opposite directions. **On its own, geometry carries real semantics**: C2 sees no semantic evidence at all — no evidence feature, no evidence mask, not even the distance to the nearest evidence — and still reaches 4.98 mIoU against the 1.09 available from class frequency alone, so the frozen completion features plus the voxel's position in the scene predict class well above chance. **Given the causal evidence, geometry adds nothing**: C4 (evidence + occupancy, no latent features) reaches 6.35 and C3 (both) 6.19 — a difference of -0.16 pp, inside the seed spread. Whatever the trunk knows about hidden class is already recoverable from where the causal evidence is and what it says.

**Q4 — where does transfer occur?**

- distance: best **2-4 m** (-0.43 pp), worst **0-1 m** (-6.29 pp)
- thing -15.68 pp, stuff -4.31 pp
- best classes: other-ground +0.85, traffic-sign +0.00, pole -0.03
- worst classes: building -19.72, truck -11.90, sidewalk -9.04

**Q5 — is this consistent with learning p(y_hidden | completed geometry, observed semantics) rather than reproducing class priors?** The class-prior control reaches 1.09 mIoU on `E2` — a single constant class for every voxel — against 6.19 for the full model and 10.18 for B0. The full model is far above the prior, so whatever it learned is voxel-specific. The selection-bias table of §6 bounds how much of any difference could be a population artefact.

## 5. E2 by distance from causal evidence

| bin | voxels | B0 | C1 prior | C2 geometry | C3 full | C4 semantics | Δ C3 full |
|---|---|---|---|---|---|---|---|
| 0-1 m | 851,732 | 13.04 | 1.10 | 5.52 | 6.75 | 7.11 | -6.29 |
| 1-2 m | 393,079 | 10.62 | 1.09 | 6.00 | 8.25 | 8.33 | -2.37 |
| 2-4 m | 301,362 | 8.78 | 1.32 | 6.62 | 8.35 | 8.27 | -0.43 |
| 4-8 m | 121,469 | 4.97 | 1.12 | 3.81 | 4.46 | 3.96 | -0.50 |
| >8 m | 199,316 | 3.35 | 0.92 | 1.76 | 0.86 | 0.40 | -2.49 |

Gate 20 found future supervision unusually strong far from causal evidence (revealability rose from 17.3 % at 0–1 m to 48.2 % beyond 8 m). This table says whether the model carried that into the never-revealed population or only smoothed B0's propagation.

## 6. E2 per class

| class | support | thing | B0 | C3 full | Δ |
|---|---|---|---|---|---|
| car | 21,089 | thing | 10.63 | 8.31 | -2.32 |
| bicycle | 48 | thing | 0.31 | 0.00 | -0.31 |
| motorcycle | 94 | thing | 4.67 | 0.00 | -4.67 |
| truck | 1,796 | thing | 11.95 | 0.05 | -11.90 |
| other-vehicle | 2,087 | thing | 1.06 | 0.46 | -0.60 |
| person | 53 | thing | 0.26 | 0.00 | -0.26 |
| road | 171,355 | stuff | 25.74 | 25.29 | -0.45 |
| parking | 127,457 | stuff | 4.42 | 1.17 | -3.25 |
| sidewalk | 94,056 | stuff | 11.52 | 2.48 | -9.04 |
| other-ground | 71,452 | stuff | 3.63 | 4.48 | +0.85 |
| building | 358,997 | stuff | 36.63 | 16.91 | -19.72 |
| fence | 13,688 | stuff | 4.12 | 0.13 | -3.99 |
| vegetation | 754,101 | stuff | 56.54 | 48.12 | -8.42 |
| terrain | 123,699 | stuff | 0.21 | 0.03 | -0.18 |
| pole | 182 | stuff | 0.03 | 0.00 | -0.03 |
| traffic-sign | 33 | stuff | 0.00 | 0.00 | +0.00 |
| other-structure | 123,924 | stuff | 10.32 | 3.90 | -6.42 |
| other-object | 2,847 | stuff | 1.22 | 0.11 | -1.11 |

`E2` mIoU is the unweighted mean of these 18 IoUs, so each class contributes its own delta divided by 18. The largest contributions are **building** -1.10 pp, **truck** -0.66 pp, **sidewalk** -0.50 pp, **vegetation** -0.47 pp; 10 of 18 classes move the metric by at least 0.05 pp. The change is therefore spread across the vocabulary rather than carried by one or two classes.

Aggregated over `E2`: thing B0 20.05 → 4.37 (-15.68 pp, 25,167 voxels); stuff B0 12.92 → 8.61 (-4.31 pp, 1,841,791 voxels). These two are mIoU over the voxels whose GT class is a thing or a stuff class respectively, each averaged over the classes with support inside that restricted region (6 and 12), so they are separate measurements and do not recombine into the `E2` figure. No class is hidden: every one of the 18 is listed, including those B0 or the model never predicts.

## 6b. Does the model collapse onto the easy classes?

mIoU averages over the classes with GT support, so a model that answers only with the few dominant classes can gain top-1 accuracy and lose mIoU. These are the numbers that tell the two apart, on `E2`.

| method | E2 mIoU | E2 top-1 | classes predicted / 18 | classes ≥1 % of predictions | dominant class | its share | prediction entropy |
|---|---|---|---|---|---|---|---|
| B0 | 10.18 | 49.06 % | 18 | 9 | vegetation | 48.9 % | 2.48 bits |
| C1 prior | 1.09 | 19.58 % | 1.0 | 1.0 | road | 100.0 % | 0.00 bits |
| C2 geometry | 4.98 | 43.13 % | 11.7 | 3.0 | vegetation | 70.1 % | 1.11 bits |
| C3 full | 6.19 | 41.03 % | 15.3 | 6.7 | vegetation | 45.1 % | 2.01 bits |
| C4 semantics | 6.35 | 41.35 % | 14.3 | 5.3 | vegetation | 43.2 % | 1.99 bits |

Top-1 and mIoU move the same way on `E2` (-8.04 pp and -3.99 pp), which settles the question this table exists to answer. A class-collapse story predicts the opposite signature — accuracy up, mIoU down, because answering only with the frequent classes is the accuracy-maximising move and the one that zeroes out the rare ones. Both fell, so the loss is a worse answer, not a differently-shaped one.

The concentration numbers are mixed and are reported in full rather than selectively: C3 full predicts 15.3 of the 18 classes against B0's 18 and has a *smaller* dominant share (45.1 % against 48.9 %), but a *lower* prediction entropy (2.01 against 2.48 bits). It is neither cleanly more nor cleanly less concentrated than B0. C2, which sees no semantic evidence, is the one that genuinely collapses: 70 % of its predictions are a single class.

mIoU was pre-registered as the primary metric and stays primary; this table says what it is measuring.

## 6c. What the model does to B0's answers

`P1` is defined as the voxels where B0's class is wrong, so each population splits exactly into the part B0 already gets right and the part it does not. The model's accuracy on each half follows from the two top-1 numbers, and it separates "uniformly worse" from "trades correct answers for fewer fixes".

| population | B0 right | B0 wrong | keeps of B0-right | fixes of B0-wrong | net voxels |
|---|---|---|---|---|---|
| E1 revealed | 346,046 | 530,208 | 54.6 % | 29.7 % | +390 |
| E2 NEVER revealed | 916,021 | 950,937 | 59.9 % | 22.8 % | -150,062 |

On `E2` the model keeps 59.9 % of the 916,021 answers B0 already had right and fixes 22.8 % of the 950,937 it had wrong — about 366,911 broken against 216,849 fixed. That is the whole of the top-1 loss, and it is not a metric artefact: the model is producing a different answer, not a more concentrated one (see §6b).

## 7. Selection bias between the training and decision populations

```
$U^R$ train (loss)   2,280,255 voxels
$U^N$ train          6,099,264
$U^R$ dev    (E1)      876,254
$U^N$ dev    (E2)    1,866,958

total-variation distance from $U^R$ train to E2
  class                26.3 %
  distance-to-evid.    23.0 %
  height               11.1 %
  range from sensor    46.8 %
  thing / stuff         0.8 %   (thing 2.13 % -> 1.35 %)
classes whose frequency changes by more than 2x: 8 of 18
```

The two populations are genuinely different, as Gate 20 predicted. Nothing is corrected for this — section 12 asks for the measurement, and an inverse-frequency weighting would be a separate ablation, not a silent change — but any failure has to be read against it.

## 8. Training curves

| step | train loss | tgt top-1 | E1 mIoU | E2 mIoU | Δ E2 |
|---|---|---|---|---|---|
| 500 | 2.1490 | 58.9 % | 5.72 | 6.25 | -5.69 |
| 1000 | 1.7856 | 72.1 % | 6.26 | 6.43 | -5.51 |
| 1500 | 1.8461 | 72.5 % | 6.30 | 6.69 | -5.24 |
| 2000 | 1.9317 | 67.6 % | 6.51 | 6.98 | -4.96 |
| 2500 | 1.8359 | 64.4 % | 6.49 | 7.04 | -4.89 |
| 3000 | 1.8621 | 71.9 % | 6.56 | 7.16 | -4.77 |

**Does E2 improve early and then overfit?** No. Averaged over the three seeds, `E2` mIoU rises monotonically across the whole schedule, 6.25 → 7.16, and its best step is the last one. Individual seeds peak earlier (seed 0: 3000, seed 1: 1000, seed 2: 2000) but that is sampling noise on a 50-anchor subsample, not a trend. The important part is the level, not the shape: at its best the model is still 4.77 pp below B0 on this subsample (B0 11.94 there), and the curve is flattening, so the deficit is not something the remaining schedule was about to close. That is evidence against more epochs as a remedy, which is one reason section 25 forbids proposing one. The reported model is the final step either way: selecting a checkpoint on a dev metric is exactly the selection the pre-registered rule forbids.

Measured on a fixed 50-anchor stride-12 subsample of 0006, swept over saved checkpoints AFTER training, so the training process never opened a file of the development drive (`drive_0006` is on its firewall's forbidden list). No checkpoint is selected from these: the reported model is the final step, as pre-registered.

## 9. The composite scene read

Over all 558,778,307 labelled voxels of 0006, with the teacher's own class at evidence voxels and the method's class elsewhere: B0 1.55 mIoU, C1 prior 0.70 (-0.85), C2 geometry 1.57 (+0.02), C3 full 1.64 (+0.09), C4 semantics 1.60 (+0.05). This is the no-regression check of section 13.3 and decides nothing.

## 10. Verification

`logs/validation.json`: **12 checks, all_ok = True**

- PASS — decision rule unchanged since it was hashed
- PASS — config thresholds appear verbatim in decision_rule.md
- PASS — no input channel is derived from a future frame
- PASS — the head input opens no file of the Gate 20 or Gate 21 reveal caches
- PASS — the prediction is bit-identical when the reveal mask is corrupted
- PASS — every training run's firewall is clean and forbids the dev drive and all GT
- PASS — the frozen geometry network is identical before, during and after every run
- PASS — E0 is bit-exactly the frozen region completed_never_observed
- PASS — the populations reproduce Gate 20's frozen counts
- PASS — the training targets are a well-formed subset of E0 with Gate 20's own readout
- PASS — C1 is a class prior: one constant class, from the training distribution
- PASS — every trained run is a distinct model

The leakage guarantee has four independent parts. (1) No input channel is derived from a future frame: the complete list is `trunk, hard_S, hard_F, occupancy, height, range, evidence_mask, evidence_dist, evidence_proj`. (2) Building that input under a file audit forbidding both reveal caches opens nothing from them. (3) The prediction is bit-identical when the reveal mask is shuffled or set to all-hidden, so the function really is the same for `E1` and `E2`. (4) Each training step fingerprints the head input and the whole causal state before the target file is opened and again after the backward pass, and requires both unchanged.

## 11. Provenance

```
gate                     gate21.1
git commit               31ee257ad5dc
config.yaml sha256       7ac918729ea19a7f...
training target sha256   3a6c7969b59071dc...
gate 20 rule sha256      086812519dae2fed...
decision rule sha256     82c42383a456ce39892b32b4a13b39867b1143b55c011c7418689e72a66deb38
geometry                 CompletionUNet seed 0 step 6000, frozen, 967,322 params
teacher                  Trident-H, 18 KITTI-360 phrases, logit scale 40
reveal definition        Gate 20 'visible' / 'mean_probability', 10 s window
training population      2,280,255 voxels over 1274 anchors of 0003/0007/0010
head parameters          C1 prior 18, C2 geometry 223,506, C3 full 257,394, C4 semantics 256,498
training                 3000 steps, AdamW lr 1e-3 wd 0.01, cosine, grad clip 1.0, 2 full-grid anchors/step
seeds                    [0, 1, 2]
dev evaluation           590 anchors of 0006, final checkpoint, no selection
bootstrap                20-anchor blocks, 2000 resamples, seed 0
```

Reproduce:

```bash
PYTHONPATH=$PWD python -m tools.future_semantic_prior.targets --drive <drive> --device cuda:0
PYTHONPATH=$PWD python -m tools.future_semantic_prior.train    --control C3_full --seed 0 --device cuda:0
PYTHONPATH=$PWD python -m tools.future_semantic_prior.curves   --control C3_full --seed 0 --device cuda:0
PYTHONPATH=$PWD python -m tools.future_semantic_prior.evaluate --control C3_full --seed 0 --device cuda:0
PYTHONPATH=$PWD python -m tools.future_semantic_prior.selection_bias
PYTHONPATH=$PWD python -m tools.future_semantic_prior.validate --device cuda:0
PYTHONPATH=$PWD python -m tools.future_semantic_prior.analyze
PYTHONPATH=$PWD python -m tools.future_semantic_prior.report
```

## 12. Recommended next step

**Close this branch.** Section 25 is explicit that a bigger network, more epochs, more data, a stronger teacher or a transformer are not automatic rescues, and none of them is recommended here. What the results rule out is worth recording, because it is what a reconsideration from first principles has to start from:

- **Not capacity.** The head reaches 67 % agreement with its own target and recovers 4.05 mIoU where B0 is wrong by construction. It is not failing to learn; it is learning something that is not better than copying.
- **Not the direction of the formulation either.** B5 tested the complementary design — an additive correction that *starts* exactly at B0 and can only move away from it — on this same population, and lost 2.38 pp. Gate 21 tests replacement and loses 2.52 pp on the same population. Both the "replace B0" and the "correct B0" formulations have now failed, so the problem is not where the prediction starts.
- **Not the revealed/never-revealed distinction.** The model performs almost identically on both halves where B0 is wrong (4.32 on `E1_P1`, 4.05 on `E2_P1`). The gate's framing assumed the interesting failure would be revealed-versus-never-revealed; it is not. It is learned-versus-copied.
- **What is left unexplained** is how small an effect any learned mechanism has ever had on this population, and how asymmetric. Against B0's 8.80 mIoU on `E0`: Gate 21 C3 full 6.28 (-2.52), B5 6.42 (-2.38), Gate 15 B1 6.54 (-2.26), B4 8.22 (-0.58), pilot B1 9.13 (+0.33), pilot B2 9.21 (+0.41), Gate 16 B3 9.38 (+0.58) — each read from that gate's own artifact, not quoted from memory. (The two entries named B1 are different methods that happen to share the label: Gate 15's feature-propagation head and the pilot's learned evidence association.) 3 of the 7 beat B0, by +0.33 to +0.58 pp; 4 lose to it, by -0.58 down to -2.52 pp, the largest loss being Gate 21 C3 full's. So a parameter-free nearest-evidence copy is not literally the strongest thing in this stack — Gate 16's B3 beats it — but nothing has beaten it by a margin that would carry a paper, and the losses are several times larger than the wins. Why *that* asymmetry holds, rather than which inductive bias to try next, is the question to answer before building anything else. Gate 20's constraint stands over whatever comes next: the camera's field of view, not the trajectory, caps revealability at 27 %, and the teacher itself reaches only 19.01 mIoU where it is measurable.

## 13. Figures

- `figures/A_training_curves.png` — training curves
- `figures/B_populations.png` — populations
- `figures/C_distance_E2.png` — E2 by distance
- `figures/D_per_class_E2.png` — E2 per class
- `figures/E_selection_bias.png` — selection bias

## 14. What this gate did not do

No change to LingBotMap, CompletionUNet, the geometry, Trident-H, B0, the voxel grid, the occupancy threshold, the evaluation masks, the class mapping, Gate 20's visibility definition or Gate 20's target construction. No SemanticKITTI, no Occ3D, no nuScenes, no extra cameras, no fisheyes. No transformer, no IGGT, no region or object queries, no diffusion, no CLIP/DINO feature regression, no learning-rate sweep, no architecture search, and no dataset growth. Future information was used to build the training loss and to split the evaluation population, and nowhere else.
