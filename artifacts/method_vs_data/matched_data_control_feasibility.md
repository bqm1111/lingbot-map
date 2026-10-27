# Matched-data control — is retraining OccAny necessary, and is it possible?

**Written 2026-10-20.** Brief §10. No OccAny training was launched; this is a feasibility and
scientific-value assessment, as instructed.

> ## Verdict
>
> **Retraining OccAny on matched data is mechanically possible, expensive, and would still
> not isolate the architecture** — because matching the *dataset* does not match the
> *supervision drawn from it* or the *pretrained initialization*, and those differ more than
> the data does.
>
> **A stronger control is already measured and costs nothing: the hard map scored alone.**
> It gives 8.82 SC IoU on SemanticKITTI against OccAny's causal 24.02, which settles the
> "you only win by remembering" objection directly. §4.

---

## 1. Mechanical feasibility

OccAny's dataset mixture is a free-form `--train_dataset` string
(`"10000 @ WaymoSeqMultiView(...) + 5000 @ VKittiSeqMultiView(...) + ..."`). Dropping four of
the five arms to leave one is a one-line edit requiring no code change. So **subset training
is mechanically trivial**, and the brief's question 10 ("availability of configurations for
training on smaller source subsets") is answered: yes, by construction.

What it would cost, from the released checkpoint's own `args`:

| | OccAny as trained | what we have |
|---|---|---|
| GPUs | **16** (`world_size=16`) | 4 × RTX PRO 6000 Blackwell (95 GiB each) |
| per-GPU batch | 2 | — |
| accumulation | 2 | would need 8 to hold effective batch 64 on 4 GPUs |
| effective batch | 64 | 64, reachable |
| epochs | 100 | — |
| sample draws | 4,500,000 (45,000 × 100) | — |
| optimizer steps | ≈70,300 | — |
| warm start | a prior recon-only OccAny checkpoint, **budget unrecorded** | we would have to start from MUSt3R only, which is **not** what they did |

Effective batch 64 is reachable on 4 GPUs with accumulation. **Wall-clock cost is not
estimable from local evidence** — no OccAny training log exists here, and inference-cost
measurements do not transfer to training. What is certain: a faithful replication is a
16-GPU-shaped job on 4 GPUs, at 4.5 M sequence draws of 5–10 views at 512×(160–336), and the
warm start cannot be reproduced at all because that checkpoint is not published and its own
budget is not recorded.

So a "matched" run would differ from the real OccAny in initialization even if the data
matched — and initialization is the one thing the brief explicitly says to record rather than
silently change.

---

## 2. Why matched data would still not isolate the architecture

Three confounds survive a dataset match. The first is the serious one.

**2.1 Matched data ≠ matched supervision.** The two methods extract different supervision from
the same bytes:

| | OccAny | ours |
|---|---|---|
| objective | 3D pointmap regression (`Regr3D_multiview`, L21, `?avg_dis`) + pose loss + SAM2 feature distillation | occupancy classification, focal + Dice |
| supervision signal | per-pixel 3D points from camera-projected depth | occupied / free / unknown from **ray carving with a sensor origin** |
| what the data must provide | a depth map per camera | raw sweeps **plus the sensor origin** |

This is not a technicality — it is exactly what blocked A2. DDAD supplies camera-projected
depth maps perfectly well, which is why OccAny trains on it; it does not supply a recoverable
sensor origin, which is why we cannot. **A dataset can be "matched" for OccAny and unusable
for us at the same time.** Any matched-data experiment must therefore state which supervision
each side could actually draw, and the two will not be equal.

**2.2 Matched data ≠ matched pretraining.** OccAny fine-tunes MUSt3R_512 (1.69 GB) and was
additionally warm-started from a prior OccAny run of unrecorded size. Our CompletionUNet
trains from scratch (967 k parameters) but sits on frozen LingBotMap and MoGe-2. Neither side
is "from scratch" as a *system*; both carry pretrained geometry priors of different origins.
Equalizing the completion-training corpus leaves these untouched.

**2.3 Matched data ≠ matched capacity.** 967 k trained parameters against a MUSt3R-scale
decoder. A dataset-matched comparison between models three orders of magnitude apart in
trained capacity attributes nothing to architecture; it measures a joint
architecture-plus-capacity-plus-pretraining difference.

**Consequence.** Outcome D in brief §11 — "both methods change substantially under matched
source-data training" — is the only outcome a matched-data run could interpret cleanly, and
even then it would license a statement about *interaction*, not about architecture alone.

---

## 3. Scientific value, honestly weighed

**What a matched-DDAD OccAny run would buy:** a defensible answer to "does OccAny also degrade
when restricted to one source dataset?" That is genuinely interesting, because our A1 result
shows *we* do, and it is currently unknown whether that is a property of single-source training
in general or of our recipe.

**What it would not buy:** the method claim. §2 shows three confounds survive.

**What makes it worse right now:** our side of the matched pair cannot be built on DDAD at all
(§2.1). So a matched-DDAD experiment would compare OccAny-on-DDAD against *nothing* of ours.
The control requires a dataset both methods can train on, which means **ONCE or PandaSet
first** — and our arm on that dataset is itself not yet authorized.

**Ordering follows from this.** The matched-data control is not the next experiment; it is at
best the one after the substitute arm exists.

---

## 4. The control that is already measured, and costs nothing

Brief §10 also asks for internal ablations isolating the LingBotMap streaming contribution.
One of them is already on disk, from the Waymo Phase 3 run:

**`map_only` — the five-frame causal hard map scored on its own, no CompletionUNet:**

| benchmark | hard map alone | + CompletionUNet (A0) | learned completion adds | OccAny causal 1-frame |
|---|---:|---:|---:|---:|
| SemanticKITTI seq 08 | **8.82** | **26.17** | **+17.35** | 24.02 [reproduced] |
| Occ3D-nuScenes val | **10.04** | **31.22** | **+21.19** | — |

Source: `artifacts/waymo_arm/results/waymo_vs_kitti360.json`, `map_only_iou`, which is
**bit-identical across the KITTI-360 and Waymo arms** (8.820500920980095 both) — as it must
be, since the hard map does not depend on the completion checkpoint. That identity is itself
the validity check on the ablation.

**This answers the "you only win by remembering" objection on its own terms.** Accumulated
reconstruction alone scores 8.82 on SemanticKITTI — **15.2 points below** OccAny's causal
24.02. Memory is not what beats them. The learned completion supplies about two thirds of our
absolute SC IoU on both benchmarks (66 % and 68 %).

**It does not answer where the *marginal advantage* lies**, which is a different question and
is the subject of `observed_vs_completed_analysis.md`. Both are needed: this ablation bounds
how much of our score completion produces; the strict split locates where our lead over
OccAny sits.

### Other internal ablations available without new training

| ablation | supported by existing code? | isolates |
|---|---|---|
| hard map alone | **yes, measured** | accumulated reconstruction |
| hard map + CompletionUNet | **yes, measured** (= A0) | the completion network's contribution |
| CompletionUNet on a 1-frame hard map | plausible — the mapper accepts any frame count — **not run** | how much of completion depends on temporal accumulation vs single-view geometry |
| frozen geometry, Waymo checkpoint | **yes, measured** (A1) | sensitivity of completion to its training source |

The third row is the one worth adding: it would separate "the CompletionUNet needs five frames
of map" from "the CompletionUNet needs any map". It requires no new training, only inference
with a truncated clip, and it is the cheapest remaining method-attribution experiment. **It has
not been run and is not claimed.**

---

## 5. Recommendation

1. **Do not retrain OccAny now.** It is expensive, cannot reproduce their warm start, and
   cannot be matched on our side for DDAD.
2. **Report the `map_only` ablation.** It is measured, it is validated by a bit-identity check,
   and it directly refutes the strongest objection to the method claim.
3. **Run the 1-frame-map ablation next** if method attribution is the priority — inference
   only, no training, and it is the one remaining control that separates accumulation from
   completion inside our own system.
4. **Revisit the matched-data control only after** a substitute source arm (ONCE or PandaSet)
   exists, since that is the first point at which a matched pair is constructible.
5. **If the matched-data control is eventually run**, record on both sides: exact sequence IDs,
   which supervision each method could draw from them, pretrained initializations, trained
   parameter counts, and the compute budget — and do not adjust either architecture or recipe
   to force equality, per the brief.
