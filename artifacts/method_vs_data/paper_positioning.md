# Paper positioning: what the measurements support, and what they do not

**2026-10-28.** Written from `runtime_comparison.md` and `observed_hidden_comparison.md` of the
same date, together with `method_vs_data_reconciled_conclusions.md`,
`training_supervision_provenance.md`, `waymo_canonical_rescore.md`, `one_frame_ablation.md`,
`once_vs_occany.md` and `occany_training_audit.md`. Geometry only.

Nothing here is a new measurement. Where a number appears it is traceable to one of those
files, and the two new ones name the JSON it came from.

---

## 1. The five claims, kept apart

These are distinct and none implies another. The table is the whole argument in short form;
§§2–6 justify each row.

| claim | status | where it belongs |
|---|---|---|
| **A. Competitive benchmark performance** | **supported, with the convention and comparator named** | main paper |
| **B. Hidden-geometry completion** | **supported, and it is the main finding — but it is a *decomposition*, not a blanket win** | main paper |
| **C. End-to-end computational efficiency** | **supported and now directly measured** | main paper |
| **D. Source-subset robustness** | **refuted on one clean arm; "the method contributes nothing" does not follow** | limitations / appendix |
| **E. Limited task-specific training requirement** | **supported as a statement about the *trainable head*, and about nothing else** | one sentence in the main paper, detail in the appendix |
| **F. Architectural novelty and causality** | **not established by any experiment here** | explicitly out of scope; appendix discussion only |

## 2. Claim A — competitive benchmark performance

SemanticKITTI 08, 801 canonical anchors, OccAny's own labels and scorer, both conventions:

| | ours (KITTI-360) | OccAny 1 causal frame | OccAny 5 forward frames |
|---|---:|---:|---:|
| SC IoU, raw | **26.40** | 20.32 | 25.21 |
| SC IoU, pooled (their published convention) | **26.34** | 24.06 | 25.93 |
| Δ vs ours, raw | — | +6.08 [+4.82, +7.33] \* | +1.19 [+0.35, +2.09] \* |
| Δ vs ours, pooled | — | +2.28 [+1.23, +3.33] \* | +0.41 [−0.34, +1.20] **ns** |

Occ3D-nuScenes val, 4,219 anchors, against their forward arm only (no causal nuScenes run
exists): **31.94 vs 20.66 raw (+11.28 \*) and 41.56 vs 23.55 pooled (+18.01 \*)**.

**What may be said.** Against OccAny's only causal mode we are ahead on SemanticKITTI in both
conventions. Against their non-causal forward mode we are ahead raw and **tied pooled** on
SemanticKITTI, and ahead in both conventions on Occ3D. The honest summary is *competitive on
SemanticKITTI and clearly ahead on Occ3D*, not *state of the art*.

**What must be said alongside.** The five-frame comparison is **not temporally matched**: their
forward arm reads `t … t+40`, 4.0 s of future, which our system never sees. And the
SemanticKITTI pooled result against that arm contains zero — no difference detected, which is
not equivalence.

## 3. Claim B — hidden-geometry completion, decomposed

This is the strongest and most defensible contribution, because it rests on a partition built
from ground truth alone and because it reports its own negative results.

SemanticKITTI, strict visibility from the five permitted causal frames, four mutually exclusive
regions that reconstruct the global confusion counts exactly. Against OccAny's **causal** arm:

| region | % of GT occupied | Δ raw | Δ pooled |
|---|---:|---|---|
| `anchor_observed` | 4.76 % | **−4.29 \*** | **−2.70 \*** |
| `earlier_only` | 6.18 % | +4.98 \* | +0.65 **ns** |
| `never_observed ∩ frustum` | **68.05 %** | +3.11 \* | **−0.25 ns** |
| `never_observed ∖ frustum` | 21.01 % | **+27.14 \*** | **+22.70 \*** |
| `never_observed` (union) | 89.06 % | +8.54 \* | +4.77 \* |

And against the hard map with the learned head removed:

| region | Δ ours − map only, raw | pooled |
|---|---|---|
| `anchor_observed` | +0.32 **ns** | −0.56 \* |
| `never_observed` | **+22.21 \*** | **+14.71 \*** |

**The three findings, in the order they should appear in a paper.**

1. **89 % of SemanticKITTI's GT occupancy was never visible to any permitted causal frame**, so
   the benchmark is a hidden-geometry task, and a completion method should be evaluated as one.
2. **The learned head contributes nothing on visible surface and everything in hidden space**
   (+0.32 ns vs +22.21 raw). Accumulated reconstruction alone scores 8.79 — 11.5 points *below*
   OccAny's causal arm. The claim "we win by remembering" is false and is already retracted in
   the record.
3. **On visible surface we are worse than OccAny**, significantly, in both conventions. This
   belongs in the main paper, not in a limitations paragraph.

**The narrowing the data forces.** Under OccAny's **published pooled** convention, against their
causal arm, the never-observed advantage lives entirely in the 21 % of hidden surface **outside
the camera frustum**; inside the frustum — which is 68 % of the GT occupancy — no difference is
detected (−0.25, interval contains zero). Raw scoring does show a small significant in-frustum
hidden win (+3.11). So:

> **Permitted claim.** "In geometry the causal input never observed, our method is ahead of the
> comparable causal baseline in both conventions; that advantage is concentrated outside the
> camera's view volume, and inside it survives only under raw scoring."
>
> **Forbidden claim.** "Our method completes occluded geometry better." The out-of-frustum
> advantage is a **coverage** result: out of frustum, OccAny's causal arm runs recall **8.38** at
> precision 62.89 — it barely predicts there at all — against our recall 52.52. Predicting where
> a competitor does not predict is a real benchmark gain and a real capability, and it is not
> evidence about reasoning through occlusion.

Occ3D supports claim B more strongly — we win every region in both conventions — but **the
comparator there is non-causal**, and **54.5 % of its GT occupancy is outside the single front
camera's frustum**, so its aggregate is dominated by the same coverage effect. Use Occ3D as
corroboration, not as the primary evidence for B.

## 4. Claim C — end-to-end computational efficiency

Measured on one idle GPU, 50 fixed anchors, 3 warm-up + 3 timed passes, batch 1, the boundary
"decoded RGB in host memory" → "occupancy grid on host".

| configuration | latency | predictions/s | peak allocated |
|---|---:|---:|---:|
| ours, 5 causal frames | **373.9 ms** | 2.67 | 11.78 GiB |
| ours, 1 causal frame | 164.4 ms | 6.08 | 11.78 GiB |
| OccAny, 1 causal frame | 1320.0 ms | 0.76 | 10.41 GiB |
| OccAny, 5 forward frames | 1704.4 ms | 0.59 | 12.66 GiB |

**4.56 ×** faster than the configuration it is compared against (3.78 × on the most generous
accounting for them), **8.03 ×** one causal frame to one causal frame, at comparable memory.

**The accuracy–latency trade is favourable and can be stated as such**, because the figure plots
each point's accuracy and latency from the *same* configuration: our five-frame point is above
and to the left of both OccAny points on raw, and above-left of their causal point on pooled.
Our one-frame configuration is 8 × faster than their one-frame configuration and not worse in
either convention (23.08 / 24.53 against 20.32 / 24.06).

**Two things must be said with it, or the claim is misleading.**

1. **Ours is the larger model.** 1,485,120,083 inference parameters against 651,129,550 —
   **2.28 ×** more. The frozen LingBotMap and MoGe-2 are counted, not treated as free. The speed
   comes from synthesising **no** novel views where OccAny generates 150 and voxelises 12.3 M
   points, not from being small.
2. **Future-frame waiting is a separate axis and is never folded in.** From the anchor instant:
   ours 0.374 s, OccAny's forward arm 4.0 s of waiting plus 1.704 s of compute.

## 5. Claim D — source dependence

**What is demonstrated.** One clean source-only substitution exists: the Waymo arm, which
preserved the frozen 21-frame / ~10 s supervision window and changed only the corpus. It costs
**2.84 / 2.97** SC IoU on SemanticKITTI and **9.09 / 3.51** on Occ3D against the KITTI-360 arm,
every interval clear of zero. Architecture, loss, optimizer, schedule, steps, batch, crop, grid,
mapper and evaluator are all held fixed, and the hard map is bit-identical across arms, so the
entire difference lives in one 967,322-parameter network.

**What is not demonstrated.** That two of OccAny's source corpora failed independently. The ONCE
arm shortened the supervision window from 21 frames / 10.44 s to 5 / 2.00 s, so it changes corpus
**and** supervision construction at once and is not a second clean substitution. It remains
exploratory and its occupied-prevalence gate remains failed.

**What the regional view adds, and it is new.** Source sensitivity is not uniform across regions:

| | K360 − Waymo, SK raw | pooled |
|---|---|---|
| `never_observed ∖ frustum` | −0.07 **ns** | −0.64 **ns** |
| `never_observed ∩ frustum` | +2.05 \* | +1.19 \* |

**The out-of-frustum completion advantage is the most source-robust thing measured** — all three
of our arms win it by 20–27 points raw against OccAny's causal mode, and the Waymo arm is not
significantly worse than the KITTI-360 arm there. What degrades when the corpus changes is
accuracy on surfaces inside the camera's view volume. That is a more useful statement than
"performance drops" and it should be the form the source result takes in the paper.

**What must not be inferred.** Falling below the KITTI-360 arm is not falling below OccAny: the
Waymo arm still beats OccAny's causal mode raw on SemanticKITTI, ties it pooled, ties the
forward arm on Occ3D raw and beats it pooled (+14.51 \*), and still wins strict never-observed
against their causal mode in both conventions (+7.02 / +4.04 \*). Dataset sensitivity does not
show the method contributes nothing, and nothing here licenses that reading.

## 6. Claim E — what the small training budget establishes

The CompletionUNet is **967,322 parameters trained for 6,000 steps at batch 4 on 128 × 128 × 32
crops**, three seeds, minutes of GPU time per seed. OccAny's released checkpoint ran ≈ **70,300
optimizer steps at effective batch 64 on 16 GPUs**, itself warm-started from a prior OccAny run
and initialised from MUSt3R-512.

**Establishes:** given a frozen general-purpose reconstruction stack and an explicit carved
occupancy map, the *completion* mapping is learnable with a very small task-specific
optimisation budget, and that is enough to beat a comparably-causal, heavily-trained end-to-end
model on this benchmark.

**Does not establish:** total training compute, data efficiency, or cheap inference. The
pretraining of LingBotMap and MoGe-2 on our side and MUSt3R on theirs is excluded from both
sides of that comparison; an optimizer-step count is not compute; and §4 shows the small
trainable head sits inside a 1.49 B-parameter inference pipeline of which it is 11.5 % of the
latency.

## 7. Claim F — architecture, and what cannot be concluded

**The distinction, described accurately.** OccAny (CVPR 2026, Valeo.ai) builds occupancy by
feed-forward multi-view reconstruction with a MUSt3R backbone and a raymap-conditioned
*generative* decoder that synthesises novel views; the occupancy grid is the voxelisation of the
confident points from reconstruction **and** generation. Ours builds occupancy by accumulating a
frozen reconstruction into an explicit ray-carved map with occupied / free / unknown states and
then **completing that map** with a compact 3D U-Net trained on LiDAR occupancy targets.
Densification by view synthesis against densification by learned occupancy completion.

**Why no causal claim follows.** Every comparison in this directory changes the backbone, the
pretraining, the supervision signal and the training corpus simultaneously. Their checkpoint
trained on Waymo + DDAD + PandaSet + ONCE + VKITTI2; ours on KITTI-360 alone. No ablation here
isolates the architectural choice, and none can be constructed from existing artifacts.

**Marked for related-work verification before any novelty claim is written.** "Frozen
reconstruction + compact learned occupancy completion" sits near a large monocular and
multi-view semantic-scene-completion literature. Nothing in this directory establishes that the
combination is new, and the claim should not be drafted until that literature has been checked
properly. The *measured* contributions in §§2–5 do not depend on it.

## 8. Recommended contributions — at most three

**1. A causal, low-latency occupancy-completion pipeline that is competitive with a much
heavier generative one.** *(main paper)* 373.9 ms per completed grid against 1704.4 ms, 4.56 ×,
at comparable memory, with SC IoU equal or better in both conventions against the causal
comparator on SemanticKITTI and clearly better on Occ3D. Carries its own caveats: larger
parameter count, untied future-frame axis.

**2. A decomposition of *where* completion accuracy comes from, on a method-independent
partition.** *(main paper — this is the methodological contribution)* Four mutually exclusive
regions from ground-truth visibility over the permitted causal window; the map-only ablation
isolating the learned head; the finding that the head contributes nothing on visible surface and
everything in hidden space; the finding that we are *worse* on visible surface; and the explicit
separation of out-of-frustum coverage from in-frustum occlusion, which shows that the pooled
hidden-space win against the causal comparator rests on the former.

**3. A measured source-dependence result for a small task-specific head.** *(limitations /
appendix)* One clean corpus substitution costs 2.84–9.09 SC IoU, all intervals clear of zero,
with everything else frozen — and the degradation is concentrated in-frustum while the
out-of-frustum completion advantage survives the corpus change.

**Not recommended as a contribution:** architectural novelty (§7), "robustness across training
sources" (the evidence points the other way), and anything resting on the ONCE arm, whose
supervision window is confounded.

## 9. The candidate direction, assessed

> *"A causal scene-completion pipeline that combines frozen monocular reconstruction with a
> compact occupancy learner trained using LiDAR supervision, with an explicitly measured
> tradeoff between hidden-geometry completion, source dependence, and inference cost."*

**Supported, with two words changed.** All three measured axes exist and are now quantified. Two
amendments the evidence requires:

- *"hidden-geometry completion"* → **"hidden-geometry coverage and completion, decomposed"**.
  The pooled advantage against the causal comparator is carried by geometry outside the camera
  frustum; inside it, no difference is detected. Writing "completion" unqualified claims more
  than is measured.
- *"tradeoff"* overstates the tension on the cost axis. There is no accuracy–latency tradeoff to
  report against OccAny: we are faster **and** equal-or-better. The real tradeoffs measured are
  **visible-surface accuracy against hidden-geometry accuracy** (we lose the first, win the
  second) and **source generality against source-specific accuracy**.

## 10. The minimum remaining evidence, and the single next action

**The one obstacle that touches every headline number.** All canonical predictions are at
`GEOMETRY_SEED = 0`, fixed in the prediction pipeline. Every interval in every report here is
evaluation-sample uncertainty. A paper that reports 26.40 against 25.21 with a +1.19 [+0.35,
+2.09] interval is asserting a 1.19-point difference without knowing the seed spread of its own
method, and a reviewer will ask.

**Single next action: generate geometry-only canonical predictions for seeds 1 and 2 of the
KITTI-360 head on both benchmarks, and report the three-seed spread beside the existing
intervals.** It resolves a specific obstacle to claims A, B and C, which is the only reason to
run anything. It is cheap and needs no new data: the frozen causal clips are already on disk, so
only the mapper, the hard-map query and the CompletionUNet forward re-run — roughly 0.1 s per
SemanticKITTI anchor and 0.17 s per Occ3D anchor, about **30 minutes of GPU time** for both
seeds and both benchmarks. The checkpoints `geo_seed{1,2}_step06000.pt` exist for all three
arms. **It is specified here and deliberately not launched.**

**Second, only if the source result goes in the main paper rather than the appendix:** the
duration-matched KITTI-360 training control specified in
`method_vs_data_reconciled_conclusions.md` §Q5 — the frozen recipe on KITTI-360 with the
supervision window truncated to ~2.1 s, everything else fixed. It isolates window length from
corpus identity and is the only experiment that would make the ONCE arm interpretable. Since §8
places the source result in the appendix and recommends dropping the ONCE arm from the argument,
this is **not** the next action.

**No further dataset arm is proposed.** Neither of the above adds a corpus.

## 11. Statements this document refuses to make

- That the method is state of the art. It is competitive on SemanticKITTI and ahead on Occ3D
  against the arms that exist locally.
- That it completes occluded geometry better than OccAny. Measured: it covers geometry outside
  the frustum that OccAny barely predicts, and inside the frustum under the published convention
  no difference is detected against the causal comparator.
- That a 967,322-parameter trainable head implies cheap inference. The pipeline is 1.49 B
  parameters and 2.28 × OccAny's.
- That the five-frame comparison is temporally matched.
- That the architecture causes the difference.
- That the method is robust across training sources.
- Anything about training-seed variation.
