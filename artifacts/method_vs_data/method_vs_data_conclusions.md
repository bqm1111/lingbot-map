# Method vs data — conclusions

**2026-10-20.** Answers to brief §13, from measured evidence only. Stages that were blocked
are reported as blocked.

| stage | outcome |
|---|---|
| OccAny audit | done — `occany_training_audit.md` |
| fair protocol | done — `fair_evaluation_protocol.md` |
| DDAD audit | done — **NO-GO**, `ddad_feasibility.md` |
| A2 DDAD-only training | **NOT EXECUTED** — gates 6 and 7 fail |
| geometry head-to-head | done — `geometry_head_to_head.md` |
| observed vs completed | done — `observed_vs_completed_analysis.md` |
| matched-data control | assessed, not run — `matched_data_control_feasibility.md` |

---



## Q1. Does our method trained on DDAD alone outperform the released OccAny under an identical monocular five-frame protocol?

**Unanswered. The experiment was not run, and should not be.**

DDAD publishes one aggregated point cloud per sample — four Luminar-H2 sensors merged into a
vehicle frame whose origin sits on the ground below the rear axle, with per-point channels
X/Y/Z/INTENSITY only. Our supervision carves free space along rays **from the sensor origin**,
and that origin is unrecoverable: no per-point identity, no per-sensor counts in the release,
and no azimuthal sector structure from which to infer it (max/min bin ratio 3.49, content-
driven, no seams). Measured elevation p1 **−1.35°** confirms the origin is on the road, so rays
to road returns graze along the road plane.

Running the frozen builder unchanged on a real DDAD scene gives occupied prevalence
**0.0042–0.0094** against KITTI-360's 0.0276 and Waymo's 0.0222 — **3–6× out of family** — and
varying the assumed mount across plausible values moves the valid fraction ~20 pp and
prevalence ~2×. An arm trained that way would confound DDAD with a constant we invented, so it
could not be the control A2 exists to be.

**Also worth knowing:** DDAD's only forward camera is **47.9° HFOV**, essentially Waymo's 50°.
Even unblocked, A2 would have re-run the input condition A1 already failed under.

---

## Q2. Does our method achieve competitive zero-shot geometry with fewer source datasets and less task-specific training data?

**Two halves, with different answers.**

**Less task-specific training: YES, verified on both sides.**

| | OccAny (released) | ours |
|---|---:|---:|
| optimizer steps | **≥ 70,300** | **6,000** |
| sample draws | 4,500,000 | 24,000 |
| effective batch | 64 (16 GPUs) | 4 |
| trained parameters | MUSt3R-scale decoder | **967 k** |
| initialization | MUSt3R_512 **+ a prior OccAny run of unrecorded size** | from scratch |
| source datasets | 5 | 1 |

"≥" is load-bearing: `args.pretrained_occany` shows a warm start whose budget is not recorded,
so OccAny's total is **unbounded above** by this audit.

Two honesty qualifiers. (a) "Task-specific" excludes our frozen LingBotMap and MoGe-2
dependencies — our *system* is not trained from scratch, our *completion network* is. (b) We
verified OccAny's sample draws per epoch, **not** its unique scenes or frames; those live in
sequence pickles not present locally. **So the efficiency claim must rest on steps and draws,
which are verified, not on dataset counts, which are not.**

**Competitive geometry: YES, with scope** — SemanticKITTI a tie, Occ3D a clear win:

| benchmark | ours causal − OccAny forward, raw | pooled |
|---|---:|---:|
| SemanticKITTI | +1.11 [+0.30, +1.93] | **−0.04** [−0.77, +0.70] (tie) |
| Occ3D | **+10.98** [+9.01, +12.97] | **+17.36** [+16.04, +18.66] |

...while consuming only past frames where their strongest released run consumes future ones.
We are behind on semantics (4.45 vs 7.28 SSC mIoU) and on surround geometry (32.62 vs 34.15);
the advantage is **single-camera only**.

**Fewer datasets *from OccAny's pool*: NOT ESTABLISHED.** KITTI-360 is not one of their five
sources, so the strong rows above cannot settle the subset question. That was A2's job.

---

## Q3. Is the advantage primarily streaming evidence accumulation or learned geometric completion?

**Learned completion, by a wide margin.** Two independent measurements agree.

**The map-only ablation.** The accumulated five-frame hard map scored on its own:

| benchmark | hard map alone | + CompletionUNet | completion adds | OccAny best released |
|---|---:|---:|---:|---:|
| SemanticKITTI | **8.79** | 26.32 | **+17.53** (67 %) | 25.21 |
| Occ3D | **9.82** | 31.63 | **+21.81** (69 %) | 20.66 |

Accumulation alone is **16.4 points below** OccAny. It is not what wins.

**The strict split** (801 anchors, vs OccAny 1-frame causal, both conventions):

| region | share of GT surface | Δ raw | Δ pooled (OccAny's published convention) |
|---|---:|---:|---:|
| anchor_observed | 4.76 % | **−4.29** [−5.08, −3.60] | **−2.70** [−3.20, −2.29] |
| earlier_only | 6.18 % | **+4.98** [+3.57, +6.64] | **+0.65** [−0.33, +1.74] — **not significant** |
| never_observed | **89.06 %** | **+8.54** [+7.01, +10.06] | **+4.77** [+3.25, +6.23] |

Each permitted frame sees ~4.8 % of the GT surface; all five together see 10.94 %. Two seconds
of forward motion barely expands coverage, so the benchmark is decided in the 89 % no frame
observed — and that is where our lead is, in **both** conventions.

**The accumulation advantage does not survive dilation.** Under OccAny's own published
convention `earlier_only` falls to +0.65 with a CI containing zero, because dilation is worth
+3.74 SC IoU to their sparse 1-frame predictions against −0.05 to our already-dense ones. The
never-observed advantage is the only one robust across both conventions. (A CI containing zero
is failure to detect, not proof of absence — the defensible statement is that accumulation's
contribution is small and convention-dependent, while completion's is not.)

**This corrects an earlier conclusion of ours.** The 2026-10-16 anchor-frame split was read as
*"we beat it by remembering, not by completing"*. That reading is withdrawn: out-of-frustum
voxels turn out to be overwhelmingly **never observed**, not remembered. The two measurements
reconcile exactly (95.240 % vs 95.331 %, differing only by the 0.091 pp between the two
anchor-observed definitions); the error was interpretive, not numerical.

---

## Q4. What evidence supports attributing the advantage to the method rather than to training data?

**None that is conclusive — and our own evidence cuts partly the other way.** This is the
weakest link in the claim and should not be overstated.

**What supports a method attribution:**

* The advantage is localized to a mechanism, not to an aggregate: learned completion in
  never-observed space (Q3), not accumulation, and not visible-surface accuracy (where we
  lose).
* `map_only` is **bit-identical** across the KITTI-360 and Waymo arms
  (8.820500920980095 both), so the reconstruction and mapping stack is **training-data
  independent** by construction. Every data-sensitive part of our system is the CompletionUNet.
* We reach the result with ≥11× fewer optimizer steps and ~187× fewer sample draws, from
  scratch, with 967 k trained parameters (Q2).

**What cuts against it:**

* **Our own A1 result shows the architecture is strongly data-sensitive.** Holding the
  architecture, loss, schedule and step budget fixed and changing only the training corpus
  moved SemanticKITTI by **−3.03** and Occ3D by **−8.36**, both with CIs clear of zero. So a
  large fraction of our performance is attributable to training data — demonstrated by us, on
  our own system. Any claim of the form "the architecture is what wins" has to survive that.
* Every comparison against OccAny crosses three confounds at once: different corpora,
  different supervision drawn from them (3D pointmap regression + distillation vs occupancy
  carving), and different pretraining. Matching one does not match the others.
* Trained capacity differs by three orders of magnitude.

**Conclusion: benchmark superiority and training-data efficiency are supported; a
methodological advantage is not.** The brief's three claims are distinct, and only the first
two are currently evidenced.

---

## Q5. Is a matched-data OccAny retraining experiment necessary to substantiate the strongest claim?

**Necessary but not sufficient, and not currently possible.**

* **Necessary** for the methodological claim: without it, Q4's confounds remain.
* **Not sufficient**: matching the dataset does not match the supervision drawn from it, the
  pretrained initialization (their warm-start checkpoint is unpublished and unquantified), or
  the trained capacity. Even a clean matched run would license a statement about
  *interaction*, not about architecture alone.
* **Not currently possible**: for DDAD, *our* side cannot be built at all (Q1), so a matched
  pair does not exist. It becomes constructible only once a substitute source arm exists.
* **Cost**: a faithful replication is a 16-GPU-shaped job (4.5 M sequence draws, 100 epochs,
  effective batch 64) on our 4 GPUs, and the warm start cannot be reproduced.

**Recommendation: do not retrain OccAny now.**

---

## Q6. What is the smallest additional experiment required for a defensible method-vs-data conclusion?

**The one-frame-map ablation. Inference only, no training, no new data.**

Run the existing CompletionUNet on a hard map built from **one** frame instead of five, and
compare against the five-frame map on identical anchors.

Why this one:

* It is the only remaining control that separates **"completion needs temporal accumulation"**
  from **"completion needs any map at all"** — the question Q3 answers only from above (8.79
  floor) and not from inside.
* It puts our system in **OccAny's exact input condition** (one frame), which is the closest
  thing to a matched-input comparison obtainable without retraining anything. If our
  completion still beats their 20.32 from a single frame, the advantage cannot be temporal
  context, and the data confound is the only one left.
* The mapper already accepts any frame count, so it needs no new code path and no training.
* It is hours of inference, against days of GPU time for any training arm.

**Then, and only if a source-subset claim is wanted:** the **ONCE** arm — single `lidar_roof`
LiDAR, so one true published sensor origin, no invented constant, and our frozen
single-sensor carve applies unchanged. PandaSet is the fallback, since it publishes a per-point
sensor ID `d` that the already-validated multisensor carve can consume. Both must clear the
supervision-comparability and sensor-origin gates predeclared in `predeclaration.md` §6 before
training, and the chosen camera's HFOV must be recorded beforehand — DDAD's 47.9° and Waymo's
50.0° are a warning that a narrow-camera substitute would have little power to discriminate
dataset identity from training FOV.

**Not recommended:** ONCE + PandaSet + VKITTI2 expansion, OccAny retraining, or any
architecture change. None is justified by the evidence above.

---

## What changed in our understanding today

1. **DDAD is unusable for our supervision**, for a structural reason, established in ~1.2 GiB
   of ranged reads instead of a 256 GiB download. The probe harness
   (`tools/method_vs_data/ddad_probe.py`) makes the same audit an afternoon's work for any
   candidate dataset, and should be run **before** writing an adapter.
2. **The released OccAny checkpoint is not what the shipped script trains**, is OccAny rather
   than OccAny+, and carries an unrecorded warm start — so its training budget is a lower
   bound, not a figure.
3. **"We win by remembering" is withdrawn.** Our lead is in never-observed geometry; the
   five-frame window adds only 6.18 pp of observed surface over a single frame, and under
   OccAny's own pooled convention the remembered-space advantage is not significant at all
   (+0.65, CI ∋ 0).
4. **The reconstruction stack is data-independent and the completion network is not** —
   `map_only` is bit-identical across arms while the trained arms differ by 3–8 SC IoU. That
   localizes both the strength and the data-sensitivity to one 967 k-parameter network.
