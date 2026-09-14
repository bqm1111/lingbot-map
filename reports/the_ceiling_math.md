# The ceiling bug, derived

*Why the model predicted a solid roof, and why the same defect cost SemanticKITTI 44 % of its
score and Occ3D 9 %. Every symbol is defined; every number is measured, with the command that
produced it.*

Companion to `reports/the_ceiling_bug.md`, which tells the story. This one does the algebra.

---

## Part 1 — Why the sky is occupied by default

### 1.1 The decision rule

The map stores **log-odds** of occupancy. For voxel $i$:

$$\ell_i = \log \frac{p(\text{occupied}_i)}{1 - p(\text{occupied}_i)}$$

A voxel nothing has observed carries no evidence, so $p = 0.5$ and

$$\boxed{\ell^{\text{base}}_i = 0 \quad \text{for every unobserved voxel}}$$

The network does not output occupancy. It outputs a **residual** $r_i$, applied through a gate
(`core/model/net.py::apply_residual`, $L = \texttt{LOCK\_LOGODDS} = 2$):

$$\ell^{\text{final}}_i \;=\; \ell^{\text{base}}_i + r_i \cdot \mathbb{1}\!\left[\,|\ell^{\text{base}}_i| < L\,\right]$$

The final decision (`core/run/eval_target.py`, `comp = (s_fin >= tau) & valid`) is

$$\hat{y}_i = \mathbb{1}\!\left[\ell^{\text{final}}_i \ge \tau\right]$$

For an unobserved voxel $\ell^{\text{base}}_i = 0$, so $|0| < L$ — the gate is **open** — and

$$\ell^{\text{final}}_i = r_i \qquad\Longrightarrow\qquad \hat{y}_i = \mathbb{1}[\,r_i \ge \tau\,]$$

Every frozen threshold is negative: $\tau = -0.125$ (seed 0), $-0.3125$ (seed 2). Therefore

$$r_i = 0 \;\;\Longrightarrow\;\; \hat{y}_i = 1$$

**An unobserved voxel is classified OCCUPIED unless the network actively pushes it down.**
Empty space is not the default; it must be earned.

**Verification.** Force $r \equiv 0$ by closing the gate everywhere:

```
$ python tools/skyfix/probe.py --limit 20 --lock 0
  above +2.0 m: 96.13% of scored voxels predicted occupied
```

96.13 %, not 100 %, because a few high voxels *are* observed — free-space rays give
$\ell^{\text{base}} < \tau$ and survive without the network's help.

### 1.2 The gradient that does not exist

Training minimises a masked loss (`core/model/losses.py`), where $w_i$ comes from
`balanced_weights`:

$$\mathcal{L} \;=\; \frac{\sum_i w_i \,\ell_{\text{voxel}}(r_i, y_i)}{\sum_i w_i},
\qquad w_i \;=\; \texttt{valid}_i \cdot \big(1 + (\omega-1)\,\mathbb{1}[\text{unobserved}]\big)$$

The first factor is the whole story:

$$\texttt{valid}_i = 0 \;\;\Longrightarrow\;\; w_i = 0 \;\;\Longrightarrow\;\;
\frac{\partial \mathcal{L}}{\partial r_i} = 0$$

In the old targets `rawtarget.py` marked everything unmeasured as `UNKNOWN`, and
$\texttt{valid} = (\texttt{state} \neq \texttt{UNKNOWN})$. Measured over 40 training samples:

| height | fraction with $\texttt{valid} = 1$ |
|---|---|
| +4.3 m | **0.98 %** |
| +3.5 m | 3.27 % |
| +2.1 m | 20.15 % |
| +0.1 m | 64.73 % |

So for ~99 % of the top layer, $\partial\mathcal{L}/\partial r_i = 0$ **exactly**, for the whole
of training.

### 1.3 The root cause, stated precisely

$r_i$ in the sky is not wrong. It is **undefined by the objective**. Its value is whatever the
convolutional inductive bias interpolates from supervised regions — an accident of
architecture, not a learned quantity. And by §1.1 the sign of that accident relative to $\tau$
decides whether the sky is solid.

$$\phi \;\equiv\; \Pr\big[\hat{y}_i = 1 \;\big|\; i \text{ scored, above } 2\,\text{m}\big]
\;\approx\; \Pr\big[\,r_i \ge \tau\,\big]$$

Measuring that distribution directly on unobserved high voxels, pre-fix model, $\tau = -0.125$:

| dataset | median $r$ | $q_{10}$ | $q_{90}$ | $\Pr[r \ge \tau]$ | measured $\phi$ |
|---|---|---|---|---|---|
| SemanticKITTI | **−0.379** | −0.90 | +0.01 | **22.60 %** | **21.74 %** |
| Occ3D | −0.938 | −1.80 | −0.26 | 6.64 % | 6.17 % |

The predicted and measured $\phi$ agree to about one point on both datasets. **The residual
distribution fully accounts for the ceiling.**

After the fix — sky voxels become `FREE`, so $\texttt{valid} = 1$, so a gradient exists:

| dataset | median $r$ | $\Pr[r \ge \tau]$ | measured $\phi$ |
|---|---|---|---|
| SemanticKITTI | **−1.547** | **0.70 %** | 1.94 % |
| Occ3D | −0.852 | 2.71 % | 2.72 % |

SemanticKITTI's sky residual moves from −0.379 to −1.547: from *hovering just under the
decision boundary by accident* to *held well below it on purpose*.

---

## Part 2 — Why the two benchmarks disagree

### 2.1 An exact decomposition of IoU

Let $P = \mathrm{TP} + \mathrm{FN}$ be the number of ground-truth occupied voxels. Define

$$t \;=\; \frac{\mathrm{TP}}{P} \;\;(\text{recall}), \qquad F \;=\; \frac{\mathrm{FP}}{P}$$

Then, with no approximation:

$$\mathrm{IoU} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}
= \frac{tP}{tP + \mathrm{FP} + (1-t)P} = \frac{tP}{P + \mathrm{FP}}
\;=\; \boxed{\;\frac{t}{1+F}\;}$$

(As a check, $\text{pred}/\text{GT} = (\mathrm{TP}+\mathrm{FP})/P = t + F$, so $F$ is readable
straight off the probe output.)

Now split false positives by height, $F = F_{\text{lo}} + F_{\text{hi}}$, and write $N_{\text{hi}}$
for the number of **scored** voxels above 2 m and $g$ for the fraction of those that are truly
occupied. A model marking fraction $\phi$ of them occupied produces
$\mathrm{FP}_{\text{hi}} \approx \phi\,N_{\text{hi}}(1-g)$. Define the benchmark's **sky burden**

$$\beta \;\equiv\; \frac{N_{\text{hi}}\,(1-g)}{P}
\;=\; \frac{\text{scored empty voxels above 2 m}}{\text{all ground-truth occupied voxels}}$$

giving the working model:

$$\boxed{\;\mathrm{IoU} \;\simeq\; \frac{t}{1 + F_{\text{lo}} + \phi\,\beta}\;}$$

Two factors control the ceiling's cost, and they are cleanly separated:

* $\phi$ — **the model's** behaviour: how much of the sky it fills (§1.3);
* $\beta$ — **the benchmark's** geometry: how much empty sky it scores, per positive voxel.

### 2.2 Measured, all four combinations

Seed 0, 40 anchors each, `tools/skyfix/probe.py`:

| | $t$ | pred/GT | $F$ | $\phi$ | $\beta$ | $\phi\beta$ | $F_{\text{lo}}$ | IoU |
|---|---|---|---|---|---|---|---|---|
| SemKITTI, pre-fix | 0.3738 | 2.34 | 1.966 | 21.74 % | 6.03 | **1.311** | 0.655 | 12.59 |
| SemKITTI, fixed | 0.4265 | 1.38 | 0.954 | 1.94 % | 6.03 | 0.117 | 0.837 | 21.87 |
| Occ3D, pre-fix | 0.3440 | 0.55 | 0.206 | 6.17 % | 1.78 | **0.110** | 0.096 | 28.56 |
| Occ3D, fixed | 0.4008 | 0.59 | 0.189 | 2.72 % | 1.78 | 0.048 | 0.141 | 33.65 |

$\beta$ is **identical across models** on each dataset (6.03 and 1.78) — as it must be: it is a
property of the benchmark's masks and labels, not of the network. That invariance is the
check that the decomposition is real and not curve-fitting.

**The ceiling's share of the IoU denominator $1+F$:**

$$\text{SemanticKITTI: } \frac{1.311}{2.966} = 44.2\,\% \qquad\qquad
\text{Occ3D: } \frac{0.110}{1.206} = 9.1\,\%$$

**That is the answer to the question.** One defect, one model, one behaviour — consuming 44 %
of one benchmark's scoring budget and 9 % of the other's.

### 2.3 The counterfactual

Remove *only* the ceiling term, holding everything else fixed:
$\mathrm{IoU}^{*} = t/(1+F_{\text{lo}})$.

| | actual pre-fix | $\mathrm{IoU}^{*}$ | predicted gain | actual gain | ratio |
|---|---|---|---|---|---|
| SemanticKITTI | 12.59 | **22.58** | +9.99 | +9.28 | **108 %** |
| Occ3D | 28.56 | **31.38** | +2.82 | +5.09 | **55 %** |

The ratio is predicted ÷ actual: how much of the observed improvement the ceiling term alone
accounts for.

On SemanticKITTI it slightly **over**-predicts (108 %). Removing the ceiling should have been
worth +9.99 and only +9.28 arrived, because the retrained model is a little worse below 2 m —
$F_{\text{lo}}$ rose from 0.655 to 0.837. Essentially the whole gain is the ceiling, minus a
small regression elsewhere.

On Occ3D it explains only 55 %. The rest is real recall the retrained model gained
(0.3440 → 0.4008), which has nothing to do with the ceiling — and note $F_{\text{lo}}$ rose
there too (0.096 → 0.141), the same trade in miniature.

### 2.4 Where $\beta$ comes from — it is prevalence

Factor $\beta$ into two measurable pieces. With $V$ the scored voxels per frame and
$p = P/V$ the **prevalence** of the positive class:

$$\beta \;=\; \frac{N_{\text{hi}}(1-g)}{P}
\;=\; \underbrace{\frac{N_{\text{hi}}(1-g)}{V}}_{\textstyle h}\;\cdot\;\underbrace{\frac{V}{P}}_{\textstyle 1/p}
\qquad\Longrightarrow\qquad \boxed{\;\beta = \frac{h}{p}\;}$$

$h$ is the share of the scored volume that is empty space above 2 m. Measured in three
independent configurations:

| configuration | scored/frame | GT occ/frame | $p$ | $h$ | $\beta = h/p$ |
|---|---|---|---|---|---|
| SemanticKITTI (`.invalid` mask) | 1,517,106 | 121,157 | 7.99 % | **48.1 %** | 6.03 |
| Occ3D, camera mask **on** | 40,730 | 9,208 | 22.61 % | **42.3 %** | 1.87 |
| Occ3D, camera mask **off** | 320,000 | 14,341 | 4.48 % | **48.6 %** | 10.83 |

$h$ is **essentially constant at 42–49 %** across all three. It is close to a property of urban
driving scenes: about half of any scored road-scene volume is empty air above head height.
$\beta$ therefore varies almost entirely as $1/p$.

**And the third row is the punchline.** Strip Occ3D's camera mask and its sky burden becomes
$\beta = 10.83$ — nearly **twice as punishing as SemanticKITTI**. Occ3D is not intrinsically
forgiving. Its mask makes it so.

The mechanism is now fully explicit. The camera mask discards 87 % of the volume
(320,000 → 40,730) but only 36 % of the occupied voxels (14,341 → 9,208), because what it keeps
is *visible structure* and what it discards is *occluded air*. That raises prevalence

$$p:\;4.48\,\% \longrightarrow 22.61\,\% \quad (5.0\times)$$

and collapses the sky burden

$$\beta:\;10.83 \longrightarrow 1.87 \quad (5.8\times)$$

### 2.5 The whole difference in one line

$$\frac{(\phi\beta)_{\text{SK}}}{(\phi\beta)_{\text{Occ}}}
= \underbrace{\frac{0.2174}{0.0617}}_{\textstyle 3.52\times\;(\text{model})}
\;\times\;
\underbrace{\frac{6.03}{1.78}}_{\textstyle 3.39\times\;(\text{benchmark})}
\;=\; 11.9\times$$

Half the gap is the model — the residual accident of §1.3 happened to land closer to $\tau$ on
SemanticKITTI inputs (median −0.379 vs −0.938). Half is the benchmark — SemanticKITTI scores
2.8× more empty sky per positive voxel, because its LiDAR-traversal mask keeps open air that
Occ3D's visibility mask throws away.

---

## Part 3 — What to take from this

**The root cause in one sentence.** An unobserved voxel decodes to *occupied* by default
($\ell^{\text{base}} = 0 \ge \tau$), and the loss placed exactly zero weight on the region where
that default applied, so nothing ever contradicted it.

**Three transferable statements:**

1. **A zero in a loss mask is not neutral.** It hands the region to whatever the decision rule
   does by default. Always ask what that default *is*: here, $\tau < 0$ made it "solid".
2. **$\beta = h/p$.** The penalty a benchmark charges for hallucinating into empty space is
   inversely proportional to its positive-class prevalence — and prevalence is set by the
   evaluation mask, which is a protocol choice, not a fact about the world.
3. **Two benchmarks disagreeing violently about one model is usually a statement about the
   benchmarks.** Occ3D's 30.89 was never evidence the geometry was sound; it was evidence that
   $\beta = 1.87$. The harsher benchmark was telling the truth.

## Reproducing every number

```bash
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
py=/home/minh/anaconda3/envs/cu128/bin/python

# §1.1  the default is occupied
$py tools/skyfix/probe.py --limit 20 --lock 0

# §2.2  t, phi, beta for all four combinations
for ds in semantickitti occ3d; do for mf in frozen_manifest frozen_manifest_sky; do
  $py tools/skyfix/probe.py --dataset $ds --limit 40 --seed 0 \
      --manifest $PWD/artifacts/gate8c1/$mf.json
done; done
```

§1.2's supervision table, §1.3's residual distributions and §2.4's mask comparison are the
inline scripts in `WALKTHROUGH.md` Stop 5 and in the session that produced this file; each is a
few lines over `artifacts/gate8c1/` and the Occ3D label files.

**Scope.** 40 anchors per configuration, seed 0, one segment of SemanticKITTI 08 and the first
Occ3D scenes. Sharp enough for the decomposition and the ordering; the published headline
numbers are the full 163-anchor and 4,819-sample runs in `REPRODUCE.md` and
`reports/the_ceiling_bug.md`. The approximation
$\mathrm{FP}_{\text{hi}} \approx \phi N_{\text{hi}}(1-g)$ assumes predictions above 2 m are
roughly independent of ground truth there, which is safe on SemanticKITTI ($g = 0.53\,\%$) and
looser on Occ3D ($g = 13.08\,\%$) — that is the main reason the §2.3 counterfactual is tighter
for SemanticKITTI than for Occ3D.
