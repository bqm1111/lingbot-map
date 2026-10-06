# Zero-shot scene completion — the results, and how to rerun them

**The claim.** A 0.99 M-parameter completion module trained on KITTI-360 alone
(drives 0003/0007/0010), evaluated on SemanticKITTI 08 and Occ3D-nuScenes val,
which it never saw. Causal: past frames and the current frame only, no future,
no LiDAR at inference, no adaptation to either target.

This is **target-domain-free transfer of the completion module**, not whole-system
zero-shot — the frozen models beneath it (LingBot-Map, MoGe-2, Trident-H) have not
had their training provenance audited, and LingBot-Map's published mixture includes
KITTI-360. That distinction is load-bearing; keep it in any writeup.

---

## 1. The results

Headline protocol is `past5` (causal, 5 past frames). Median over 3 seeds ± sd.

| target | ours SC IoU % | ours SSC mIoU % | OccAny published SC IoU % | fraction |
|---|---|---|---|---|
| SemanticKITTI 08 | **12.74 ± 1.03** | 2.65 | 25.91 | 49 % |
| Occ3D-nuScenes val | **30.89 ± 0.29** | 4.13 | 23.55 | 131 % |

OccAny trains on both targets and runs non-causally with ≈1.5 B parameters.

All three protocols, for context:

| protocol | causal | SemanticKITTI 08 | Occ3D-nuScenes val |
|---|---|---|---|
| `past5` (headline) | yes | 12.74 ± 1.03 | 30.89 ± 0.29 |
| `stream` | yes | 12.83 ± 0.81 | 31.98 ± 0.44 |
| `occany_fwd` | no | 13.94 ± 1.12 | 27.56 ± 0.05 |

**Verdict on record: FAIL** against the pre-registered continue bar — it beats OccAny
on Occ3D but reaches 49 % of it on SemanticKITTI, and does not beat the strongest
non-learned baseline on both. The full decision table is in
`artifacts/gate8c1/report.md` § 1.

Every number above is derived from `artifacts/gate8c1/gate8c1_results.json`, which is
aggregated from the 18 `artifacts/gate8c1/eval_<target>_<protocol>_seed<n>.json` runs.

---

## 2. Rerunning it

Three levels, cheapest first. All need the environment in § 3.

### A. Rebuild the report from the recorded runs — seconds, no GPU

```bash
$PY tools/gate8c1/aggregate.py          # 18 eval JSONs  -> gate8c1_results.json
$PY tools/gate8c1/report.py             # results        -> artifacts/gate8c1/report.md
```

Both are deterministic: given the same eval outputs they regenerate byte-identical
files. Note that `eval_target.py` writes four npz families alongside each JSON
(`scores_*`, `bincounts_*`, `counts_*`, `newly_*`) and `aggregate.py` recomputes the
threshold-free metrics from them — so back up all of them together, or the aggregate
will mix runs.

### B. Rerun the target evaluation — ~67 GPU-min total

```bash
bash tools/gate8c1/run_eval.sh          # 2 targets x 3 protocols x 3 seeds, one seed per GPU
$PY tools/gate8c1/aggregate.py && $PY tools/gate8c1/report.py
```

Or a single run:

```bash
$PY tools/gate8c1/eval_target.py --dataset semantickitti --mode past5 --seed 0 --device cuda:0
```

`eval_target.py` refuses to start without `artifacts/gate8c1/frozen_manifest.json`.
This reads the target datasets; everything before it does not.

### C. Full rebuild from KITTI-360 — ~2.2 GPU-h

```bash
bash tools/gate8c1/run_targets.sh       # 0.06 GPU-h   raw-Velodyne supervision
$PY tools/gate8c1/validate_targets.py   #              invariant checks before training
bash tools/gate8c1/run_samples.sh       # 0.29 GPU-h   causal 32-channel inputs + targets
bash tools/gate8c1/run_train.sh         # 0.72 GPU-h   3 seeds, 6000 steps each
bash tools/gate8c1/run_eval.sh          # 1.12 GPU-h   the 18 target runs
```

Not rebuilt by any of these: the frozen per-frame caches (LingBot-Map depth/pose,
MoGe-2 depth, Trident-H semantics) under `/media/SSD1/MINH_DATASETS/lingbot_gate7b/`
and `.../lingbot_gate8/`. Those are the expensive upstream artifacts and the pipeline
reads them as given.

**Before running C, read § 5.** The target builder changed after the cached targets
were written, and rebuilding restates the source-domain numbers.

---

## 3. Environment

```bash
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
PY=/home/minh/anaconda3/envs/cu128/bin/python     # bare `python` is not on PATH
```

conda env `cu128`, torch 2.7.1+cu128, 4× RTX PRO 6000 Blackwell (sm_120).
`tools/gate8/gpus.sh` maps logical seed index → physical GPU.

Data roots, all read-only:

| path | holds |
|---|---|
| `/media/SSD1/MINH_DATASETS/lingbot_gate7b/{stream,scale}/` | LingBot depth/pose + metric scale, per frame |
| `/media/SSD1/MINH_DATASETS/lingbot_gate8/` | Trident-H semantics, MoGe-2 depth |
| `/media/SSD1/MINH_DATASETS/lingbot_gate8c1/{samples,targets}/` | the KITTI-360 training pairs |
| `/media/SSD1/MINH_DATASETS/sscbench_kitti360/` | SSCBench-KITTI-360 (its `_1_1.npy` labels are never opened) |
| `/media/welf/MINH/datasets/kitti360/KITTI-360/` | KITTI-360 raw Velodyne + GT poses |
| `data/kitti/dataset/` | SemanticKITTI 08 |

---

## 4. What "reproducible" means here, measured

Re-running `eval_target.py --dataset semantickitti --mode stream --seed 0` and diffing
all 2 600 numeric fields against the recorded JSON:

| | |
|---|---|
| SC IoU | 0.1148505 → 0.1148530 (relative 2.2e-5) |
| SSC mIoU | 0.0241106 → 0.0241106 (relative 1.2e-6) |
| max relative change, any field | 1.2e-3 |
| median relative change | 3.9e-6 |

Re-running one eval and re-aggregating moved every headline SC IoU in § 1 by
**0.00000 pp** at 5-decimal precision, on both targets and all three protocols.

So the results reproduce to 4–5 significant figures, roughly a thousand times tighter
than the ±1.03 seed spread. The residual is GPU float non-associativity, not a
correctness problem. **Do not expect bit-identical eval JSONs.**

The report and aggregate steps (§ 2A) *are* byte-identical on rerun, as are all 155
figures and reports across every gate.

---

## 5. Known issues — read before rebuilding

**The target builder changed after the cached targets were written.**
`gates/gate8c1/rawtarget.py` gained an open-sky rule, which marks the space above the
local roofline FREE rather than UNKNOWN. The cached targets predate it — the file with
the rule disabled reproduces their `valid` and `occupied` bit-exactly. A rebuild is
therefore *not* a no-op: on a 6-anchor sample the source-domain (KITTI-360 drive 0006)
completion SC IoU goes 9.29 % → 4.67 %.

That drop is the rule working as designed, not a regression. The sky was previously
UNKNOWN and excluded from scoring, and the sky is exactly where this model
hallucinates. The source-domain numbers were flattered by not scoring the region with
the worst errors. **The target-domain results in § 1 are unaffected** — they score
against SemanticKITTI and Occ3D ground truth, not against rebuilt KITTI-360 targets.

A genuine bug in that rule (it could label a voxel holding a LiDAR return as free) was
fixed in commit `647bcf3`.

**Two files still differ from the frozen manifest**, from earlier work:
`gate8/net.py` (`4cbfced7` → `56852e4a`, optional padding variants; numerically benign
— the source IoU reproduces to 15 digits) and `gate8c1/sources.py`
(`1cfa2be4` → `c23f2813`, uncharacterised). Those original versions are unrecoverable:
the manifest stores hashes only and the tree was untracked before commit `f7b7552`.
`tests/gate8c1::test_gate8c0_artifacts_remain_unchanged` fails for exactly this reason
and should keep failing until the manifest is re-frozen.

**The 3D figures cannot be regenerated.** `figure_3d_compare.py` and
`figure_3d_gallery.py` are non-deterministic — 4 of 6 outputs differ between
back-to-back runs with identical arguments — and the stored versions were made with
`--n 8` (not the default `--n 3`) by code that has since drifted. They are gitignored,
so they exist only on disk. Do not delete them.

---

## 6. Where everything is

```
core/          the live pipeline, 57 files reorganised by role — start at core/README.md
gates/         the research history, one frozen package per gate — gates/README.md
tools/<gate>/  each gate's scripts and runners
configs/<gate>/  frozen configuration
artifacts/<gate>/  results; the bulk .npz/.pt are gitignored
reports/<gate>/  written reports
tests/<gate>/  each gate's tests
```

Reading order for someone new:

1. this file — the claim and the numbers
2. `core/README.md` — the pipeline, the 32 input channels, the residual lock
3. `artifacts/gate8c1/report.md` — the full gate report and decision table
4. `gates/README.md` — why each earlier gate existed and what it found
5. `artifacts/gate8c1_source_sanity/report.md` — the open source-domain question

Test suite: `$PY -m pytest tests/ -q` → 861 passed, 6 failed. The 6 are known: one
manifest-hash test (§ 5) and five `semantic_sidecar` tests that fail on a missing
`ninja` binary, unrelated to any of this.
