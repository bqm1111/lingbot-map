# Owning this codebase in one sitting

You do not need to read 603 files. You need eight, and you need to have *moved* each one.

This is a hands-on tour: read a file, run a command, watch a number change. Every exercise
runs in seconds, writes nothing you care about, and can be undone by rerunning without the
flag. Do them in order — the data flows that way.

**Setup, once per shell:**

```bash
cd /home/minh/workspace/lingbot-map_fork
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST="12.0a" HF_HUB_OFFLINE=1
export G8C1_MANIFEST=$PWD/artifacts/gate8c1/frozen_manifest_sky.json
py=/home/minh/anaconda3/envs/cu128/bin/python      # use as $py below
```

That `G8C1_MANIFEST` line points every tool at the **fixed** model. Without it you get the
model the README still describes, which scores 12.74 instead of 22.71. See `reports/the_ceiling_bug.md`.

**The system in five lines.** Three frozen models turn camera frames into depth, pose and
semantics — run once, offline, cached as `.npz`; nothing here ever runs them live. An
incremental mapper fuses those into a 256×256×32 log-odds grid at 0.2 m. A 986,114-parameter
3D U-Net reads 32 channels of that grid and predicts a **residual** that fills in voxels
nobody observed. Threshold at τ. Supervision comes from *future* LiDAR the model never sees.

---

# Part 1 — inference: what actually runs

## Stop 1 · `core/mapping/mapper.py` — frames become a grid

**The question it answers:** how do cached per-frame predictions turn into one volume?

**Read:** `IncrementalMapper.step` (how one frame is integrated) and `.query` (how the volume
is read back). Note `scale_state` — monocular depth has no absolute scale, so a scale factor
is estimated once from the first five frames and then *frozen*.

**Run:**

```bash
$py tools/skyfix/probe.py --limit 20 --frames 5     # the shipped setting
$py tools/skyfix/probe.py --limit 20 --frames 1     # one frame only
```

|  | map only IoU | completion IoU |
|---|---|---|
| 5 frames | 9.77 | 22.19 |
| 1 frame | 5.48 | 22.26 |

**What just happened.** Starving the map of history nearly halves the map's own score — and
the completion *does not care*. That is a real finding, not a demo: the network is leaning far
more on a learned prior than on accumulated evidence. Sit with that. It is the kind of thing
that should make you suspicious of the whole method, and it is exactly the sort of question
you are now equipped to chase.

## Stop 2 · `core/model/net.py` — the grid becomes a prediction

**The question:** what does the network actually see, and what is it allowed to change?

**Read:** `Completer.raw`. The `torch.stack([...])` is the input: 7 geometry channels
(log-odds, free-weight, observed, unobserved, observation count, age, semantic weight) plus 25
semantic channels = **32**. Then `apply_residual`:

```python
def apply_residual(base_logodds, occ_res):
    gate = (base_logodds.abs() < LOCK_LOGODDS).to(occ_res.dtype)
    return base_logodds + occ_res * gate
```

The network does not output occupancy. It outputs a *correction*, and it is **locked out** of
any voxel the map already feels strongly about (`|log-odds| ≥ 2.0`). It may only edit the
uncertain middle.

**Run:**

```bash
$py tools/skyfix/probe.py --limit 20 --lock 0       # freeze the network out entirely
```

| | IoU | precision | recall | pred/GT | above +2 m |
|---|---|---|---|---|---|
| network on | 22.19 | 33.57 | 39.56 | 1.18 | **1.47 %** |
| network off (`--lock 0`) | 8.24 | 8.28 | 94.31 | 11.39 | **96.13 %** |

**What just happened — this is the most important thing in the tour.** With the network gone,
94 % recall and 11× over-prediction: every unobserved voxel sits at log-odds 0, which is above
τ, so *the default state of the grid is "everything is solid"*. Above 2 m it is 96 % occupied.

**The ceiling is not something the network invents. It is what the network is there to carve
away.** The bug in `reports/the_ceiling_bug.md` was never "the model hallucinated a roof" — it
was "the model was never given a reason to carve one particular region, so the default
survived." Once you see that, the whole failure makes sense.

## Stop 3 · `core/run/eval_target.py` — a prediction becomes a score

**The question:** what exactly is being measured, and under what rules?

**Read:** the module docstring — three protocols (`stream`, `past5`, `occany_fwd`) and which
are causal. Then in `run()`, find `comp = (s_fin >= tau) & valid`. That single line is the
prediction every headline number scores. Note that `tau` is read from the manifest and there
is deliberately **no `--threshold` flag**: the threshold was frozen on KITTI-360 before any
target dataset was opened.

**Run** — sweep τ straight off the dumped log-odds, no GPU, no re-inference:

```bash
$py - <<'EOF'
import numpy as np, glob
fs=sorted(glob.glob('output/skyfix/pred_seed2/*.npz'))[:40]
lo,gt=[],[]
for f in fs:
    z=np.load(f); n=int(np.prod(tuple(z['dims'])))
    k=np.unpackbits(z['keep'])[:n].astype(bool)
    lo.append(z['logodds'].reshape(-1)[k].astype(np.float32))
    gt.append(z['gt_label'].reshape(-1)[k]!=0)
lo,gt=np.concatenate(lo),np.concatenate(gt)
for t in (-1.0,-0.5,-0.3125,-0.125,0.0,0.5,1.0):
    p=lo>=t; tp=(p&gt).sum(); fp=(p&~gt).sum(); fn=(~p&gt).sum()
    print(f"{t:+7.3f}  IoU {tp/(tp+fp+fn)*100:6.2f}  P {tp/(tp+fp)*100:6.2f}  R {tp/(tp+fn)*100:6.2f}")
EOF
```

```
 -1.000  IoU  14.14  P  14.42  R  87.92
 -0.500  IoU  23.26  P  31.19  R  47.78
 -0.312  IoU  22.07  P  33.24  R  39.63     <- the frozen threshold
 -0.125  IoU  20.22  P  34.22  R  33.08
 +1.000  IoU  12.24  P  44.71  R  14.42
```

**What just happened.** τ = −0.5 would score better *on this data*. Choosing it would also be
cheating — you would have tuned on the target dataset, and the number would no longer mean
"transferred without adaptation". This is the discipline the whole `gates/` directory exists
to enforce. The gap between the frozen and the best-possible threshold is how you *measure*
your calibration, not how you improve your score.

## Stop 4 · `core/evaluation/metrics.py` — how a number is made

**The question:** can you derive the headline yourself?

**Read:** the docstring. Counts are stored as integers per clip and summed *before* any
division — "averaging per-clip IoUs would silently weight a clip with 200 occupied voxels
like one with 80,000." Then `summarize`: `iou = tp / (tp + fp + fn)`, and `ssc_miou` is the
plain mean over classes present.

**Run** — rebuild a published number from raw counts:

```bash
$py - <<'EOF'
import numpy as np, json
z=np.load('artifacts/gate8c1/counts_semantickitti_past5_seed0_completion.npz')
tp,fp,fn=[z[k].astype(np.int64).sum(0) for k in ('tp','fp','fn')]
den=tp+fp+fn; iou=np.where(den>0, tp/np.maximum(den,1), np.nan)
print("recomputed mIoU:", float(np.nanmean(iou[den>0])))
print("published      :", json.load(open('artifacts/gate8c1/gate8c1_results.json'))
      ['runs']['semantickitti']['past5']['0']['semantic']['completion']['ssc_miou'])
EOF
```

Both print `0.026502...`. **Every published number in this repo can be rebuilt from stored
integer counts.** You never have to take a JSON file's word for anything.

---

# Part 2 — training: where the model came from

## Stop 5 · `core/supervision/rawtarget.py` — where truth comes from

**The question:** what does "ground truth" mean when the dataset's own labels were rejected?

**Read:** the three states at the top — `UNKNOWN=0`, `FREE=1`, `OCCUPIED=2` — then the block
at line 172, `open sky is FREE, not unknown`. This is the fix that was worth +78 %.

The insight in one sentence: **unmeasured is not occluded.** A Velodyne ray never climbs above
its topmost ring, so sky is unmeasured — but nothing stands between it and the sensor, and the
column beneath it was measured down to a surface. So it can be declared free.

**Run** — measure the supervision gap that caused the bug:

```bash
$py - <<'EOF'
import numpy as np, glob, random
random.seed(0)
fs=sum([sorted(glob.glob(f'/media/SSD1/MINH_DATASETS/lingbot_gate8c1/samples/2013_05_28_drive_{d}_sync/*.npz'))
        for d in ('0003','0007','0010')], [])
V=np.zeros(32)
for p in random.sample(fs,40):
    V+=np.unpackbits(np.load(p)['gt_valid'])[:2097152].reshape(256,256,32).sum((0,1))
for k in (31,27,20,10):
    print(f"z={k:2d}  height {-2+(k+0.5)*0.2:+5.2f} m   {V[k]/(40*65536)*100:6.2f}% supervised")
EOF
```

```
z=31  height +4.30 m     0.98% supervised
z=27  height +3.50 m     3.27% supervised
z=20  height +2.10 m    20.15% supervised
z=10  height +0.10 m    64.73% supervised
```

Those are the **old** targets. 0.98 % supervision at the top of the grid, against SemanticKITTI
scoring 98.99 % of that layer and calling it empty. That mismatch *is* the bug.

## Stop 6 · `core/model/losses.py` — what it optimizes, and where it is silent

**Read** `balanced_weights`, at line 37. Three lines, and the second is the whole story:

```python
w = gt_valid.float()
```

The loss weight is **exactly zero** on `UNKNOWN` voxels, in both `focal_bce` and `soft_dice`.
Put this next to Stop 5 and the failure is fully explained: 99 % of the top layer was UNKNOWN,
UNKNOWN gets zero gradient, so nothing ever told the model the sky was empty — and Stop 2
showed that "solid" is the grid's default. Three files, one bug.

**This is the pairing to remember.** Not a fact about this codebase — a habit. When a model
fails somewhere strange, ask what the loss was *silent* about there.

## Stop 7 · `core/run/train.py` — the loop

**Read:** the docstring, then how a batch is drawn and how `balanced_weights` is applied. Don't
run it — a seed is roughly 25 GPU-min. Read it to know where you *would* intervene.

## Stop 8 · `core/datasets/kitti360_splits.py` — the rules of the game

**The question:** what is this code forbidden to look at, and how is that enforced?

**Read:** `FORBIDDEN_PATH_TOKENS` and `assert_no_target_access`. The claim of the whole project
is "trained and selected on KITTI-360 alone, then evaluated on untouched targets". A claim like
that is worthless unless something *mechanically* prevents violating it.

**Run:**

```bash
$py -c "
import sys; sys.path.insert(0,'.')
from core.datasets.kitti360_splits import assert_no_target_access, TargetAccessViolation
for p in ('/media/SSD1/MINH_DATASETS/sscbench_kitti360/preprocess/labels/x_1_1.npy',
          'data/kitti/dataset/sequences/08/voxels/000000.label',
          '/media/welf/MINH/datasets/kitti360/KITTI-360/data_3d_raw/x/velodyne_points/data/0.bin'):
    try: assert_no_target_access([p]); print('ALLOWED', p)
    except TargetAccessViolation: print('BLOCKED', p)
"
```

SSCBench's own completion labels: **blocked**. SemanticKITTI voxels: **blocked**. Raw KITTI-360
Velodyne: allowed. `core/datasets/audit.py` goes further and monkeypatches `builtins.open`,
`np.load` and `np.fromfile` during the prediction stage, so a violation raises rather than
silently producing a better number.

---

## What you now own

The eight files, and the three ideas that make them a method rather than a pile:

1. **The network only edits what the map is unsure about** (`apply_residual`). Everything else
   passes through untouched.
2. **The default state of an unobserved voxel is occupied**, so the model's real job is
   carving, not filling. Every failure mode reads differently once you know this.
3. **Zero loss weight is invisible until something else scores that region.** The ceiling bug
   lived at the seam between what training ignored and what evaluation measured.

## Where to go next

Turn a dial and predict the answer *before* you run it. You will learn more from one wrong
prediction than from rereading any of this.

- `--pad-z 4` — pads the grid with a correctly-encoded "unobserved" margin so the convolution
  boundary sits outside the real volume. Does the ceiling change? Should it, given Stop 2?
- `--frames 20` — if 1 frame ≈ 5 frames, does 20 help at all? What does that say about the
  method?
- `--lock 5.0` — let the network overwrite voxels the map is confident about. Better or worse,
  and what does the answer tell you about where the map is trustworthy?
- The real open problem: **63.9 % of remaining false positives sit at or below −1.0 m**, in
  layers carrying 82.9 % of all true positives. The road surface is being predicted too thick.
  That is the live research question, and it is fully yours.

Reference: `README.md` (orientation) · `REPRODUCE.md` (numbers, currently pre-fix) ·
`reports/the_ceiling_bug.md` (the full case study) · `core/README.md` (module map) ·
`gates/README.md` (why the history looks like that).
