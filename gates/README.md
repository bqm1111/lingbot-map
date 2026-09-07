# gates/ — the research history

Each gate asked **one question**, answered it, and was then frozen. A gate is not a
library: later gates deliberately do not edit earlier ones, so an old gate keeps working
against the artifacts it produced. If you want the pipeline that is *currently running*
rather than the history, read [`../core/README.md`](../core/README.md).

Every gate has four siblings outside this directory, all keyed by the same name:

```
gates/<gate>/        the package        tests/<gate>/      its tests
tools/<gate>/        its scripts        configs/<gate>/    its frozen config
                     artifacts/<gate>/  its results        reports/<gate>/  its write-up
```

## The gates

| gate | question it asked |
|---|---|
| `depth_gate` | which monocular depth estimator, under which gauge |
| `scale_gate` | the metric-scale problem, on KITTI odometry |
| `voxel_gate` | voxel-space features and the inference-computable correction region |
| `voxel_gate_validation` | does that correction region survive validation |
| `gate6` | frozen Trident semantic lifting; multi-dataset coverage decomposition |
| `gate7a` | completion reachability and oracle-envelope diagnosis (oracle analysis, not deployable) |
| `gate7b` | native causal streaming replay; metric-gauge stability; complementary depth |
| `gate7c` | a bounded, training-free diagnostic on the 7B streaming system |
| `gate8` | causal semantic memory with privileged completion |
| `gate8a` | source-only prior and calibration ablation |
| `gate8b` | leave-one-dataset-out transfer |
| `gate8c0` | KITTI-360 alignment and learnability audit |
| `gate8c1` | KITTI-360-only training on rebuilt raw-LiDAR supervision |
| `gate8d` | dense targets and the future-frame protocol |

Nothing in `gate6`, `gate7a`, `gate7b` or `gate7c` is trained: no optimizer, no backward
pass. They run frozen checkpoints and fixed analytic fusion rules declared in
`configs/` *before* evaluation.

## The line that is still live

```
gate7b  →  gate8  →  gate8a  →  gate8c0  →  gate8c1
```

Each handed the next a specific problem:

- **gate8** trained a completion module whose held-out KITTI-360 IoU (0.2098) fell *below*
  the trivial "declare every valid voxel occupied" line (0.2509), while beating it on both
  training sources.
- **gate8a** asked whether that was sampling/calibration or representation. Every
  source-side fix left the completion at chance on held-out KITTI-360.
- **gate8b** asked whether that was KITTI-360-specific or general, via the two missing
  leave-one-dataset-out folds.
- **gate8c0** found the reason it looked unlearnable: SSCBench-KITTI-360's own `_1_1.npy`
  completion label **contradicts its own LiDAR** — a ground-truth sweep lands on a
  label-*free* voxel 71% of the time, against 1.7% on SemanticKITTI.
- **gate8c1** therefore never opens that label. It rebuilds supervision from raw Velodyne,
  trains and selects on KITTI-360 alone behind a dataset firewall, and only then evaluates
  on untouched SemanticKITTI and Occ3D.

Where that ended up: `artifacts/gate8c1_source_sanity/report.md`. Short version — a fresh
head cannot fit sixteen training clips, and the incremental mapper has 11.66% precision
against the raw-LiDAR target, so the fault sits upstream of the network.

## Two things to know before you touch this

**Gate 8C-1's completed audits import from a snapshot, not from here.**
`tools/gate8c1/{source_sanity,padding_source_probe,groupnorm_stat_probe}.py` set

```python
sys.path[:0] = [str(REF), str(ROOT), str(ROOT / 'gates')]
```

where `REF = artifacts/gate8c1_original_audit/code_used/`. `gate8` resolves from that
snapshot (it carries an `__init__.py`); everything else falls through to the live tree
here. Their `invariants.json` asserts exactly that. Do not "clean up" those imports to
`gates.gate8` — it would silently re-point three finished audits at live code.

**Paths are relative to the repo root, not to this directory.** Modules here compute the
repo root as `dirname(__file__)/../..`. If you move a package between `gates/` and the
top level again, fix those first — `gates/scale_gate/config.py` defines `REPO_ROOT`, and
most of `tools/` reads its `artifacts/` paths from it.
