# Gate 8C-1 — the ceiling slab is a convolution-padding bug

Reported 2026-09-04: "the ceiling of the environment in SemanticKITTI turns all red".
It is a real defect, not a scene prior. This is the diagnosis, the fix, and what the fix
does and does not recover.

## The bug

`gate8/net.py::_block` builds every convolution as `nn.Conv3d(..., 3, padding=1)`, which
pads with **zeros**. Input channels 2 and 3 are `observed` and `1 - observed`
(`gate8/targets.py::unpack_sample`), so in every voxel that can occur they **sum to 1**.
Zero padding manufactures a boundary vector where **both are 0** — a state absent from the
training distribution. The network's response to that impossible input propagates inward,
and since the grid is only **32 voxels deep in z**, it reaches a large share of the ceiling
and the floor.

## Evidence it is a boundary effect, not a learned prior

Median final log-odds by height, SemanticKITTI, 12 anchors, frozen checkpoint:

| z (m) | +1.9 | +2.5 | +3.1 | +3.5 | +3.9 | +4.3 |
|---|---|---|---|---|---|---|
| median final log-odds | −1.06 | −0.78 | −0.54 | −0.30 | −0.14 | **+0.13** |
| ground-truth occupied | 1.98 % | 0.88 % | 0.20 % | 0.06 % | 0.02 % | **0.01 %** |

The network's confidence rises monotonically toward the boundary exactly where the evidence
falls to nothing. It does the same at the floor. No scene prior has that shape; a zero-padded
convolution does.

Two things then compound it: the frozen threshold is **negative** (τ = −0.125), so a voxel
with no opinion is called occupied; and the ceiling is **almost unsupervised** in training —
in the KITTI-360 targets the top layer is only **1.12 %** supervised (the loss masks the rest
correctly, so it receives no gradient at all).

## The fix

`padding_mode="replicate"` on every 3×3×3 convolution — the truthful statement "outside the
grid looks like its edge" instead of an impossible vector. Now configurable
(`configs/gate8c1_padfix/`), stored in the checkpoint, and **defaulting to `"zeros"` so every
earlier gate's numbers stay bit-identical**.

An inference-only patch was tried first and rejected: the model was *trained* with the
defect and had adapted to it, so patching only at test time is a train/test mismatch and
costs source-domain IoU (11.33 → 8.99). The fix requires retraining, which it got — three
seeds, same config, same data, 15 min each, **0 firewall violations**.

## What the fix recovers

Threshold re-selected per seed by the gate's own rule (maximise full-grid IoU on KITTI-360
drive 0006 — source domain only, firewall intact). 24 SemanticKITTI anchors.

Occupancy by height, seed 0 (% of evaluated voxels predicted occupied):

| z (m) | ground truth | zeros | **replicate** |
|---|---|---|---|
| +3.1 | 0.20 | 2.28 | **1.27** |
| +3.3 | 0.11 | 10.34 | **1.23** |
| +3.5 | 0.06 | 42.95 | **1.09** |
| +3.7 | 0.03 | 66.72 | **1.09** |
| +3.9 | 0.02 | 90.99 | **1.43** |
| +4.1 | 0.02 | 97.94 | 73.94 |
| +4.3 | 0.01 | 98.25 | 73.32 |

**The slab is gone from +3.1 to +3.9 m** — 66.7 % → 1.1 % at +3.7 m. SemanticKITTI IoU:

| seed | zeros | replicate |
|---|---|---|
| 0 | 10.55 | **14.70** |
| 1 | 14.85 | **17.10** |
| 2 | 14.32 | 12.83 |
| median | 14.32 | **14.70** |

Source-domain IoU is unchanged (10.4–10.7 either way), so this is not a source/target trade.

## What it does not fix

**The outermost two layers still flood (≈74 %).** Replicate padding removes the impossible
state but the array edge is still a distinguishable place, and those layers get ~1 %
supervision, so nothing constrains them. That residue is a **supervision-coverage** problem
in the target builder, not a padding problem: `gate8c1/rawtarget.py` marks voxels no LiDAR
ray reached as UNKNOWN and excludes them from the loss, while both target benchmarks mark
the same sky voxels VALID and EMPTY and score us there. Closing that is the next fix.

Seed 2 also regresses (14.32 → 12.83), so the improvement is real but not uniform across
seeds — worth stating rather than quoting the median alone.

## Files

`gate8/net.py` (padding_mode, checkpointed), `configs/gate8c1_padfix/seed{0,1,2}.yaml`,
`artifacts/gate8c1/checkpoints/padfix_seed*_best.pt`,
`tools/gate8c1/{padding_fix,reselect_threshold,padfix_evaluate}.py`,
`artifacts/gate8c1/{padding_fix,reselect_threshold,padfix_evaluate}_*.json`.
The frozen `seed*_best.pt` checkpoints and every gate artifact are untouched.
