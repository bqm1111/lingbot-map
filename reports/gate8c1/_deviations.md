1. **The released OccAny model could not be re-run.** Its stack is missing `croco`,
   `dust3r`, `mast3r`, `depth_anything_3`, `diffusers` and `torchsparse`; two build CUDA
   extensions. Running it would also need 6.4 GB of weights and a GroundingDINO box pass
   over all 1 345 evaluation clips, then a generative diffusion model over them — several
   GPU-hours plus install risk, outside a gate whose instruction is to stop after
   reporting. **This was not attempted rather than attempted-and-failed**, and the
   comparison is therefore *published numbers vs ours*, with OccAny's own evaluator and
   pooling applied to our predictions. `occany_reproduction.md` records exactly what is
   missing and what would be required.

2. **A within-sweep carving exception, stated explicitly.** The brief's rule 8 sends
   occupied-and-free voxels to unknown. Applied naively that also fires *within* a single
   sweep, where a grazing ray passes through a voxel another ray of the same sweep
   terminated in; that made ~73 % of surface voxels self-conflicting and collapsed the
   target. A sweep therefore never carves a voxel it measured itself — standard occupancy
   mapping, and not a conflict resolution, since a measured return *is* a direct
   observation of occupancy. Genuine cross-sweep disagreement (dynamic objects, thin
   structure, occlusion boundaries) is still left unknown, at ~8 % of touched voxels.

3. **The future window is 20 *stream* frames, not 20 native frames.** Gate 8's privileged
   horizon is frozen at 20 frames of the stream, which for KITTI-360 is stride 5 in native
   frames — about 100 native frames, ~10 s. Gate 8C-0 showed the sensor leaves the 51.2 m
   box after 11–20 stream frames, so this is the horizon that actually covers the volume.
   A 20-*native*-frame window would cover only a fifth of it.

4. **Anchors are all eligible *stream* frames, not all raw frames.** The frozen LingBot
   stream for these drives is sampled at SSCBench-index stride 5, and re-streaming at
   stride 1 would change a frozen cache. "All eligible raw frames" is therefore realised as
   all eligible frames of the frozen stream: 1 358 training anchors versus the 1 276 that
   SSCBench labels would have allowed, with the three mechanical exclusions (scale warm-up,
   future-window availability, pose availability) recorded per drive in the audit.

5. **Drive 0006 targets are built at anchor stride 3** (590 of 1 768 eligible anchors) to
   bound build cost. The subsample is deterministic and was fixed before any model existed,
   so it cannot have been chosen against a result.

6. **`tools/gate8c1/select.py` had to be renamed `selection.py`.** Python puts a script's
   own directory first on `sys.path`, so a module named `select` shadows the standard
   library's and breaks `subprocess` for every other tool in that directory. It cost one
   failed validation-sample build. The same trap was recorded in Gate 8A.

7. **One crash during the validation-sample build**, fixed and re-run: the per-frame
   `T_cam_to_grid` is populated only for official SSCBench anchor frames, and Gate 8C-1
   deliberately anchors on frames that are not. For KITTI-360 that transform is a per-drive
   calibration constant, so it is now read from `DriveGeometry.rect_cam_to_velo`; it is
   byte-identical to the per-frame value on every frame that had one, so the 26 samples
   written before the fix remain valid.

8. **`np.fromstring` deprecation warning** from the frozen
   `sscbench_kitti360.adapter.parse_calibration`. Pre-existing, out of scope, parses
   correctly.
