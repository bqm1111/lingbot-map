1. **Stage 7 was not run, by the brief's own precondition.** It is authorised "only if
   Stages 1–6 show correct alignment and the model passes fixed-batch memorization".
   Memorization passes, but the KITTI-360 target is not geometrically consistent with its
   own LiDAR, so a KITTI-360-only training run would measure the defect rather than
   learnability. The descriptive map-channel comparison Stage 7 also asked for is
   independent of that gate and is reported (`stage7_channel_stats.json`,
   `fig_channels.png`); no conclusion rests on it.

2. **`gate8c0.transforms.inv` uses `np.linalg.inv`, not `(Rᵀ, −Rᵀt)`.** This matches the
   production path (`tools/gate8/evaluate._grid_to_world`). KITTI-360's shipped matrices are
   only approximately rigid — `R_rect_00` has determinant 0.999999548 and `cam0_to_world`
   1.000000611, because the published files carry six decimals — so a rigid inverse leaves a
   7e-5 m round-trip residual at 40 m. `inv_rigid` is kept and Stage 1 reports the difference
   rather than hiding it. The round-trip tolerance is 1 mm (1/200 voxel), chosen so it cannot
   be met by accident; measured worst error is 1.4e-12 m.

3. **The audit subset is 9 anchors for Stage 1, 37 for Stage 2, 24 for Stage 3, 8 for
   Stage 4, 40 samples for Stage 5 and 8 for Stage 6.** Stages 2–4 are O(minutes) per sample
   because each rebuilds oracle volumes and Euclidean distance transforms over a 2.1 M-voxel
   grid. The effects reported are 2–20× in size and consistent across every sample and every
   partition, so the subset sizes are not the limiting factor; the per-sample tables are in
   the artifacts for inspection.

4. **"All available past" is bounded.** A sweep taken more than 51.2 m back contributes no
   voxel to this grid, so Stage 3 walks back until the sensor leaves the box (cap 200 native
   frames). On these drives it resolves to 11–20 stream frames, which is why `all_past` and
   `20` agree to four decimals — that is the correct answer, not a truncation artefact.

5. **Two additive, behaviour-preserving edits from Gate 8B are still in place** and are
   re-hashed here: the `register_source` hook in `gate8/sources.py` and the `sequence`
   argument of `sscbench_kitti360.adapter.load_target`. Gate 8C-0 added none of its own; the
   `hashes.json` frozen list covers 29 files and reports drift on re-run.

6. **`sscbench_kitti360.adapter.parse_calibration` emits a NumPy deprecation warning**
   (`np.fromstring` on a text buffer). It is pre-existing, frozen by this gate's scope, and
   parses correctly; it surfaces as the single warning in the test run.

7. **The Stage-2 shift scan reports `(0, 0, 1)` as the best-scoring alignment and it was not
   adopted.** The brief forbids adopting a better-scoring alignment without confirmation from
   official metadata, and the confirmation went the other way: our unshifted voxelization
   reproduces SSCBench's own `.bin` at 99.90 % recall, while their `.bin` and their label
   disagree by the same +1 z. Shifting would break agreement with the official input to chase
   agreement with an inconsistent label.
