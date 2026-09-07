1. **`gate8/net.py` was refactored, not changed.** The evaluator needs the continuous final
   log-odds, which Gate 8's `Completer.complete` threw away. `complete` was split into
   `raw(q, grid) -> (final_logodds, probs)` plus the frozen `> 0` decision, so the score
   path and the Gate 8 decision path cannot drift apart. The 25 Gate 8 tests pass unchanged
   and `complete()` is now literally `raw()` followed by `> 0`. `stage0_audit.json` records
   the post-refactor hash; this is the only edit to a frozen file in the gate.

2. **The checkpoint pool is {best, last} per cell, not every 500-step checkpoint.** Gate 8's
   trainer keeps only the lowest-validation-loss checkpoint and the final one, and the brief
   says to reuse the Gate 8 run rather than retrain it. Selecting over per-500-step
   checkpoints would therefore have given the three new cells a larger pool than cell A. The
   Gate 8A trainer keeps exactly the same two, so all four cells enter selection with two
   candidates each, chosen by the same within-cell rule.

3. **Cell A was not retrained.** Its two checkpoints are Gate 8's own
   (`artifacts/gate8/checkpoints/completion_{best,last}.pt`). The Gate 8A trainer's
   `focal_dice` path reproduces Gate 8's objective term for term, and
   `configs/gate8a/cellA_occcentred_focal.yaml` is asserted by test to agree with
   `configs/gate8/completion.yaml` on every shared key.

4. **The editable region on Occ3D's coarse evaluation grid.** The residual acts at 0.2 m and
   the benchmark scores at 0.4 m under the frozen any-sub-voxel occupancy rule. A coarse
   voxel is therefore called editable if *any* of its eight children is editable, and
   *forced occupied* if any child is locked occupied -- occupancy the residual cannot
   remove, which no baseline is allowed to remove either. On SemanticKITTI and KITTI-360 the
   two grids coincide and the region is exactly `abs(base_logodds) < 2.0`.

5. **Scores are histogrammed, not stored per voxel.** Storing the final log-odds for every
   voxel of every anchor would be ~40 GB. They are accumulated per anchor into 1 024 bins on
   [-16, +16] with 0.0 on a bin boundary, which makes AP, PR, AUROC and every threshold
   sweep exact to a bin width of 0.03125 log-odds. Brier and ECE are accumulated exactly,
   not from the histogram. A test checks AP and AUROC against scikit-learn on identically
   quantised scores.

6. **`tools/gate8a/select.py` had to be renamed `selection.py`.** Python puts a script's own
   directory first on `sys.path`, so a module named `select` shadows the stdlib `select` and
   breaks `subprocess` for every other tool in the same directory. This cost one failed run
   of `sampler_stats.py`.

7. **The uniform sampler sometimes draws a crop with no valid future-teacher voxel**, in
   which case the teacher-KL term for that sample is zero because its mask is empty. That is
   the existing `gate8.losses.semantic_kl` behaviour under an empty mask, not a Gate 8A
   change, but it is worth knowing when reading the training logs.

8. **The frozen selection file was rewritten once, before KITTI-360 was opened, to fix a
   provenance label.** `selection.py` looked up the winning cell's training record by the
   candidate name (`cellB_last`) instead of its config tag (`cellB_uniform_focal`), so the
   first file recorded cell A's sampler, loss and config path next to cell B's checkpoint.
   The checkpoint path, its sha256 and both thresholds were correct and are byte-identical
   after the fix; the diff is three provenance lines. Selection is a deterministic function
   of the source score blocks, so re-running it reproduced the same choice.

9. **The mapper's source-calibrated threshold degenerates to "everything occupied"**
   (τ = −16.0, the bottom of the score range). This is the declared rule applied honestly,
   not a bug: the incremental map's log-odds carry very little ranking information
   (AUROC 0.552 on SemanticKITTI, and its IoU on Occ3D is maximised by predicting every
   voxel), so no single global threshold weighted equally across the two sources beats the
   trivial line. The consequence is that "incremental mapper with its own source-calibrated
   threshold" and "every valid voxel occupied" are the *same* prediction on the held-out
   set, and the report treats `mapper_native` and the 0.4 m dilation as the meaningful
   mapper baselines.
