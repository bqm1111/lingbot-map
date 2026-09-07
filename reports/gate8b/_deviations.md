1. **Four additive edits to files Gate 8A treated as frozen, each behaviour-preserving for
   every existing caller.** (a) `gate8/sources.py` gained a `register_source` hook so a later
   gate can add a source without touching the existing branches; every pre-existing path and
   segment builder is unchanged and a test asserts it. (b) `sscbench_kitti360/adapter.py::
   load_target` gained a `sequence` argument defaulting to the validation drive, and (c)
   `gate6/targets.py` passes `rec.get("sequence", <val drive>)` -- every official validation
   record already carries the val drive, so Gates 5.2-8A resolve byte-identically (Gate 6 and
   Gate 8 tests pass unchanged). (d) `tools/gate8a/evaluate.py::run` gained an `art` output
   directory argument defaulting to the Gate 8A root, so Gate 8B could reuse the same
   streaming evaluator without writing into Gate 8A's artifacts.

2. **KITTI-360 training partition = official SSCBench train drives 0003, 0007, 0010; source-
   validation = the official val drive 0006.** All seven official train drives have raw images
   locally; the three chosen give 1 289 labelled anchors, matching `sk_train` (1 656) and
   `occ3d_train` (1 570) in size and keeping every stream inside the RoPE budget at keyframe
   interval 1. Their occupancy targets (`_1_1.npy`) were fetched from the official archive by
   byte-range reads (201 MB, 364 requests, 6 min); the drive poses came from the local
   `data_poses.zip`. The stream is defined by the pose file alone (every 5th SSCBench index);
   a target only marks an anchor.

3. **Equal per-source sampling probability** (p = 1/2 per training source, uniform within) is
   the brief's rule and differs from Gate 8A's uniform-over-files draw, which weighted a
   source by its sample count. Recorded so the two data draws are not conflated.

4. **Balanced within-fold validation set.** Gate 8A's trainer took the first 120 files of the
   concatenated validation list, which -- by construction of that list -- was SemanticKITTI
   only. With two validation domains per fold the same rule would silently validate on one
   domain, so each fold takes the first 60 files of *each* source-validation domain. This
   affects only which of {best, last} is called `best`; the cross-candidate selection is by
   macro AP on the full grids of both domains, exactly as in Gate 8A.

5. **The matched five-frame setting uses the Gate 5.2/6 per-clip LingBot caches, not the
   causal stream.** That is the point of the setting -- LingBot sees exactly the five frames --
   but it means the depth and poses of a given frame differ from the streaming setting's.
   The scale is the pinned G51-B scalar of every prior gate, forced into `ScaleState` (the
   five per-frame candidates would median to the same value; forcing avoids a float round
   trip). Every clip of all three benchmarks has a valid G51-B scale, so no clip is skipped.

6. **OccAny's "separate" pooling is two different operators depending on a flag.** With
   `is_geometry_only=True` its default is `use_dilation=True`: a 3x3x3 `max_pool3d`, i.e. a
   one-voxel Chebyshev dilation, not a vote. The vote (`use_dilation=False`) never removes
   occupancy and needs > 13.5 of 27 neighbours. In semantic mode the operator relabels
   occupied voxels and leaves occupancy untouched. All three are reproduced and checked
   bit-for-bit against the official function in the OccAny environment; the report gives
   dilation and vote as separate columns and never uses either for the primary result.

7. **The comparison with OccAny is a reference comparison, not an exact one.** OccAny's five
   frames start at the target and run *forward* (SemanticKITTI: offsets 0..+40 at stride 10;
   nuScenes: 5 samples ~1 s apart); ours end at the target and run *backward* (offsets -20..0
   at stride 5; nuScenes 0.5 s apart). The anchor sets, grids, masks, occupancy definition,
   pooling and aggregation match; the temporal sampling and the metric-scale source do not.

8. **Whole-system zero-shot is not claimed for any fold.** LingBot-Map's published training
   mixture includes KITTI-360, so the existing KITTI-360 fold is not whole-system zero-shot;
   MoGe-2's and Trident-H's training data were not audited here. Every fold is described as
   leave-one-dataset-out transfer with no target-domain training or adaptation of the
   completion module.

9. **The teacher-accuracy diagnostic reduces Occ3D through the frozen any-sub-voxel rule**
   (`reduce_occ3d_probs`: mean union probability over evidence-bearing sub-voxels, then
   argmax), exactly as every evaluated prediction is, rather than a majority of sub-voxel
   labels. It is an evaluation-only use of semantic ground truth.

10. **One diagnostic crashed and was re-run.** `teacher_accuracy.py` reduced the causally
    observed mask to the Occ3D evaluation grid only when the anchor had teacher evidence, so an
    anchor with none raised a shape error after the smoke test had passed. The reduction was
    moved outside that condition and the Occ3D and KITTI-360 runs restarted. The diagnostic
    is evaluation-only and touches nothing on the prediction path.
