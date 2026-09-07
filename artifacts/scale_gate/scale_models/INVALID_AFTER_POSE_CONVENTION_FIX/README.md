# INVALID_AFTER_POSE_CONVENTION_FIX

These checkpoints and training records were produced from LingBot caches in which
`pred_pose_c2w` held the **inverse** of the intended transform: `pose_encoding_to_extri_intri`
returns camera-to-world, but the cache builder inverted it following `demo.py`'s
"convert w2c to c2w" comment.

Consequences for these artifacts:

* `pose_features` (path length, span, step statistics) were computed from wrong camera
  centres, so every `combined_mlp` input vector is stale.
* `depth_mlp` inputs are unaffected (depth statistics only), but its *evaluation* used the
  wrong fusion transforms, so its downstream occupancy numbers are stale.
* Scale **targets** are largely unaffected: the pose estimator uses translation magnitudes,
  which the inversion approximately preserved (agreement median 0.0288 -> 0.0277).

Retained for provenance only. Do not load these for any result.
Superseded by the models in the parent directory, retrained on repaired caches.
