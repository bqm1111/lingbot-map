"""Gate 8B — leave-one-dataset-out transfer evaluation.

Gate 8A showed the completion collapses to chance on held-out KITTI-360 after every
source-side fix. Gate 8B asks whether that is a KITTI-360-specific failure or a general
one, by completing the two missing leave-one-dataset-out folds with exactly Gate 8A's
frozen method, and it adds the matched five-frame setting needed for an honest reference
comparison with OccAny.
"""
