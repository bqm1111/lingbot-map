"""The executable stages, in pipeline order.

    build_targets  ->  validate_targets  ->  build_samples  ->  train  ->  eval_target

A leading underscore marks a module inherited from an earlier gate: it is
still executed (train.py takes its step_loss from _gate8a_train, which takes
its batching from _gate8_train) but it is not the entry point for this gate.
"""
