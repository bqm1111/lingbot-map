"""The research gates, in order. Each one asked a single question and answered it.

Every gate is a self-contained package plus its scripts in ``tools/<gate>/``, its tests
in ``tests/<gate>/``, its config in ``configs/<gate>/`` and its results in
``artifacts/<gate>/``. Nothing here is a library for general use: a gate is a frozen
record of one experiment, and later gates deliberately do not edit earlier ones.

    depth_gate              monocular depth: which estimator, under which gauge
    scale_gate              the metric-scale problem, on KITTI odometry
    voxel_gate              voxel-space features and the correction region
    voxel_gate_validation   whether that correction region survives validation

    gate6                   frozen Trident semantic lifting; multi-dataset coverage
    gate7a                  envelopes, transport and frustum geometry over gate6's state
    gate7b                  native causal streaming replay; metric-gauge stability
    gate7c                  (placeholder)
    gate8                   causal semantic memory + privileged completion
    gate8a                  source-only prior and calibration ablation
    gate8b                  leave-one-dataset-out transfer
    gate8c0                 KITTI-360 alignment and learnability audit
    gate8c1                 KITTI-360-only training on rebuilt raw-LiDAR supervision
    gate8d                  dense targets and future-frame protocol

The chain that is still live is gate7b -> gate8 -> gate8a -> gate8c0 -> gate8c1.
``core/`` holds exactly that chain, extracted and reorganised by role; read
``core/README.md`` first if you want the pipeline rather than the history.
"""
