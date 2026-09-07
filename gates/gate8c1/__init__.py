"""Gate 8C-1 — KITTI-360-only training with rebuilt raw-LiDAR supervision.

Gate 8C-0 proved SSCBench-KITTI-360's `_1_1.npy` completion label contradicts its own
LiDAR (a ground-truth sweep lands on a label-*free* voxel 71 % of the time, against 1.7 %
on SemanticKITTI). This gate therefore never opens that label. It rebuilds KITTI-360
supervision directly from raw Velodyne sweeps through the chain Gate 8C-0 verified, trains
and selects on KITTI-360 alone behind a dataset firewall, and only then evaluates the
frozen models on untouched SemanticKITTI 08 and Occ3D-nuScenes.
"""
