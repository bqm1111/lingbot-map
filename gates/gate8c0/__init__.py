"""Gate 8C-0 — KITTI-360 alignment and learnability audit.

Gate 8B left one thing unexplained: completion is at chance on KITTI-360 (AP/prevalence
1.04x held out) and only 1.19-1.31x even when KITTI-360 drives are a *training* source.
This gate separates a pipeline defect from an unlearnable target, a cross-drive shift and
an information-poor causal input, without changing a single frozen component.
"""
