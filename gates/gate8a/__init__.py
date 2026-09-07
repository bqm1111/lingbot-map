"""Gate 8A — source-only prior and calibration ablation.

Gate 8 trained a completion module whose held-out KITTI-360 binary IoU (0.2098) fell
*below* the trivial "declare every valid voxel occupied" line (0.2509) while beating it on
both training sources. Gate 8A asks one question: is that a sampling/calibration artefact
or a representation failure? Everything upstream of the completion head stays frozen; only
the crop sampler, the occupancy loss and the decision threshold move, and every choice is
made on SemanticKITTI 08 + Occ3D validation alone.
"""
