"""The feature objective, and the sparse read of the feature head that makes it cheap.

``L_feature`` is the mean cosine distance between the L2-normalised predicted feature and
the L2-normalised target feature, averaged over the voxels that carry an auxiliary target:

.. math:: L_\\text{feature} = \\frac{1}{|M|}\\sum_{v \\in M} \\left(1 - \\hat f_v \\cdot t_v\\right)

Targets are unit vectors already (``build_feature_targets`` normalises the fused mean), so
only the prediction is normalised here.

The head is a 1x1x1 convolution, which is a linear map applied independently per voxel, so
evaluating it at |M| ~ 30 000 supervised voxels is the same arithmetic as a dense forward
over 2 million and ~70x cheaper. :func:`features_at` does that; ``tests`` in
``verify.py`` asserts it matches ``CompletionUNet.predict_features`` exactly.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def features_at(net, trunk: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``[M, n_feat]``: the feature head read at flat voxel positions of the batch.

    ``trunk`` is ``[B, w, X, Y, Z]``; ``index`` indexes the flattened ``B*X*Y*Z`` lattice
    in the same C order as ``trunk.permute(0, 2, 3, 4, 1).reshape(-1, w)``.
    """
    head = net.head_feat
    w = head.weight.reshape(head.out_channels, head.in_channels)      # 1x1x1 -> [F, w]
    B, C = trunk.shape[0], trunk.shape[1]
    t = trunk.permute(0, 2, 3, 4, 1).reshape(-1, C)[index].float()
    return t @ w.float().T + head.bias.float()


def feature_cosine(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean cosine distance over supervised voxels. Empty support -> exactly 0."""
    if pred.numel() == 0:
        return pred.sum() * 0.0
    p = F.normalize(pred.float(), dim=1, eps=1e-8)
    t = F.normalize(target.float(), dim=1, eps=1e-8)
    return (1.0 - (p * t).sum(1)).mean()


@torch.no_grad()
def feature_diagnostics(pred: torch.Tensor, target: torch.Tensor,
                        train_mean: torch.Tensor) -> dict:
    """Cosine agreement, and the same number for a constant predictor.

    A high raw cosine similarity means little on its own: the training-set mean feature is a
    constant vector that already scores well, because SAM features occupy a narrow cone. The
    gap between the two is the part a prediction actually earned.
    """
    if pred.numel() == 0:
        return {}
    p = F.normalize(pred.float(), dim=1, eps=1e-8)
    t = F.normalize(target.float(), dim=1, eps=1e-8)
    m = F.normalize(train_mean.float().reshape(1, -1), dim=1, eps=1e-8)
    cos = (p * t).sum(1)
    return {"cosine_mean": float(cos.mean()), "cosine_median": float(cos.median()),
            "cosine_mean_predictor": float((m * t).sum(1).mean()),
            "pred_dispersion": float(p.std(0).mean()),
            "target_dispersion": float(t.std(0).mean()),
            "n": int(pred.shape[0])}


__all__ = ["features_at", "feature_cosine", "feature_diagnostics"]
