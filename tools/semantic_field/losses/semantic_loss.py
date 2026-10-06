"""B4 losses. All features are 1024-d; text embeddings [K, 1024] unit; ``mu`` [1024] is the frozen training-drive mean.

    text logits         logit_k = scale * cos(f, text_k)                  (raw unit features, scale = 40)
    text distillation   L_text = KL(P_teacher || P_student)
    centered feature    L_feature = 1 - cos(normalize(f_teacher - mu), normalize(f_student - mu))
    masked target (A)   L_A = lambda_text * L_text + lambda_feature * L_feature
    delayed target (B)  L_B = KL(q_delayed || P_student)   (the stored delayed target is a class distribution)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def text_log_probs(features: torch.Tensor, text: torch.Tensor, scale: float) -> torch.Tensor:
    """[N, 1024] -> [N, K] log-probabilities of the frozen text bank."""
    return torch.log_softmax(scale * (F.normalize(features, dim=-1) @ text.T), -1)


def text_distillation(teacher_probs: torch.Tensor, student_features: torch.Tensor, text: torch.Tensor, scale: float) -> torch.Tensor:
    """Mean KL(P_teacher || P_student); ``teacher_probs`` [N, K] rows sum to 1."""
    t = teacher_probs.float().clamp_min(1e-8)
    return (t * (t.log() - text_log_probs(student_features.float(), text, scale))).sum(-1).mean()


def centre(features: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    return F.normalize(features - mu, dim=-1)


def centered_feature_loss(teacher_features: torch.Tensor, student_features: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    return (1 - (centre(teacher_features.float(), mu) * centre(student_features.float(), mu)).sum(-1)).mean()


def masked_reconstruction_loss(teacher_features, student_features, text, mu, scale, lambda_text, lambda_feature) -> dict:
    p_teacher = torch.softmax(scale * (F.normalize(teacher_features.float(), dim=-1) @ text.T), -1)
    lt = text_distillation(p_teacher, student_features, text, scale); lf = centered_feature_loss(teacher_features, student_features, mu)
    return {"loss": lambda_text * lt + lambda_feature * lf, "text": lt, "feature": lf}


def delayed_loss(delayed_probs, student_features, text, scale) -> dict:
    lt = text_distillation(delayed_probs, student_features, text, scale)
    return {"loss": lt, "text": lt}


__all__ = ["text_log_probs", "text_distillation", "centre", "centered_feature_loss", "masked_reconstruction_loss", "delayed_loss"]
