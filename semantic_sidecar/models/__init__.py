"""Sidecar architectures: everything trainable in this project lives here."""

from semantic_sidecar.models.linear_sidecar import LinearSidecar, RidgeAccumulator
from semantic_sidecar.models.semantic_sidecar import SemanticSidecar, build_sidecar, parameter_report

__all__ = [
    "LinearSidecar",
    "RidgeAccumulator",
    "SemanticSidecar",
    "build_sidecar",
    "parameter_report",
]
