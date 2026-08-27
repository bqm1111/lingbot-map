"""Sensor-conditioned metric anchoring for frozen LingbotMap.

A minimal, evidence-driven prototype that tests whether sparse metric depth or
pose observations can anchor LingbotMap's scale and reduce long-horizon drift on
subsequent image-only frames, without retraining LingbotMap.

Nothing in this package modifies, wraps or backpropagates through LingbotMap.
It consumes cached predictions produced by ``scripts/cache_lingbot_predictions.py``.
"""

__all__ = [
    "conventions",
    "datasets",
    "prompts",
    "anchors",
    "metrics",
    "model",
]
