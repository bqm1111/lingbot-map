"""Semantic Sidecar: parameter-efficient open-vocabulary 3D reconstruction on top
of a frozen LingBot-MAP streaming geometry model.

The package never mutates :mod:`lingbot_map`; frozen features are captured with a
forward hook and all geometry is consumed read-only.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
