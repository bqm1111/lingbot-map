"""Gate 7A — completion reachability and oracle-envelope diagnosis.

Target-dependent **oracle analysis**, not a deployable prediction. Nothing here is
trained, no optimizer is constructed and no backward pass is run; the Gate-6 predictions,
caches, manifests and evaluation masks are read-only inputs.
"""
__all__ = ["config", "distance", "envelopes", "transport", "frustum", "pipeline"]
