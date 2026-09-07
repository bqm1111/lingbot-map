"""Gate 7B — native causal streaming replay, metric-gauge stability, complementary depth.

Nothing here is trained: no optimizer, no backward pass, no network is created. The
frozen LingBot-Map checkpoint is run in its **native direct streaming mode** over
chronological, deduplicated frame streams, and every fusion rule is a fixed analytic
update declared in ``configs/gate7b/streaming_replay_precommit.yaml`` before evaluation.
"""
__all__ = ["config", "streams", "scale", "voxmap", "rays", "evidence", "replay"]
