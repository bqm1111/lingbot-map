"""Gate 7C — bounded, training-free diagnostic on the Gate-7B streaming system.

Nothing is trained; no selector is learned. Ground truth is read only inside
``gate6.targets`` (metrics) and ``gate7c.oracle`` (an explicitly isolated diagnostic).
"""
