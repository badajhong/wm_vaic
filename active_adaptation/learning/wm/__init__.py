"""Differentiable world-model (BPTT) teacher training.

``WM``/``WMConfig`` stay importable from this package path: Hydra targets
``active_adaptation.learning.wm.WM`` and saved checkpoints pickle
``active_adaptation.learning.wm.WMConfig``.
"""

from .algorithm import WM, WMConfig

__all__ = ["WM", "WMConfig"]
