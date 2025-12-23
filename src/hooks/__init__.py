"""Hooks infrastructure for experimental modifications to training behavior.

This module provides a hook system that enables low-level experimentation by
allowing hooks to intercept and observe (or modify) training behavior without
changing core training code. Forward hooks can log activation statistics for
debugging and analysis, while update hooks can transform gradients for
experimental learning rules.
"""

from src.hooks.forward_hooks import ActivationStatsHook, ForwardHook
from src.hooks.registry import HookRegistry
from src.hooks.update_hooks import IdentityUpdateHook, UpdateHook

__all__ = [
    "HookRegistry",
    "ForwardHook",
    "ActivationStatsHook",
    "UpdateHook",
    "IdentityUpdateHook",
]

