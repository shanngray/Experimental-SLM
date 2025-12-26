"""Forward hooks for observing activations during training.

This module provides forward hooks that can observe activations during the
forward pass without modifying outputs. Forward hooks are read-only observers
that enable debugging and analysis of model behavior.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class ForwardHook(ABC):
    """Base class for forward hooks.
    
    Forward hooks observe activations during the forward pass without
    modifying outputs. They are called after activations are computed
    but before loss computation.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this hook."""
        pass
    
    @abstractmethod
    def __call__(self, activations: Any) -> None:
        """Call the hook with activations.
        
        Args:
            activations: Activations from the forward pass. Can be any type
                depending on what the hook observes (e.g., tensor, dict, etc.).
        """
        pass


class ActivationStatsHook(ForwardHook):
    """Forward hook that logs activation statistics.
    
    Computes and logs mean and standard deviation of activations for
    debugging and analysis. This hook is read-only and does not modify
    activations.
    
    Attributes:
        log_interval: Log statistics every N steps (default: 1, logs every step).
        step_counter: Internal counter for tracking steps.
    """
    
    def __init__(self, log_interval: int = 1):
        """Initialize the activation stats hook.
        
        Args:
            log_interval: Log statistics every N steps (default: 1).
        """
        self.log_interval = log_interval
        self.step_counter = 0
    
    @property
    def name(self) -> str:
        """Return the name of this hook."""
        return "activation_stats"
    
    def __call__(self, activations: Any) -> None:
        """Log activation statistics.
        
        Computes mean and standard deviation of activations if they are
        tensors. Logs statistics according to log_interval.
        
        Args:
            activations: Activations to analyze. Can be a tensor or dict of tensors.
        """
        self.step_counter += 1
        
        if self.step_counter % self.log_interval != 0:
            return
        
        if isinstance(activations, torch.Tensor):
            self._log_tensor_stats(activations, "activations")
        elif isinstance(activations, dict):
            for key, value in activations.items():
                if isinstance(value, torch.Tensor):
                    self._log_tensor_stats(value, f"activations[{key}]")
    
    def _log_tensor_stats(self, tensor: torch.Tensor, name: str) -> None:
        """Log statistics for a tensor.
        
        Args:
            tensor: Tensor to analyze.
            name: Name to use in log message.
        """
        # Immediately convert to numpy/cpu to completely isolate from computation graph
        # This ensures we don't affect gradients, RNG state, or any PyTorch internals
        # by avoiding any tensor operations that might interact with the graph
        with torch.inference_mode():
            # Convert to numpy array immediately - this creates a completely independent
            # copy that has no connection to the computation graph
            numpy_array = tensor.detach().cpu().numpy()
            mean = float(np.mean(numpy_array))
            std = float(np.std(numpy_array))
            print(f"[ActivationStatsHook] {name}: mean={mean:.6f}, std={std:.6f}")


class QuantizationStatsHook(ForwardHook):
    """Forward hook that logs quantization-related statistics.
    
    Observes activations and logs statistics relevant to quantization,
    such as activation ranges, which can help with quantization calibration
    and debugging quantization-aware training.
    
    Attributes:
        log_interval: Log statistics every N steps (default: 1).
        step_counter: Internal counter for tracking steps.
    """
    
    def __init__(self, log_interval: int = 1):
        """Initialize the quantization stats hook.
        
        Args:
            log_interval: Log statistics every N steps (default: 1).
        """
        self.log_interval = log_interval
        self.step_counter = 0
    
    @property
    def name(self) -> str:
        """Return the name of this hook."""
        return "quantization_stats"
    
    def __call__(self, activations: Any) -> None:
        """Log quantization-related statistics.
        
        Computes min, max, and range of activations which are useful
        for understanding quantization behavior and calibration.
        
        Args:
            activations: Activations to analyze. Can be a tensor or dict of tensors.
        """
        self.step_counter += 1
        
        if self.step_counter % self.log_interval != 0:
            return
        
        if isinstance(activations, torch.Tensor):
            self._log_quantization_stats(activations, "activations")
        elif isinstance(activations, dict):
            for key, value in activations.items():
                if isinstance(value, torch.Tensor):
                    self._log_quantization_stats(value, f"activations[{key}]")
    
    def _log_quantization_stats(self, tensor: torch.Tensor, name: str) -> None:
        """Log quantization statistics for a tensor.
        
        Args:
            tensor: Tensor to analyze.
            name: Name to use in log message.
        """
        with torch.inference_mode():
            numpy_array = tensor.detach().cpu().numpy()
            min_val = float(np.min(numpy_array))
            max_val = float(np.max(numpy_array))
            mean_val = float(np.mean(numpy_array))
            std_val = float(np.std(numpy_array))
            range_val = max_val - min_val
            
            print(
                f"[QuantizationStatsHook] {name}: "
                f"min={min_val:.6f}, max={max_val:.6f}, "
                f"range={range_val:.6f}, mean={mean_val:.6f}, std={std_val:.6f}"
            )

