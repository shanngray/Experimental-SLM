"""Update hooks for transforming gradients during training.

This module provides update hooks that can receive and transform gradients
during the optimizer step. Update hooks enable experimental learning rules
and gradient modifications.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class UpdateHook(ABC):
    """Base class for update hooks.
    
    Update hooks receive and can transform gradients during the optimizer step.
    They are called after backward pass but before parameter update.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this hook."""
        pass
    
    @abstractmethod
    def __call__(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Transform gradients.
        
        Args:
            gradients: Dictionary mapping parameter names to gradient tensors.
        
        Returns:
            Transformed gradients dictionary. Can be the same dictionary
            (for identity/read-only hooks) or a new dictionary with modified gradients.
        """
        pass


class IdentityUpdateHook(UpdateHook):
    """Identity update hook that passes gradients through unchanged.
    
    This is the default update hook that doesn't modify gradients.
    Training behavior with this hook is identical to no hooks.
    """
    
    @property
    def name(self) -> str:
        """Return the name of this hook."""
        return "identity"
    
    def __call__(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Pass gradients through unchanged.
        
        Args:
            gradients: Dictionary mapping parameter names to gradient tensors.
        
        Returns:
            The same gradients dictionary, unchanged.
        """
        return gradients

