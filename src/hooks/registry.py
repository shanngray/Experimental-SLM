"""Hook registry system for managing hooks during training.

This module provides the HookRegistry class that manages hook registration,
loading from configuration, toggling, and execution.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.hooks.forward_hooks import ForwardHook
    from src.hooks.update_hooks import UpdateHook


class HookRegistry:
    """Registry for managing training hooks.
    
    Manages forward hooks (for observing activations) and update hooks
    (for transforming gradients). Supports loading hooks from configuration,
    enabling/disabling hooks, and executing hooks in registration order.
    
    Attributes:
        forward_hooks: List of registered forward hooks with their enabled status.
        update_hooks: List of registered update hooks with their enabled status.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the hook registry.
        
        Args:
            config: Optional configuration dictionary with hook definitions.
                Expected format:
                {
                    "hooks": {
                        "forward": [
                            {"name": "activation_stats", "enabled": True, ...}
                        ],
                        "update": [
                            {"name": "identity", "enabled": True, ...}
                        ]
                    }
                }
        """
        from src.hooks.forward_hooks import ForwardHook
        from src.hooks.update_hooks import UpdateHook
        
        self.forward_hooks: List[tuple[ForwardHook, bool]] = []
        self.update_hooks: List[tuple[UpdateHook, bool]] = []
        
        if config:
            self.load_from_config(config)
    
    def load_from_config(self, config: Dict[str, Any]) -> None:
        """Load hooks from configuration dictionary.
        
        Loads hook definitions from config and registers them according to
        their type (forward hooks, update hooks). Hooks can be enabled or
        disabled via configuration.
        
        Args:
            config: Configuration dictionary with hook definitions.
                Expected format:
                {
                    "hooks": {
                        "forward": [
                            {"name": "activation_stats", "enabled": True, ...}
                        ],
                        "update": [
                            {"name": "identity", "enabled": True, ...}
                        ]
                    }
                }
        """
        hooks_config = config.get("hooks", {})
        
        # Load forward hooks
        forward_configs = hooks_config.get("forward", [])
        for hook_config in forward_configs:
            hook = self._create_forward_hook(hook_config)
            enabled = hook_config.get("enabled", True)
            self.register_forward_hook(hook, enabled=enabled)
        
        # Load update hooks
        update_configs = hooks_config.get("update", [])
        for hook_config in update_configs:
            hook = self._create_update_hook(hook_config)
            enabled = hook_config.get("enabled", True)
            self.register_update_hook(hook, enabled=enabled)
    
    def _create_forward_hook(self, config: Dict[str, Any]) -> "ForwardHook":
        """Create a forward hook from configuration.
        
        Args:
            config: Hook configuration dictionary with "name" and other parameters.
        
        Returns:
            ForwardHook instance.
        
        Raises:
            ValueError: If hook name is unknown.
        """
        from src.hooks.forward_hooks import ActivationStatsHook, QuantizationStatsHook
        
        name = config.get("name")
        if name == "activation_stats":
            return ActivationStatsHook(**{k: v for k, v in config.items() if k != "name" and k != "enabled"})
        elif name == "quantization_stats":
            return QuantizationStatsHook(**{k: v for k, v in config.items() if k != "name" and k != "enabled"})
        else:
            raise ValueError(f"Unknown forward hook name: {name}")
    
    def _create_update_hook(self, config: Dict[str, Any]) -> "UpdateHook":
        """Create an update hook from configuration.
        
        Args:
            config: Hook configuration dictionary with "name" and other parameters.
        
        Returns:
            UpdateHook instance.
        
        Raises:
            ValueError: If hook name is unknown.
        """
        from src.hooks.update_hooks import IdentityUpdateHook
        
        name = config.get("name")
        if name == "identity":
            return IdentityUpdateHook(**{k: v for k, v in config.items() if k != "name" and k != "enabled"})
        else:
            raise ValueError(f"Unknown update hook name: {name}")
    
    def register_forward_hook(self, hook: "ForwardHook", enabled: bool = True) -> None:  # type: ignore
        """Register a forward hook.
        
        Args:
            hook: ForwardHook instance to register.
            enabled: Whether the hook is enabled (default: True).
        """
        self.forward_hooks.append((hook, enabled))
    
    def register_update_hook(self, hook: "UpdateHook", enabled: bool = True) -> None:  # type: ignore
        """Register an update hook.
        
        Args:
            hook: UpdateHook instance to register.
            enabled: Whether the hook is enabled (default: True).
        """
        self.update_hooks.append((hook, enabled))
    
    def enable_forward_hook(self, index: int) -> None:
        """Enable a forward hook by index.
        
        Args:
            index: Index of the hook in the registration order.
        
        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= len(self.forward_hooks):
            raise IndexError(f"Forward hook index {index} out of range")
        hook, _ = self.forward_hooks[index]
        self.forward_hooks[index] = (hook, True)
    
    def disable_forward_hook(self, index: int) -> None:
        """Disable a forward hook by index.
        
        Args:
            index: Index of the hook in the registration order.
        
        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= len(self.forward_hooks):
            raise IndexError(f"Forward hook index {index} out of range")
        hook, _ = self.forward_hooks[index]
        self.forward_hooks[index] = (hook, False)
    
    def enable_update_hook(self, index: int) -> None:
        """Enable an update hook by index.
        
        Args:
            index: Index of the hook in the registration order.
        
        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= len(self.update_hooks):
            raise IndexError(f"Update hook index {index} out of range")
        hook, _ = self.update_hooks[index]
        self.update_hooks[index] = (hook, True)
    
    def disable_update_hook(self, index: int) -> None:
        """Disable an update hook by index.
        
        Args:
            index: Index of the hook in the registration order.
        
        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= len(self.update_hooks):
            raise IndexError(f"Update hook index {index} out of range")
        hook, _ = self.update_hooks[index]
        self.update_hooks[index] = (hook, False)
    
    def call_forward_hooks(self, activations: Any) -> None:
        """Call all enabled forward hooks in registration order.
        
        Args:
            activations: Activations from the forward pass to pass to hooks.
        """
        for hook, enabled in self.forward_hooks:
            if enabled:
                hook(activations)
    
    def call_update_hooks(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Call all enabled update hooks in registration order.
        
        Update hooks can transform gradients. The output of one hook is passed
        as input to the next hook.
        
        Args:
            gradients: Dictionary mapping parameter names to gradient tensors.
        
        Returns:
            Transformed gradients dictionary.
        """
        result = gradients
        for hook, enabled in self.update_hooks:
            if enabled:
                result = hook(result)
        return result
    
    def get_active_hooks(self) -> Dict[str, List[str]]:
        """Get list of active (enabled) hooks.
        
        Returns:
            Dictionary with "forward" and "update" keys, each containing
            a list of hook names that are currently enabled.
        """
        active_forward = [
            hook.name for hook, enabled in self.forward_hooks if enabled
        ]
        active_update = [
            hook.name for hook, enabled in self.update_hooks if enabled
        ]
        return {
            "forward": active_forward,
            "update": active_update,
        }

