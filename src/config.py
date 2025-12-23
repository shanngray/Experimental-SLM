"""Configuration management for training hyperparameters.

This module provides a configuration system for managing training
hyperparameters such as learning rate, weight decay, optimizer betas, etc.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters.
    
    This class holds all hyperparameters needed for training the transformer model.
    Default values are set according to common practices for transformer training.
    
    Attributes:
        learning_rate: Learning rate for optimizer (default: 3e-4).
        weight_decay: Weight decay for L2 regularization (default: 0.1).
        beta1: First momentum coefficient for AdamW (default: 0.9).
        beta2: Second momentum coefficient for AdamW (default: 0.95).
        batch_size: Batch size for training (default: 16).
        max_seq_len: Maximum sequence length (default: 256).
        seed: Random seed for reproducibility (default: None).
        hooks: Hook configuration dictionary (default: None).
            Expected format:
            {
                "forward": [
                    {"name": "activation_stats", "enabled": True, ...}
                ],
                "update": [
                    {"name": "identity", "enabled": True, ...}
                ]
            }
        eval_cadence: Evaluation cadence in steps (default: None, disabled).
            If set, validation loss is computed every N steps.
        sampling_cadence: Sampling cadence in steps (default: None, disabled).
            If set, text samples are generated every N steps.
        sampling_temperature: Temperature for text sampling (default: 1.0).
        sampling_prompt: Fixed prompt for text sampling (default: "The").
        sampling_max_length: Maximum number of tokens to generate (default: 100).
        sampling_seed: Random seed for sampling reproducibility (default: 42).
    """
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    batch_size: int = 16
    max_seq_len: int = 256
    seed: Optional[int] = None
    hooks: Optional[Dict[str, List[Dict[str, Any]]]] = None
    eval_cadence: Optional[int] = None
    sampling_cadence: Optional[int] = None
    sampling_temperature: float = 1.0
    sampling_prompt: str = "The"
    sampling_max_length: int = 100
    sampling_seed: int = 42
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create TrainingConfig from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters.
                Unknown keys are ignored.
        
        Returns:
            TrainingConfig instance with values from dictionary.
        """
        # Only use keys that are valid attributes
        valid_keys = {
            "learning_rate", "weight_decay", "beta1", "beta2",
            "batch_size", "max_seq_len", "seed", "hooks",
            "eval_cadence", "sampling_cadence", "sampling_temperature",
            "sampling_prompt", "sampling_max_length", "sampling_seed"
        }
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)
    
    def to_dict(self) -> dict:
        """Convert TrainingConfig to dictionary.
        
        Returns:
            Dictionary with all configuration parameters.
        """
        result = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "batch_size": self.batch_size,
            "max_seq_len": self.max_seq_len,
            "seed": self.seed,
            "eval_cadence": self.eval_cadence,
            "sampling_cadence": self.sampling_cadence,
            "sampling_temperature": self.sampling_temperature,
            "sampling_prompt": self.sampling_prompt,
            "sampling_max_length": self.sampling_max_length,
            "sampling_seed": self.sampling_seed,
        }
        if self.hooks is not None:
            result["hooks"] = self.hooks
        return result
    
    def get_hooks_config(self) -> Dict[str, Any]:
        """Get hooks configuration dictionary for HookRegistry.
        
        Returns:
            Dictionary with "hooks" key containing hook definitions,
            or empty dict if no hooks configured.
        """
        if self.hooks is None:
            return {}
        return {"hooks": self.hooks}

