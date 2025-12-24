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
        # Model Architecture Hyperparameters
        n_layers: Number of transformer blocks (default: 4).
            Controls model depth. More layers increase capacity but slow training/inference.
            Typical range: 2-12 for small models, 12-96+ for large models.
            Must be a positive integer.
        d_model: Model dimension / embedding size (default: 256).
            Controls the width of the model. Must be divisible by n_heads.
            Typical range: 128-2048+ depending on model size.
            Constraint: d_model % n_heads == 0.
        n_heads: Number of attention heads (default: 4).
            Controls multi-head attention. Must divide d_model evenly.
            Typical range: 2-32+ depending on d_model.
            Constraint: d_model % n_heads == 0.
        d_ff: Feed-forward network dimension (default: 1024).
            Controls the width of the MLP layers. Typically 4x d_model.
            Must be a positive integer.
        dropout: Dropout probability (default: 0.1).
            Applied to attention and MLP layers for regularization.
            Constraint: 0.0 <= dropout <= 1.0.
        
        # Dataset Hyperparameters
        train_ratio: Fraction of data to use for training (default: 0.95).
            Remaining fraction (1 - train_ratio) is used for validation.
            Constraint: 0.0 < train_ratio < 1.0.
        
        # Training Loop Hyperparameters
        max_steps: Maximum number of training steps (default: 10000).
            Training stops after this many steps. Must be a positive integer.
        checkpoint_cadence: Steps between checkpoint saves (default: 1000).
            If None, checkpointing is disabled. Must be positive integer or None.
        
        # Optimizer Hyperparameters
        learning_rate: Learning rate for optimizer (default: 3e-4).
        weight_decay: Weight decay for L2 regularization (default: 0.1).
        beta1: First momentum coefficient for AdamW (default: 0.9).
        beta2: Second momentum coefficient for AdamW (default: 0.95).
        batch_size: Batch size for training (default: 16).
        
        # Sequence Hyperparameters
        max_seq_len: Maximum sequence length (default: 256).
        
        # Other Hyperparameters
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
    # Model Architecture Hyperparameters
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    
    # Dataset Hyperparameters
    train_ratio: float = 0.95
    
    # Training Loop Hyperparameters
    max_steps: int = 10000
    checkpoint_cadence: Optional[int] = 1000
    
    # Optimizer Hyperparameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    batch_size: int = 16
    
    # Sequence Hyperparameters
    max_seq_len: int = 256
    
    # Other Hyperparameters
    seed: Optional[int] = None
    hooks: Optional[Dict[str, List[Dict[str, Any]]]] = None
    eval_cadence: Optional[int] = None
    sampling_cadence: Optional[int] = None
    sampling_temperature: float = 1.0
    sampling_prompt: str = "The"
    sampling_max_length: int = 100
    sampling_seed: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate positive integers first (before divisibility checks)
        if self.n_layers <= 0:
            raise ValueError(
                f"n_layers ({self.n_layers}) must be positive"
            )
        
        if self.d_model <= 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be positive"
            )
        
        if self.n_heads <= 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be positive"
            )
        
        if self.d_ff <= 0:
            raise ValueError(
                f"d_ff ({self.d_ff}) must be positive"
            )
        
        # Validate divisibility constraint (after ensuring n_heads > 0)
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        
        # Validate dropout range
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(
                f"dropout ({self.dropout}) must be between 0.0 and 1.0"
            )
        
        # Validate train_ratio range
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError(
                f"train_ratio ({self.train_ratio}) must be between 0.0 and 1.0 (exclusive)"
            )
        
        # Validate max_steps
        if self.max_steps <= 0:
            raise ValueError(
                f"max_steps ({self.max_steps}) must be positive"
            )
        
        # Validate checkpoint_cadence
        if self.checkpoint_cadence is not None and self.checkpoint_cadence <= 0:
            raise ValueError(
                f"checkpoint_cadence ({self.checkpoint_cadence}) must be positive or None"
            )
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create TrainingConfig from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters.
                Unknown keys are ignored. Missing keys use default values.
        
        Returns:
            TrainingConfig instance with values from dictionary.
            Validation is performed via __post_init__.
        """
        # Only use keys that are valid attributes
        valid_keys = {
            # Model Architecture
            "n_layers", "d_model", "n_heads", "d_ff", "dropout",
            # Dataset
            "train_ratio",
            # Training Loop
            "max_steps", "checkpoint_cadence",
            # Optimizer
            "learning_rate", "weight_decay", "beta1", "beta2", "batch_size",
            # Sequence
            "max_seq_len",
            # Other
            "seed", "hooks",
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
        result: Dict[str, Any] = {
            # Model Architecture
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            # Dataset
            "train_ratio": self.train_ratio,
            # Training Loop
            "max_steps": self.max_steps,
            "checkpoint_cadence": self.checkpoint_cadence,
            # Optimizer
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "batch_size": self.batch_size,
            # Sequence
            "max_seq_len": self.max_seq_len,
            # Other
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

