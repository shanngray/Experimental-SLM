"""Adapter for custom Transformer architecture.

This adapter wraps the existing custom Transformer implementation to provide
the BaseAdapter interface, maintaining backward compatibility with existing code.
"""

from typing import Dict, Optional
import torch

from src.model.transformer import Transformer
from src.model.adapters.base import BaseAdapter


class CustomTransformerAdapter(BaseAdapter):
    """Adapter for custom Transformer architecture.
    
    Wraps the existing Transformer implementation to provide the BaseAdapter
    interface while maintaining backward compatibility with existing behavior.
    
    Attributes:
        model: The underlying Transformer model.
        architecture_type: Always "custom-transformer".
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 256,
        n_layers: int = 4,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        seed: Optional[int] = None
    ):
        """Initialize CustomTransformerAdapter.
        
        Args:
            vocab_size: Size of the vocabulary.
            max_seq_len: Maximum sequence length.
            n_layers: Number of transformer blocks.
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ff: Feed-forward hidden dimension.
            dropout: Dropout probability.
            seed: Random seed for deterministic initialization.
        """
        super().__init__()
        self.model = Transformer(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            seed=seed
        )
        self._architecture_type = "custom-transformer"
        self._vocab_size = vocab_size
        self._max_seq_len = max_seq_len
        self._n_layers = n_layers
        self._d_model = d_model
        self._n_heads = n_heads
        self._d_ff = d_ff
        self._dropout = dropout
        self._seed = seed
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the transformer model.
        
        Args:
            input_ids: Input tensor of shape [batch_size, seq_len] with token IDs.
            attention_mask: Optional attention mask (not used by custom Transformer,
                but included for interface compatibility).
        
        Returns:
            Logits tensor of shape [batch_size, seq_len, vocab_size].
        """
        # Custom Transformer doesn't use attention_mask, but we accept it
        # for interface compatibility
        return self.model(input_ids)
    
    def get_config(self) -> Dict:
        """Get model configuration.
        
        Returns:
            Dictionary containing model configuration parameters.
        """
        return {
            'vocab_size': self._vocab_size,
            'max_seq_len': self._max_seq_len,
            'n_layers': self._n_layers,
            'd_model': self._d_model,
            'n_heads': self._n_heads,
            'd_ff': self._d_ff,
            'dropout': self._dropout,
            'seed': self._seed,
        }
    
    def save_checkpoint(self, path: str) -> None:
        """Save model state to disk.
        
        Args:
            path: Directory path where checkpoint should be saved.
        """
        import os
        from pathlib import Path
        
        checkpoint_dir = Path(path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_path = checkpoint_dir / "model.pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Save config
        config_path = checkpoint_dir / "config.json"
        import json
        with open(config_path, 'w') as f:
            json.dump(self.get_config(), f, indent=2)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model state from disk.
        
        Args:
            path: Directory path where checkpoint is stored.
        """
        from pathlib import Path
        
        checkpoint_dir = Path(path)
        
        # Load model weights
        model_path = checkpoint_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
    
    def get_architecture_type(self) -> str:
        """Get architecture type identifier.
        
        Returns:
            "custom-transformer"
        """
        return self._architecture_type
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters in the model.
        
        Returns:
            Total parameter count as integer.
        """
        return sum(p.numel() for p in self.model.parameters())
    
    def parameters(self):
        """Get model parameters for optimizer.
        
        Returns:
            Iterator over model parameters.
        """
        return self.model.parameters()
    
    def train(self, mode: bool = True):
        """Set training mode.
        
        Args:
            mode: If True, set to training mode; if False, set to eval mode.
        
        Returns:
            Self for method chaining.
        """
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode.
        
        Returns:
            Self for method chaining.
        """
        return self.train(False)

