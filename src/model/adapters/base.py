"""Base adapter interface for multi-architecture model support.

This module defines the BaseAdapter interface that all architecture adapters
must implement to provide a unified API for training and inference.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch


class BaseAdapter(ABC):
    """Base interface for architecture adapters.
    
    All architecture adapters must implement this interface to provide
    a unified API for training and inference across different model architectures.
    
    Adapters handle architecture-specific details like layer naming, activation
    functions, and tokenizer integration while exposing a consistent interface
    to the training loop.
    """
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Make adapter callable like a PyTorch module.
        
        Args:
            input_ids: Input tensor of shape [batch_size, seq_len] with token IDs.
            attention_mask: Optional attention mask tensor.
        
        Returns:
            Logits tensor from forward pass.
        """
        return self.forward(input_ids, attention_mask)
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            input_ids: Input tensor of shape [batch_size, seq_len] with token IDs.
            attention_mask: Optional attention mask tensor of shape [batch_size, seq_len].
                Values should be 0 for masked positions and 1 for unmasked positions.
        
        Returns:
            Logits tensor of shape [batch_size, seq_len, vocab_size].
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict:
        """Get model configuration.
        
        Returns:
            Dictionary containing model configuration parameters.
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save model state to disk.
        
        Args:
            path: Directory path where checkpoint should be saved.
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load model state from disk.
        
        Args:
            path: Directory path where checkpoint is stored.
        """
        pass
    
    @abstractmethod
    def get_architecture_type(self) -> str:
        """Get architecture type identifier.
        
        Returns:
            String identifier for the architecture (e.g., "custom-transformer", "qwen").
        """
        pass
    
    @abstractmethod
    def get_num_parameters(self) -> int:
        """Get total number of parameters in the model.
        
        Returns:
            Total parameter count as integer.
        """
        pass
    
    def get_tokenizer(self):
        """Get the tokenizer associated with this model.
        
        Returns:
            Tokenizer object. Default implementation returns None.
            Subclasses should override to return appropriate tokenizer.
        """
        return None
    
    def parameters(self):
        """Get model parameters for optimizer.
        
        Returns:
            Iterator over model parameters. Default implementation returns
            parameters() from underlying model if it's a torch.nn.Module.
            Subclasses should override if needed.
        """
        # Default implementation assumes adapter wraps a torch.nn.Module
        if hasattr(self, 'model') and isinstance(self.model, torch.nn.Module):
            return self.model.parameters()
        raise NotImplementedError(
            "Adapter must implement parameters() or wrap a torch.nn.Module"
        )
    
    def train(self, mode: bool = True):
        """Set training mode.
        
        Args:
            mode: If True, set to training mode; if False, set to eval mode.
        
        Returns:
            Self for method chaining.
        """
        # Default implementation assumes adapter wraps a torch.nn.Module
        if hasattr(self, 'model') and isinstance(self.model, torch.nn.Module):
            self.model.train(mode)
            return self
        raise NotImplementedError(
            "Adapter must implement train() or wrap a torch.nn.Module"
        )
    
    def eval(self):
        """Set evaluation mode.
        
        Returns:
            Self for method chaining.
        """
        return self.train(False)

