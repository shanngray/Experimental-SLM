"""Layer normalization module for transformer models."""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization module.
    
    Normalizes inputs across the last dimension with learnable scale and shift
    parameters. This stabilizes training in transformer architectures.
    
    Attributes:
        normalized_shape: Shape of the last dimension to normalize over.
        eps: Small epsilon value to prevent division by zero (default: 1e-5).
        weight: Learnable scale parameter (gamma).
        bias: Learnable shift parameter (beta).
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """Initialize LayerNorm.
        
        Args:
            normalized_shape: Size of the last dimension to normalize over.
            eps: Small epsilon value to prevent division by zero (default: 1e-5).
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters: scale (gamma) and shift (beta)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization to input.
        
        Args:
            x: Input tensor of shape [..., normalized_shape].
        
        Returns:
            Normalized tensor of the same shape as input.
        """
        # Compute mean and variance across the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize: (x - mean) / sqrt(var + eps)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable scale and shift
        return self.weight * normalized + self.bias
