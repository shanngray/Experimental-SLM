"""Multi-layer perceptron (MLP) module for transformer models."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron (feed-forward network) for transformer blocks.
    
    Implements a two-layer MLP with GELU activation. The hidden dimension
    (d_ff) is configurable and typically larger than the model dimension (d_model).
    
    Architecture:
        input -> linear(d_model -> d_ff) -> GELU -> linear(d_ff -> d_model) -> output
    
    Attributes:
        d_model: Model dimension (input and output size).
        d_ff: Feed-forward hidden dimension.
        gate_proj: Linear projection for the gate (first layer).
        up_proj: Linear projection for up projection (first layer).
        down_proj: Linear projection for down projection (second layer).
    """
    
    def __init__(self, d_model: int, d_ff: int, seed: int = None):
        """Initialize MLP.
        
        Args:
            d_model: Model dimension (input and output size).
            d_ff: Feed-forward hidden dimension.
            seed: Random seed for deterministic initialization (default: None).
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Set seed for deterministic initialization if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Two-layer MLP with GELU activation
        # Using gate_proj and up_proj pattern (common in modern transformers)
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)
        
        # Reset seed after initialization to avoid affecting other components
        if seed is not None:
            torch.manual_seed(seed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP transformation to input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].
        
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model].
        """
        # Gate and up projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # GELU activation on gate, then multiply with up
        activated = torch.nn.functional.gelu(gate) * up
        
        # Down projection
        output = self.down_proj(activated)
        
        return output
