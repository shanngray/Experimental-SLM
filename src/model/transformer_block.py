"""Transformer block module combining attention, MLP, and residual connections."""

import torch
import torch.nn as nn

from src.model.attention import MultiHeadAttention
from src.model.mlp import MLP
from src.model.layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    """Transformer block combining attention, MLP, and residual connections.
    
    Implements a single transformer block with:
    - Multi-head self-attention with causal masking
    - Residual connection around attention
    - Layer normalization before attention
    - MLP (feed-forward network)
    - Residual connection around MLP
    - Layer normalization before MLP
    
    Architecture:
        x -> LayerNorm -> Attention -> + (residual) -> LayerNorm -> MLP -> + (residual) -> output
    
    Attributes:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        attn_norm: Layer normalization before attention.
        attention: Multi-head attention module.
        mlp_norm: Layer normalization before MLP.
        mlp: MLP (feed-forward network) module.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, seed: int = None):
        """Initialize TransformerBlock.
        
        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ff: Feed-forward hidden dimension.
            seed: Random seed for deterministic initialization (default: None).
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        # Set seed for deterministic initialization if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Pre-attention layer normalization
        self.attn_norm = LayerNorm(d_model)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, n_heads, seed=seed)
        
        # Pre-MLP layer normalization
        self.mlp_norm = LayerNorm(d_model)
        
        # MLP (feed-forward network)
        self.mlp = MLP(d_model, d_ff, seed=seed)
        
        # Reset seed after initialization to avoid affecting other components
        if seed is not None:
            torch.manual_seed(seed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block transformation.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].
        
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model].
        """
        # Attention block with residual connection
        # Pre-norm architecture: norm -> attention -> residual
        attn_input = self.attn_norm(x)
        attn_output = self.attention(attn_input)
        x = x + attn_output  # Residual connection
        
        # MLP block with residual connection
        # Pre-norm architecture: norm -> MLP -> residual
        mlp_input = self.mlp_norm(x)
        mlp_output = self.mlp(mlp_input)
        x = x + mlp_output  # Residual connection
        
        return x
