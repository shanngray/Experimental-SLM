"""Multi-head attention module with causal masking for transformer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking.
    
    Implements scaled dot-product attention with multiple heads and causal
    masking to prevent positions from attending to future positions. This is
    essential for decoder-only transformer models.
    
    Architecture:
        - Split d_model into n_heads heads, each with dimension d_k = d_model // n_heads
        - Compute Q, K, V projections for each head
        - Apply causal masking (position i cannot attend to position > i)
        - Compute scaled dot-product attention
        - Concatenate heads and project to d_model
    
    Attributes:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_k: Dimension of each head (d_model // n_heads).
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        out_proj: Output projection.
    """
    
    def __init__(self, d_model: int, n_heads: int, seed: int = None):
        """Initialize MultiHeadAttention.
        
        Args:
            d_model: Model dimension (must be divisible by n_heads).
            n_heads: Number of attention heads.
            seed: Random seed for deterministic initialization (default: None).
        
        Raises:
            ValueError: If d_model is not divisible by n_heads.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Set seed for deterministic initialization if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Reset seed after initialization to avoid affecting other components
        if seed is not None:
            torch.manual_seed(seed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention with causal masking.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].
        
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model].
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V projections
        Q = self.q_proj(x)  # [B, seq_len, d_model]
        K = self.k_proj(x)  # [B, seq_len, d_model]
        V = self.v_proj(x)  # [B, seq_len, d_model]
        
        # Reshape to separate heads: [B, seq_len, n_heads, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Transpose to [B, n_heads, seq_len, d_k] for attention computation
        Q = Q.transpose(1, 2)  # [B, n_heads, seq_len, d_k]
        K = K.transpose(1, 2)  # [B, n_heads, seq_len, d_k]
        V = V.transpose(1, 2)  # [B, n_heads, seq_len, d_k]
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # scores: [B, n_heads, seq_len, seq_len]
        
        # Apply causal mask: position i cannot attend to position > i
        # Create mask where mask[i, j] = 0 if j <= i, else -inf
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [B, n_heads, seq_len, seq_len]
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, n_heads, seq_len, d_k]
        
        # Concatenate heads: transpose back and reshape
        attn_output = attn_output.transpose(1, 2)  # [B, seq_len, n_heads, d_k]
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(attn_output)  # [B, seq_len, d_model]
        
        return output
