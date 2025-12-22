"""Full transformer model assembly for decoder-only language modeling.

This module provides the complete Transformer model that combines embeddings,
transformer blocks, and output projection to produce logits over the vocabulary.
"""

import torch
import torch.nn as nn

from src.model.embeddings import TokenEmbedding, PositionalEmbedding
from src.model.transformer_block import TransformerBlock
from src.model.layer_norm import LayerNorm


class Transformer(nn.Module):
    """Complete decoder-only Transformer model for language modeling.
    
    Assembles the full transformer architecture:
    - Token embeddings (maps token IDs to dense vectors)
    - Positional embeddings (learned absolute positional encodings)
    - N transformer blocks (with attention, MLP, and residual connections)
    - Final layer normalization
    - Language modeling head (output projection to vocab_size)
    
    Architecture:
        token_ids -> TokenEmbedding -> + PositionalEmbedding -> Dropout ->
        -> TransformerBlock[0] -> ... -> TransformerBlock[N-1] ->
        -> LayerNorm -> LMHead -> logits
    
    Attributes:
        vocab_size: Size of the vocabulary.
        max_seq_len: Maximum sequence length.
        n_layers: Number of transformer blocks.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
        token_embedding: Token embedding layer.
        pos_embedding: Positional embedding layer.
        blocks: List of transformer blocks.
        final_norm: Final layer normalization.
        lm_head: Language modeling head (output projection).
        dropout_layer: Dropout layer.
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
        seed: int = None
    ):
        """Initialize Transformer model.
        
        Args:
            vocab_size: Size of the vocabulary (number of unique tokens).
            max_seq_len: Maximum sequence length (default: 256).
            n_layers: Number of transformer blocks (default: 4).
            d_model: Model dimension (default: 256).
            n_heads: Number of attention heads (default: 4).
            d_ff: Feed-forward hidden dimension (default: 1024).
            dropout: Dropout probability (default: 0.1).
            seed: Random seed for deterministic initialization (default: None).
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Set seed for deterministic initialization if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Token embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model, seed=seed)
        
        # Positional embeddings
        self.pos_embedding = PositionalEmbedding(max_seq_len, d_model, seed=seed)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, seed=seed)
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.final_norm = LayerNorm(d_model)
        
        # Language modeling head (output projection to vocab_size)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Reset seed after initialization to avoid affecting other components
        if seed is not None:
            torch.manual_seed(seed)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer model.
        
        Args:
            token_ids: Input tensor of shape [batch_size, seq_len] with token IDs.
        
        Returns:
            Logits tensor of shape [batch_size, seq_len, vocab_size].
        """
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings: [B, seq_len] -> [B, seq_len, d_model]
        token_embeds = self.token_embedding(token_ids)
        
        # Positional embeddings: [seq_len, d_model]
        pos_embeds = self.pos_embedding(seq_len)
        
        # Add positional embeddings to token embeddings
        # pos_embeds is [seq_len, d_model], broadcast to [B, seq_len, d_model]
        x = token_embeds + pos_embeds.unsqueeze(0)  # [B, seq_len, d_model]
        
        # Apply dropout
        x = self.dropout_layer(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)  # [B, seq_len, d_model]
        
        # Final layer normalization
        x = self.final_norm(x)  # [B, seq_len, d_model]
        
        # Language modeling head: [B, seq_len, d_model] -> [B, seq_len, vocab_size]
        logits = self.lm_head(x)  # [B, seq_len, vocab_size]
        
        return logits

