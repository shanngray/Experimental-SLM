"""Embedding modules for transformer models.

This module provides token and positional embeddings for the transformer architecture.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token embedding layer that maps token IDs to dense vector representations.
    
    Maps integer token IDs to learnable embedding vectors of dimension d_model.
    This is the standard embedding layer used in transformer models.
    
    Attributes:
        vocab_size: Size of the vocabulary (number of unique tokens).
        d_model: Dimension of the embedding vectors.
        embedding: Embedding layer mapping token IDs to vectors.
    """
    
    def __init__(self, vocab_size: int, d_model: int, seed: int = None):
        """Initialize TokenEmbedding.
        
        Args:
            vocab_size: Size of the vocabulary (number of unique token IDs).
            d_model: Dimension of the embedding vectors.
            seed: Random seed for deterministic initialization (default: None).
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Set seed for deterministic initialization if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Create embedding layer: vocab_size x d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Reset seed after initialization to avoid affecting other components
        if seed is not None:
            torch.manual_seed(seed)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs to embedding vectors.
        
        Args:
            token_ids: Input tensor of shape [batch_size, seq_len] with token IDs.
        
        Returns:
            Embedding tensor of shape [batch_size, seq_len, d_model].
        """
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """Learned absolute positional embedding layer.
    
    Provides learnable positional embeddings for sequence position encoding.
    Each position in the sequence gets a unique learnable embedding vector.
    
    Attributes:
        max_seq_len: Maximum sequence length (maximum number of positions).
        d_model: Dimension of the embedding vectors.
        embedding: Embedding layer mapping position indices to vectors.
    """
    
    def __init__(self, max_seq_len: int, d_model: int, seed: int = None):
        """Initialize PositionalEmbedding.
        
        Args:
            max_seq_len: Maximum sequence length (maximum number of positions).
            d_model: Dimension of the embedding vectors.
            seed: Random seed for deterministic initialization (default: None).
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Set seed for deterministic initialization if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Create embedding layer: max_seq_len x d_model
        # Position indices are 0, 1, 2, ..., max_seq_len-1
        self.embedding = nn.Embedding(max_seq_len, d_model)
        
        # Reset seed after initialization to avoid affecting other components
        if seed is not None:
            torch.manual_seed(seed)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """Get positional embeddings for a sequence.
        
        Args:
            seq_len: Length of the sequence (must be <= max_seq_len).
        
        Returns:
            Positional embedding tensor of shape [seq_len, d_model].
        """
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=self.embedding.weight.device)
        return self.embedding(positions)

