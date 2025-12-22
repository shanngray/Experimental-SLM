"""Loss computation for next-token prediction.

This module provides cross-entropy loss computation for language modeling
over all sequence positions.
"""

import torch
import torch.nn.functional as F


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss for next-token prediction over all positions.
    
    The loss is computed for each position in the sequence, predicting the
    next token. This is standard for autoregressive language modeling.
    
    Args:
        logits: Model output logits of shape [B, seq_len, vocab_size].
        targets: Target token IDs of shape [B, seq_len] with next-token labels.
    
    Returns:
        Scalar loss tensor suitable for backward pass.
    
    Example:
        >>> vocab_size = 1000
        >>> batch_size = 2
        >>> seq_len = 256
        >>> logits = torch.randn(batch_size, seq_len, vocab_size)
        >>> targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        >>> loss = compute_loss(logits, targets)
        >>> loss.shape
        torch.Size([])
    """
    batch_size, seq_len, vocab_size = logits.shape
    target_batch, target_seq_len = targets.shape
    
    # Verify shapes are compatible
    if batch_size != target_batch:
        raise ValueError(
            f"Batch size mismatch: logits batch_size={batch_size}, "
            f"targets batch_size={target_batch}"
        )
    if seq_len != target_seq_len:
        raise ValueError(
            f"Sequence length mismatch: logits seq_len={seq_len}, "
            f"targets seq_len={target_seq_len}"
        )
    
    # Reshape logits to [B * seq_len, vocab_size] and targets to [B * seq_len]
    # This allows cross-entropy to compute loss over all positions
    logits_flat = logits.view(-1, vocab_size)  # [B * seq_len, vocab_size]
    targets_flat = targets.view(-1)  # [B * seq_len]
    
    # Compute cross-entropy loss
    # This computes loss for each position and returns the mean
    loss = F.cross_entropy(logits_flat, targets_flat)
    
    return loss

