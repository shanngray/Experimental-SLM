"""Evaluator for computing validation loss on validation datasets.

This module provides functionality to compute validation loss for assessing
model generalization beyond the training set.
"""

import torch

from src.dataloader import DataLoader
from src.training.loss import compute_loss


def compute_val_loss(
    model: torch.nn.Module,
    val_dataloader: DataLoader
) -> float:
    """Compute validation loss on validation dataset.
    
    Computes cross-entropy loss on the validation set by iterating through
    all validation batches. The model is set to evaluation mode (no gradients)
    during computation. Returns the average loss across all validation batches.
    
    Args:
        model: Transformer model to evaluate (must be a PyTorch nn.Module).
        val_dataloader: DataLoader providing validation batches.
            Each batch should be a tensor of shape [B, seq_len] with token IDs.
    
    Returns:
        Average validation loss as a float scalar.
    
    Raises:
        ValueError: If validation dataset is empty.
    
    Example:
        >>> model = Transformer(vocab_size=100, max_seq_len=256)
        >>> val_dataset = WindowDataset(val_corpus, context_length=256)
        >>> val_loader = DataLoader(val_dataset, batch_size=16)
        >>> val_loss = compute_val_loss(model, val_loader)
        >>> isinstance(val_loss, float)
        True
    """
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for batch in val_dataloader:
            # Handle empty dataset
            if batch.numel() == 0:
                continue
            
            # Create targets by shifting inputs by 1 position
            # Same logic as in training_step
            targets = torch.zeros_like(batch)
            targets[:, :-1] = batch[:, 1:]
            targets[:, -1] = batch[:, -1]
            
            # Forward pass
            logits = model(batch)  # [B, seq_len, vocab_size]
            
            # Compute loss
            loss = compute_loss(logits, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    # Restore model to training mode
    model.train()
    
    # Handle empty validation set
    if num_batches == 0:
        raise ValueError("Validation dataset is empty")
    
    # Return average loss
    avg_loss = total_loss / num_batches
    return avg_loss

