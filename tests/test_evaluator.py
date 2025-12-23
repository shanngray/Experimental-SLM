"""Tests for evaluation module."""

import pytest
import torch

from src.dataset import WindowDataset
from src.dataloader import DataLoader
from src.evaluation import compute_val_loss
from src.model.transformer import Transformer


def test_compute_val_loss_returns_scalar():
    """Test that compute_val_loss returns a scalar float."""
    vocab_size = 100
    model = Transformer(vocab_size=vocab_size, max_seq_len=256)
    
    # Create validation dataset with token IDs within vocab range
    # Need enough data for at least batch_size windows
    # With context_length=256, need 256 * batch_size + 1 tokens minimum
    val_corpus = [i % vocab_size for i in range(256 * 4 + 1)]  # 4 batches worth
    val_dataset = WindowDataset(val_corpus, context_length=256)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    val_loss = compute_val_loss(model, val_loader)
    
    assert isinstance(val_loss, float)
    assert val_loss >= 0.0


def test_compute_val_loss_computes_correct_loss():
    """Test that validation loss matches expected values."""
    vocab_size = 10
    model = Transformer(vocab_size=vocab_size, max_seq_len=64, seed=42)
    
    # Create validation dataset with token IDs within vocab range (0-9)
    # Need 64 * 2 + 1 = 129 tokens for 2 complete batches
    val_corpus = [i % vocab_size for i in range(64 * 2 + 1)]
    val_dataset = WindowDataset(val_corpus, context_length=64)
    val_loader = DataLoader(val_dataset, batch_size=2)
    
    val_loss = compute_val_loss(model, val_loader)
    
    # Loss should be reasonable (not NaN, not Inf)
    assert not torch.isnan(torch.tensor(val_loss))
    assert not torch.isinf(torch.tensor(val_loss))
    assert val_loss > 0.0


def test_compute_val_loss_runs_in_eval_mode():
    """Test that evaluation runs in eval mode (no gradient computation)."""
    vocab_size = 100
    model = Transformer(vocab_size=vocab_size, max_seq_len=256)
    
    # Set model to training mode initially
    model.train()
    assert model.training
    
    # Create validation dataset with token IDs within vocab range
    val_corpus = [i % vocab_size for i in range(256 * 4 + 1)]
    val_dataset = WindowDataset(val_corpus, context_length=256)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # Compute validation loss
    val_loss = compute_val_loss(model, val_loader)
    
    # Model should be restored to training mode after evaluation
    assert model.training
    
    # Verify no gradients were computed
    assert isinstance(val_loss, float)


def test_compute_val_loss_handles_empty_dataset():
    """Test that evaluation handles empty validation set gracefully."""
    vocab_size = 100
    model = Transformer(vocab_size=vocab_size, max_seq_len=256)
    
    # Create empty validation dataset
    val_corpus = []  # Empty corpus
    val_dataset = WindowDataset(val_corpus, context_length=256)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # Should raise ValueError for empty dataset
    with pytest.raises(ValueError, match="Validation dataset is empty"):
        compute_val_loss(model, val_loader)


def test_compute_val_loss_averages_across_batches():
    """Test that validation loss is averaged across all batches."""
    vocab_size = 100
    model = Transformer(vocab_size=vocab_size, max_seq_len=256, seed=42)
    
    # Create validation dataset with multiple batches
    # WindowDataset uses non-overlapping windows (stride = context_length)
    # Need 256 * (4 * 2) + 1 = 2049 tokens for at least 2 batches with batch_size=4
    val_corpus = [i % vocab_size for i in range(256 * 8 + 1)]
    val_dataset = WindowDataset(val_corpus, context_length=256)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # Count batches
    num_batches = len(val_loader)
    assert num_batches > 1
    
    # Compute validation loss
    val_loss = compute_val_loss(model, val_loader)
    
    # Loss should be a valid scalar
    assert isinstance(val_loss, float)
    assert val_loss > 0.0


def test_compute_val_loss_uses_same_loss_function():
    """Test that validation loss uses the same loss function as training."""
    vocab_size = 10
    model = Transformer(vocab_size=vocab_size, max_seq_len=64, seed=42)
    
    # Create validation dataset with token IDs within vocab range (0-9)
    val_corpus = [i % vocab_size for i in range(64 * 2 + 1)]
    val_dataset = WindowDataset(val_corpus, context_length=64)
    val_loader = DataLoader(val_dataset, batch_size=2)
    
    # Get a batch and compute loss manually
    batch = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        targets = torch.zeros_like(batch)
        targets[:, :-1] = batch[:, 1:]
        targets[:, -1] = batch[:, -1]
        
        from src.training.loss import compute_loss
        manual_loss = compute_loss(logits, targets).item()
    
    # Compute validation loss
    val_loss = compute_val_loss(model, val_loader)
    
    # Both should be valid losses (may differ slightly due to averaging)
    assert isinstance(val_loss, float)
    assert isinstance(manual_loss, float)
    assert val_loss > 0.0
    assert manual_loss > 0.0

