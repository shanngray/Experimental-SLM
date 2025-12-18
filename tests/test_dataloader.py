"""Tests for dataloader module."""

import pytest
import torch

from src.dataloader import DataLoader
from src.dataset import WindowDataset


def test_dataloader_init():
    """Test DataLoader initialization."""
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    dataloader = DataLoader(dataset, batch_size=16)
    
    assert dataloader.dataset == dataset
    assert dataloader.batch_size == 16
    assert dataloader.seed is None


def test_dataloader_init_default_batch_size():
    """Test DataLoader with default batch size."""
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    dataloader = DataLoader(dataset)
    
    assert dataloader.batch_size == 16


def test_dataloader_init_custom_batch_size():
    """Test DataLoader with custom batch size."""
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    dataloader = DataLoader(dataset, batch_size=32)
    
    assert dataloader.batch_size == 32


def test_dataloader_init_with_seed():
    """Test DataLoader initialization with seed."""
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    dataloader = DataLoader(dataset, batch_size=16, seed=42)
    
    assert dataloader.seed == 42


def test_dataloader_invalid_batch_size():
    """Test invalid batch_size raises error."""
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    
    with pytest.raises(ValueError, match="batch_size must be positive"):
        DataLoader(dataset, batch_size=0)
    
    with pytest.raises(ValueError, match="batch_size must be positive"):
        DataLoader(dataset, batch_size=-1)


def test_dataloader_batch_shapes():
    """Test that batches have correct shape [B, 256]."""
    corpus = list(range(10000))
    dataset = WindowDataset(corpus, context_length=256)
    dataloader = DataLoader(dataset, batch_size=16)
    
    for batch in dataloader:
        assert batch.shape == (16, 256)
        assert batch.dtype == torch.int64


def test_dataloader_batch_dtype():
    """Test that batches are dtype int64."""
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    dataloader = DataLoader(dataset, batch_size=16)
    
    for batch in dataloader:
        assert batch.dtype == torch.int64


def test_dataloader_configurable_batch_size():
    """Test that batch size is configurable."""
    corpus = list(range(10000))
    dataset = WindowDataset(corpus, context_length=256)
    
    # Test batch size 8
    dataloader_8 = DataLoader(dataset, batch_size=8)
    for batch in dataloader_8:
        assert batch.shape[0] == 8
    
    # Test batch size 32
    dataloader_32 = DataLoader(dataset, batch_size=32)
    for batch in dataloader_32:
        assert batch.shape[0] == 32


def test_dataloader_drops_incomplete_batch():
    """Test that incomplete last batch is dropped."""
    # Create dataset with size that doesn't divide evenly by batch_size
    # Dataset length: (1000 - 1) // 256 = 3 sequences
    # With batch_size=2, we should get 1 complete batch (2 sequences)
    # and drop the incomplete batch (1 sequence)
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    dataloader = DataLoader(dataset, batch_size=2)
    
    batches = list(dataloader)
    
    # Should have 1 complete batch (3 // 2 = 1)
    assert len(batches) == 1
    assert batches[0].shape == (2, 256)
    
    # Verify no incomplete batch
    assert all(batch.shape[0] == 2 for batch in batches)


def test_dataloader_iteration():
    """Test that can iterate through all batches."""
    corpus = list(range(10000))
    dataset = WindowDataset(corpus, context_length=256)
    dataloader = DataLoader(dataset, batch_size=16)
    
    count = 0
    for batch in dataloader:
        assert batch.shape == (16, 256)
        assert batch.dtype == torch.int64
        count += 1
    
    # Dataset length: (10000 - 1) // 256 = 39 sequences
    # Number of batches: 39 // 16 = 2 complete batches
    expected_batches = len(dataset) // 16
    assert count == expected_batches


def test_dataloader_deterministic():
    """Test that same seed produces same batches."""
    corpus = list(range(10000))
    dataset = WindowDataset(corpus, context_length=256)
    
    # Create two dataloaders with same seed
    dataloader1 = DataLoader(dataset, batch_size=16, seed=42)
    dataloader2 = DataLoader(dataset, batch_size=16, seed=42)
    
    batches1 = list(dataloader1)
    batches2 = list(dataloader2)
    
    assert len(batches1) == len(batches2)
    
    for b1, b2 in zip(batches1, batches2):
        assert torch.equal(b1, b2)


def test_dataloader_deterministic_different_seeds():
    """Test that different seeds produce same batches (no shuffling yet)."""
    # Since we're not shuffling, same dataset should produce same batches
    # regardless of seed
    corpus = list(range(10000))
    dataset = WindowDataset(corpus, context_length=256)
    
    dataloader1 = DataLoader(dataset, batch_size=16, seed=42)
    dataloader2 = DataLoader(dataset, batch_size=16, seed=123)
    
    batches1 = list(dataloader1)
    batches2 = list(dataloader2)
    
    assert len(batches1) == len(batches2)
    
    for b1, b2 in zip(batches1, batches2):
        assert torch.equal(b1, b2)


def test_dataloader_empty_dataset():
    """Test that empty dataset is handled gracefully."""
    # Corpus too small to create any windows
    corpus = list(range(100))
    dataset = WindowDataset(corpus, context_length=256)
    dataloader = DataLoader(dataset, batch_size=16)
    
    # Should iterate without errors but yield nothing
    batches = list(dataloader)
    assert len(batches) == 0


def test_dataloader_length():
    """Test DataLoader __len__ method."""
    corpus = list(range(10000))
    dataset = WindowDataset(corpus, context_length=256)
    dataloader = DataLoader(dataset, batch_size=16)
    
    # Dataset length: (10000 - 1) // 256 = 39 sequences
    # Number of batches: 39 // 16 = 2
    expected_batches = len(dataset) // 16
    assert len(dataloader) == expected_batches
    
    # Verify it matches actual iteration
    actual_batches = len(list(dataloader))
    assert len(dataloader) == actual_batches


def test_dataloader_length_empty():
    """Test DataLoader length with empty dataset."""
    corpus = list(range(100))
    dataset = WindowDataset(corpus, context_length=256)
    dataloader = DataLoader(dataset, batch_size=16)
    
    assert len(dataloader) == 0


def test_dataloader_model_input_readiness():
    """Test that batches are ready for model input."""
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    dataloader = DataLoader(dataset, batch_size=16)
    
    for batch in dataloader:
        # Verify shape is [B, 256]
        assert len(batch.shape) == 2
        assert batch.shape[0] == 16  # batch size
        assert batch.shape[1] == 256  # sequence length
        
        # Verify dtype is int64 (required for model input)
        assert batch.dtype == torch.int64
        
        # Verify tensor is contiguous (good for model input)
        assert batch.is_contiguous()


def test_dataloader_integration():
    """Integration test: create dataset then dataloader."""
    # Create a larger corpus
    corpus = list(range(10000))
    
    # Create dataset
    dataset = WindowDataset(corpus, context_length=256)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=16, seed=42)
    
    # Verify we can iterate
    batch_count = 0
    for batch in dataloader:
        assert batch.shape == (16, 256)
        assert batch.dtype == torch.int64
        batch_count += 1
    
    assert batch_count > 0
    assert batch_count == len(dataloader)


def test_dataloader_batch_content():
    """Test that batch content matches dataset sequences."""
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    dataloader = DataLoader(dataset, batch_size=2)
    
    # Get first batch
    batches = list(dataloader)
    assert len(batches) > 0
    
    first_batch = batches[0]
    
    # Get first two sequences from dataset
    x0, _ = dataset[0]
    x1, _ = dataset[1]
    
    # Verify batch contains correct sequences
    assert torch.equal(first_batch[0], torch.tensor(x0, dtype=torch.int64))
    assert torch.equal(first_batch[1], torch.tensor(x1, dtype=torch.int64))
