"""Tests for dataset module."""

import pytest

from src.dataset import WindowDataset, split_corpus


def test_split_corpus_basic():
    """Test basic corpus splitting."""
    corpus = list(range(1000))
    train, val = split_corpus(corpus, train_ratio=0.95)
    
    assert len(train) == 950
    assert len(val) == 50
    assert train == list(range(950))
    assert val == list(range(950, 1000))


def test_split_corpus_custom_ratio():
    """Test split with custom ratio."""
    corpus = list(range(1000))
    train, val = split_corpus(corpus, train_ratio=0.9)
    
    assert len(train) == 900
    assert len(val) == 100


def test_split_corpus_deterministic():
    """Test that same seed produces same split."""
    corpus = list(range(1000))
    
    train1, val1 = split_corpus(corpus, train_ratio=0.95, seed=42)
    train2, val2 = split_corpus(corpus, train_ratio=0.95, seed=42)
    
    assert train1 == train2
    assert val1 == val2


def test_split_corpus_different_seeds():
    """Test that different seeds produce same split (contiguous split)."""
    corpus = list(range(1000))
    
    train1, val1 = split_corpus(corpus, train_ratio=0.95, seed=42)
    train2, val2 = split_corpus(corpus, train_ratio=0.95, seed=123)
    
    # Contiguous split should be deterministic regardless of seed
    assert train1 == train2
    assert val1 == val2


def test_split_corpus_empty():
    """Test splitting empty corpus raises error."""
    with pytest.raises(ValueError, match="Cannot split empty corpus"):
        split_corpus([])


def test_split_corpus_small():
    """Test splitting very small corpus."""
    corpus = [1, 2, 3]
    train, val = split_corpus(corpus, train_ratio=0.95)
    
    # With 3 tokens and 0.95 ratio, split_idx = int(3 * 0.95) = 2
    assert len(train) == 2
    assert len(val) == 1
    assert train == [1, 2]
    assert val == [3]


def test_split_corpus_invalid_ratio():
    """Test invalid train_ratio raises error."""
    corpus = list(range(100))
    
    with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
        split_corpus(corpus, train_ratio=0.0)
    
    with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
        split_corpus(corpus, train_ratio=1.0)
    
    with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
        split_corpus(corpus, train_ratio=1.5)


def test_window_dataset_init():
    """Test WindowDataset initialization."""
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    
    assert dataset.corpus == corpus
    assert dataset.context_length == 256


def test_window_dataset_invalid_context_length():
    """Test invalid context_length raises error."""
    corpus = list(range(1000))
    
    with pytest.raises(ValueError, match="context_length must be positive"):
        WindowDataset(corpus, context_length=0)
    
    with pytest.raises(ValueError, match="context_length must be positive"):
        WindowDataset(corpus, context_length=-1)


def test_window_dataset_length():
    """Test dataset length calculation."""
    # Corpus of 1000 tokens, context_length=256
    # Need 257 tokens per window (256 for x + 1 for y)
    # Number of windows: (1000 - 1) // 256 = 999 // 256 = 3
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    
    assert len(dataset) == 3


def test_window_dataset_length_exact_multiple():
    """Test dataset length when corpus is exact multiple of context_length."""
    # Corpus of 512 tokens, context_length=256
    # Need 257 tokens per window
    # Number of windows: (512 - 1) // 256 = 511 // 256 = 1
    corpus = list(range(512))
    dataset = WindowDataset(corpus, context_length=256)
    
    assert len(dataset) == 1


def test_window_dataset_length_small_corpus():
    """Test dataset length with corpus smaller than context_length."""
    corpus = list(range(100))
    dataset = WindowDataset(corpus, context_length=256)
    
    assert len(dataset) == 0


def test_window_dataset_length_exact_boundary():
    """Test dataset length at exact boundary."""
    # Corpus of 257 tokens, context_length=256
    # Number of windows: (257 - 1) // 256 = 256 // 256 = 1
    corpus = list(range(257))
    dataset = WindowDataset(corpus, context_length=256)
    
    assert len(dataset) == 1


def test_window_dataset_shapes():
    """Test that windows have correct shape."""
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    
    for x, y in dataset:
        assert len(x) == 256
        assert len(y) == 256


def test_window_dataset_y_shifted():
    """Test that y is correctly shifted (y[i] == x[i+1])."""
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    
    for x, y in dataset:
        for i in range(255):  # Check all but last position
            assert y[i] == x[i + 1], f"Mismatch at position {i}: y[{i}]={y[i]}, x[{i+1}]={x[i+1]}"


def test_window_dataset_non_overlapping():
    """Test that windows are non-overlapping."""
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    
    windows = list(dataset)
    
    # Check first token of each window
    first_tokens = [x[0] for x, _ in windows]
    
    # Should be: 0, 256, 512, ...
    expected_first_tokens = [0, 256, 512]
    assert first_tokens == expected_first_tokens


def test_window_dataset_boundaries():
    """Test that incomplete last window is dropped."""
    # Corpus of 1000 tokens, context_length=256
    # Last complete window starts at index 768 (768 + 256 = 1024 > 1000)
    # So we should have windows starting at 0, 256, 512, 768
    # But 768 + 256 = 1024 > 1000, so we can't create a full window
    # Actually: (1000 - 1) // 256 = 999 // 256 = 3 windows
    # Windows: [0:256], [256:512], [512:768]
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    
    windows = list(dataset)
    
    # Should have exactly 3 windows
    assert len(windows) == 3
    
    # Last window should end before corpus end
    last_x, last_y = windows[-1]
    assert last_x[-1] == 767  # Last token of x in last window
    assert last_y[-1] == 768  # Last token of y in last window


def test_window_dataset_iteration():
    """Test that dataset can be iterated without errors."""
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    
    count = 0
    for x, y in dataset:
        assert len(x) == 256
        assert len(y) == 256
        count += 1
    
    assert count == len(dataset)


def test_window_dataset_getitem():
    """Test accessing dataset by index."""
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    
    x0, y0 = dataset[0]
    assert len(x0) == 256
    assert len(y0) == 256
    assert x0[0] == 0
    assert y0[0] == 1
    
    x1, y1 = dataset[1]
    assert len(x1) == 256
    assert len(y1) == 256
    assert x1[0] == 256
    assert y1[0] == 257


def test_window_dataset_getitem_out_of_range():
    """Test that out-of-range index raises IndexError."""
    corpus = list(range(1000))
    dataset = WindowDataset(corpus, context_length=256)
    
    with pytest.raises(IndexError):
        _ = dataset[10]  # Only 3 windows available
    
    with pytest.raises(IndexError):
        _ = dataset[-1]  # Negative indexing not supported


def test_window_dataset_empty():
    """Test dataset with corpus smaller than context_length."""
    corpus = list(range(100))
    dataset = WindowDataset(corpus, context_length=256)
    
    assert len(dataset) == 0
    
    # Should iterate without errors but yield nothing
    windows = list(dataset)
    assert len(windows) == 0


def test_window_dataset_small_context():
    """Test dataset with smaller context length."""
    corpus = list(range(100))
    dataset = WindowDataset(corpus, context_length=10)
    
    # (100 - 1) // 10 = 99 // 10 = 9 windows
    assert len(dataset) == 9
    
    windows = list(dataset)
    assert len(windows) == 9
    
    for x, y in windows:
        assert len(x) == 10
        assert len(y) == 10
        for i in range(9):
            assert y[i] == x[i + 1]


def test_window_dataset_integration():
    """Integration test: split corpus then create dataset."""
    # Create a larger corpus
    corpus = list(range(10000))
    
    # Split into train/val
    train, val = split_corpus(corpus, train_ratio=0.95)
    
    assert len(train) == 9500
    assert len(val) == 500
    
    # Create datasets
    train_dataset = WindowDataset(train, context_length=256)
    val_dataset = WindowDataset(val, context_length=256)
    
    # Check train dataset
    assert len(train_dataset) == (len(train) - 1) // 256
    
    # Check val dataset
    assert len(val_dataset) == (len(val) - 1) // 256
    
    # Verify windows are correct
    for x, y in train_dataset:
        assert len(x) == 256
        assert len(y) == 256
        for i in range(255):
            assert y[i] == x[i + 1]
