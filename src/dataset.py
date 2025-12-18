"""Dataset module for creating training sequences from tokenized corpus.

This module provides functionality to split a tokenized corpus into train/validation
sets and create sliding window sequences for next-token prediction.
"""

from typing import List, Tuple


def split_corpus(
    corpus: List[int],
    train_ratio: float = 0.95,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Split a tokenized corpus into training and validation sets.
    
    Uses a contiguous split strategy where the first portion goes to training
    and the remaining portion goes to validation. The split is deterministic
    when the same seed is used.
    
    Args:
        corpus: List of token IDs representing the full corpus.
        train_ratio: Fraction of corpus to use for training (default: 0.95).
        seed: Random seed for deterministic splitting (default: 42).
            Note: Currently unused as split is deterministic, but kept for
            future extensibility and API consistency.
    
    Returns:
        Tuple of (train_corpus, val_corpus) where each is a list of token IDs.
    
    Raises:
        ValueError: If train_ratio is not between 0 and 1, or if corpus is empty.
    
    Examples:
        >>> corpus = list(range(1000))
        >>> train, val = split_corpus(corpus, train_ratio=0.9)
        >>> len(train) == 900
        True
        >>> len(val) == 100
        True
    """
    if not corpus:
        raise ValueError("Cannot split empty corpus")
    
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    # Contiguous split: first portion for training, rest for validation
    split_idx = int(len(corpus) * train_ratio)
    
    train_corpus = corpus[:split_idx]
    val_corpus = corpus[split_idx:]
    
    return train_corpus, val_corpus


class WindowDataset:
    """Dataset that creates (x, y) sequence pairs from a tokenized corpus.
    
    Creates sliding windows of fixed size from the corpus, where each window
    is used to predict the next token. The dataset yields (x, y) pairs where:
    - x is a window of tokens [t0, t1, ..., t255]
    - y is x shifted by 1 position [t1, t2, ..., t256]
    
    Windows are non-overlapping (stride equals window size). Incomplete
    windows at the end of the corpus are dropped.
    
    Attributes:
        corpus: The tokenized corpus as a list of token IDs.
        context_length: Size of each window (default: 256).
    """
    
    def __init__(self, corpus: List[int], context_length: int = 256):
        """Initialize the window dataset.
        
        Args:
            corpus: List of token IDs representing the corpus.
            context_length: Size of each window (default: 256).
        
        Raises:
            ValueError: If context_length is not positive.
        """
        if context_length <= 0:
            raise ValueError(f"context_length must be positive, got {context_length}")
        
        self.corpus = corpus
        self.context_length = context_length
    
    def __len__(self) -> int:
        """Return the number of windows in the dataset.
        
        Returns:
            Number of complete windows that can be created from the corpus.
        """
        if len(self.corpus) < self.context_length + 1:
            return 0
        # Number of windows: (corpus_length - 1) // context_length
        # We need context_length + 1 tokens to create one (x, y) pair
        return (len(self.corpus) - 1) // self.context_length
    
    def __iter__(self):
        """Iterate through the dataset, yielding (x, y) pairs.
        
        Yields:
            Tuple of (x, y) where:
            - x: List of token IDs of length context_length
            - y: List of token IDs of length context_length (x shifted by 1)
        """
        # Create non-overlapping windows with stride = context_length
        for i in range(0, len(self.corpus) - self.context_length, self.context_length):
            # Extract window of size context_length + 1
            # We need one extra token for y (next-token prediction)
            window = self.corpus[i:i + self.context_length + 1]
            
            # x is the first context_length tokens
            x = window[:self.context_length]
            # y is x shifted by 1 (last context_length tokens)
            y = window[1:self.context_length + 1]
            
            yield x, y
    
    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        """Get a specific (x, y) pair by index.
        
        Args:
            idx: Index of the window to retrieve.
        
        Returns:
            Tuple of (x, y) token ID lists.
        
        Raises:
            IndexError: If idx is out of range.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        start_idx = idx * self.context_length
        window = self.corpus[start_idx:start_idx + self.context_length + 1]
        
        x = window[:self.context_length]
        y = window[1:self.context_length + 1]
        
        return x, y
