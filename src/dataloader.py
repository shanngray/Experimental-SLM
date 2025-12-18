"""DataLoader module for creating batches from datasets.

This module provides functionality to convert individual sequence pairs from
a dataset into batched tensors ready for model training.
"""

import random
from typing import Iterator, Optional

import torch

from src.dataset import WindowDataset


class DataLoader:
    """DataLoader that creates batches of sequences from a WindowDataset.
    
    Converts individual (x, y) sequence pairs from a dataset into batched
    tensors of shape [B, 256] where B is the batch size. Batches are
    deterministic when the same seed is used.
    
    Attributes:
        dataset: The WindowDataset to create batches from.
        batch_size: Number of sequences per batch (default: 16).
        seed: Random seed for deterministic batching (default: None).
    """
    
    def __init__(
        self,
        dataset: WindowDataset,
        batch_size: int = 16,
        seed: Optional[int] = None
    ):
        """Initialize the DataLoader.
        
        Args:
            dataset: WindowDataset to create batches from.
            batch_size: Number of sequences per batch (default: 16).
            seed: Random seed for deterministic batching (default: None).
        
        Raises:
            ValueError: If batch_size is not positive.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        
        # Set random seed if provided for deterministic batching
        if seed is not None:
            random.seed(seed)
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate through batches, yielding tensors of shape [B, 256].
        
        Yields:
            Tensor of shape [B, 256] where B is batch_size, dtype int64.
            Incomplete last batch is dropped.
        """
        # Reset seed if provided for deterministic iteration
        if self.seed is not None:
            random.seed(self.seed)
        
        # Collect all sequences from dataset
        sequences = []
        for x, _ in self.dataset:
            sequences.append(x)
        
        # Handle empty dataset
        if len(sequences) == 0:
            return
        
        # Create batches
        num_complete_batches = len(sequences) // self.batch_size
        
        for i in range(num_complete_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_sequences = sequences[start_idx:end_idx]
            
            # Convert to tensor: [B, 256]
            batch_tensor = torch.tensor(batch_sequences, dtype=torch.int64)
            yield batch_tensor
    
    def __len__(self) -> int:
        """Return the number of complete batches.
        
        Returns:
            Number of complete batches (incomplete last batch excluded).
        """
        dataset_len = len(self.dataset)
        return dataset_len // self.batch_size
