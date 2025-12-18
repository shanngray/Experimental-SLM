# Change: Add Phase 1 Session 2 - Dataset Preparation

## Why
Phase 1 Session 2 implements the dataset preparation infrastructure needed to convert tokenized corpus text into training sequences. This includes splitting the corpus into train/validation sets and creating sliding window sequences for next-token prediction. This capability is essential for feeding data into the model during training and is a prerequisite for Session 3 (Batching Infrastructure).

## What Changes
- Add dataset capability with train/val split functionality
- Implement contiguous train/val split (e.g., 95%/5%) with deterministic seed-based splitting
- Implement `WindowDataset` class that creates (x, y) pairs from corpus
- Windows have shape [256] (context length)
- Sliding windows with stride 256 (non-overlapping)
- y is x shifted by 1 position (next-token prediction)
- Handle corpus boundaries correctly (drop incomplete last window)
- Comprehensive test coverage for split and windowing functionality

## Impact
- Affected specs: New `dataset` capability specification
- Affected code:
  - `src/dataset.py` (new)
  - `tests/test_dataset.py` (new)
- Dependencies: Session 1 (Tokenizer) - requires tokenized corpus input
- Future impact: Session 3 (Batching Infrastructure) depends on this dataset capability
