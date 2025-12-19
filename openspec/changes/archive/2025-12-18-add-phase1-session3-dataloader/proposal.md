# Change: Add Phase 1 Session 3 - Batching Infrastructure

## Why
Phase 1 Session 3 implements the DataLoader infrastructure needed to convert individual sequence pairs from the dataset into batched tensors ready for model training. This capability is essential for efficient training as it groups multiple sequences together, enabling parallel processing and better GPU utilization. This is a prerequisite for Session 6 (Basic Training Loop) and depends on Session 2 (Dataset Preparation).

## What Changes
- Add dataloader capability with batch creation functionality
- Implement `DataLoader` class that produces batches of shape [B, 256] from dataset sequences
- Batch size B is configurable (default: 16)
- Tensors are dtype `int64`
- Handle last batch deterministically (drop incomplete batch)
- Support iteration through batches
- Comprehensive test coverage for batching functionality

## Impact
- Affected specs: New `dataloader` capability specification
- Affected code:
  - `src/dataloader.py` (new)
  - `tests/test_dataloader.py` (new)
- Dependencies: Session 2 (Dataset) - requires WindowDataset as input
- Future impact: Session 6 (Basic Training Loop) depends on this dataloader capability
