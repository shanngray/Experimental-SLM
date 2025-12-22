# Change: Add Phase 1 Session 6 - Basic Training Loop

## Why
Phase 1 Session 6 implements the basic training infrastructure needed to train the Transformer model. This includes loss computation, optimizer setup, and a training step that performs forward pass, loss calculation, backward pass, and optimizer updates. This session is essential for enabling the model to learn from data and is a prerequisite for Session 7 (Checkpointing & Resume) and Session 9 (Evaluation & Sampling). Without a working training loop, the model cannot be trained and the project cannot progress beyond Phase 1.

## What Changes
- Add training capability with basic training loop infrastructure
- Implement `compute_loss()` function for cross-entropy next-token prediction over all positions
- Implement `Trainer` class with training step functionality
- Training step: forward → loss → backward → optimizer step
- Configure AdamW optimizer: lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95)
- Implement step counter that increments correctly
- Add basic logging (loss per step)
- Implement config system for hyperparameters
- Comprehensive test coverage for training components

## Impact
- Affected specs: New `training` capability specification
- Affected code:
  - `src/training/__init__.py` (new)
  - `src/training/trainer.py` (new)
  - `src/training/loss.py` (new)
  - `src/config.py` (new)
  - `tests/test_trainer.py` (new)
- Dependencies: 
  - Session 3 (DataLoader) - provides batched data for training
  - Session 5 (Model) - provides Transformer model for training
- Future impact: 
  - Session 7 (Checkpointing & Resume) depends on training infrastructure
  - Session 8 (Hooks Infrastructure) depends on training loop
  - Session 9 (Evaluation & Sampling) depends on training step

