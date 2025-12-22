## 1. Implementation

### 1.1 Loss Computation
- [x] 1.1.1 Create `src/training/loss.py` with `compute_loss()` function
- [x] 1.1.2 Implement cross-entropy loss for next-token prediction over all positions
- [x] 1.1.3 Ensure loss computation handles logits shape [B, 256, vocab_size] and targets shape [B, 256]
- [x] 1.1.4 Verify loss computation is correct (manual check on known inputs)

### 1.2 Configuration System
- [x] 1.2.1 Create `src/config.py` for configuration management
- [x] 1.2.2 Implement config system for hyperparameters (learning rate, weight decay, betas, etc.)
- [x] 1.2.3 Ensure config can be loaded from file or passed as parameters
- [x] 1.2.4 Document config parameters

### 1.3 Trainer Class
- [x] 1.3.1 Create `src/training/__init__.py`
- [x] 1.3.2 Create `src/training/trainer.py` with `Trainer` class
- [x] 1.3.3 Implement training step:
  - [x] Forward pass through model
  - [x] Loss computation
  - [x] Backward pass (gradient computation)
  - [x] Optimizer step (parameter update)
- [x] 1.3.4 Implement step counter that increments correctly
- [x] 1.3.5 Add basic logging (loss per step)
- [x] 1.3.6 Ensure Trainer can be instantiated with model, optimizer, and config

### 1.4 Optimizer Setup
- [x] 1.4.1 Configure AdamW optimizer with:
  - [x] lr=3e-4
  - [x] weight_decay=0.1
  - [x] betas=(0.9, 0.95)
- [x] 1.4.2 Ensure optimizer state is tracked correctly
- [x] 1.4.3 Verify optimizer updates model parameters

### 1.5 Testing
- [x] 1.5.1 Write `tests/test_trainer.py` with training tests
- [x] 1.5.2 Test loss computation is correct (manual check on known inputs)
- [x] 1.5.3 Test training step completes without errors
- [x] 1.5.4 Test optimizer updates model parameters
- [x] 1.5.5 Test step counter increments
- [x] 1.5.6 Test loss decreases on simple synthetic data (smoke test)
- [x] 1.5.7 Verify all tests pass

### 1.6 Documentation & Cleanup
- [x] 1.6.1 Add docstrings to all functions and classes
- [x] 1.6.2 Verify code follows project style conventions
- [x] 1.6.3 Run linter and fix any errors
- [x] 1.6.4 Verify can run training for a few steps on tiny dataset
- [x] 1.6.5 Verify loss is computed and logged
- [x] 1.6.6 Commit with message: "Session 6: Basic training loop"

