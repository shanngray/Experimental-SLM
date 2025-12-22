## 1. Implementation
- [x] 1.1 Create `src/training/checkpoint.py` with checkpoint save/load functions
- [x] 1.2 Implement `save_checkpoint()` function
  - [x] 1.2.1 Save model state_dict
  - [x] 1.2.2 Save optimizer state_dict
  - [x] 1.2.3 Save training config (serialize to JSON-compatible format)
  - [x] 1.2.4 Save vocabulary (from tokenizer)
  - [x] 1.2.5 Save step counter
  - [x] 1.2.6 Create checkpoint directory structure (checkpoints/)
  - [x] 1.2.7 Use PyTorch format for binary weights (torch.save)
  - [x] 1.2.8 Include JSON metadata file
- [x] 1.3 Implement `load_checkpoint()` function
  - [x] 1.3.1 Load model state_dict
  - [x] 1.3.2 Load optimizer state_dict
  - [x] 1.3.3 Load training config
  - [x] 1.3.4 Load vocabulary
  - [x] 1.3.5 Load step counter
  - [x] 1.3.6 Return checkpoint data structure
- [x] 1.4 Integrate checkpointing with Trainer class
  - [x] 1.4.1 Add checkpoint save method to Trainer (or use standalone functions)
  - [x] 1.4.2 Add resume capability to Trainer initialization
  - [x] 1.4.3 Ensure step counter resumes correctly

## 2. Testing
- [x] 2.1 Create `tests/test_checkpoint.py`
- [x] 2.2 Test checkpoint save creates files
  - [x] 2.2.1 Verify checkpoint directory is created
  - [x] 2.2.2 Verify model weights file exists
  - [x] 2.2.3 Verify metadata file exists
- [x] 2.3 Test checkpoint load restores state
  - [x] 2.3.1 Verify model state is restored correctly
  - [x] 2.3.2 Verify optimizer state is restored correctly
  - [x] 2.3.3 Verify config is restored correctly
  - [x] 2.3.4 Verify vocabulary is restored correctly
  - [x] 2.3.5 Verify step counter is restored correctly
- [x] 2.4 Test resume functionality
  - [x] 2.4.1 Train for N steps, save checkpoint
  - [x] 2.4.2 Load checkpoint and resume training
  - [x] 2.4.3 Verify step counter continues from saved step
  - [x] 2.4.4 Verify loss progression is identical (within tolerance) to uninterrupted training
- [x] 2.5 Test edge cases
  - [x] 2.5.1 Handle missing checkpoint files gracefully
  - [x] 2.5.2 Handle corrupted checkpoint files gracefully
  - [x] 2.5.3 Verify checkpoint format compatibility

## 3. Documentation & Cleanup
- [x] 3.1 Add docstrings to checkpoint functions
- [x] 3.2 Update `src/training/__init__.py` to export checkpoint functions
- [x] 3.3 Run all tests and verify they pass
- [x] 3.4 Fix any linter errors
- [x] 3.5 Verify code follows project style guidelines

