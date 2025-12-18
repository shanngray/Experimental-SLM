## 1. Implementation

### 1.1 DataLoader Core
- [x] 1.1.1 Create `DataLoader` class in `src/dataloader.py`
- [x] 1.1.2 Implement batch creation that produces batches of shape [B, 256]
- [x] 1.1.3 Make batch size B configurable (default: 16)
- [x] 1.1.4 Ensure tensors are dtype `int64`
- [x] 1.1.5 Handle last incomplete batch (drop it deterministically)
- [x] 1.1.6 Ensure DataLoader is iterable

### 1.2 Batch Handling
- [x] 1.2.1 Implement deterministic batch creation (same seed → same batches)
- [x] 1.2.2 Handle empty dataset gracefully
- [x] 1.2.3 Ensure batches are ready for model input

### 1.3 Testing
- [x] 1.3.1 Write `tests/test_dataloader.py` with comprehensive dataloader tests
- [x] 1.3.2 Test batch shapes are correct [B, 256]
- [x] 1.3.3 Test batch size is configurable
- [x] 1.3.4 Test last incomplete batch is dropped
- [x] 1.3.5 Test can iterate through all batches
- [x] 1.3.6 Test batches are deterministic (same seed → same batches)
- [x] 1.3.7 Test handles empty dataset gracefully
- [x] 1.3.8 Verify all tests pass

### 1.4 Documentation & Cleanup
- [x] 1.4.1 Add docstrings to all functions and classes
- [x] 1.4.2 Verify code follows project style conventions
- [x] 1.4.3 Run linter and fix any errors
- [x] 1.4.4 Verify can create DataLoader from Dataset
- [x] 1.4.5 Verify batches are ready for model input
- [x] 1.4.6 Commit with message: "Session 3: Batching infrastructure"
