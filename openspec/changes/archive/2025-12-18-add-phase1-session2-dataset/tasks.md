## 1. Implementation

### 1.1 Train/Val Split
- [x] 1.1.1 Implement `split_corpus()` function in `src/dataset.py`
- [x] 1.1.2 Implement contiguous split (e.g., 95%/5% or configurable ratio)
- [x] 1.1.3 Ensure split is deterministic using seed-based randomization
- [x] 1.1.4 Handle edge cases: empty corpus, very small corpus

### 1.2 Window Dataset
- [x] 1.2.1 Create `WindowDataset` class in `src/dataset.py`
- [x] 1.2.2 Implement window creation with shape [256] (context length)
- [x] 1.2.3 Implement sliding windows with stride 256 (non-overlapping)
- [x] 1.2.4 Implement y as x shifted by 1 position (next-token prediction)
- [x] 1.2.5 Handle corpus boundaries: drop incomplete last window
- [x] 1.2.6 Ensure dataset is iterable

### 1.3 Testing
- [x] 1.3.1 Write `tests/test_dataset.py` with comprehensive dataset tests
- [x] 1.3.2 Test split produces expected sizes (95%/5% or configurable)
- [x] 1.3.3 Test same seed → same split (determinism)
- [x] 1.3.4 Test window shapes are correct [256]
- [x] 1.3.5 Test y is correctly shifted (y[i] == x[i+1])
- [x] 1.3.6 Test window boundaries handled (last incomplete window dropped)
- [x] 1.3.7 Test can iterate through dataset without errors
- [x] 1.3.8 Verify all tests pass

### 1.4 Documentation & Cleanup
- [x] 1.4.1 Add docstrings to all functions and classes
- [x] 1.4.2 Verify code follows project style conventions
- [x] 1.4.3 Run linter and fix any errors
- [x] 1.4.4 Verify can create dataset from text file
- [x] 1.4.5 Commit with message: "Session 2: Dataset preparation"
