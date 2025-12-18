## 1. Implementation

### 1.1 Text Normalization
- [x] 1.1.1 Implement `normalize_text()` function in `src/normalize.py`
- [x] 1.1.2 Implement ASCII policy: printable chars 32-126 + `\n` + `\t`
- [x] 1.1.3 Map unknown characters to `<UNK>` placeholder
- [x] 1.1.4 Handle edge cases: empty strings, non-ASCII, control chars, unicode

### 1.2 Tokenizer Core
- [x] 1.2.1 Create `Tokenizer` class in `src/tokenizer.py`
- [x] 1.2.2 Define vocab: `<PAD>=0`, `<UNK>=1`, then ASCII chars in order
- [x] 1.2.3 Implement `encode()` method: text → list of token IDs
- [x] 1.2.4 Implement `decode()` method: list of token IDs → text
- [x] 1.2.5 Ensure round-trip: `decode(encode(text))` matches normalized text

### 1.3 Vocab Persistence
- [x] 1.3.1 Implement `save_vocab()` method: save vocab to JSON file
- [x] 1.3.2 Implement `load_vocab()` method: load vocab from JSON file
- [x] 1.3.3 Ensure vocab file is human-readable JSON format
- [x] 1.3.4 Verify save/load produces identical tokenization

### 1.4 Testing
- [x] 1.4.1 Write `tests/test_normalize.py` with normalization edge cases
- [x] 1.4.2 Write `tests/test_tokenizer.py` with comprehensive tokenizer tests
- [x] 1.4.3 Test encode produces correct token IDs
- [x] 1.4.4 Test decode produces correct text
- [x] 1.4.5 Test round-trip preservation
- [x] 1.4.6 Test special tokens (`<PAD>`, `<UNK>`) handling
- [x] 1.4.7 Test vocab save/load produces identical tokenization
- [x] 1.4.8 Verify all tests pass

### 1.5 Documentation & Cleanup
- [x] 1.5.1 Add docstrings to all functions and classes
- [x] 1.5.2 Verify code follows project style conventions
- [x] 1.5.3 Run linter and fix any errors
- [x] 1.5.4 Verify can tokenize sample text files
- [x] 1.5.5 Commit with message: "Session 1: Tokenizer foundation"
