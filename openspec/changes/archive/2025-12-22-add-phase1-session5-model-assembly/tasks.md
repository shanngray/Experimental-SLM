## 1. Implementation

### 1.1 Token Embeddings
- [x] 1.1.1 Create `src/model/embeddings.py` with `TokenEmbedding` class
- [x] 1.1.2 Implement token ID to embedding mapping
- [x] 1.1.3 Ensure embedding shapes are correct [B, seq_len, d_model]
- [x] 1.1.4 Implement deterministic initialization (seed-based)

### 1.2 Positional Embeddings
- [x] 1.2.1 Implement `PositionalEmbedding` class in `src/model/embeddings.py`
- [x] 1.2.2 Implement learned absolute positional embeddings
- [x] 1.2.3 Ensure positional embeddings are learnable parameters
- [x] 1.2.4 Ensure embedding shapes are correct [seq_len, d_model]

### 1.3 Transformer Model Assembly
- [x] 1.3.1 Create `src/model/transformer.py` with `Transformer` class
- [x] 1.3.2 Assemble full decoder-only model:
  - [x] Token embeddings
  - [x] Positional embeddings (added to token embeddings)
  - [x] N transformer blocks (n_layers=4)
  - [x] Final layer norm
  - [x] LM head (output projection to vocab_size)
- [x] 1.3.3 Configure model hyperparameters:
  - [x] n_layers=4
  - [x] d_model=256
  - [x] n_heads=4
  - [x] d_ff=1024
  - [x] dropout=0.1
- [x] 1.3.4 Ensure forward pass returns logits [B, 256, vocab_size]
- [x] 1.3.5 Implement deterministic initialization (seed-based)

### 1.4 Testing
- [x] 1.4.1 Write `tests/test_embeddings.py` with embedding tests
- [x] 1.4.2 Test token embedding shapes are correct
- [x] 1.4.3 Test positional embeddings are learnable parameters
- [x] 1.4.4 Test embedding combination (token + positional)
- [x] 1.4.5 Write `tests/test_transformer.py` with full model tests
- [x] 1.4.6 Test model forward pass produces correct logit shapes [B, 256, vocab_size]
- [x] 1.4.7 Test model can process batches from DataLoader
- [x] 1.4.8 Test initialization is deterministic
- [x] 1.4.9 Verify all tests pass

### 1.5 Documentation & Cleanup
- [x] 1.5.1 Add docstrings to all functions and classes
- [x] 1.5.2 Verify code follows project style conventions
- [x] 1.5.3 Run linter and fix any errors
- [x] 1.5.4 Verify model can process a batch end-to-end
- [x] 1.5.5 Verify logits are correct shape and dtype
- [x] 1.5.6 Commit with message: "Session 5: Model assembly"

