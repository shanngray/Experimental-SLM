## 1. Implementation

### 1.1 Multi-Head Attention
- [x] 1.1.1 Create `src/model/attention.py` with `MultiHeadAttention` class
- [x] 1.1.2 Implement configurable number of attention heads
- [x] 1.1.3 Implement causal masking (position i cannot attend to position > i)
- [x] 1.1.4 Ensure attention output shapes are correct
- [x] 1.1.5 Implement deterministic initialization (seed-based)

### 1.2 MLP (Feed-Forward Network)
- [x] 1.2.1 Create `src/model/mlp.py` with `MLP` class
- [x] 1.2.2 Implement configurable d_ff (feed-forward dimension)
- [x] 1.2.3 Ensure MLP output shapes are correct
- [x] 1.2.4 Implement deterministic initialization (seed-based)

### 1.3 Layer Normalization
- [x] 1.3.1 Create `src/model/layer_norm.py` with `LayerNorm` class
- [x] 1.3.2 Implement layer normalization functionality
- [x] 1.3.3 Ensure correct normalization behavior

### 1.4 Transformer Block
- [x] 1.4.1 Create `src/model/transformer_block.py` with `TransformerBlock` class
- [x] 1.4.2 Combine attention + MLP + residuals
- [x] 1.4.3 Ensure residual connections work correctly
- [x] 1.4.4 Ensure forward pass works end-to-end
- [x] 1.4.5 Create `src/model/__init__.py` to export components

### 1.5 Testing
- [x] 1.5.1 Write `tests/test_attention.py` with comprehensive attention tests
- [x] 1.5.2 Test attention output shapes are correct
- [x] 1.5.3 Test causal masking prevents future attention (manual mask check)
- [x] 1.5.4 Write `tests/test_mlp.py` with MLP tests
- [x] 1.5.5 Test MLP output shapes are correct
- [x] 1.5.6 Write `tests/test_transformer_block.py` with block tests
- [x] 1.5.7 Test transformer block forward pass works
- [x] 1.5.8 Test residual connections work correctly
- [x] 1.5.9 Test initialization is deterministic
- [x] 1.5.10 Verify all tests pass

### 1.6 Documentation & Cleanup
- [x] 1.6.1 Add docstrings to all functions and classes
- [x] 1.6.2 Verify code follows project style conventions
- [x] 1.6.3 Run linter and fix any errors
- [x] 1.6.4 Verify components can be instantiated and run forward pass
- [x] 1.6.5 Verify causal masking with test
- [x] 1.6.6 Commit with message: "Session 4: Model core (transformer blocks)"
