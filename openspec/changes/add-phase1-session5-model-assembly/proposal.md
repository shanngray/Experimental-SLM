# Change: Add Phase 1 Session 5 - Model Assembly

## Why
Phase 1 Session 5 assembles the full decoder-only Transformer model by implementing token and positional embeddings, and combining them with the transformer blocks from Session 4. This completes the model architecture, enabling end-to-end forward passes that produce logits over the vocabulary. This session is essential for Session 6 (Basic Training Loop), which requires a complete model to train.

## What Changes
- Add transformer capability with full model assembly
- Implement `TokenEmbedding` to map token IDs to embeddings
- Implement `PositionalEmbedding` with learned absolute positional embeddings
- Implement `Transformer` class assembling the full decoder-only model
- Model hyperparameters: n_layers=4, d_model=256, n_heads=4, d_ff=1024, dropout=0.1
- Forward pass returns logits [B, 256, vocab_size]
- Model initialization is deterministic (seed-based)
- Comprehensive test coverage for embeddings and full model

## Impact
- Affected specs: New `transformer` capability specification
- Affected code:
  - `src/model/embeddings.py` (new)
  - `src/model/transformer.py` (new)
  - `tests/test_embeddings.py` (new)
  - `tests/test_transformer.py` (new)
- Dependencies: Session 4 (Model Core) - uses TransformerBlock, MultiHeadAttention, MLP, LayerNorm
- Future impact: Session 6 (Basic Training Loop) depends on complete model assembly

