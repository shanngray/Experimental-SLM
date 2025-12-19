# Change: Add Phase 1 Session 4 - Model Core (Transformer Blocks)

## Why
Phase 1 Session 4 implements the core Transformer components needed for the decoder-only language model. This includes multi-head attention with causal masking, feed-forward networks (MLP), layer normalization, and the transformer block that combines these components. These are the fundamental building blocks that will be assembled into the full model in Session 5. This session focuses on pure model code with no training dependencies, making it independently testable and modular.

## What Changes
- Add model-core capability with Transformer building blocks
- Implement `MultiHeadAttention` with configurable heads and causal masking
- Implement `MLP` (feed-forward network) with configurable d_ff
- Implement `LayerNorm` for normalization
- Implement `TransformerBlock` combining attention + MLP + residuals
- All components initialized deterministically (seed-based)
- Comprehensive test coverage for all components

## Impact
- Affected specs: New `model-core` capability specification
- Affected code:
  - `src/model/__init__.py` (new)
  - `src/model/attention.py` (new)
  - `src/model/mlp.py` (new)
  - `src/model/layer_norm.py` (new)
  - `src/model/transformer_block.py` (new)
  - `tests/test_attention.py` (new)
  - `tests/test_mlp.py` (new)
  - `tests/test_transformer_block.py` (new)
- Dependencies: None (pure model code, no training dependencies)
- Future impact: Session 5 (Model Assembly) depends on these components
