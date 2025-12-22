# model-core Specification

## Purpose
TBD - created by archiving change add-phase1-session4-model-core. Update Purpose after archive.
## Requirements
### Requirement: Multi-Head Attention
The system SHALL provide multi-head attention with causal masking for decoder-only transformer models.

#### Scenario: Configurable attention heads
- **WHEN** `MultiHeadAttention` is instantiated with a specified number of heads
- **THEN** the attention mechanism uses that number of heads
- **AND** the output shape accounts for the number of heads

#### Scenario: Causal masking prevents future attention
- **WHEN** attention is computed with causal masking enabled
- **THEN** position i cannot attend to positions greater than i
- **AND** attention weights for future positions are masked (set to -inf or 0)

#### Scenario: Correct output shapes
- **WHEN** `MultiHeadAttention` processes input tensors
- **THEN** output shapes are correct for the given input dimensions
- **AND** batch and sequence dimensions are preserved

#### Scenario: Deterministic initialization
- **WHEN** `MultiHeadAttention` is initialized with the same seed
- **THEN** initial weights are identical
- **AND** forward pass produces identical outputs for identical inputs

### Requirement: MLP (Feed-Forward Network)
The system SHALL provide a feed-forward network (MLP) with configurable hidden dimension.

#### Scenario: Configurable feed-forward dimension
- **WHEN** `MLP` is instantiated with a specified d_ff (feed-forward dimension)
- **THEN** the MLP uses that dimension for its hidden layer
- **AND** the output shape matches the input shape

#### Scenario: Correct output shapes
- **WHEN** `MLP` processes input tensors
- **THEN** output shapes match input shapes
- **AND** batch and sequence dimensions are preserved

#### Scenario: Deterministic initialization
- **WHEN** `MLP` is initialized with the same seed
- **THEN** initial weights are identical
- **AND** forward pass produces identical outputs for identical inputs

### Requirement: Layer Normalization
The system SHALL provide layer normalization for stabilizing transformer training.

#### Scenario: Normalization behavior
- **WHEN** `LayerNorm` processes input tensors
- **THEN** outputs are normalized across the specified dimension
- **AND** normalization statistics are computed correctly

### Requirement: Transformer Block
The system SHALL provide a transformer block that combines attention, MLP, and residual connections.

#### Scenario: Forward pass works end-to-end
- **WHEN** `TransformerBlock` processes input tensors
- **THEN** forward pass completes without errors
- **AND** output shapes match input shapes

#### Scenario: Residual connections work correctly
- **WHEN** `TransformerBlock` processes input tensors
- **THEN** residual connections add input to attention output
- **AND** residual connections add attention output to MLP output
- **AND** gradients can flow through residual connections

#### Scenario: Component integration
- **WHEN** `TransformerBlock` processes input tensors
- **THEN** attention is applied first with residual connection
- **AND** MLP is applied second with residual connection
- **AND** layer normalization is applied appropriately

#### Scenario: Deterministic initialization
- **WHEN** `TransformerBlock` is initialized with the same seed
- **THEN** initial weights of all components are identical
- **AND** forward pass produces identical outputs for identical inputs

