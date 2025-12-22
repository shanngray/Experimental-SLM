## ADDED Requirements

### Requirement: Token Embeddings
The system SHALL provide token embeddings that map token IDs to dense vector representations.

#### Scenario: Correct embedding shapes
- **WHEN** `TokenEmbedding` processes token IDs of shape [B, seq_len]
- **THEN** output embeddings have shape [B, seq_len, d_model]
- **AND** each token ID maps to a unique embedding vector

#### Scenario: Deterministic initialization
- **WHEN** `TokenEmbedding` is initialized with the same seed
- **THEN** initial embedding weights are identical
- **AND** forward pass produces identical outputs for identical inputs

### Requirement: Positional Embeddings
The system SHALL provide learned absolute positional embeddings for sequence position encoding.

#### Scenario: Positional embeddings are learnable
- **WHEN** `PositionalEmbedding` is instantiated
- **THEN** positional embeddings are learnable parameters
- **AND** embeddings have shape [seq_len, d_model] where seq_len is the maximum sequence length

#### Scenario: Positional embeddings combine with token embeddings
- **WHEN** token embeddings and positional embeddings are combined
- **THEN** the sum produces embeddings that encode both token identity and position
- **AND** the combined embeddings have shape [B, seq_len, d_model]

### Requirement: Transformer Model Assembly
The system SHALL provide a complete decoder-only Transformer model that processes token sequences and produces logits over the vocabulary.

#### Scenario: Model architecture components
- **WHEN** `Transformer` model is instantiated
- **THEN** it includes:
  - **AND** token embeddings
  - **AND** positional embeddings (added to token embeddings)
  - **AND** N transformer blocks (n_layers=4)
  - **AND** final layer normalization
  - **AND** language modeling head (output projection to vocab_size)

#### Scenario: Model hyperparameters
- **WHEN** `Transformer` model is instantiated
- **THEN** it uses the following hyperparameters:
  - **AND** n_layers=4
  - **AND** d_model=256
  - **AND** n_heads=4
  - **AND** d_ff=1024
  - **AND** dropout=0.1

#### Scenario: Forward pass produces correct logits
- **WHEN** `Transformer` processes input token IDs of shape [B, 256]
- **THEN** forward pass returns logits of shape [B, 256, vocab_size]
- **AND** logits are suitable for next-token prediction (cross-entropy loss)

#### Scenario: Model processes DataLoader batches
- **WHEN** `Transformer` processes batches from DataLoader
- **THEN** it handles batches of shape [B, 256] correctly
- **AND** produces logits of shape [B, 256, vocab_size]
- **AND** forward pass completes without errors

#### Scenario: Deterministic initialization
- **WHEN** `Transformer` is initialized with the same seed
- **THEN** initial weights of all components are identical
- **AND** forward pass produces identical outputs for identical inputs

