# hyperparameter-audit Specification

## Purpose
TBD - created by archiving change add-phase2-session1-hyperparameter-audit. Update Purpose after archive.
## Requirements
### Requirement: Hyperparameter Audit Documentation
The system SHALL provide comprehensive documentation of all hyperparameters in the codebase, including their current locations, default values, categorization, and recommendations for configuration.

#### Scenario: Complete hyperparameter inventory
- **WHEN** Phase 2 Session 1 audit is performed
- **THEN** all hyperparameters in the codebase are documented with:
  - Current value
  - Current location (file:line)
  - Category (Model Architecture, Training, Dataset, Evaluation, Sampling, Checkpointing, Other)
  - Whether it should be configurable
  - Suggested default value
  - Value constraints/notes

#### Scenario: Hyperparameter categorization
- **WHEN** hyperparameters are identified during audit
- **THEN** each hyperparameter is categorized into one of:
  - Model Architecture (n_layers, d_model, n_heads, d_ff, dropout)
  - Training (learning_rate, weight_decay, beta1, beta2, batch_size, max_steps)
  - Dataset (max_seq_len, train_ratio)
  - Evaluation (eval_cadence)
  - Sampling (sampling_cadence, sampling_temperature, sampling_prompt, sampling_max_length, sampling_seed)
  - Checkpointing (checkpoint_cadence)
  - Other (seed)

#### Scenario: Identification of missing configurable hyperparameters
- **WHEN** audit compares current config system against codebase
- **THEN** all hardcoded hyperparameters that should be configurable are identified and documented, including:
  - Model architecture parameters (n_layers, d_model, n_heads, d_ff, dropout)
  - Dataset parameters (train_ratio)
  - Training loop parameters (max_steps default, checkpoint_cadence)

#### Scenario: Documentation of hardcoded hyperparameters
- **WHEN** audit identifies hyperparameters that should remain hardcoded
- **THEN** each hardcoded hyperparameter is documented with justification for why it should not be configurable

#### Scenario: Value range documentation
- **WHEN** hyperparameters are documented
- **THEN** reasonable value ranges and constraints are documented for each hyperparameter (e.g., dropout between 0 and 1, n_heads must divide d_model)

#### Scenario: Audit summary for Phase 2 Sessions 2-6
- **WHEN** audit is complete
- **THEN** summary document is created that:
  - Lists all hyperparameters already in config (with checkmarks)
  - Lists all hyperparameters that need to be added to config
  - Provides clear categorization
  - Includes file locations for easy reference
  - Is formatted to support Phase 2 Sessions 2-6 implementation

