## ADDED Requirements
### Requirement: Comprehensive README Documentation
The system SHALL provide comprehensive hyperparameter documentation in README.md, including detailed descriptions of all hyperparameters, their default values, reasonable ranges, usage examples, and troubleshooting guidance. The documentation SHALL enable users to understand and modify hyperparameters effectively without needing to read source code.

#### Scenario: README contains comprehensive configuration guide
- **WHEN** users examine README.md
- **THEN** README contains a "Configuration" section with:
  - Overview of the config system
  - How to use config files (--config flag)
  - How to create custom configs
  - How to override values via CLI
  - Link to configs/README.md for detailed config documentation
- **AND** documentation is clear and comprehensive

#### Scenario: README contains detailed hyperparameter reference
- **WHEN** users examine README.md
- **THEN** README contains a detailed hyperparameter reference section documenting:
  - Model Architecture hyperparameters: n_layers (default: 4), d_model (default: 256), n_heads (default: 4), d_ff (default: 1024), dropout (default: 0.1)
  - Training hyperparameters: learning_rate (default: 3e-4), weight_decay (default: 0.1), beta1 (default: 0.9), beta2 (default: 0.95), batch_size (default: 16), max_steps (default: 10000)
  - Dataset hyperparameters: max_seq_len (default: 256), train_ratio (default: 0.95)
  - Evaluation/Sampling hyperparameters: eval_cadence (default: null), sampling_cadence (default: null), sampling_temperature (default: 1.0), sampling_prompt (default: "The"), sampling_max_length (default: 100), sampling_seed (default: 42)
  - Checkpointing hyperparameters: checkpoint_cadence (default: 1000)
  - Other hyperparameters: seed (default: null)
- **AND** each hyperparameter includes: description, default value, reasonable value range, notes on interactions with other hyperparameters
- **AND** constraints are documented (e.g., d_model must be divisible by n_heads, dropout between 0.0 and 1.0)

#### Scenario: README contains practical examples
- **WHEN** users examine README.md
- **THEN** README contains an examples section with:
  - Example: Using default config (command and explanation)
  - Example: Using custom config file (command and explanation)
  - Example: Modifying specific hyperparameters (minimal YAML config)
  - Example: Creating a small model config (values and trade-offs)
  - Example: Creating a large model config (values and trade-offs)
- **AND** examples are clear and practical
- **AND** examples use correct YAML syntax

#### Scenario: README contains troubleshooting guidance
- **WHEN** users examine README.md
- **THEN** README contains a troubleshooting section covering:
  - Common config errors (file not found, invalid YAML, invalid values, unknown hyperparameters)
  - How to validate config files (check YAML syntax, verify constraints, test loading)
  - How to check which config is active (logging shows active config values)
- **AND** troubleshooting guidance is helpful and actionable

#### Scenario: README Quick Start mentions config files
- **WHEN** users examine README.md Quick Start section
- **THEN** Quick Start section mentions:
  - Using --config flag to specify config files
  - Link to config examples
  - Note that default config is used if none specified

#### Scenario: README links to example config files
- **WHEN** users examine README.md
- **THEN** README contains links to example config files:
  - configs/default.yaml
  - configs/small-model.yaml
  - configs/large-model.yaml
  - configs/fast-training.yaml
  - configs/detailed-eval.yaml
  - configs/README.md
- **AND** links are correct and functional

#### Scenario: README documentation matches actual system behavior
- **WHEN** users follow README documentation
- **THEN** all hyperparameter defaults match actual TrainingConfig defaults
- **AND** all examples work correctly
- **AND** all file paths are correct
- **AND** documentation accurately describes config system behavior

