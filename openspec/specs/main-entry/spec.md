# main-entry Specification

## Purpose
TBD - created by archiving change add-main-entry-point. Update Purpose after archive.
## Requirements
### Requirement: Training Pipeline Orchestration
The system SHALL provide a main entry point that orchestrates all Phase 1 components into a complete training pipeline.

#### Scenario: End-to-end training execution
- **WHEN** `main.py` is executed
- **THEN** it loads configuration, data, creates tokenizer, datasets, dataloaders, model, optimizer, and trainer
- **AND** runs training loop iterating over batches and calling `trainer.training_step()`
- **AND** training completes successfully

#### Scenario: Component integration
- **WHEN** main.py orchestrates components
- **THEN** it uses existing components: Tokenizer, split_corpus, WindowDataset, DataLoader, Transformer, create_optimizer, Trainer
- **AND** components are initialized in correct order with proper dependencies
- **AND** all components are already implemented and tested (from Phase 1)

### Requirement: Configuration Loading
The system SHALL load training configuration, supporting YAML files or defaults, with command-line overrides. TrainingConfig SHALL include all hyperparameters needed for training, including model architecture, training loop, dataset, evaluation, sampling, and checkpointing parameters. The system SHALL provide example YAML configuration files demonstrating all hyperparameters and a standard config file format with documentation.

#### Scenario: Load configuration from YAML file
- **WHEN** `main.py` is invoked with `--config <path>` argument
- **THEN** configuration is loaded from YAML file and converted to TrainingConfig
- **AND** loaded values override TrainingConfig defaults
- **AND** TrainingConfig includes model architecture hyperparameters (n_layers, d_model, n_heads, d_ff, dropout)
- **AND** TrainingConfig includes dataset hyperparameters (train_ratio, max_seq_len)
- **AND** TrainingConfig includes training loop hyperparameters (max_steps, checkpoint_cadence, learning_rate, weight_decay, beta1, beta2, batch_size)
- **AND** TrainingConfig includes evaluation and sampling hyperparameters (eval_cadence, sampling_cadence, sampling_temperature, sampling_prompt, sampling_max_length, sampling_seed)

#### Scenario: Use default configuration
- **WHEN** `main.py` is invoked without `--config` argument
- **THEN** default TrainingConfig is used
- **AND** training proceeds with sensible defaults for all hyperparameters
- **AND** default values match current hardcoded values (n_layers=4, d_model=256, n_heads=4, d_ff=1024, dropout=0.1, train_ratio=0.95, max_steps=10000, checkpoint_cadence=1000)

#### Scenario: Command-line argument overrides
- **WHEN** `main.py` is invoked with `--max-steps <N>`
- **THEN** max_steps override is applied to configuration
- **AND** training runs for exactly N steps

#### Scenario: TrainingConfig includes all hyperparameters
- **WHEN** TrainingConfig is instantiated
- **THEN** it includes model architecture fields: n_layers (default: 4), d_model (default: 256), n_heads (default: 4), d_ff (default: 1024), dropout (default: 0.1)
- **AND** it includes dataset fields: train_ratio (default: 0.95), max_seq_len (default: 256)
- **AND** it includes training loop fields: max_steps (default: 10000), checkpoint_cadence (default: 1000), learning_rate, weight_decay, beta1, beta2, batch_size
- **AND** it includes evaluation and sampling fields: eval_cadence, sampling_cadence, sampling_temperature, sampling_prompt, sampling_max_length, sampling_seed

#### Scenario: TrainingConfig serialization and deserialization
- **WHEN** TrainingConfig.from_dict() is called with a dictionary containing new hyperparameters
- **THEN** all new hyperparameters (n_layers, d_model, n_heads, d_ff, dropout, train_ratio, max_steps, checkpoint_cadence) are recognized and loaded
- **AND** missing hyperparameters use default values
- **WHEN** TrainingConfig.to_dict() is called
- **THEN** all hyperparameters including new ones are serialized to dictionary
- **AND** serialized dictionary can be used to recreate TrainingConfig

#### Scenario: Backward compatibility with existing configs
- **WHEN** an existing config file (missing new hyperparameters) is loaded
- **THEN** TrainingConfig.from_dict() uses default values for missing hyperparameters
- **AND** training proceeds successfully with defaults
- **AND** no errors occur due to missing fields

#### Scenario: Default YAML config file exists
- **WHEN** users examine the `configs/` directory
- **THEN** `configs/default.yaml` exists with all hyperparameters
- **AND** each hyperparameter includes inline comments explaining its purpose, default value, and when to change it
- **AND** config file is organized into logical sections (model architecture, training, dataset, evaluation, sampling, checkpointing)
- **AND** config file demonstrates the standard YAML format for all hyperparameters

#### Scenario: Example YAML config files demonstrate different use cases
- **WHEN** users examine the `configs/` directory
- **THEN** `configs/small-model.yaml` exists demonstrating smaller model configuration
- **AND** `configs/large-model.yaml` exists demonstrating larger model configuration
- **AND** `configs/fast-training.yaml` exists demonstrating faster training settings
- **AND** `configs/detailed-eval.yaml` exists demonstrating more frequent evaluation/sampling
- **AND** each example config includes comments explaining its use case and trade-offs

#### Scenario: Config directory documentation exists
- **WHEN** users examine the `configs/` directory
- **THEN** `configs/README.md` exists documenting:
  - How to use config files (--config flag)
  - How to create custom configs
  - How configs are loaded and merged with defaults
  - Examples of common modifications
  - Links to example config files
  - Troubleshooting common config errors
- **AND** documentation is clear and comprehensive
- **AND** examples match actual config file format

### Requirement: Data Loading and Setup
The system SHALL load text data, initialize tokenizer, and create train/val datasets and dataloaders using hyperparameters from config.

#### Scenario: Load data and create datasets
- **WHEN** data file path is provided (via --data or default)
- **THEN** text data is loaded from file
- **AND** Tokenizer is initialized and corpus is tokenized
- **AND** corpus is split using `split_corpus()` with `train_ratio` from config (default: 0.95)
- **AND** WindowDataset instances are created for train and val
- **AND** DataLoader instances are created with batch_size from config

### Requirement: Model and Trainer Initialization
The system SHALL initialize the Transformer model, optimizer, and Trainer using hyperparameters from config.

#### Scenario: Create model and trainer
- **WHEN** vocab_size and configuration are available
- **THEN** Transformer model is created with vocab_size and all architecture hyperparameters from config:
  - `n_layers` from `config.n_layers` (default: 4)
  - `d_model` from `config.d_model` (default: 256)
  - `n_heads` from `config.n_heads` (default: 4)
  - `d_ff` from `config.d_ff` (default: 1024)
  - `dropout` from `config.dropout` (default: 0.1)
  - `max_seq_len` from `config.max_seq_len` (default: 256)
  - `seed` from `config.seed` (default: None)
- **AND** AdamW optimizer is created using `create_optimizer()` with config
- **AND** Trainer is initialized with model, optimizer, config, val_dataloader, and tokenizer
- **AND** model architecture hyperparameters are logged when model is created

### Requirement: Checkpoint Resume
The system SHALL support resuming training from a saved checkpoint.

#### Scenario: Resume from checkpoint
- **WHEN** `main.py` is invoked with `--resume <checkpoint_path>` argument
- **THEN** checkpoint is loaded using `Trainer.from_checkpoint()`
- **AND** training continues from restored step
- **AND** all state (model, optimizer, config, step) is restored correctly

#### Scenario: Checkpoint file errors
- **WHEN** checkpoint file is missing or invalid
- **THEN** an error message is displayed
- **AND** program exits gracefully with non-zero exit code

### Requirement: Training Loop Execution
The system SHALL execute the training loop by iterating over batches and calling trainer methods, using hyperparameters from config.

#### Scenario: Run training loop
- **WHEN** trainer is initialized
- **THEN** training loop iterates over batches from training DataLoader
- **AND** `trainer.training_step(batch)` is called for each batch
- **AND** loop continues until `max_steps` from config is reached (default: 10000)
- **AND** `max_steps` can be overridden via CLI `--max-steps` argument (CLI override takes precedence)
- **AND** Trainer handles logging, evaluation cadence, and sampling cadence internally

#### Scenario: Periodic checkpoint saving
- **WHEN** training is in progress
- **THEN** checkpoints are saved periodically using `checkpoint_cadence` from config (default: 1000 steps)
- **AND** if `config.checkpoint_cadence` is None, periodic checkpointing is disabled
- **AND** `trainer.save_checkpoint()` is called with tokenizer at the specified cadence
- **AND** final checkpoint is always saved at the end of training regardless of cadence
- **AND** checkpoint path is logged

### Requirement: Structured Logging
The system SHALL ensure training logs are structured and parseable for experiment tracking and reproducibility.

#### Scenario: Log format is parseable
- **WHEN** training logs are output
- **THEN** log format is structured (JSON-like or consistent key-value format)
- **AND** logs can be parsed programmatically
- **AND** key fields (run_id, step, loss, etc.) are easily extractable

#### Scenario: Complete metadata logging
- **WHEN** training starts
- **THEN** run metadata is logged including: run_id, config_hash, git_commit
- **AND** active hooks list is logged
- **AND** configuration summary is logged

#### Scenario: Training progress logging
- **WHEN** training is in progress
- **THEN** each step logs: step number, loss value
- **AND** validation loss is logged when evaluation occurs
- **AND** generated text samples are logged when sampling occurs
- **AND** all log entries include timestamp or step number for traceability

### Requirement: Command-Line Interface
The system SHALL provide a command-line interface for common operations.

#### Scenario: Resume from checkpoint argument
- **WHEN** `main.py` is invoked with `--resume <path>`
- **THEN** training resumes from the specified checkpoint path

#### Scenario: Configuration file argument
- **WHEN** `main.py` is invoked with `--config <path>`
- **THEN** configuration is loaded from the specified YAML file

#### Scenario: Data file argument
- **WHEN** `main.py` is invoked with `--data <path>`
- **THEN** training data is loaded from the specified file path

#### Scenario: Max steps argument
- **WHEN** `main.py` is invoked with `--max-steps <N>`
- **THEN** training runs for exactly N steps (overrides config value)

#### Scenario: Help/usage information
- **WHEN** `main.py` is invoked with `--help` or invalid arguments
- **THEN** usage information is displayed
- **AND** available command-line options are listed

### Requirement: Error Handling
The system SHALL handle common error cases gracefully.

#### Scenario: Missing data file
- **WHEN** data file path does not exist
- **THEN** an error message is displayed
- **AND** program exits gracefully with non-zero exit code

#### Scenario: Invalid configuration
- **WHEN** configuration file is invalid or missing required fields
- **THEN** an error message is displayed
- **AND** program exits gracefully with non-zero exit code

### Requirement: Integration Testing
The system SHALL provide comprehensive integration tests that verify the full training pipeline end-to-end.

#### Scenario: End-to-end training pipeline test
- **WHEN** integration test runs with tiny corpus
- **THEN** test executes: load data → tokenize → create datasets → train → checkpoint → resume → verify
- **AND** training completes without errors
- **AND** full pipeline works correctly

#### Scenario: Checkpoint resume produces identical results
- **WHEN** training is interrupted and resumed from checkpoint
- **THEN** resumed training produces identical loss progression
- **AND** step counter continues correctly
- **AND** results are reproducible

#### Scenario: Reproducibility verification
- **WHEN** multiple training runs use same seed and configuration
- **THEN** all runs produce identical loss values at each step
- **AND** generated samples are identical (when using fixed sampling seed)
- **AND** results are deterministic

#### Scenario: Training dynamics verification (smoke test)
- **WHEN** training runs on simple synthetic or real data
- **THEN** loss decreases over time
- **AND** model demonstrates learning capability
- **AND** generated text samples improve qualitatively

#### Scenario: Logging completeness verification
- **WHEN** training runs and logs are captured
- **THEN** logs include: run_id, config_hash, git_commit, step, loss
- **AND** val_loss is logged when eval_cadence triggers
- **AND** sample_text is logged when sampling_cadence triggers
- **AND** log format is parseable (structured text or JSON-like format)
- **AND** all required metadata is present for reproducibility

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

