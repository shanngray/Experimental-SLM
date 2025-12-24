## MODIFIED Requirements

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

