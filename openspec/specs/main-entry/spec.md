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
The system SHALL load training configuration, supporting YAML files or defaults, with command-line overrides. TrainingConfig SHALL include model selection field (model_name) to specify which model from the registry to load. The system SHALL provide example YAML configuration files demonstrating model selection for different architectures.

#### Scenario: Load configuration from YAML file
- **WHEN** `main.py` is invoked with `--config <path>` argument
- **THEN** configuration is loaded from YAML file and converted to TrainingConfig
- **AND** loaded values override TrainingConfig defaults
- **AND** TrainingConfig includes model_name field for model selection
- **AND** TrainingConfig includes all existing hyperparameters

#### Scenario: Use default configuration
- **WHEN** `main.py` is invoked without `--config` argument
- **THEN** default TrainingConfig is used
- **AND** model_name defaults to None (uses custom Transformer)
- **AND** training proceeds with sensible defaults for all hyperparameters

#### Scenario: Load configuration with model_name
- **WHEN** YAML config includes `model_name: "qwen-0.5b-base"`
- **THEN** TrainingConfig.model_name is set to "qwen-0.5b-base"
- **AND** system will load specified model from registry
- **AND** model's architecture config takes precedence over TrainingConfig architecture params

#### Scenario: Command-line argument overrides
- **WHEN** `main.py` is invoked with `--max-steps <N>`
- **THEN** max_steps override is applied to configuration
- **AND** training runs for exactly N steps

#### Scenario: TrainingConfig serialization includes model_name
- **WHEN** TrainingConfig.to_dict() is called
- **THEN** model_name field is included in serialized dictionary
- **AND** model_name can be None for custom Transformer

#### Scenario: Example configs for different models
- **WHEN** users examine the `configs/` directory
- **THEN** example configs exist for different model types:
  - `configs/custom-transformer.yaml` - using custom architecture
  - `configs/qwen-base.yaml` - using imported Qwen base model
  - `configs/qwen-finetuned.yaml` - using fine-tuned Qwen variant
- **AND** each config includes comments explaining model_id usage

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
The system SHALL provide comprehensive integration tests that verify the full training pipeline end-to-end, including thorough testing and validation of the configuration system to ensure all hyperparameters are correctly loaded, config system works end-to-end with all components, backward compatibility is maintained, and error handling is robust.

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

#### Scenario: TrainingConfig includes all new hyperparameters
- **WHEN** TrainingConfig tests run
- **THEN** tests verify all new hyperparameters are included: n_layers (default: 4), d_model (default: 256), n_heads (default: 4), d_ff (default: 1024), dropout (default: 0.1), train_ratio (default: 0.95), max_steps (default: 10000), checkpoint_cadence (default: 1000)
- **AND** tests verify defaults match expected values
- **AND** tests verify `from_dict` correctly loads new fields
- **AND** tests verify `to_dict` includes new fields

#### Scenario: Config loading from YAML works correctly
- **WHEN** config loading tests run
- **THEN** tests verify default config (`configs/default.yaml`) loads successfully
- **AND** tests verify custom config files load successfully
- **AND** tests verify missing fields use TrainingConfig defaults
- **AND** tests verify invalid YAML syntax raises clear error messages
- **AND** tests verify invalid value types raise appropriate errors

#### Scenario: Config integration with main.py works correctly
- **WHEN** integration tests run with config files
- **THEN** tests verify `main.py` uses config for model creation (all architecture hyperparameters: n_layers, d_model, n_heads, d_ff, dropout)
- **AND** tests verify `main.py` uses config for dataset split (train_ratio)
- **AND** tests verify `main.py` uses config for checkpoint cadence (checkpoint_cadence)
- **AND** tests verify CLI override (`--max-steps`) still works and takes precedence
- **AND** tests verify default config is used when no config file provided

#### Scenario: Backward compatibility with old configs
- **WHEN** old config files (missing new hyperparameters) are loaded
- **THEN** configs load successfully without errors
- **AND** missing fields use TrainingConfig defaults
- **AND** training proceeds normally with defaults
- **AND** default behavior matches previous hardcoded values when no config provided

#### Scenario: Example config files work correctly
- **WHEN** each example config file is loaded and used
- **THEN** `configs/default.yaml` loads and works correctly
- **AND** `configs/small-model.yaml` loads and creates smaller model with correct architecture
- **AND** `configs/large-model.yaml` loads and creates larger model with correct architecture
- **AND** `configs/fast-training.yaml` loads and uses faster training settings
- **AND** `configs/detailed-eval.yaml` loads and uses more frequent evaluation/sampling
- **AND** model architecture matches config values for each example
- **AND** training uses correct hyperparameters from config
- **AND** checkpointing uses correct cadence from config
- **AND** evaluation uses correct settings from config

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

### Requirement: Model Loading and Initialization
The system SHALL initialize models based on config.model_name, loading from registry for imported models or creating custom Transformer when model_name is None. The system SHALL use appropriate architecture adapter for each model type.

#### Scenario: Load custom Transformer when model_name is None
- **WHEN** config.model_name is None or not specified
- **THEN** system creates custom Transformer using architecture params from config
- **AND** uses CustomTransformerAdapter to wrap model
- **AND** model is initialized with config.n_layers, config.d_model, etc.
- **AND** training proceeds as before (backward compatible behavior)

#### Scenario: Load model from registry by model_name
- **WHEN** config.model_name is specified (e.g., "qwen-0.5b-base")
- **THEN** system queries model registry for model_name
- **AND** retrieves model metadata (model_id, architecture_type, local_path, etc.)
- **AND** loads model weights from local_path
- **AND** selects appropriate adapter based on architecture_type
- **AND** initializes adapter with loaded model

#### Scenario: Model registry lookup fails
- **WHEN** config.model_name references non-existent model in registry
- **THEN** system displays clear error message indicating model not found
- **AND** lists available model_names from registry
- **AND** exits gracefully without proceeding to training

#### Scenario: Model files missing for registry entry
- **WHEN** registry entry exists but local model files are missing
- **THEN** system displays clear error message indicating missing files
- **AND** suggests re-importing model or running validate-registry
- **AND** exits gracefully

#### Scenario: Initialize optimizer with adapter model
- **WHEN** model is loaded via adapter
- **THEN** optimizer is created with adapter's trainable parameters
- **AND** optimizer works correctly regardless of architecture
- **AND** gradients flow correctly during training

### Requirement: Model Metadata Logging
The system SHALL log comprehensive model metadata at training start, including model source, architecture type, and fine-tuning lineage.

#### Scenario: Log custom Transformer metadata
- **WHEN** training starts with custom Transformer (model_name is None)
- **THEN** logs indicate architecture_type is "custom-transformer"
- **AND** logs include model architecture params (n_layers, d_model, n_heads, etc.)
- **AND** logs indicate model is trained from scratch

#### Scenario: Log imported model metadata
- **WHEN** training starts with imported model from registry
- **THEN** logs include model_name
- **AND** logs include model_id (original HuggingFace repo ID)
- **AND** logs include architecture_type (e.g., "qwen")
- **AND** logs include source (e.g., "huggingface")
- **AND** logs include model size (parameters, disk space)

#### Scenario: Log fine-tuning lineage
- **WHEN** training starts with a fine-tuned model or creating new fine-tune
- **THEN** logs include fine_tuned_from field showing parent model
- **AND** logs include full fine-tuning chain if applicable
- **AND** logs distinguish between resuming existing fine-tune vs starting new fine-tune

### Requirement: Fine-Tuning Workflow
The system SHALL support fine-tuning imported models and automatically track fine-tuning lineage in checkpoints and registry.

#### Scenario: Start fine-tuning from base model
- **WHEN** training starts with base model from registry (e.g., config.model_name = "qwen-0.5b-base")
- **THEN** system loads base model weights
- **AND** training proceeds with loaded model
- **AND** saved checkpoints include fine_tuned_from = "qwen-0.5b-base"
- **AND** checkpoints track fine-tuning start timestamp

#### Scenario: Register fine-tuned model after training
- **WHEN** training completes or checkpoint is saved
- **THEN** user can register fine-tuned checkpoint as new model in registry
- **AND** new model entry includes fine_tuned_from lineage
- **AND** new model can be used as model_name in subsequent configs

#### Scenario: Continue fine-tuning from checkpoint
- **WHEN** training resumes from checkpoint of fine-tuned model
- **THEN** fine-tuning lineage is preserved from checkpoint
- **AND** subsequent checkpoints maintain lineage chain
- **AND** logs indicate continuing existing fine-tuning run

#### Scenario: Start new fine-tuning branch from checkpoint
- **WHEN** user loads checkpoint but specifies new model_name
- **THEN** system creates new fine-tuning branch
- **AND** new branch's fine_tuned_from points to checkpoint's model_name
- **AND** new registry entry created for the branch
- **AND** original model remains unchanged

### Requirement: Model Management CLI Commands
The system SHALL provide CLI commands for managing models: importing, listing, viewing details, and deleting.

#### Scenario: Import model command
- **WHEN** user runs `python main.py import-model Qwen/Qwen-0.5B`
- **THEN** model is downloaded and imported as described in model-import spec
- **AND** command is integrated into main.py as subcommand

#### Scenario: List models command
- **WHEN** user runs `python main.py list-models`
- **THEN** displays all models from registry as described in model-registry spec
- **AND** command is integrated into main.py as subcommand

#### Scenario: Model info command
- **WHEN** user runs `python main.py model-info <model-name>`
- **THEN** displays detailed model information as described in model-registry spec
- **AND** command is integrated into main.py as subcommand

#### Scenario: Delete model command
- **WHEN** user runs `python main.py delete-model <model-name>`
- **THEN** deletes model as described in model-registry spec
- **AND** command is integrated into main.py as subcommand

#### Scenario: Main.py subcommand routing
- **WHEN** main.py is invoked with subcommand (e.g., `python main.py import-model`, `python main.py list-models`)
- **THEN** main.py routes to appropriate handler function
- **AND** default behavior (no subcommand) runs training as before
- **AND** `--help` displays all available subcommands

### Requirement: Tokenizer Handling for Different Architectures
The system SHALL support different tokenizers for different model architectures, using model's native tokenizer by default with optional override capability.

#### Scenario: Use model's native tokenizer
- **WHEN** model is loaded from registry with native tokenizer
- **THEN** system uses model's tokenizer for encoding/decoding
- **AND** tokenizer is loaded from model's directory
- **AND** tokenizer is used consistently throughout training and inference

#### Scenario: Use char-level tokenizer for custom Transformer
- **WHEN** custom Transformer is used (model_name is None)
- **THEN** system uses char-level tokenizer as before
- **AND** behavior is backward compatible with existing implementation

#### Scenario: Override tokenizer in config
- **WHEN** config includes tokenizer override option
- **THEN** system uses specified tokenizer instead of model's native tokenizer
- **AND** override is documented in training metadata
- **AND** warnings are logged about potential compatibility issues

#### Scenario: Tokenizer in checkpoint
- **WHEN** checkpoint is saved
- **THEN** checkpoint includes tokenizer identifier
- **AND** checkpoint can be loaded with correct tokenizer
- **AND** generated samples use correct tokenizer for decoding

### Requirement: Integration Testing for Multi-Model Support
The system SHALL provide comprehensive integration tests verifying multi-model support end-to-end, including model import, loading, fine-tuning, and checkpoint resume for different architectures.

#### Scenario: End-to-end custom Transformer (backward compatibility)
- **WHEN** integration test runs without model_name
- **THEN** custom Transformer is created and trained
- **AND** behavior matches pre-multi-model implementation
- **AND** checkpoints save and load correctly

#### Scenario: End-to-end Qwen model import and inference
- **WHEN** integration test imports Qwen model and runs inference
- **THEN** model imports successfully
- **AND** model loads correctly from registry
- **AND** forward pass produces logits
- **AND** text generation works

#### Scenario: End-to-end Qwen model fine-tuning
- **WHEN** integration test fine-tunes imported Qwen model
- **THEN** model trains successfully
- **AND** loss decreases over training
- **AND** checkpoints save correctly with metadata
- **AND** fine-tuning lineage is tracked

#### Scenario: Checkpoint resume with Qwen model
- **WHEN** integration test interrupts and resumes Qwen fine-tuning
- **THEN** training resumes correctly from checkpoint
- **AND** model metadata is preserved
- **AND** fine-tuning lineage is maintained
- **AND** loss progression continues identically

#### Scenario: Switch between model architectures
- **WHEN** integration test trains custom Transformer, then switches to Qwen model
- **THEN** both models train successfully in separate runs
- **AND** checkpoints for each are isolated and correct
- **AND** configs correctly specify model_name for each

#### Scenario: Model registry CLI integration test
- **WHEN** integration test exercises registry CLI commands
- **THEN** list-models works correctly
- **AND** model-info displays accurate information
- **AND** delete-model removes models correctly
- **AND** registry remains consistent after operations

