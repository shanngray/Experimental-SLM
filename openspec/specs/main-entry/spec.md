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
The system SHALL load training configuration, supporting YAML files or defaults, with command-line overrides.

#### Scenario: Load configuration from YAML file
- **WHEN** `main.py` is invoked with `--config <path>` argument
- **THEN** configuration is loaded from YAML file and converted to TrainingConfig
- **AND** loaded values override TrainingConfig defaults

#### Scenario: Use default configuration
- **WHEN** `main.py` is invoked without `--config` argument
- **THEN** default TrainingConfig is used
- **AND** training proceeds with sensible defaults

#### Scenario: Command-line argument overrides
- **WHEN** `main.py` is invoked with `--max-steps <N>`
- **THEN** max_steps override is applied to configuration
- **AND** training runs for exactly N steps

### Requirement: Data Loading and Setup
The system SHALL load text data, initialize tokenizer, and create train/val datasets and dataloaders.

#### Scenario: Load data and create datasets
- **WHEN** data file path is provided (via --data or default)
- **THEN** text data is loaded from file
- **AND** Tokenizer is initialized and corpus is tokenized
- **AND** corpus is split using `split_corpus()` (95%/5% train/val)
- **AND** WindowDataset instances are created for train and val
- **AND** DataLoader instances are created with batch_size from config

### Requirement: Model and Trainer Initialization
The system SHALL initialize the Transformer model, optimizer, and Trainer.

#### Scenario: Create model and trainer
- **WHEN** vocab_size and configuration are available
- **THEN** Transformer model is created with vocab_size and config parameters
- **AND** AdamW optimizer is created using `create_optimizer()`
- **AND** Trainer is initialized with model, optimizer, config, val_dataloader, and tokenizer

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
The system SHALL execute the training loop by iterating over batches and calling trainer methods.

#### Scenario: Run training loop
- **WHEN** trainer is initialized
- **THEN** training loop iterates over batches from training DataLoader
- **AND** `trainer.training_step(batch)` is called for each batch
- **AND** loop continues until max_steps is reached
- **AND** Trainer handles logging, evaluation cadence, and sampling cadence internally

#### Scenario: Periodic checkpoint saving
- **WHEN** training is in progress
- **THEN** checkpoints are saved periodically (e.g., every N steps or at end)
- **AND** `trainer.save_checkpoint()` is called with tokenizer
- **AND** checkpoint path is logged

#### Scenario: Graceful shutdown on KeyboardInterrupt
- **WHEN** training is interrupted with Ctrl+C (KeyboardInterrupt)
- **THEN** current checkpoint is saved before exit
- **AND** program exits gracefully
- **AND** training can be resumed from saved checkpoint

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

