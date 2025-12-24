## MODIFIED Requirements

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

