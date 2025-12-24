## 1. Update main.py to use config for model architecture
- [x] 1.1 Update `Transformer` instantiation (line ~238, ~253) to pass config hyperparameters:
  - [x] `n_layers=config.n_layers`
  - [x] `d_model=config.d_model`
  - [x] `n_heads=config.n_heads`
  - [x] `d_ff=config.d_ff`
  - [x] `dropout=config.dropout`
- [x] 1.2 Remove hardcoded model architecture defaults from `Transformer` calls

## 2. Update main.py to use config for dataset hyperparameters
- [x] 2.1 Update `split_corpus` call (line ~218) to use `config.train_ratio` instead of hardcoded 0.95
- [x] 2.2 Update logging to show train_ratio being used

## 3. Update main.py to use config for training loop hyperparameters
- [x] 3.1 Update `max_steps` handling (line ~187):
  - [x] Read from `config.max_steps` by default
  - [x] Allow CLI override via `--max-steps` argument (prefer CLI if provided)
  - [x] Update logic: `max_steps = args.max_steps if args.max_steps is not None else config.max_steps`
- [x] 3.2 Update checkpoint saving logic (line ~286) to use `config.checkpoint_cadence`:
  - [x] Replace hardcoded `1000` with `config.checkpoint_cadence`
  - [x] Handle `None` case (if checkpoint_cadence is None, disable periodic checkpointing)
  - [x] Ensure final checkpoint is still saved regardless of cadence

## 4. Add config value logging and validation
- [x] 4.1 Add logging to show active config values:
  - [x] Log model architecture hyperparameters when model is created
  - [x] Log dataset hyperparameters when splitting corpus
  - [x] Log training loop hyperparameters (max_steps, checkpoint_cadence) at start
- [x] 4.2 Add validation that required config values are present (if needed)
- [x] 4.3 Ensure error messages are helpful if config is invalid

## 5. Ensure backward compatibility
- [x] 5.1 Verify default config is used if no config file provided (already works, verify)
- [x] 5.2 Verify missing config values use defaults from `TrainingConfig` (already works, verify)
- [x] 5.3 Test that existing config files without new fields still work

## 6. Testing and validation
- [x] 6.1 Test that model is created with correct architecture from config
- [x] 6.2 Test that dataset split uses config.train_ratio
- [x] 6.3 Test that checkpoint cadence uses config.checkpoint_cadence
- [x] 6.4 Test that CLI override for max_steps still works
- [x] 6.5 Test that default config is used when no config file provided
- [x] 6.6 Test that missing config values use defaults
- [x] 6.7 Verify no hardcoded hyperparameter values remain in main.py

