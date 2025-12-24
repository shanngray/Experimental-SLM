# Change: Update main.py Integration

## Why
Phase 2 Session 2 extended `TrainingConfig` to include all hyperparameters (model architecture, dataset, training loop), and Session 3 created YAML config files. However, `main.py` still contains hardcoded values that should come from config. This session updates `main.py` to use config for all hyperparameters, removing hardcoded values and ensuring the config system works end-to-end.

## What Changes
- Update `Transformer` instantiation in `main.py` to use config hyperparameters:
  - `n_layers` from `config.n_layers`
  - `d_model` from `config.d_model`
  - `n_heads` from `config.n_heads`
  - `d_ff` from `config.d_ff`
  - `dropout` from `config.dropout`
- Update `split_corpus` call to use `config.train_ratio` instead of hardcoded 0.95
- Update checkpoint saving logic to use `config.checkpoint_cadence` instead of hardcoded 1000
- Update `max_steps` handling to read from `config.max_steps` by default, with CLI override support
- Add logging to show which config values are being used for transparency
- Add validation that required config values are present
- Ensure backward compatibility (default config used if none provided, missing values use defaults)

## Impact
- Affected specs: `main-entry` (Model and Trainer Initialization, Training Loop Execution, Data Loading and Setup requirements)
- Affected code: `main.py` (removes hardcoded hyperparameters, uses config throughout)
- This change completes the hyperparameter centralization goal of Phase 2, enabling users to modify all hyperparameters via YAML files without code changes

