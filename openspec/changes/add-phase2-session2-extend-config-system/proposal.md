# Change: Extend Config System

## Why
Phase 2 Session 1 audit identified 8 hyperparameters that are currently hardcoded but should be configurable via YAML files. These include model architecture parameters (n_layers, d_model, n_heads, d_ff, dropout), dataset parameters (train_ratio), and training loop parameters (max_steps, checkpoint_cadence). Extending `TrainingConfig` to include these hyperparameters is the foundation for making all hyperparameters configurable without code changes.

## What Changes
- Extend `TrainingConfig` class with 8 new hyperparameter fields:
  - Model architecture: `n_layers`, `d_model`, `n_heads`, `d_ff`, `dropout`
  - Dataset: `train_ratio`
  - Training loop: `max_steps`, `checkpoint_cadence`
- Update `from_dict()` method to handle new fields
- Update `to_dict()` method to serialize new fields
- Add comprehensive docstrings explaining each hyperparameter
- Maintain backward compatibility with existing configs

## Impact
- Affected specs: `main-entry` (Configuration Loading requirement)
- Affected code: `src/config.py` (TrainingConfig class)
- This change enables Phase 2 Sessions 3-6 (YAML configs, main.py integration, documentation, testing)

