# Change: Update README Documentation

## Why
Phase 2 Sessions 2-4 have extended the configuration system to include all hyperparameters, created YAML config files, and updated main.py to use config throughout. However, the README.md still has only basic configuration documentation. Users need comprehensive hyperparameter documentation to understand what each hyperparameter does, what values they take, and how to modify them effectively. This session updates README.md with complete hyperparameter reference documentation, examples, and troubleshooting guidance.

## What Changes
- Replace existing "Configuration" section in README.md with comprehensive guide:
  - Overview of config system
  - How to use config files
  - How to create custom configs
  - How to override via CLI
- Add detailed hyperparameter reference section documenting all hyperparameters:
  - Model Architecture: n_layers, d_model, n_heads, d_ff, dropout
  - Training: learning_rate, weight_decay, beta1, beta2, batch_size, max_steps
  - Dataset: max_seq_len, train_ratio
  - Evaluation/Sampling: eval_cadence, sampling_cadence, sampling_temperature, sampling_prompt, sampling_max_length, sampling_seed
  - Checkpointing: checkpoint_cadence
  - Other: seed
  - Each hyperparameter includes: description, default value, reasonable range, notes on interactions
- Add examples section with practical use cases:
  - Using default config
  - Using custom config file
  - Modifying specific hyperparameters
  - Creating small/large model configs
- Add troubleshooting section:
  - Common config errors
  - How to validate config files
  - How to check which config is active
- Update Quick Start section to mention config files
- Add links to example config files in configs/ directory

## Impact
- Affected specs: `main-entry` (adds requirement for comprehensive README documentation)
- Affected code: `README.md` (major documentation update)
- This change completes Phase 2 by providing users with complete hyperparameter documentation, enabling them to effectively use and modify the configuration system without needing to read source code

