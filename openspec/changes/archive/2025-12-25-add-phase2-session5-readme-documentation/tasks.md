## 1. Replace existing Configuration section with comprehensive guide
- [x] 1.1 Replace existing "Configuration" section (lines ~210-233) with comprehensive guide:
  - [x] Overview of config system
  - [x] How to use config files (--config flag)
  - [x] How to create custom configs
  - [x] How to override via CLI (--max-steps, etc.)
  - [x] Link to configs/README.md for detailed config documentation

## 2. Add detailed hyperparameter reference section
- [x] 2.1 Add Model Architecture hyperparameters section:
  - [x] `n_layers` - description, default (4), range, notes
  - [x] `d_model` - description, default (256), range, notes (must be divisible by n_heads)
  - [x] `n_heads` - description, default (4), range, notes
  - [x] `d_ff` - description, default (1024), range, notes
  - [x] `dropout` - description, default (0.1), range (0.0-1.0), notes
- [x] 2.2 Add Training hyperparameters section:
  - [x] `learning_rate` - description, default (3e-4), range, notes
  - [x] `weight_decay` - description, default (0.1), range, notes
  - [x] `beta1` - description, default (0.9), range, notes
  - [x] `beta2` - description, default (0.95), range, notes
  - [x] `batch_size` - description, default (16), range, notes
  - [x] `max_steps` - description, default (10000), range, notes (CLI override available)
- [x] 2.3 Add Dataset hyperparameters section:
  - [x] `max_seq_len` - description, default (256), range, notes
  - [x] `train_ratio` - description, default (0.95), range (0.0-1.0), notes
- [x] 2.4 Add Evaluation/Sampling hyperparameters section:
  - [x] `eval_cadence` - description, default (null), range, notes (null disables)
  - [x] `sampling_cadence` - description, default (null), range, notes (null disables)
  - [x] `sampling_temperature` - description, default (1.0), range, notes
  - [x] `sampling_prompt` - description, default ("The"), range, notes
  - [x] `sampling_max_length` - description, default (100), range, notes
  - [x] `sampling_seed` - description, default (42), range, notes
- [x] 2.5 Add Checkpointing hyperparameters section:
  - [x] `checkpoint_cadence` - description, default (1000), range, notes (null disables periodic)
- [x] 2.6 Add Other hyperparameters section:
  - [x] `seed` - description, default (null), range, notes

## 3. Add examples section
- [x] 3.1 Add example: Using default config
  - [x] Show command: `uv run python main.py`
  - [x] Explain that default config is used automatically
- [x] 3.2 Add example: Using custom config file
  - [x] Show command: `uv run python main.py --config configs/my-config.yaml`
  - [x] Explain how to create custom config
- [x] 3.3 Add example: Modifying specific hyperparameters
  - [x] Show minimal YAML config with just changed values
  - [x] Explain partial configs work
- [x] 3.4 Add example: Creating a small model config
  - [x] Show example config values (e.g., 2 layers, 128 dim)
  - [x] Explain use case and trade-offs
- [x] 3.5 Add example: Creating a large model config
  - [x] Show example config values (e.g., 6 layers, 512 dim)
  - [x] Explain use case and trade-offs

## 4. Add troubleshooting section
- [x] 4.1 Add common config errors:
  - [x] Config file not found
  - [x] Invalid YAML syntax
  - [x] Invalid hyperparameter values (e.g., d_model not divisible by n_heads)
  - [x] Unknown hyperparameters (ignored)
- [x] 4.2 Add how to validate config files:
  - [x] Check YAML syntax
  - [x] Verify hyperparameter constraints
  - [x] Test loading config
- [x] 4.3 Add how to check which config is active:
  - [x] Explain logging shows active config values
  - [x] Show where to find config summary in logs

## 5. Update Quick Start section
- [x] 5.1 Update Quick Start section (lines ~164-187) to mention config files:
  - [x] Add note about using --config flag
  - [x] Link to config examples
  - [x] Mention that default config is used if none specified

## 6. Add links to example config files
- [x] 6.1 Add links to example config files in configs/ directory:
  - [x] Link to configs/default.yaml
  - [x] Link to configs/small-model.yaml
  - [x] Link to configs/large-model.yaml
  - [x] Link to configs/fast-training.yaml
  - [x] Link to configs/detailed-eval.yaml
  - [x] Link to configs/README.md for detailed config documentation

## 7. Ensure documentation consistency
- [x] 7.1 Verify all hyperparameter defaults match actual TrainingConfig defaults
- [x] 7.2 Verify all examples use correct YAML syntax
- [x] 7.3 Verify all file paths are correct
- [x] 7.4 Verify documentation matches actual config system behavior
- [x] 7.5 Ensure formatting is clear and readable (tables, code blocks, etc.)

