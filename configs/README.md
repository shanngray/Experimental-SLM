# Configuration Files

This directory contains YAML configuration files for training the transformer language model. Configuration files allow you to customize hyperparameters without modifying code.

## Quick Start

### Using a Config File

To use a configuration file, pass it to `main.py` with the `--config` flag:

```bash
uv run python main.py --config configs/default.yaml
```

### Using Example Configs

Several example configurations are provided:

- **`default.yaml`** - Default configuration with all hyperparameters documented
- **`small-model.yaml`** - Smaller model for quick experimentation
- **`large-model.yaml`** - Larger model for better quality
- **`fast-training.yaml`** - Faster training settings
- **`detailed-eval.yaml`** - More frequent evaluation and sampling

Example usage:

```bash
# Train with a small model
uv run python main.py --config configs/small-model.yaml

# Train with detailed evaluation
uv run python main.py --config configs/detailed-eval.yaml
```

## How Configs Work

### Loading Process

1. When you specify `--config <path>`, the system loads the YAML file
2. The YAML values are merged with `TrainingConfig` defaults
3. Any values specified in the YAML file override the defaults
4. Missing values use the default from `TrainingConfig`

### Creating Custom Configs

To create a custom configuration:

1. **Start from an example**: Copy one of the example configs as a starting point
   ```bash
   cp configs/default.yaml configs/my-config.yaml
   ```

2. **Modify values**: Edit the YAML file to change hyperparameters
   ```yaml
   # Example: Change learning rate and batch size
   learning_rate: 1.0e-4  # Lower learning rate
   batch_size: 32  # Larger batch size
   ```

3. **Use your config**: Run training with your custom config
   ```bash
   uv run python main.py --config configs/my-config.yaml
   ```

### Partial Configs

You don't need to specify all hyperparameters. Only include the ones you want to change:

```yaml
# Minimal config - only override what you need
learning_rate: 1.0e-4
batch_size: 32
max_steps: 5000
```

All other values will use defaults from `TrainingConfig`.

## Configuration Sections

### Model Architecture

Control the structure and size of the transformer:

- `n_layers` - Number of transformer blocks (default: 4)
- `d_model` - Model dimension (default: 256)
- `n_heads` - Number of attention heads (default: 4)
- `d_ff` - Feed-forward dimension (default: 1024)
- `dropout` - Dropout probability (default: 0.1)

**Constraints:**
- `d_model` must be divisible by `n_heads`
- `dropout` must be between 0.0 and 1.0

### Dataset

Control data splitting and processing:

- `train_ratio` - Fraction of data for training (default: 0.95)
- `max_seq_len` - Maximum sequence length (default: 256)

**Constraints:**
- `train_ratio` must be between 0.0 and 1.0 (exclusive)

### Training Loop

Control the training process:

- `max_steps` - Maximum training steps (default: 10000)
- `checkpoint_cadence` - Steps between checkpoints (default: 1000)

**Constraints:**
- `max_steps` must be positive
- `checkpoint_cadence` must be positive or `null` (to disable)

### Optimizer

Control the AdamW optimizer:

- `learning_rate` - Learning rate (default: 3.0e-4)
- `weight_decay` - L2 regularization (default: 0.1)
- `beta1` - First momentum coefficient (default: 0.9)
- `beta2` - Second momentum coefficient (default: 0.95)
- `batch_size` - Batch size (default: 16)

### Evaluation and Sampling

Control when and how the model is evaluated:

- `eval_cadence` - Steps between evaluations (default: `null`, disabled)
- `sampling_cadence` - Steps between text sampling (default: `null`, disabled)
- `sampling_temperature` - Sampling temperature (default: 1.0)
- `sampling_prompt` - Prompt for sampling (default: "The")
- `sampling_max_length` - Max tokens to generate (default: 100)
- `sampling_seed` - Seed for sampling (default: 42)

### Other

- `seed` - Random seed for reproducibility (default: `null`)
- `hooks` - Hook configuration dictionary (default: `null`)

## Common Modifications

### Increase Model Size

```yaml
n_layers: 6
d_model: 512
n_heads: 8
d_ff: 2048
batch_size: 8  # Reduce batch size due to memory constraints
```

### Faster Training

```yaml
max_steps: 5000
batch_size: 32  # Larger batches = fewer steps needed
eval_cadence: null  # Disable evaluation to save time
sampling_cadence: null
```

### More Frequent Monitoring

```yaml
eval_cadence: 500  # Evaluate every 500 steps
sampling_cadence: 500  # Sample every 500 steps
checkpoint_cadence: 500  # Checkpoint more frequently
```

### Lower Learning Rate

```yaml
learning_rate: 1.0e-4  # More conservative learning rate
```

### Longer Sequences

```yaml
max_seq_len: 512  # Longer context window
# Note: This increases memory usage significantly
```

## Command-Line Overrides

You can also override specific values from the command line:

```bash
# Override max_steps even when using a config file
uv run python main.py --config configs/default.yaml --max-steps 5000
```

## Troubleshooting

### Config File Not Found

**Error:** `Config file not found: configs/my-config.yaml`

**Solution:** Check that the file path is correct and the file exists.

### Invalid YAML Syntax

**Error:** `Invalid YAML in config file: ...`

**Solution:** Check your YAML syntax. Common issues:
- Missing colons after keys
- Incorrect indentation
- Unquoted strings with special characters

### Invalid Hyperparameter Values

**Error:** `ValueError: d_model (128) must be divisible by n_heads (3)`

**Solution:** Ensure constraints are met:
- `d_model` must be divisible by `n_heads`
- `dropout` must be between 0.0 and 1.0
- `train_ratio` must be between 0.0 and 1.0
- All positive integer fields must be > 0

### Unknown Hyperparameters

**Note:** Unknown hyperparameters in YAML files are ignored. Only valid `TrainingConfig` fields are used.

## Example Files

- [default.yaml](default.yaml) - Complete reference with all hyperparameters
- [small-model.yaml](small-model.yaml) - Small model example
- [large-model.yaml](large-model.yaml) - Large model example
- [fast-training.yaml](fast-training.yaml) - Fast training example
- [detailed-eval.yaml](detailed-eval.yaml) - Detailed evaluation example

## Reference

For the complete list of available hyperparameters and their defaults, see:
- `src/config.py` - `TrainingConfig` class definition
- `configs/default.yaml` - All hyperparameters with inline documentation

