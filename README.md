# Experimental SLM

A small, locally runnable language model built from first principles for learning and experimentation.

## Project Overview

This project aims to build a small language model from scratch to deeply understand how language models work end-to-end. The emphasis is on learning and experimentation, not on competing with large-scale production models.

**Goals:**
- Deeply understand how language models work end-to-end
- Experiment with architectural and algorithmic changes
- Innovate at both high and low levels of the training and learning process

## Setup Instructions

### Prerequisites

- **Python 3.13** or higher
- **uv** - Fast Python package installer and resolver

### Installing uv

If you don't have `uv` installed, you can install it using one of the following methods:

#### macOS/Linux (using curl)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### macOS (using Homebrew)
```bash
brew install uv
```

#### Windows (using PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Using pip
```bash
pip install uv
```

After installation, restart your terminal or run `source ~/.bashrc` (or equivalent) to ensure `uv` is in your PATH.

### Setting Up the Project

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd experimental-slm
   ```

2. **Install dependencies using uv**:
   ```bash
   uv sync
   ```
   
   This command will:
   - Create a virtual environment (if one doesn't exist)
   - Install Python 3.13 (if not already installed)
   - Install all project dependencies (including PyTorch)
   - Make the project available in the environment

3. **Activate the virtual environment** (if needed):
   
   After running `uv sync`, you can activate the environment:
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```
   
   Alternatively, you can run commands directly with `uv run`:
   ```bash
   uv run python main.py
   ```

4. **Verify installation**:
   ```bash
   uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

### Adding New Dependencies

To add a new dependency to the project:

1. **Add it to pyproject.toml**:
   ```bash
   uv add <package-name>
   ```
   
   Or manually edit `pyproject.toml` and add it to the `dependencies` list, then run:
   ```bash
   uv sync
   ```

2. **Remove a dependency**:
   ```bash
   uv remove <package-name>
   ```

### Common uv Commands

- `uv sync` - Install all dependencies and sync the environment
- `uv add <package>` - Add a new dependency
- `uv remove <package>` - Remove a dependency
- `uv run <command>` - Run a command in the project's virtual environment
- `uv pip list` - List installed packages

## Project Status

**Phase 1: Complete** ✅

All core functionality has been implemented and tested:
- ✅ Character-level tokenizer with ASCII policy
- ✅ Dataset preparation with train/val split
- ✅ DataLoader with batching infrastructure
- ✅ Decoder-only Transformer model (4 layers, 256 dim)
- ✅ Training loop with AdamW optimizer
- ✅ Checkpointing and resume functionality
- ✅ Hooks system for experimentation
- ✅ Evaluation and text sampling

## Project Structure

```
experimental-slm/
├── src/                          # Source code
│   ├── tokenizer.py             # Character-level tokenizer
│   ├── normalize.py             # Text normalization (ASCII policy)
│   ├── dataset.py               # Dataset with train/val split
│   ├── dataloader.py            # Batching infrastructure
│   ├── config.py                # Configuration management
│   ├── model/                   # Transformer model components
│   │   ├── attention.py         # Multi-head attention (causal)
│   │   ├── mlp.py              # Feed-forward network
│   │   ├── layer_norm.py       # Layer normalization
│   │   ├── embeddings.py       # Token & positional embeddings
│   │   ├── transformer_block.py # Single transformer block
│   │   └── transformer.py      # Full model assembly
│   ├── training/                # Training infrastructure
│   │   ├── trainer.py          # Training loop
│   │   ├── loss.py             # Cross-entropy loss
│   │   └── checkpoint.py       # Save/load checkpoints
│   ├── hooks/                   # Hook system for experiments
│   │   ├── registry.py         # Hook management
│   │   ├── forward_hooks.py    # Forward pass hooks
│   │   └── update_hooks.py     # Gradient update hooks
│   ├── evaluation/              # Evaluation utilities
│   │   └── evaluator.py        # Validation loss computation
│   └── sampling/                # Text generation
│       └── sampler.py          # Temperature-based sampling
├── tests/                       # Comprehensive unit tests
├── configs/                     # YAML configuration files
├── data/                        # Dataset storage
├── logs/                        # Training logs
├── checkpoints/                 # Model checkpoints
├── main.py                      # Main training script
└── pyproject.toml              # Project dependencies (uv)
```

## Quick Start

### 1. Prepare Your Dataset

Place your training text file(s) in the `data/` directory:

```bash
# Example: download a sample text corpus
curl -o data/sample.txt https://www.gutenberg.org/files/1342/1342-0.txt
```

### 2. Run Training

```bash
# Use default configuration
uv run python main.py

# Or use a custom configuration file
uv run python main.py --config configs/small-model.yaml
```

This will:
- Create a character-level tokenizer
- Split the corpus into train/val sets (95%/5%)
- Initialize a 4-layer Transformer (256 dim, 4 heads) by default
- Train for the configured number of steps
- Save checkpoints periodically
- Log training progress and generate text samples

See the [Configuration](#configuration) section below for details on customizing hyperparameters. Example config files are available in the [`configs/`](configs/) directory.

### 3. Resume from Checkpoint

```bash
uv run python main.py --resume checkpoints/latest.pt
```

### 4. Run Tests

Run the full test suite to verify everything works:

```bash
uv run pytest tests/ -v
```

Run specific test modules:

```bash
uv run pytest tests/test_tokenizer.py -v
uv run pytest tests/test_transformer.py -v
```

## Configuration

The model uses a YAML-based configuration system that allows you to customize all hyperparameters without modifying code. Configuration files provide a clean, version-controlled way to manage training settings.

### Overview

The configuration system works by:
1. Loading default values from `TrainingConfig` class
2. Optionally loading values from a YAML config file (via `--config` flag)
3. Optionally overriding specific values via command-line arguments (e.g., `--max-steps`)
4. Validating all values and constraints before training starts

### Using Config Files

To use a configuration file, pass it to `main.py` with the `--config` flag:

```bash
uv run python main.py --config configs/default.yaml
```

If no config file is specified, the system uses default values from `TrainingConfig`.

### Creating Custom Configs

You can create custom configuration files by:

1. **Starting from an example**: Copy one of the example configs as a starting point
   ```bash
   cp configs/default.yaml configs/my-config.yaml
   ```

2. **Modifying values**: Edit the YAML file to change hyperparameters
   ```yaml
   # Example: Change learning rate and batch size
   learning_rate: 1.0e-4
   batch_size: 32
   ```

3. **Using your config**: Run training with your custom config
   ```bash
   uv run python main.py --config configs/my-config.yaml
   ```

**Partial configs are supported**: You don't need to specify all hyperparameters. Only include the ones you want to change. All other values will use defaults from `TrainingConfig`.

### Overriding via CLI

You can override specific hyperparameters from the command line even when using a config file:

```bash
# Override max_steps even when using a config file
uv run python main.py --config configs/default.yaml --max-steps 5000
```

### Example Config Files

Several example configurations are provided in the [`configs/`](configs/) directory:

- **[`configs/default.yaml`](configs/default.yaml)** - Complete reference with all hyperparameters documented
- **[`configs/small-model.yaml`](configs/small-model.yaml)** - Smaller model for quick experimentation
- **[`configs/large-model.yaml`](configs/large-model.yaml)** - Larger model for better quality
- **[`configs/fast-training.yaml`](configs/fast-training.yaml)** - Faster training settings
- **[`configs/detailed-eval.yaml`](configs/detailed-eval.yaml)** - More frequent evaluation and sampling

For detailed information about configuration files, see [`configs/README.md`](configs/README.md).

## Hyperparameter Reference

This section documents all available hyperparameters, their default values, reasonable ranges, and important constraints.

### Model Architecture Hyperparameters

These control the structure and size of the transformer model.

| Hyperparameter | Default | Range | Description |
|---------------|---------|-------|-------------|
| `n_layers` | 4 | 2-12+ | Number of transformer blocks. Controls model depth. More layers increase capacity but slow training/inference. |
| `d_model` | 256 | 128-2048+ | Model dimension / embedding size. Controls the width of the model. **Must be divisible by `n_heads`**. |
| `n_heads` | 4 | 2-32+ | Number of attention heads. Controls multi-head attention. **Must divide `d_model` evenly**. |
| `d_ff` | 1024 | 256-8192+ | Feed-forward network dimension. Controls the width of the MLP layers. Typically 4x `d_model`. |
| `dropout` | 0.1 | 0.0-1.0 | Dropout probability. Applied to attention and MLP layers for regularization. **Constraint: 0.0 ≤ dropout ≤ 1.0** |

**Important Constraints:**
- `d_model` must be divisible by `n_heads` (e.g., if `d_model=256` and `n_heads=4`, then 256 % 4 == 0 ✓)
- `dropout` must be between 0.0 and 1.0 (inclusive)

### Training Hyperparameters

These control the training process and optimizer behavior.

| Hyperparameter | Default | Range | Description |
|---------------|---------|-------|-------------|
| `learning_rate` | 3e-4 | 1e-5 to 1e-3 | Learning rate for AdamW optimizer. Common range: 1e-5 to 1e-3. |
| `weight_decay` | 0.1 | 0.0-1.0 | Weight decay for L2 regularization. Helps prevent overfitting. |
| `beta1` | 0.9 | 0.8-0.99 | First momentum coefficient for AdamW. Controls exponential decay rate for first moment estimates. |
| `beta2` | 0.95 | 0.9-0.999 | Second momentum coefficient for AdamW. Controls exponential decay rate for second moment estimates. |
| `batch_size` | 16 | 1-128+ | Batch size for training. Larger batches use more memory but provide more stable gradients. |
| `max_steps` | 10000 | 1+ | Maximum number of training steps. Training stops after this many steps. **Can be overridden via CLI with `--max-steps`**. |

### Dataset Hyperparameters

These control how the training data is split and processed.

| Hyperparameter | Default | Range | Description |
|---------------|---------|-------|-------------|
| `max_seq_len` | 256 | 64-2048+ | Maximum sequence length (context window). Longer sequences allow the model to see more context but use more memory. |
| `train_ratio` | 0.95 | 0.0-1.0 (exclusive) | Fraction of data to use for training. Remaining fraction (1 - `train_ratio`) is used for validation. **Constraint: 0.0 < train_ratio < 1.0** |

### Evaluation/Sampling Hyperparameters

These control when and how the model is evaluated and sampled during training.

| Hyperparameter | Default | Range | Description |
|---------------|---------|-------|-------------|
| `eval_cadence` | null | positive integer or null | Evaluation cadence in steps. If set, validation loss is computed every N steps. If `null`, evaluation is disabled. |
| `sampling_cadence` | null | positive integer or null | Sampling cadence in steps. If set, text samples are generated every N steps. If `null`, sampling is disabled. |
| `sampling_temperature` | 1.0 | 0.1-2.0 | Temperature for text sampling. Lower values (0.1-0.5) make output more deterministic. Higher values (1.0-2.0) make output more creative/random. |
| `sampling_prompt` | "The" | any string | Fixed prompt for text sampling. The model will generate text starting from this prompt. |
| `sampling_max_length` | 100 | 1-1000+ | Maximum number of tokens to generate during sampling. |
| `sampling_seed` | 42 | any integer or null | Random seed for sampling reproducibility. Set to `null` for non-deterministic sampling. |

### Checkpointing Hyperparameters

These control checkpoint saving behavior.

| Hyperparameter | Default | Range | Description |
|---------------|---------|-------|-------------|
| `checkpoint_cadence` | 1000 | positive integer or null | Steps between checkpoint saves. If `null`, periodic checkpointing is disabled (checkpoints are still saved at the end of training). |

### Other Hyperparameters

| Hyperparameter | Default | Range | Description |
|---------------|---------|-------|-------------|
| `seed` | null | any integer or null | Random seed for reproducibility. Set to `null` for non-deterministic training. |

## Configuration Examples

This section provides practical examples of using the configuration system.

### Example 1: Using Default Config

The simplest way to train is to use the default configuration:

```bash
uv run python main.py
```

This automatically uses default values from `TrainingConfig` (4 layers, 256 dim, 4 heads, etc.). No config file is needed.

### Example 2: Using Custom Config File

To use a custom configuration file:

```bash
uv run python main.py --config configs/my-config.yaml
```

Create `configs/my-config.yaml` with your desired hyperparameters. Only specify the values you want to change; defaults are used for everything else.

### Example 3: Modifying Specific Hyperparameters

You can create a minimal config file that only overrides specific values:

```yaml
# configs/custom-lr.yaml - Only change learning rate and batch size
learning_rate: 1.0e-4  # Lower learning rate
batch_size: 32  # Larger batch size
```

Then use it:

```bash
uv run python main.py --config configs/custom-lr.yaml
```

All other hyperparameters will use their default values.

### Example 4: Creating a Small Model Config

For quick experimentation or limited computational resources, use a smaller model:

```yaml
# configs/my-small-model.yaml
n_layers: 2  # Fewer transformer blocks
d_model: 128  # Smaller model dimension
n_heads: 2  # Fewer attention heads (must divide d_model)
d_ff: 512  # Smaller feed-forward dimension
batch_size: 32  # Can use larger batch size with smaller model
```

**Trade-offs:**
- ✅ Faster training and inference
- ✅ Lower memory usage
- ❌ Less model capacity (may not capture complex patterns)

### Example 5: Creating a Large Model Config

For better quality results with sufficient computational resources:

```yaml
# configs/my-large-model.yaml
n_layers: 6  # More transformer blocks
d_model: 512  # Larger model dimension
n_heads: 8  # More attention heads (must divide d_model)
d_ff: 2048  # Larger feed-forward dimension
batch_size: 8  # Smaller batch size due to memory constraints
max_steps: 20000  # May need more steps to converge
```

**Trade-offs:**
- ✅ Better model capacity and quality
- ❌ Slower training and inference
- ❌ Higher memory usage
- ❌ Requires more training steps to converge

## Troubleshooting

This section covers common configuration issues and how to resolve them.

### Common Config Errors

#### Config File Not Found

**Error:** `Config file not found: configs/my-config.yaml`

**Solution:** Check that the file path is correct and the file exists. Use relative paths from the project root or absolute paths.

#### Invalid YAML Syntax

**Error:** `Invalid YAML in config file: ...`

**Solution:** Check your YAML syntax. Common issues:
- Missing colons after keys
- Incorrect indentation (YAML is indentation-sensitive)
- Unquoted strings with special characters
- Mixing tabs and spaces (use spaces only)

**Example of correct YAML:**
```yaml
learning_rate: 3.0e-4
batch_size: 16
n_layers: 4
```

#### Invalid Hyperparameter Values

**Error:** `ValueError: d_model (128) must be divisible by n_heads (3)`

**Solution:** Ensure constraints are met:
- `d_model` must be divisible by `n_heads` (e.g., if `n_heads=4`, `d_model` must be 4, 8, 12, 16, 20, 24, 28, 32, ...)
- `dropout` must be between 0.0 and 1.0 (inclusive)
- `train_ratio` must be between 0.0 and 1.0 (exclusive, i.e., not 0.0 or 1.0)
- All positive integer fields (`n_layers`, `d_model`, `n_heads`, `d_ff`, `batch_size`, `max_steps`, etc.) must be > 0

#### Unknown Hyperparameters

**Note:** Unknown hyperparameters in YAML files are silently ignored. Only valid `TrainingConfig` fields are used. This allows you to add comments or temporary values without breaking the config loader.

### How to Validate Config Files

Before running training, you can validate your config file:

1. **Check YAML syntax**: Use a YAML validator or try loading it:
   ```bash
   python -c "import yaml; yaml.safe_load(open('configs/my-config.yaml'))"
   ```

2. **Verify hyperparameter constraints**: Ensure:
   - `d_model % n_heads == 0`
   - `0.0 <= dropout <= 1.0`
   - `0.0 < train_ratio < 1.0`
   - All positive integer fields are > 0

3. **Test loading config**: The config loader will validate values when you run training. If there are errors, they will be reported before training starts.

### How to Check Which Config is Active

When you run training, the system logs the active configuration:

1. **Check startup logs**: At the beginning of training, the system logs a summary of the active configuration, including all hyperparameter values.

2. **Check config hash**: Each run logs a `config_hash` that uniquely identifies the configuration used. This helps ensure reproducibility.

3. **Review config file**: If you specified `--config`, the values from that file (merged with defaults) are what's active. If you didn't specify a config file, defaults from `TrainingConfig` are used.

## Architecture Overview

### Tokenizer

- **Type:** Character-level
- **Policy:** ASCII printable (32-126) + newline + tab
- **Special tokens:** `<PAD>` (0), `<UNK>` (1)
- **Vocab size:** ~100 tokens
- **Format:** JSON vocab file for persistence

### Model

- **Architecture:** Decoder-only Transformer (GPT-style)
- **Layers:** 4 transformer blocks
- **Attention:** Multi-head causal attention (4 heads)
- **Embeddings:** Learned token + learned positional
- **Normalization:** Layer normalization (pre-norm)
- **Activation:** GELU (in MLP)
- **Parameters:** ~2-3M (small, locally runnable)

### Training

- **Objective:** Next-token prediction (cross-entropy)
- **Optimizer:** AdamW with weight decay
- **Batching:** Fixed-size sequences (256 tokens)
- **Windowing:** Non-overlapping sliding windows
- **Checkpointing:** Model, optimizer, config, step counter

### Hooks System

A flexible hook system allows experimentation without modifying core code:

- **Forward hooks:** Log activation statistics without changing outputs
- **Update hooks:** Transform gradients before applying updates
- **Registry:** Enable/disable hooks via configuration
- **Safety logging:** Every run logs run_id, git_commit, config_hash, hook_list

### Evaluation & Sampling

- **Validation:** Computed on held-out 5% of corpus
- **Sampling:** Temperature-based multinomial sampling
- **Reproducibility:** Fixed seeds for deterministic generation
- **Temperature:** 1.0 (default, configurable)

## Development Workflow

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_trainer.py -v
```

### Code Quality

The project follows clean code practices:
- Comprehensive unit tests for all modules
- Type hints throughout the codebase
- Docstrings for all public functions and classes
- Modular, testable components

### Reproducibility

All experiments are reproducible:
- Seed-based initialization for all randomness
- Deterministic dataset splits
- Checkpoint/resume produces identical results
- Every run logged with git commit and config hash

## Logging

Training produces structured logs including:
- `run_id`: Unique identifier for each run
- `git_commit`: Git commit hash at training time
- `config_hash`: Hash of configuration for reproducibility
- `step`: Current training step
- `train_loss`: Training loss
- `val_loss`: Validation loss (periodic)
- `sample_text`: Generated text samples (periodic)
- `hook_list`: Active hooks for the run

## Next Steps (Phase 2+)

Potential enhancements and experiments:
- Larger models (more layers, larger d_model)
- Advanced sampling strategies (top-k, nucleus)
- Learning rate scheduling
- Gradient clipping
- Batch accumulation for larger effective batch sizes
- BPE or WordPiece tokenization
- Flash attention for efficiency
- Distributed training support

## Contributing

This is a learning and experimentation project. Feel free to fork, modify, and experiment! 

If you find bugs or have suggestions:
1. Document the issue with reproduction steps
2. Include relevant logs and configuration
3. Propose a fix if possible

## Key Features

### 🎯 Built from First Principles
- All components implemented from scratch (no high-level frameworks)
- Every line of code is understandable and modifiable
- Clear separation of concerns for easy experimentation

### 🔬 Designed for Experimentation
- Modular architecture allows swapping components
- Hook system enables non-invasive modifications
- Comprehensive test coverage ensures changes don't break functionality

### 📊 Reproducibility First
- Deterministic training with seed control
- Every run tracked with git commit and config hash
- Checkpoint/resume produces identical results

### 🚀 Locally Runnable
- Small model size (~2-3M parameters)
- Runs on CPU or GPU
- Fast iteration cycles for learning

### 📝 Well-Documented
- Comprehensive docstrings
- Unit tests serve as usage examples
- Clear architecture documentation

## References

This project implements concepts from:
- "Attention Is All You Need" (Vaswani et al., 2017)
- GPT-style decoder-only architecture
- Modern best practices for LLM training

## License

This project is provided as-is for learning and experimentation purposes.
