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
uv run python main.py
```

This will:
- Create a character-level tokenizer
- Split the corpus into train/val sets (95%/5%)
- Initialize a 4-layer Transformer (256 dim, 4 heads)
- Train for the configured number of steps
- Save checkpoints periodically
- Log training progress and generate text samples

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

The model can be configured by editing configuration files or passing parameters. Key hyperparameters:

**Model Architecture:**
- `n_layers`: 4 (transformer blocks)
- `d_model`: 256 (embedding dimension)
- `n_heads`: 4 (attention heads)
- `d_ff`: 1024 (feed-forward dimension)
- `dropout`: 0.1
- `context_length`: 256 tokens

**Training:**
- `batch_size`: 16
- `learning_rate`: 3e-4
- `weight_decay`: 0.1
- `betas`: (0.9, 0.95)
- `max_steps`: Configurable

**Evaluation:**
- Validation loss computed every 200 steps
- Text samples generated periodically
- Fixed seed for reproducibility

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
