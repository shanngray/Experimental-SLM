# Project Context

## Purpose
Build a small, locally runnable language model from first principles in order to:
- Deeply understand how language models work end-to-end
- Experiment with architectural and algorithmic changes
- Innovate at both high and low levels of the training and learning process

The emphasis is on learning and experimentation, not on competing with large-scale production models.

## Tech Stack
- **Python** - Primary language for fast iteration and experimentation
- **PyTorch** - ML framework for tensor operations, GPU acceleration, and autograd infrastructure
  - Used for baseline operations while allowing custom overrides for experimental learning rules
- **Target Hardware**: Mac CPU with integrated memory (initially), with modular device backend for future GPU support

## Project Conventions

### Code Style
- **Transparency over abstraction**: Prefer explicit implementations over opaque "magic" abstractions
- **Inspectability**: Each major component (tokenization, model architecture, training loop, backprop, etc.) should be inspectable and modifiable
- **Pythonic style**: Follow Python conventions and PEP 8 where appropriate
- **Modular interfaces**: Clear module boundaries with well-defined input/output contracts

### Architecture Patterns
- **Decoder-only Transformer**: Causal transformer architecture as baseline
- **Modular component design**: 
  - Data Ingest → Tokenizer → Sequence Builder → Batcher → Model Core → Loss → Training Loop
  - Each component has clear interfaces and can be modified independently
- **Experiment Hooks Layer**: Pluggable intervention points for:
  - Forward hooks (modify activations, attention patterns, norms, residuals)
  - Backward hooks (custom gradients via torch.autograd.Function)
  - Update rule hooks (optimizer variants / gradient transforms)
  - Auxiliary losses (regularizers, constraints) toggleable via config
- **Device abstraction**: CPU-first design with modular device layer for future MPS/CUDA support
- **Config-driven**: Model size, data, training, and hooks controlled via YAML configuration files (not JSON)
  - All hyperparameters must be configurable via YAML files in `configs/` directory
  - Hyperparameters are categorized: Model Architecture, Training, Dataset, Evaluation, Sampling, Checkpointing, Other
  - Config system centralized in `TrainingConfig` class (`src/config.py`)
  - YAML files include inline comments documenting each hyperparameter (purpose, defaults, ranges, constraints)
  - Example configs for different use cases (small-model, large-model, fast-training, detailed-eval)
  - Validation logic enforces constraints (e.g., `d_model % n_heads == 0`, `0.0 <= dropout <= 1.0`)
  - CLI overrides permitted for convenience (e.g., `--max-steps`) but config remains source of truth
  - Backward compatibility required: missing fields use sensible defaults from `TrainingConfig`
  - No hardcoded hyperparameter values in main entry points; all flow from config

### Testing Strategy
- **Reproducibility**: Strict random seed management and deterministic operations
- **Experiment tracking**: Config hashes per run to ensure hooks don't silently invalidate comparisons
- **Baseline comparisons**: Maintain baseline runs for clean experimental comparisons
- **One-change-per-run policy**: Enforce clean ablation studies

### Git Workflow
- Use meaningful commit messages that describe changes clearly
- Branch strategy: TBD (consider feature branches for experimental variants)
- Version configs for reproducibility

## Domain Context
- **Model Architecture**: Decoder-only Transformer with:
  - Embedding + Positional Encoding (sinusoidal or learned, TBD)
  - N Transformer blocks (self-attention + MLP)
  - LayerNorm + LM head → logits over vocab
- **Tokenization**: Character-level ASCII initially (vocab ~128-256), with migration path for future strategies
- **Context Length**: 256 tokens (computationally manageable on CPU while showing sequence effects)
- **Training**: Next-token prediction with cross-entropy loss
- **Sequence Building**: Creates (x, y) pairs where x=[t0..t255], y=[t1..t256] (shifted)
- **Batch Processing**: Fixed or dynamic batch sizing (TBD) producing tensors on current device
- **Evaluation**: Perplexity/avg loss on validation set + qualitative text generation (greedy/temperature sampling)

## Important Constraints
- **From scratch, but not from nothing**: Do not start from existing pretrained models
- **Avoid unnecessary re-implementation**: Use well-established libraries (PyTorch) for tensor operations, but understand and control their usage
- **CPU-first**: Initial target is Mac CPU; architecture should scale but remain runnable locally
- **Small scale**: Start with minimal parameter count and vocabulary, designed for local hardware
- **Experimental flexibility**: System must allow low-level experimentation (modifying learning dynamics, backpropagation behavior, architectural assumptions)
- **Scalability by design**: Architecture should be scalable incrementally, limited primarily by local hardware or affordable rented compute
- **Transparency requirement**: Each component must be inspectable and modifiable, avoiding opaque abstractions

## External Dependencies
- **PyTorch**: Core ML framework for tensor operations and autograd
- **Python standard library**: For basic utilities
- **Future considerations**: 
  - Text datasets (TBD - simple, readily available datasets for initial training)
  - Potentially lightweight experiment tracking tools (TBD - may start with simple filesystem logs)
  - No plans for lower-level (C++/CUDA) components initially, but architecture should allow for targeted extensions if needed later
