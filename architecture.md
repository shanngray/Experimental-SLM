Updated architecture sketch (v1)

System-level blocks
	1.	Data Ingest
	•	Raw text → normalization (ASCII-only for now) → train/val split
	2.	Tokenizer
	•	ASCII char-level → token_id: int (vocab ~128–256 depending on control chars policy)
	3.	Sequence Builder
	•	Build 256-token context windows
	•	Create (x, y) pairs for next-token prediction:
	•	x: tokens [t0..t255]
	•	y: tokens [t1..t256] (shifted)
	4.	Batcher
	•	Packs multiple windows into batches
	•	Produces tensors on the current device (initially CPU; later adaptable)
	5.	Model Core
	•	Decoder-only Transformer (causal)
	•	Embedding + Positional Encoding
	•	N Transformer blocks (self-attn + MLP)
	•	LayerNorm + LM head → logits over vocab for each position
	6.	Loss
	•	Cross-entropy on next-token prediction over all positions (optionally ignore padding if used)
	7.	Training Loop
	•	Forward → loss → backward → optimizer step → logging → checkpointing
	8.	Evaluation & Sampling
	•	Perplexity/avg loss on val set
	•	Text generation (greedy / temperature sampling) for qualitative checks
	9.	Experiment Hooks Layer (Option E)
	•	“Intervention points” you can turn on/off without rewriting the system:
	•	Forward hooks (modify activations, attention patterns, norms, residuals)
	•	Backward hooks (custom gradients via torch.autograd.Function on selected ops)
	•	Update rule hooks (optimizer variants / gradient transforms before stepping)
	•	Aux losses (regularizers, constraints) toggled via config

⸻

Why this is the right choice for your constraints
	•	256 tokens keeps things computationally manageable on CPU while still being long enough to show non-trivial sequence effects.
	•	Decoder-only Transformer gives you a well-understood baseline and easy scaling knobs (depth/width/heads).
	•	Pluggable learning rules lets you innovate where you care (learning dynamics) while keeping the rest stable for clean comparisons.
	•	CPU-first, modular device layer avoids painting yourself into a corner: the same code path can later target MPS/CUDA/remote.

⸻

Concrete module interfaces (so this becomes “spec-like”)

Tokenizer
	•	Input: str
	•	Output: List[int] (token IDs)
	•	Also: decode(List[int]) -> str

SequenceBuilder
	•	Input: List[int]
	•	Output: iterable of (x: List[int], y: List[int]) where len(x)=len(y)=256

Batcher
	•	Input: iterable of (x, y)
	•	Output:
	•	x: Tensor[int64] shape [B, 256]
	•	y: Tensor[int64] shape [B, 256]
	•	meta (optional): offsets, source doc id, etc.

Model
	•	Input: x: Tensor[int64] [B, 256]
	•	Output: logits: Tensor[float] [B, 256, vocab_size]

Loss
	•	Input: logits, y
	•	Output: loss: Tensor[float] scalar

ExperimentHooks
	•	Config-driven registry, e.g.
	•	hooks.forward = [HookA, HookB]
	•	hooks.backward = [CustomGrad1]
	•	hooks.update = [GradTransformX]

⸻

Decisions recorded from you
	•	Architecture: A + E (Decoder-only Transformer + Pluggable learning rules)
	•	Context length: 256 tokens
	•	Experiment surface: open for now
	•	Target hardware (v1): Mac CPU with integrated memory, but keep device backend modular for future

⸻

Open questions to keep in the outline
	1.	Tokenizer policy: ASCII exactly which set? (printable only vs include newline/tab; what about unknown chars?)
	2.	Batch size strategy on CPU: fixed batch size vs dynamic based on throughput/memory
	3.	Baseline model size config: tiny but meaningful defaults (layers, heads, d_model, d_ff)
	4.	Evaluation protocol: what “good” means for v1 (loss targets, qualitative sampling cadence)
	5.	Hook safety: rules for ensuring hooks don’t silently invalidate comparisons (e.g., run IDs + config hashes)

⸻

If you want the next step to be maximally useful: I can turn this into a Phase 1 build plan (ordered milestones) plus a single config schema (YAML/JSON-ish) that controls model size, data, training, and hooks—so you can scale and experiment just by editing config.

---

## Multi-Model Architecture

The system supports multiple model architectures through an adapter pattern, enabling importing pretrained models from HuggingFace and fine-tuning them.

### Architecture Adapter System

- **BaseAdapter**: Common interface for all architectures (`src/model/adapters/base.py`)
  - Defines `forward()`, `get_config()`, `save_checkpoint()`, `load_checkpoint()`
  - Provides unified API for training loop
- **CustomTransformerAdapter**: Wraps the original Transformer architecture
  - Used when `model_name` is `null` in config (backward compatible)
- **QwenAdapter**: Supports Qwen family models from HuggingFace
  - Handles Qwen-specific layer naming and tokenizer integration
- **Extensible**: New architectures can be added by implementing BaseAdapter

### Model Registry

- **Location**: `models/registry.json`
- **Purpose**: Tracks available models and their metadata
- **Dual Identifiers**: 
  - `model_name`: User-friendly identifier for configs (e.g., "qwen-0.5b-base")
  - `model_id`: Original identifier from source (e.g., "Qwen/Qwen-0.5B")
- **Metadata**: Architecture type, source, fine-tuning lineage, timestamps, license

### Model Loading Flow

1. Config specifies `model_name` (or `null` for custom Transformer)
2. If `model_name` is set, registry resolves to model entry
3. Adapter is selected based on `architecture_type` field
4. Model weights and config loaded from `models/{model_name}/` directory
5. Tokenizer selected:
   - Native tokenizer for imported models (e.g., Qwen's tokenizer)
   - Character-level tokenizer for custom Transformer

### Checkpointing with Model Metadata

Checkpoints include:
- **Model metadata**: `model_name`, `model_id`, `model_source`
- **Fine-tuning lineage**: `fine_tuned_from` (immediate parent only)
- **Standard checkpoint data**: weights, optimizer state, config, step counter

This enables:
- Resuming training with correct model architecture
- Tracking fine-tuning chains (model provenance)
- Understanding model relationships

### Fine-Tuning Workflow

1. Import base model: `python main.py import-model Qwen/Qwen-0.5B`
2. Create config with `model_name: "qwen-0.5b-base"`
3. Train: Checkpoints automatically track `fine_tuned_from`
4. Resume: System loads correct model based on checkpoint metadata