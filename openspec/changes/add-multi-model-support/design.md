# Design: Multi-Model Support with HuggingFace Integration

## Context
The current system is designed around a single custom Transformer architecture. To support importing and using pretrained models from HuggingFace (specifically the Qwen family initially), we need to:
1. Download and convert models from HuggingFace format to our local format
2. Support different model architectures beyond the custom Transformer
3. Track model metadata (source, architecture, fine-tuning history)
4. Select models at runtime via configuration
5. Maintain a registry of available models

The system must support both training (fine-tuning) and inference for imported models, while still supporting the original custom Transformer architecture.

## Goals / Non-Goals

### Goals
- Download models from HuggingFace using CLI tool
- Support Qwen family of models initially
- Track model source, architecture type, and fine-tuning lineage
- Select models via config file for both training and inference
- Extensible architecture adapter system for adding more model families
- Model registry to track available local models
- Metadata persistence in checkpoints

### Non-Goals
- Backward compatibility with existing checkpoints (acceptable breaking change)
- Parallel/batch fine-tuning of multiple models (separate runs are sufficient)
- Support for non-HuggingFace model sources in initial implementation
- Automatic model recommendation or selection
- Model merging or ensemble techniques
- Support for architectures beyond Qwen family in initial implementation (but design should be extensible)

## Decisions

### Decision 1: Two-Phase Model Loading (CLI Import → Config Selection)
**Rationale**: Separates the one-time download/conversion step from the runtime model selection step.
- Phase 1: CLI tool downloads from HuggingFace and converts to local format (stored in `models/` directory)
  - Command format: `python main.py import-model <huggingface-url-or-id>`
  - Example: `python main.py import-model Qwen/Qwen-0.5B`
- Phase 2: Config file specifies which local model to load for training/inference
  - Config field: `model_name: "qwen-0.5b-base"`
- **Why**: Avoids repeated downloads, allows offline operation, enables conversion/validation before use

**Alternatives considered**:
- Direct HuggingFace loading at runtime: Would require network access for every run, slower startup
- Config-only approach with automatic download: Less control, harder to debug conversion issues

### Decision 2: Architecture Adapter Pattern
**Rationale**: Use adapter pattern to normalize different architectures to a common interface.
- Each architecture family has an adapter (e.g., `QwenAdapter`, `CustomTransformerAdapter`)
- Adapters implement common interface: `forward()`, `get_config()`, `save_checkpoint()`, `load_checkpoint()`
- Adapters handle architecture-specific details (layer naming, activation functions, etc.)
- **Why**: Allows adding new architectures without modifying core training loop

**Alternatives considered**:
- Direct architecture switching in model core: Would bloat model-core with architecture-specific logic
- Separate training pipelines per architecture: Would duplicate training code, hard to maintain

### Decision 3: Model Registry as JSON Manifest
**Rationale**: Simple JSON file tracking available models and their metadata.
- Registry stored at `models/registry.json`
- Each entry contains: model_name, model_id, source (huggingface repo or "custom"), architecture_type, local_path, metadata (created_at, fine-tuned_from, etc.)
- CLI tool updates registry when importing models
- Config references models by model_name
- **Why**: Simple, human-readable, easy to inspect and edit, no database dependency

**Alternatives considered**:
- SQLite database: Overkill for current scale, adds dependency
- Filename-based convention only: Loses metadata, harder to track fine-tuning lineage

### Decision 4: Model Storage Structure
**Rationale**: Organize models in `models/` directory with clear naming convention.
```
models/
├── registry.json                          # Model registry manifest
├── custom-transformer/                    # Custom architecture
│   ├── config.json
│   └── weights.pt
├── qwen-0.5b-base/                        # Imported base model
│   ├── config.json
│   ├── weights.pt
│   └── metadata.json
├── qwen-0.5b-finetuned-001/              # Fine-tuned variant
│   ├── config.json
│   ├── weights.pt
│   └── metadata.json
└── ...
```
- Each model has its own directory with consistent structure
- Metadata file tracks source, fine-tuning history, etc.
- **Why**: Clear organization, easy to manage, supports model versioning

**Alternatives considered**:
- Flat directory with prefixed filenames: Harder to manage multiple files per model
- Nested by architecture: Harder to navigate when fine-tuning creates variations

### Decision 5: Config-Based Model Selection
**Rationale**: Add `model_name` field to TrainingConfig to specify which model to use.
```yaml
# Example config
model_name: "qwen-0.5b-base"  # References entry in models/registry.json by model_name
# OR
model_name: null  # Use default custom architecture
```
- If `model_name` is null/missing, use custom Transformer with architecture params from config
- If `model_name` is specified, load that model from registry (architecture params from model's config)
- Registry entries contain both `model_name` (user-friendly reference) and `model_id` (original HuggingFace ID)
- **Why**: Clean separation of concerns, consistent with existing config system, human-readable

**Alternatives considered**:
- Command-line only: Would require passing model path/id every time, less reproducible
- Automatic detection: Too magical, harder to debug
- Use HuggingFace model_id directly: Less flexible, harder to manage local fine-tuned variants

### Decision 6: Architecture-Specific vs Shared Hyperparameters
**Rationale**: Some hyperparameters are architecture-specific, others are universal.
- **Shared** (in TrainingConfig): learning_rate, batch_size, max_steps, etc.
- **Architecture-specific** (in model's config.json): n_layers, d_model, n_heads, etc.
- When loading a model, architecture params come from model's config, not TrainingConfig
- When using custom Transformer, architecture params come from TrainingConfig (existing behavior)
- **Why**: Different architectures have different valid parameter ranges and names

**Alternatives considered**:
- All params in TrainingConfig: Would require complex validation per architecture
- All params in model config: Would make command-line overrides harder for custom architecture

### Decision 7: HuggingFace Dependencies
**Rationale**: Use official HuggingFace libraries for model download and loading.
- `transformers`: Model loading and architecture definitions
- `huggingface_hub`: Model downloading and authentication
- `safetensors`: Safe model weight loading (preferred over pickle)
- **Why**: Battle-tested, handles authentication, caching, format conversions

**Alternatives considered**:
- Manual implementation: Reinventing the wheel, error-prone, doesn't handle edge cases
- Requests + manual parsing: Would miss important details, authentication complexity

### Decision 8: Qwen Adapter Implementation
**Rationale**: Start with Qwen family support, design for extensibility.
- QwenAdapter wraps HuggingFace Qwen models
- Handles conversion between HuggingFace format and our checkpoint format
- Maps Qwen-specific layer names to our standardized naming
- Handles tokenizer integration (Qwen uses different tokenizer than our char-level)
- **Why**: Qwen is the initial requirement, but design allows adding more adapters

## Risks / Trade-offs

### Risk 1: Model Size and Memory
- **Risk**: HuggingFace models can be large (multi-GB), may exceed local hardware capacity
- **Mitigation**: 
  - Start with smaller Qwen models (0.5B, 1.8B variants)
  - Document memory requirements in model import CLI
  - Add model size validation before import
  - Support quantization for larger models (leverage existing quantization support)

### Risk 2: Architecture Compatibility
- **Risk**: Some model architectures may not be compatible with our training loop or hooks system
- **Mitigation**:
  - Start with Qwen family only (well-understood architecture)
  - Document adapter requirements for adding new architectures
  - Test thoroughly with integration tests
  - Design adapters to handle architecture-specific edge cases

### Risk 3: Breaking Changes to Existing Checkpoints
- **Risk**: Existing checkpoints won't work with new system
- **Mitigation**: Acceptable per requirements; document in migration guide

### Risk 4: Tokenizer Incompatibility
- **Risk**: HuggingFace models use different tokenizers than our char-level tokenizer
- **Mitigation**:
  - Import and store model's tokenizer alongside weights
  - Adapter handles tokenizer selection
  - Config specifies whether to use model's tokenizer or override
  - Document tokenizer considerations

### Risk 5: License and Attribution
- **Risk**: Imported models have various licenses that must be respected
- **Mitigation**:
  - Model metadata includes license information from HuggingFace
  - CLI displays license during import and requires acknowledgment
  - Registry tracks license for each model
  - Documentation includes attribution requirements

## Migration Plan

### Phase 1: Foundation (Model Registry + Import CLI)
1. Create model registry system (`src/model/registry.py`)
2. Implement HuggingFace import CLI (`scripts/import_model.py` or `main.py import` subcommand)
3. Test with downloading Qwen models
4. Update dependencies (`pyproject.toml`)

### Phase 2: Architecture Adapters
1. Design and implement adapter interface (`src/model/adapters/base.py`)
2. Implement custom Transformer adapter (wrap existing implementation)
3. Implement Qwen adapter (`src/model/adapters/qwen.py`)
4. Test adapter switching

### Phase 3: Integration
1. Update TrainingConfig with `model_id` field
2. Modify main.py to load models via registry
3. Update checkpointing to save model metadata
4. Update trainer to work with different architectures
5. Integration testing with both custom and Qwen models

### Phase 4: Documentation and Polish
1. Document model import CLI usage
2. Document adding new architecture adapters
3. Create example configs for different models
4. Add model management utilities (list, delete, info)

### Rollback
If critical issues arise:
1. Remove `model_id` from configs (fall back to custom Transformer)
2. Keep imported models but don't load them
3. Revert to previous checkpoint format for custom architecture

## Resolved Questions

1. **Tokenizer handling**: Use model's native tokenizer by default, document how to override if needed
   - **Decision**: Models will use their native tokenizers (e.g., Qwen's tokenizer for Qwen models)
   - Override mechanism available in config for special cases
   - Custom Transformer continues using char-level tokenizer

2. **Model naming convention**: CLI suggests name like `{base-model}-finetuned-{timestamp}`, user can override
   - **Decision**: Default naming follows pattern for clarity and traceability
   - Users can specify custom names with `--name` flag
   - Names must be unique in registry

3. **Checkpoint format**: Unified format with architecture field, adapters handle loading
   - **Decision**: Single checkpoint format for all architectures
   - Architecture field in metadata determines which adapter to use
   - Adapters handle architecture-specific loading details

4. **CLI location**: Integrate into main.py as `python main.py import-model URL` for consistency
   - **Decision**: Main.py supports subcommands for model management
   - Cleaner interface, consistent with typical ML tool patterns
   - Format: `python main.py import-model <url>`

5. **Model updates**: CLI can re-import with version suffix (e.g., `qwen-0.5b-v2`), registry tracks version
   - **Decision**: Re-importing creates new entry with version suffix
   - Registry tracks all versions separately
   - Enables A/B comparison of model versions

6. **Fine-tuning lineage**: Track immediate parent only initially, can extend if needed
   - **Decision**: Single-level parent tracking (`fine_tuned_from` field)
   - Can be extended to full chain traversal if needed
   - Sufficient for most use cases, keeps implementation simple

