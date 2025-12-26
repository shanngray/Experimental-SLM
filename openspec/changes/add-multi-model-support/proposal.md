# Change: Add Multi-Model Support with HuggingFace Integration

## Why
The system currently only supports training a custom Transformer architecture from scratch. Users need the ability to import pretrained models from HuggingFace (starting with the Qwen family), fine-tune them, and select which model to use at runtime for both training and inference. This enables leveraging existing pretrained models, comparing different model variants, and building on state-of-the-art architectures rather than always starting from scratch.

## What Changes
- Add CLI tool for downloading and converting models from HuggingFace to local format
  - Command format: `python main.py import-model <huggingface-url-or-id>`
- Create model registry system to track available models with dual identifiers:
  - `model_name`: User-friendly name for referencing in configs (e.g., "qwen-0.5b-base")
  - `model_id`: Original identifier like HuggingFace repo ID (e.g., "Qwen/Qwen-0.5B")
- Add architecture adapter system to support different model architectures (Qwen family initially, extensible for future architectures)
- Uplift model-core to support multiple architectures beyond the custom Transformer
- Extend checkpointing to save/load model metadata and fine-tuning lineage (immediate parent tracking)
- Extend main-entry to support model selection via config file using `model_name` field
- Add comprehensive model metadata tracking (source repository, architecture type, fine-tuning history, license, etc.)
- Tokenizer handling: Use model's native tokenizer by default with optional override
- Maintain ability to use the existing custom Transformer architecture (when `model_name` is None)
- **BREAKING**: Existing checkpoints will not be compatible with the new system (acceptable per requirements)

## Impact
- Affected specs: model-import (NEW), model-registry (NEW), model-core (MODIFIED), checkpointing (MODIFIED), main-entry (MODIFIED)
- Affected code:
  - New: `src/model/registry.py` - Model registry with model_name/model_id tracking
  - New: `src/model/adapters/` - Architecture adapters (BaseAdapter, CustomTransformerAdapter, QwenAdapter)
  - Modified: `src/model/transformer.py` - Wrapped by CustomTransformerAdapter
  - Modified: `src/training/checkpoint.py` - Add model metadata (model_name, model_id, fine_tuned_from, etc.)
  - Modified: `src/config.py` - Add model_name field for model selection
  - Modified: `main.py` - Add subcommands (import-model, list-models, model-info, delete-model) and model loading logic
- New dependencies: `transformers`, `safetensors`, `huggingface_hub`
- Key concepts:
  - `model_name`: User-friendly identifier used in configs (e.g., "qwen-0.5b-base", "my-custom-finetune")
  - `model_id`: Original identifier from source (e.g., "Qwen/Qwen-0.5B" for HuggingFace models)
  - Registry tracks both, config references by model_name

