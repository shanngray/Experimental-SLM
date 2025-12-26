# Implementation Tasks: Multi-Model Support

## 1. Project Setup
- [ ] 1.1 Add HuggingFace dependencies to pyproject.toml (transformers, huggingface_hub, safetensors)
- [ ] 1.2 Create models/ directory structure
- [ ] 1.3 Update .gitignore to exclude large model files but track registry.json

## 2. Model Registry System
- [ ] 2.1 Create src/model/registry.py with ModelRegistry class
- [ ] 2.2 Implement registry.json schema and loading/saving (with model_name and model_id fields)
- [ ] 2.3 Implement add_model(), get_model(), list_models(), delete_model() methods
- [ ] 2.4 Implement metadata tracking (model_name, model_id, source, architecture, created_at, fine_tuned_from, etc.)
- [ ] 2.5 Write tests for model registry (test_registry.py)

## 3. Architecture Adapter System
- [ ] 3.1 Create src/model/adapters/ directory
- [ ] 3.2 Define BaseAdapter interface in src/model/adapters/base.py
- [ ] 3.3 Implement CustomTransformerAdapter wrapping existing Transformer
- [ ] 3.4 Create src/model/adapters/qwen.py for Qwen adapter
- [ ] 3.5 Implement Qwen model loading from HuggingFace format
- [ ] 3.6 Implement Qwen forward pass integration
- [ ] 3.7 Implement Qwen checkpoint save/load
- [ ] 3.8 Handle Qwen tokenizer integration
- [ ] 3.9 Write tests for adapters (test_adapters.py)

## 4. Model Import CLI
- [ ] 4.1 Add import-model subcommand to main.py (format: `python main.py import-model <url-or-id>`)
- [ ] 4.2 Implement HuggingFace model download using huggingface_hub
- [ ] 4.3 Implement model conversion to local format
- [ ] 4.4 Implement license display and acknowledgment
- [ ] 4.5 Implement automatic registry update on successful import with model_name generation
- [ ] 4.6 Add validation (model size, architecture support, etc.)
- [ ] 4.7 Add progress indicators for download and conversion
- [ ] 4.8 Support --name flag for custom model_name
- [ ] 4.9 Write tests for import CLI

## 5. Configuration Updates
- [ ] 5.1 Add model_name field to TrainingConfig (references registry by model_name)
- [ ] 5.2 Add tokenizer_override field if needed
- [ ] 5.3 Update TrainingConfig.from_dict() to handle model_name
- [ ] 5.4 Update TrainingConfig.to_dict() to serialize model_name
- [ ] 5.5 Update config validation to check model_name references exist in registry
- [ ] 5.6 Create example configs for different models (custom-transformer.yaml, qwen-base.yaml, qwen-finetuned.yaml)
- [ ] 5.7 Update tests for config changes

## 6. Checkpointing Updates
- [ ] 6.1 Extend checkpoint metadata to include model_name, model_id, and model_source
- [ ] 6.2 Extend checkpoint metadata to include fine_tuned_from lineage (immediate parent only)
- [ ] 6.3 Update save_checkpoint() to save model metadata
- [ ] 6.4 Update load_checkpoint() to load model metadata
- [ ] 6.5 Update checkpoint tests for new metadata fields

## 7. Main Entry Point Updates
- [ ] 7.1 Add model loading logic based on config.model_name
- [ ] 7.2 Integrate with model registry to resolve model_name to local path and metadata
- [ ] 7.3 Load appropriate adapter based on architecture type
- [ ] 7.4 Handle custom Transformer fallback when model_name is None
- [ ] 7.5 Update model initialization to use adapter interface
- [ ] 7.6 Implement tokenizer selection (native vs override)
- [ ] 7.7 Update integration tests for multi-model support

## 8. Model Management Utilities
- [ ] 8.1 Add list-models subcommand to display available models
- [ ] 8.2 Add model-info subcommand to show detailed model metadata
- [ ] 8.3 Add delete-model subcommand to remove models from registry
- [ ] 8.4 Add validate-model subcommand to check model integrity

## 9. Documentation
- [ ] 9.1 Document model import CLI in README.md (format: `python main.py import-model <url>`)
- [ ] 9.2 Document model selection via config in README.md (using model_name field)
- [ ] 9.3 Document supported architectures (Qwen family) and how to add more
- [ ] 9.4 Create tutorial for importing and fine-tuning a Qwen model
- [ ] 9.5 Document tokenizer handling (native by default, override mechanism)
- [ ] 9.6 Document license and attribution requirements
- [ ] 9.7 Update architecture.md with multi-model design
- [ ] 9.8 Create models/README.md explaining model directory structure and registry format
- [ ] 9.9 Document model_name vs model_id distinction

## 10. Integration Testing
- [ ] 10.1 Test importing Qwen model from HuggingFace
- [ ] 10.2 Test loading imported model for inference
- [ ] 10.3 Test fine-tuning imported model
- [ ] 10.4 Test saving and loading fine-tuned model
- [ ] 10.5 Test switching between custom Transformer and imported model
- [ ] 10.6 Test checkpoint resume with imported model
- [ ] 10.7 Test model metadata tracking through fine-tuning
- [ ] 10.8 Test error handling (missing model, corrupt files, etc.)

## 11. Polish and Optimization
- [ ] 11.1 Add progress bars for model download
- [ ] 11.2 Add model caching to avoid re-downloads
- [ ] 11.3 Optimize model loading performance
- [ ] 11.4 Add model validation checksums
- [ ] 11.5 Add support for model deletion cleaning up disk space
- [ ] 11.6 Add warnings for large model sizes

## 12. Future Extensibility (Optional for initial implementation)
- [ ] 12.1 Document how to add new architecture adapters
- [ ] 12.2 Create adapter template for common patterns
- [ ] 12.3 Add adapter registration system for cleaner architecture discovery

