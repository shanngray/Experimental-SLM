# Multi-Model Support: Session Breakdown

## Overview
This change is large enough to warrant multiple implementation sessions. This document breaks down the work into logical, testable phases.

## Session 1: Foundation (Model Registry + Adapter Interface)
**Goal**: Build the foundational infrastructure that everything else depends on.

### Specs to Implement
1. **model-registry** (NEW) - Core registry system
2. **model-core** (partial) - BaseAdapter interface + CustomTransformerAdapter wrapper

### Tasks (from tasks.md)
- ✅ **1. Project Setup** (all)
  - Add HuggingFace dependencies
  - Create models/ directory structure
  - Update .gitignore
  
- ✅ **2. Model Registry System** (all)
  - Create `src/model/registry.py`
  - Implement registry.json schema
  - Implement CRUD operations
  - Write tests

- ✅ **3. Architecture Adapter System** (partial: 3.1-3.3, 3.9)
  - Create adapters directory
  - Define BaseAdapter interface
  - Implement CustomTransformerAdapter (wrap existing Transformer)
  - Write adapter tests

### Success Criteria
- [ ] Registry can add/get/list/delete models
- [ ] Registry persists to JSON correctly
- [ ] BaseAdapter interface is well-defined
- [ ] CustomTransformerAdapter wraps existing Transformer without breaking behavior
- [ ] All tests pass
- [ ] Existing training pipeline still works (backward compatible)

### Deliverables
- `src/model/registry.py` - Model registry implementation
- `src/model/adapters/base.py` - BaseAdapter interface
- `src/model/adapters/custom_transformer.py` - CustomTransformerAdapter
- `tests/test_registry.py` - Registry tests
- `tests/test_adapters.py` - Adapter tests (at least BaseAdapter and CustomTransformerAdapter)
- `models/registry.json` - Empty registry file
- Updated `pyproject.toml` - HuggingFace dependencies

### Estimated Complexity
- **Low-Medium**: Mostly infrastructure code, well-defined interfaces
- **Risk**: Low - doesn't touch existing training code yet
- **Time**: 2-4 hours depending on test coverage depth

---

## Session 2: Model Import + Qwen Adapter
**Goal**: Enable importing models from HuggingFace and support Qwen architecture.

### Specs to Implement
1. **model-import** (NEW) - HuggingFace import CLI
2. **model-core** (partial) - QwenAdapter implementation

### Tasks
- ✅ **4. Model Import CLI** (all)
  - Add import-model subcommand to main.py
  - Implement HuggingFace download
  - Implement model conversion
  - License handling
  - Registry integration
  - Tests

- ✅ **3. Architecture Adapter System** (remaining: 3.4-3.8)
  - Create QwenAdapter
  - Implement Qwen loading
  - Implement Qwen forward pass
  - Implement Qwen checkpoint save/load
  - Handle Qwen tokenizer

- ✅ **8. Model Management Utilities** (8.1-8.2)
  - list-models subcommand
  - model-info subcommand

### Success Criteria
- [ ] Can import Qwen model from HuggingFace: `python main.py import-model Qwen/Qwen-0.5B`
- [ ] Imported model appears in registry
- [ ] Can list imported models
- [ ] Can view model info
- [ ] QwenAdapter can load and run forward pass
- [ ] Qwen tokenizer works correctly
- [ ] All tests pass

### Deliverables
- `main.py` - import-model, list-models, model-info subcommands
- `src/model/adapters/qwen.py` - QwenAdapter implementation
- Updated `tests/test_adapters.py` - QwenAdapter tests
- Updated `tests/test_integration.py` - Import CLI tests
- Example: Successfully imported Qwen model in `models/` directory

### Estimated Complexity
- **Medium-High**: HuggingFace integration, architecture-specific code
- **Risk**: Medium - first external model integration, tokenizer differences
- **Time**: 4-6 hours (longer if debugging HuggingFace integration)

---

## Session 3: Configuration + Main Entry Integration
**Goal**: Wire everything together so models can be selected via config and used in training.

### Specs to Implement
1. **main-entry** (MODIFIED) - Model loading and selection logic
2. **model-core** (complete) - Full adapter integration

### Tasks
- ✅ **5. Configuration Updates** (all)
  - Add model_name to TrainingConfig
  - Update config loading/saving
  - Config validation
  - Example configs

- ✅ **7. Main Entry Point Updates** (all)
  - Model loading based on config.model_name
  - Registry integration
  - Adapter selection
  - Custom Transformer fallback
  - Tokenizer selection

- ✅ **8. Model Management Utilities** (remaining: 8.3-8.4)
  - delete-model subcommand
  - validate-model subcommand

### Success Criteria
- [ ] Config with `model_name: "qwen-0.5b-base"` loads Qwen model
- [ ] Config without model_name uses custom Transformer (backward compatible)
- [ ] Training works with both custom Transformer and imported Qwen model
- [ ] Tokenizer selection works correctly
- [ ] All integration tests pass

### Deliverables
- Updated `src/config.py` - model_name field
- Updated `main.py` - Model loading logic
- `configs/qwen-base.yaml` - Example config
- Updated `tests/test_config.py` - Config tests
- Updated `tests/test_integration.py` - Multi-model integration tests

### Estimated Complexity
- **Medium**: Integration work, need to be careful about backward compatibility
- **Risk**: Medium - touching main entry point, need to preserve existing behavior
- **Time**: 3-5 hours

---

## Session 4: Checkpointing + Polish
**Goal**: Complete the feature with checkpointing support and polish.

### Specs to Implement
1. **checkpointing** (MODIFIED) - Model metadata in checkpoints
2. **main-entry** (complete) - Fine-tuning workflow

### Tasks
- ✅ **6. Checkpointing Updates** (all)
  - Extend checkpoint metadata
  - Save/load model metadata
  - Fine-tuning lineage tracking
  - Tests

- ✅ **9. Documentation** (all)
  - README updates
  - Tutorial
  - Architecture docs

- ✅ **10. Integration Testing** (all)
  - End-to-end tests
  - Fine-tuning tests
  - Checkpoint resume tests

- ✅ **11. Polish and Optimization** (as needed)
  - Progress bars
  - Error messages
  - Performance optimizations

### Success Criteria
- [ ] Checkpoints save model_name, model_id, fine_tuned_from
- [ ] Can resume training from checkpoint with correct model
- [ ] Fine-tuning lineage is tracked correctly
- [ ] Documentation is complete
- [ ] All integration tests pass
- [ ] Feature is production-ready

### Deliverables
- Updated `src/training/checkpoint.py` - Metadata support
- Updated `tests/test_checkpoint.py` - Metadata tests
- Updated `README.md` - Multi-model documentation
- `docs/multi-model-tutorial.md` - Tutorial
- Updated `architecture.md` - Multi-model design

### Estimated Complexity
- **Low-Medium**: Mostly extending existing checkpointing, documentation
- **Risk**: Low - checkpointing is well-understood
- **Time**: 3-4 hours

---

## Recommended Session Order

### Option A: Sequential (Recommended)
1. **Session 1** → Foundation
2. **Session 2** → Import + Qwen
3. **Session 3** → Integration
4. **Session 4** → Polish

**Pros**: Each session builds on previous, clear dependencies, testable milestones  
**Cons**: Can't use imported models until Session 3

### Option B: Parallel (Advanced)
- **Session 1** + **Session 2** in parallel (if two developers)
- Then **Session 3** + **Session 4**

**Pros**: Faster overall  
**Cons**: Requires coordination, higher risk

---

## Testing Strategy Per Session

### Session 1
- Unit tests for registry (CRUD operations)
- Unit tests for BaseAdapter interface
- Unit tests for CustomTransformerAdapter
- Integration test: Existing training still works

### Session 2
- Unit tests for import CLI
- Unit tests for QwenAdapter
- Integration test: Import Qwen model successfully
- Integration test: QwenAdapter forward pass works

### Session 3
- Unit tests for config.model_name
- Integration test: Load model from config
- Integration test: Training with imported model
- Integration test: Backward compatibility

### Session 4
- Unit tests for checkpoint metadata
- Integration test: Save/load checkpoint with metadata
- Integration test: Fine-tuning lineage
- End-to-end test: Full workflow

---

## Rollback Strategy

If issues arise in any session:

### Session 1
- No rollback needed (additive only, doesn't touch existing code)

### Session 2
- Can disable import CLI, keep registry
- QwenAdapter can be disabled if broken

### Session 3
- Can revert config changes, keep registry/import
- Main entry changes can be feature-flagged

### Session 4
- Checkpointing changes can be optional
- Can ship without full documentation

---

---

## Session 6: Polish and Optimization
**Goal**: Complete polish features and optimizations for production readiness.

### Specs to Implement
1. **model-import** (MODIFIED) - Enhanced progress, caching, checksums
2. **model-registry** (MODIFIED) - Checksum validation support

### Tasks
- ✅ **11. Polish and Optimization** (all)
  - Add progress bars for model download
  - Improve model caching to avoid re-downloads
  - Optimize model loading performance
  - Add model validation checksums
  - Verify model deletion disk cleanup
  - Improve warnings for large model sizes

### Success Criteria
- [ ] Progress bars show during model download
- [ ] Models are cached and re-used when already downloaded
- [ ] Model loading is optimized
- [ ] Checksums validate model integrity
- [ ] Model deletion properly cleans up disk space
- [ ] Clear warnings for large model sizes

### Deliverables
- Updated `main.py` - Enhanced import-model with progress bars, caching, checksums
- Updated `src/model/registry.py` - Checksum support in metadata
- Updated `src/model/adapters/qwen.py` - Optimized loading

### Estimated Complexity
- **Low-Medium**: Mostly enhancements to existing code
- **Risk**: Low - additive improvements
- **Time**: 2-3 hours

---

## Success Metrics

After all sessions:
- ✅ Can import Qwen models from HuggingFace
- ✅ Can select models via config
- ✅ Can fine-tune imported models
- ✅ Fine-tuning lineage is tracked
- ✅ Checkpoints preserve model metadata
- ✅ Backward compatible with existing workflows
- ✅ Documentation is complete
- ✅ Polish features complete (progress bars, caching, checksums, optimizations)

