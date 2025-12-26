# Design: Quantization Support

## Context
This project is an experimental SLM built from first principles with emphasis on transparency and inspectability. The codebase currently uses full-precision (FP32) models. Adding quantization support will enable:
- Reduced model size (4x reduction for INT8, 8x for INT4)
- Faster inference on CPU
- Memory-efficient deployment
- Fine-tuning of quantized models for task adaptation

The implementation must maintain the project's principles: transparency, inspectability, and modularity.

## Goals / Non-Goals

### Goals
- Support post-training quantization (PTQ) for converting trained FP32 models to INT8/INT4
- Support quantization-aware training (QAT) for training models with quantization simulation
- Enable fine-tuning of quantized models (continuing training after quantization)
- Preserve checkpoint compatibility (quantized checkpoints clearly marked)
- Maintain transparency (quantization operations are inspectable)
- Support both static and dynamic quantization modes

### Non-Goals
- Custom quantization algorithms (use PyTorch's built-in quantization)
- Per-channel quantization initially (start with per-tensor)
- Quantization of embeddings initially (focus on linear layers first)
- Automatic quantization selection (user must explicitly configure)

## Decisions

### Decision: Use PyTorch's Native Quantization APIs
**What**: Leverage `torch.ao.quantization` (formerly `torch.quantization`) for quantization infrastructure.

**Why**: 
- Well-tested and maintained by PyTorch team
- Supports both static and dynamic quantization
- Provides QAT and PTQ workflows
- Maintains compatibility with PyTorch ecosystem
- Reduces implementation complexity while maintaining transparency

**Alternatives considered**:
- Custom quantization implementation: Too complex, reinventing the wheel
- Third-party libraries (e.g., TensorRT): Adds external dependency, less transparent

### Decision: Support INT8 and INT4 Precision
**What**: Implement both INT8 (8-bit) and INT4 (4-bit) quantization modes.

**Why**:
- INT8: Standard quantization, good quality/size trade-off, widely supported
- INT4: Maximum compression, useful for deployment scenarios
- Both are common in production systems

**Alternatives considered**:
- INT8 only: Less flexible, misses INT4 benefits
- FP16/BF16 only: Less compression, different use case (mixed precision training)

### Decision: Per-Tensor Quantization Initially
**What**: Start with per-tensor quantization (single scale/zero-point per tensor).

**Why**:
- Simpler to implement and understand
- Sufficient for initial use cases
- Can extend to per-channel later if needed

**Alternatives considered**:
- Per-channel quantization: More complex, better quality but harder to debug

### Decision: Quantize Linear Layers Only Initially
**What**: Focus quantization on `nn.Linear` layers (attention projections, MLP layers, LM head).

**Why**:
- Linear layers are the largest memory consumers
- Embeddings are small and quantization benefits are minimal
- LayerNorm and attention ops are harder to quantize effectively
- Can extend to other layers later

**Alternatives considered**:
- Quantize all layers: More complex, diminishing returns
- Quantize only MLP: Misses attention layer benefits

### Decision: Separate Quantization Module
**What**: Create `src/quantization/` module with separate files for PTQ, QAT, and core quantization logic.

**Why**:
- Clear separation of concerns
- Maintains modularity
- Easy to test independently
- Follows project's modular architecture pattern

**Alternatives considered**:
- Integrate into training module: Less modular, harder to reuse
- Single quantization file: Too large, harder to navigate

### Decision: Configuration-Driven Quantization
**What**: Add quantization settings to `TrainingConfig` with clear defaults (quantization disabled by default).

**Why**:
- Consistent with project's config-driven approach
- Easy to enable/disable via YAML configs
- Maintains backward compatibility (default = no quantization)

**Configuration fields**:
- `quantization_mode`: None | "ptq" | "qat" | "none" (default: None)
- `quantization_bits`: 8 | 4 (default: 8)
- `quantization_type`: "static" | "dynamic" (default: "static")
- `enable_quantized_finetuning`: bool (default: False)

### Decision: Checkpoint Format Extension
**What**: Extend checkpoint format to include quantization metadata and quantized state_dict.

**Why**:
- Need to distinguish quantized vs full-precision checkpoints
- Must store quantization parameters (scales, zero-points)
- Maintains backward compatibility (old checkpoints still loadable)

**Checkpoint changes**:
- Add `quantization_metadata.json` to checkpoint directory
- Store quantized model state_dict separately or mark in metadata
- Version checkpoint format to handle migration

### Decision: Fine-Tuning Support for Quantized Models
**What**: Allow continuing training (fine-tuning) of quantized models.

**Why**:
- Enables task adaptation without dequantizing
- Maintains quantization benefits during fine-tuning
- Important for practical deployment scenarios

**Implementation approach**:
- Use QAT mode for fine-tuning (simulates quantization during training)
- Store quantized weights but compute gradients in higher precision
- Re-quantize after optimizer step

## Risks / Trade-offs

### Risk: Numerical Instability
**Mitigation**: 
- Use PyTorch's well-tested quantization APIs
- Provide validation tests comparing quantized vs full-precision outputs
- Allow users to disable quantization if issues occur

### Risk: Checkpoint Compatibility
**Mitigation**:
- Version checkpoint format
- Clear error messages for incompatible checkpoints
- Migration utilities if needed

### Risk: Performance Overhead
**Mitigation**:
- Quantization should improve inference speed, not training speed
- QAT adds overhead but is optional
- Users can choose PTQ-only workflow

### Risk: Complexity Increase
**Mitigation**:
- Keep quantization optional (default disabled)
- Clear documentation and examples
- Modular design allows ignoring quantization if not needed

## Migration Plan

### Phase 1: Core Quantization Infrastructure
1. Add quantization module with basic PTQ support
2. Add quantization config fields (disabled by default)
3. Tests for PTQ conversion

### Phase 2: Checkpoint Integration
1. Extend checkpoint format for quantized models
2. Update save/load functions
3. Tests for quantized checkpoint round-trip

### Phase 3: QAT Support
1. Add QAT utilities
2. Integrate QAT into training loop
3. Tests for QAT training

### Phase 4: Fine-Tuning Support
1. Add quantized model fine-tuning capability
2. Tests for fine-tuning quantized models
3. Documentation and examples

### Rollback Plan
- Quantization is opt-in (default disabled)
- Old checkpoints remain compatible
- Can disable quantization via config if issues arise

## Open Questions
- Should we support per-channel quantization from the start? (Defer: start with per-tensor)
- Should quantization be applied to embeddings? (Defer: focus on linear layers first)
- Should we support custom quantization schemes? (Defer: use PyTorch defaults initially)
- How to handle quantization of LayerNorm? (Defer: focus on linear layers first)

