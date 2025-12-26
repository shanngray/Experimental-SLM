# Change: Add Quantization Support

## Why
Quantization is essential for reducing model size and enabling efficient inference on resource-constrained devices. This change adds support for post-training quantization (PTQ) and quantization-aware training (QAT), allowing models to be quantized to INT8 or INT4 precision. Additionally, it enables fine-tuning of quantized models, which is crucial for adapting quantized models to specific tasks or datasets without losing the memory and speed benefits of quantization. This capability is important for making the experimental SLM more practical for deployment scenarios while maintaining the project's emphasis on transparency and inspectability.

## What Changes
- Add quantization capability with support for INT8 and INT4 precision
- Implement post-training quantization (PTQ) for converting trained models to quantized format
- Implement quantization-aware training (QAT) for training models with quantization simulation
- Add fine-tuning support for quantized models (allowing continued training of quantized models)
- Extend checkpointing to save/load quantized model states
- Add quantization configuration options to TrainingConfig
- Support mixed precision training (FP16/BF16) as a stepping stone to quantization
- Add quantization utilities for converting between quantized and full-precision formats
- Comprehensive test coverage for quantization operations

## Impact
- Affected specs: 
  - New `quantization` capability specification
  - Modified `checkpointing` capability (quantized model save/load)
  - Modified `training` capability (quantization-aware training, fine-tuning quantized models)
  - Modified `model-core` capability (quantized operations)
- Affected code:
  - `src/quantization/` (new directory)
    - `__init__.py` (new)
    - `quantizer.py` (new) - Core quantization logic
    - `qat.py` (new) - Quantization-aware training utilities
    - `ptq.py` (new) - Post-training quantization utilities
  - `src/config.py` - Add quantization configuration fields
  - `src/training/checkpoint.py` - Support quantized model save/load
  - `src/training/trainer.py` - Support QAT and quantized model fine-tuning
  - `src/model/transformer.py` - Optional quantized forward pass
  - `tests/test_quantization.py` (new)
  - `tests/test_checkpoint.py` - Add quantized checkpoint tests
  - `tests/test_trainer.py` - Add QAT and quantized fine-tuning tests
- Dependencies:
  - PyTorch quantization APIs (torch.quantization, torch.ao.quantization)
  - Existing training, checkpointing, and model infrastructure
- Future impact:
  - Enables deployment of smaller models
  - Foundation for further optimization (pruning, distillation)
  - Supports experimentation with quantization-aware learning dynamics

