"""Quantization support for transformer models.

This module provides functionality for:
- Post-training quantization (PTQ): Convert trained FP32 models to INT8/INT4
- Quantization-aware training (QAT): Train models with quantization simulation
- Quantized model fine-tuning: Continue training quantized models
- Quantization utilities: Convert between quantized and FP32 formats
"""

from src.quantization.quantizer import (
    Quantizer,
    quantize_model_ptq,
    dequantize_model,
    get_quantization_info,
    is_model_quantized,
    prepare_model_for_quantization,
)
from src.quantization.qat import (
    prepare_model_for_qat,
    convert_qat_model,
    is_qat_model,
)

__all__ = [
    "Quantizer",
    "quantize_model_ptq",
    "dequantize_model",
    "get_quantization_info",
    "is_model_quantized",
    "prepare_model_for_quantization",
    "prepare_model_for_qat",
    "convert_qat_model",
    "is_qat_model",
]

