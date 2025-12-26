"""Post-training quantization (PTQ) utilities.

This module provides utilities for converting trained FP32 models to quantized
format after training is complete.
"""

from typing import Optional, List
import torch
import torch.nn as nn

from src.quantization.quantizer import quantize_model_ptq as _quantize_model_ptq


def quantize_model_ptq(
    model: nn.Module,
    quantization_bits: int = 8,
    quantization_type: str = "static",
    calibration_data: Optional[List[torch.Tensor]] = None
) -> nn.Module:
    """Convert a trained FP32 model to quantized format (post-training quantization).
    
    This function performs post-training quantization, converting a fully trained
    FP32 model to INT8 or INT4 quantized format. For static quantization, calibration
    data is required to compute quantization parameters.
    
    Args:
        model: Trained FP32 model to quantize.
        quantization_bits: Number of bits for quantization (8 or 4, default: 8).
        quantization_type: Type of quantization ("static" or "dynamic", default: "static").
        calibration_data: Optional list of input tensors for calibration (required for static).
            Each element should be a tensor of shape [batch_size, seq_len] with token IDs.
    
    Returns:
        Quantized model ready for inference.
    
    Raises:
        ValueError: If calibration_data is None for static quantization.
    
    Example:
        >>> # Static quantization with calibration data
        >>> calibration_data = [torch.randint(0, vocab_size, (batch_size, seq_len)) 
        ...                     for _ in range(100)]
        >>> quantized_model = quantize_model_ptq(
        ...     model, quantization_bits=8, quantization_type="static",
        ...     calibration_data=calibration_data
        ... )
        
        >>> # Dynamic quantization (no calibration needed)
        >>> quantized_model = quantize_model_ptq(
        ...     model, quantization_bits=8, quantization_type="dynamic"
        ... )
    """
    return _quantize_model_ptq(
        model=model,
        quantization_bits=quantization_bits,
        quantization_type=quantization_type,
        calibration_data=calibration_data
    )

