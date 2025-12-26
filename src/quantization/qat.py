"""Quantization-aware training (QAT) utilities.

This module provides utilities for training models with quantization simulation,
allowing models to learn quantization-aware representations.
"""

import warnings
import torch
import torch.nn as nn

# Suppress deprecation warnings for torch.ao.quantization
# These APIs are deprecated but still functional until PyTorch 2.10
# TODO: Migrate to torchao's new quantize_ API in the future
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*torch.ao.quantization.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.ao.quantization")
warnings.filterwarnings("ignore", message=".*reduce_range.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Please use quant_min and quant_max.*", category=UserWarning)

# Try torchao first (new API), fall back to torch.ao.quantization for compatibility
try:
    from torchao.quantization import (
        get_default_qat_qconfig,
        QConfigMapping,
        prepare,
        convert,
    )
    from torchao.quantization import FakeQuantize
except ImportError:
    # Fallback to torch.ao.quantization if torchao is not available
    from torch.ao.quantization import (
        get_default_qat_qconfig,
        QConfigMapping,
        prepare,
        convert,
    )
    from torch.ao.quantization import FakeQuantize

from src.quantization.quantizer import Quantizer


def prepare_model_for_qat(
    model: nn.Module,
    quantization_bits: int = 8
) -> nn.Module:
    """Prepare a model for quantization-aware training (QAT).
    
    This function prepares the model for QAT by adding fake quantization operations
    that simulate quantization during training. The model will learn to be robust
    to quantization effects.
    
    Note: QAT works on all platforms including macOS, as it uses fake quantization
    (simulation) rather than actual quantization engines. However, converting a QAT
    model to a quantized model (via convert_qat_model) requires quantization engines
    and may not work on macOS.
    
    Args:
        model: Model to prepare for QAT.
        quantization_bits: Number of bits for quantization (8 or 4, default: 8).
            Note: INT4 QAT requires custom qconfig, currently using INT8.
    
    Returns:
        Model prepared for QAT (with fake quantization operations).
    
    Example:
        >>> model = Transformer(vocab_size=1000, ...)
        >>> qat_model = prepare_model_for_qat(model, quantization_bits=8)
        >>> # Train qat_model normally - quantization is simulated during forward pass
        >>> # After training, convert to quantized model (may not work on macOS):
        >>> quantized_model = convert_qat_model(qat_model)
    """
    if quantization_bits not in (4, 8):
        raise ValueError(f"quantization_bits must be 8 or 4, got {quantization_bits}")
    
    # Get QAT qconfig
    # For INT4, we'd need custom qconfig, but PyTorch's default is INT8
    if quantization_bits == 4:
        # INT4 QAT requires custom implementation
        # For now, use INT8 as fallback
        qconfig = get_default_qat_qconfig("fbgemm")
    else:
        qconfig = get_default_qat_qconfig("fbgemm")
    
    # Assign qconfig directly to Linear modules
    # QAT qconfigs include fake quantization, so using prepare() with QAT qconfigs
    # will add fake quantization modules (same as prepare_qat)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.qconfig = qconfig
    
    # Use prepare() with QAT qconfigs instead of prepare_qat()
    # This works because QAT qconfigs already include fake quantization observers
    # prepare_qat() has a bug where it passes QConfigMapping to convert() which expects a dict
    qconfig_mapping = QConfigMapping()
    qat_model = prepare(model, qconfig_mapping)
    
    return qat_model


def convert_qat_model(qat_model: nn.Module) -> nn.Module:
    """Convert a QAT-trained model to quantized format.
    
    After training with QAT, this function converts the model to actual quantized
    format for efficient inference.
    
    Note: This function requires PyTorch quantization engines and may not work
    on macOS. On platforms without quantization engines, the model will remain
    in fake quantization mode (FP32 with quantization simulation).
    
    Args:
        qat_model: Model trained with QAT.
    
    Returns:
        Quantized model ready for inference (or fake-quantized model on macOS).
    
    Raises:
        RuntimeError: If quantization engines are not available and conversion fails.
    
    Example:
        >>> # After QAT training
        >>> quantized_model = convert_qat_model(qat_model)
    """
    from src.quantization.quantizer import _check_quantization_engine_available
    
    # Check if engines are available
    if not _check_quantization_engine_available():
        import warnings
        warnings.warn(
            "Quantization engines not available (common on macOS). "
            "Cannot convert QAT model to quantized format. "
            "Model will remain in fake quantization mode (FP32 with quantization simulation).",
            UserWarning
        )
        # Return the QAT model as-is (it's already in fake quantization mode)
        return qat_model
    
    # Convert QAT model to quantized model
    try:
        quantized_model = convert(qat_model)
    except RuntimeError as e:
        if "NoQEngine" in str(e) or "quantized" in str(e).lower():
            import warnings
            warnings.warn(
                "QAT conversion failed: quantization engines not available. "
                "Returning model in fake quantization mode.",
                UserWarning
            )
            return qat_model
        raise
    
    return quantized_model


def is_qat_model(model: nn.Module) -> bool:
    """Check if a model is prepared for QAT.
    
    Args:
        model: Model to check.
    
    Returns:
        True if model is prepared for QAT, False otherwise.
    """
    # Check for fake quantization modules
    for module in model.modules():
        if hasattr(module, 'activation_post_process'):
            # This is a QAT module
            return True
        if isinstance(module, FakeQuantize):
            return True
    
    return False

