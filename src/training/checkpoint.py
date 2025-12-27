"""Checkpointing functionality for saving and loading training state.

This module provides functions to save and load complete training state,
including model weights, optimizer state, training configuration, vocabulary,
step counter, and random number generator states. This enables resuming 
training from any saved checkpoint with full reproducibility.
"""

import json
import random
from pathlib import Path
from typing import Optional

import torch

from src.config import TrainingConfig
from src.tokenizer import Tokenizer
from src.quantization import is_model_quantized, get_quantization_info


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    tokenizer: Tokenizer,
    step: int,
    checkpoint_dir: str | Path = "checkpoints",
    checkpoint_name: Optional[str] = None,
    model_name: Optional[str] = None,
    model_id: Optional[str] = None,
    model_source: Optional[str] = None,
    fine_tuned_from: Optional[str] = None
) -> Path:
    """Save complete training state to disk.
    
    Saves model state_dict, optimizer state_dict, training configuration,
    vocabulary, step counter, and RNG states to a checkpoint directory. 
    Uses PyTorch format (torch.save) for binary data and JSON format for metadata.
    This ensures full reproducibility when resuming training.
    
    Args:
        model: The transformer model to save.
        optimizer: The optimizer to save state for.
        config: Training configuration to save.
        tokenizer: Tokenizer containing vocabulary to save.
        step: Current training step counter.
        checkpoint_dir: Directory to save checkpoints in (default: "checkpoints").
        checkpoint_name: Optional name for checkpoint. If None, uses "checkpoint_step_{step}".
        model_name: Optional model name from registry (e.g., "qwen-0.5b-base").
        model_id: Optional original model identifier (e.g., "Qwen/Qwen-0.5B").
        model_source: Optional model source (e.g., "huggingface", "custom", "finetuned").
        fine_tuned_from: Optional model_name of the parent model this was fine-tuned from.
    
    Returns:
        Path to the saved checkpoint directory.
    
    Raises:
        IOError: If checkpoint directory cannot be created or files cannot be written.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if checkpoint_name is None:
        checkpoint_name = f"checkpoint_step_{step}"
    
    checkpoint_path = checkpoint_dir / checkpoint_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Save model state_dict (binary PyTorch format)
    model_path = checkpoint_path / "model.pt"
    torch.save(model.state_dict(), model_path)
    
    # Save optimizer state_dict (binary PyTorch format)
    optimizer_path = checkpoint_path / "optimizer.pt"
    torch.save(optimizer.state_dict(), optimizer_path)
    
    # Save RNG states (binary PyTorch format)
    rng_path = checkpoint_path / "rng.pt"
    rng_state = {
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    # Save CUDA RNG state if CUDA is available and being used
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        rng_state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(rng_state, rng_path)
    
    # Save vocabulary (JSON format)
    vocab_path = checkpoint_path / "vocab.json"
    tokenizer.save_vocab(vocab_path)
    
    # Save metadata (JSON format)
    metadata = {
        "step": step,
        "config": config.to_dict(),
        "checkpoint_version": "1.0",  # Version for backward compatibility
    }
    
    # Add model metadata if provided
    if model_name is not None:
        metadata["model_name"] = model_name
    if model_id is not None:
        metadata["model_id"] = model_id
    if model_source is not None:
        metadata["model_source"] = model_source
    if fine_tuned_from is not None:
        metadata["fine_tuned_from"] = fine_tuned_from
    
    # Check if model is quantized and save quantization metadata
    if is_model_quantized(model):
        quantization_info = get_quantization_info(model)
        metadata["quantization"] = quantization_info
        
        # Save quantization metadata separately
        quantization_metadata_path = checkpoint_path / "quantization_metadata.json"
        with open(quantization_metadata_path, "w", encoding="utf-8") as f:
            json.dump(quantization_info, f, indent=2)
    
    metadata_path = checkpoint_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokenizer: Tokenizer
) -> dict:
    """Load complete training state from disk.
    
    Loads model state_dict, optimizer state_dict, training configuration,
    vocabulary, step counter, and RNG states from a checkpoint directory.
    This restores the exact training state for full reproducibility.
    
    Args:
        checkpoint_path: Path to checkpoint directory.
        model: Model to load state into (will be modified in-place).
        optimizer: Optimizer to load state into (will be modified in-place).
        tokenizer: Tokenizer to load vocabulary into (will be modified in-place).
    
    Returns:
        Dictionary containing:
            - "step": int - Training step counter
            - "config": TrainingConfig - Training configuration
            - "model": torch.nn.Module - Model (same reference as input)
            - "optimizer": torch.optim.Optimizer - Optimizer (same reference as input)
            - "tokenizer": Tokenizer - Tokenizer (same reference as input)
    
    Note:
        This function also restores Python and PyTorch RNG states as a side effect.
    
    Raises:
        FileNotFoundError: If checkpoint files are missing.
        RuntimeError: If checkpoint files are corrupted or invalid.
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_path}"
        )
    
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint path is not a directory: {checkpoint_path}"
        )
    
    # Load metadata first to check for quantization
    metadata_path = checkpoint_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata checkpoint file not found: {metadata_path}"
        )
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load metadata checkpoint from {metadata_path}: {e}"
        ) from e
    
    # Check for quantization metadata
    quantization_metadata = None
    quantization_metadata_path = checkpoint_path / "quantization_metadata.json"
    if quantization_metadata_path.exists():
        try:
            with open(quantization_metadata_path, "r", encoding="utf-8") as f:
                quantization_metadata = json.load(f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load quantization metadata from {quantization_metadata_path}: {e}"
            ) from e
    
    # Load model state_dict
    model_path = checkpoint_path / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint file not found: {model_path}"
        )
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)  # Use strict=False for quantized models
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model checkpoint from {model_path}: {e}"
        ) from e
    
    # Load optimizer state_dict
    optimizer_path = checkpoint_path / "optimizer.pt"
    if not optimizer_path.exists():
        raise FileNotFoundError(
            f"Optimizer checkpoint file not found: {optimizer_path}"
        )
    try:
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
    except Exception as e:
        raise RuntimeError(
            f"Failed to load optimizer checkpoint from {optimizer_path}: {e}"
        ) from e
    
    # Load RNG states
    rng_path = checkpoint_path / "rng.pt"
    if not rng_path.exists():
        raise FileNotFoundError(
            f"RNG state checkpoint file not found: {rng_path}"
        )
    try:
        rng_state = torch.load(rng_path, map_location="cpu")
        # Restore Python RNG state
        if "python_rng_state" in rng_state:
            random.setstate(rng_state["python_rng_state"])
        # Restore PyTorch CPU RNG state
        if "torch_rng_state" in rng_state:
            torch.set_rng_state(rng_state["torch_rng_state"])
        # Restore CUDA RNG state if present
        if "cuda_rng_state" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["cuda_rng_state"])
    except Exception as e:
        raise RuntimeError(
            f"Failed to load RNG state checkpoint from {rng_path}: {e}"
        ) from e
    
    # Load vocabulary
    vocab_path = checkpoint_path / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocabulary checkpoint file not found: {vocab_path}"
        )
    try:
        tokenizer.load_vocab(vocab_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load vocabulary checkpoint from {vocab_path}: {e}"
        ) from e
    
    # Extract step and config (metadata already loaded above)
    step = metadata.get("step")
    if step is None:
        raise RuntimeError(
            f"Step counter not found in metadata: {metadata_path}"
        )
    
    config_dict = metadata.get("config")
    if config_dict is None:
        raise RuntimeError(
            f"Config not found in metadata: {metadata_path}"
        )
    
    config = TrainingConfig.from_dict(config_dict)
    
    result = {
        "step": step,
        "config": config,
        "model": model,
        "optimizer": optimizer,
        "tokenizer": tokenizer,
    }
    
    # Add model metadata if present
    if "model_name" in metadata:
        result["model_name"] = metadata["model_name"]
    if "model_id" in metadata:
        result["model_id"] = metadata["model_id"]
    if "model_source" in metadata:
        result["model_source"] = metadata["model_source"]
    if "fine_tuned_from" in metadata:
        result["fine_tuned_from"] = metadata["fine_tuned_from"]
    
    # Add quantization metadata if present
    if quantization_metadata is not None:
        result["quantization_metadata"] = quantization_metadata
    
    return result


def load_checkpoint_config(
    checkpoint_path: str | Path
) -> TrainingConfig:
    """Load only the training configuration from a checkpoint.
    
    Useful when you need to create an optimizer with the exact same
    configuration that was used to save the checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory.
    
    Returns:
        TrainingConfig loaded from the checkpoint.
    
    Raises:
        FileNotFoundError: If checkpoint files are missing.
        RuntimeError: If checkpoint files are corrupted or invalid.
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_path}"
        )
    
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint path is not a directory: {checkpoint_path}"
        )
    
    # Load metadata
    metadata_path = checkpoint_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata checkpoint file not found: {metadata_path}"
        )
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load metadata checkpoint from {metadata_path}: {e}"
        ) from e
    
    config_dict = metadata.get("config")
    if config_dict is None:
        raise RuntimeError(
            f"Config not found in metadata: {metadata_path}"
        )
    
    return TrainingConfig.from_dict(config_dict)

