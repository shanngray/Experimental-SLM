"""Trainer class for training transformer models.

This module provides the Trainer class that orchestrates the training process,
including forward pass, loss computation, backward pass, and optimizer updates.
"""

import hashlib
import json
import subprocess
import uuid
from pathlib import Path
from typing import Optional

import torch
import torch.optim

from src.config import TrainingConfig
from src.dataloader import DataLoader
from src.hooks.registry import HookRegistry
from src.tokenizer import Tokenizer
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.loss import compute_loss
from src.quantization import (
    prepare_model_for_qat,
    is_model_quantized,
    is_qat_model,
)


def create_optimizer(
    model: torch.nn.Module,
    config: TrainingConfig
) -> torch.optim.AdamW:
    """Create AdamW optimizer with specified hyperparameters.
    
    Configures AdamW optimizer with:
    - learning_rate: from config (default: 3e-4)
    - weight_decay: from config (default: 0.1)
    - betas: (config.beta1, config.beta2) (default: (0.9, 0.95))
    
    Args:
        model: Model whose parameters will be optimized.
        config: Training configuration with optimizer hyperparameters.
    
    Returns:
        Configured AdamW optimizer.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )


class Trainer:
    """Trainer for transformer model training.
    
    Manages the training loop including forward pass, loss computation,
    backward pass, and optimizer updates. Tracks training step counter
    and logs loss values. Supports hooks for experimental modifications.
    
    Attributes:
        model: The transformer model to train.
        optimizer: Optimizer for parameter updates.
        config: Training configuration with hyperparameters.
        step: Current training step counter.
        hook_registry: Registry for managing training hooks.
        run_id: Unique identifier for this training run.
        _run_logged: Whether run metadata has been logged.
        val_dataloader: Optional validation dataloader for evaluation.
        tokenizer: Optional tokenizer for text sampling.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        step: int = 0,
        val_dataloader: Optional[DataLoader] = None,
        tokenizer: Optional[Tokenizer] = None
    ):
        """Initialize the Trainer.
        
        Args:
            model: Transformer model to train (must be a PyTorch nn.Module).
            optimizer: Optimizer for parameter updates (e.g., AdamW).
            config: Training configuration with hyperparameters.
            step: Initial training step counter (default: 0). Used when resuming.
            val_dataloader: Optional validation dataloader for evaluation.
            tokenizer: Optional tokenizer for text sampling.
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.step = step
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        
        # Initialize hook registry from config
        hooks_config = config.get_hooks_config()
        self.hook_registry = HookRegistry(hooks_config)
        
        # Prepare model for QAT if enabled
        if config.quantization_mode == "qat":
            if not is_qat_model(model):
                # Prepare model for QAT if not already prepared
                self.model = prepare_model_for_qat(
                    model,
                    quantization_bits=config.quantization_bits
                )
                # Recreate optimizer with new model parameters
                self.optimizer = create_optimizer(self.model, config)
        
        # Check if model is quantized and fine-tuning is enabled
        self._is_quantized = is_model_quantized(model) or is_qat_model(model)
        self._quantized_finetuning_enabled = (
            self._is_quantized and config.enable_quantized_finetuning
        )
        
        # Generate run ID and log run metadata (only once)
        self.run_id = str(uuid.uuid4())
        self._run_logged = False
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        tokenizer: Tokenizer
    ) -> "Trainer":
        """Create a Trainer instance from a saved checkpoint.
        
        Loads model state, optimizer state, config, vocabulary, and step counter
        from a checkpoint and creates a Trainer instance ready to resume training.
        
        Args:
            checkpoint_path: Path to checkpoint directory.
            model: Model to load state into (will be modified in-place).
            optimizer: Optimizer to load state into (will be modified in-place).
            tokenizer: Tokenizer to load vocabulary into (will be modified in-place).
        
        Returns:
            Trainer instance with restored state.
        
        Raises:
            FileNotFoundError: If checkpoint files are missing.
            RuntimeError: If checkpoint files are corrupted or invalid.
        """
        checkpoint_data = load_checkpoint(checkpoint_path, model, optimizer, tokenizer)
        return cls(
            model=checkpoint_data["model"],
            optimizer=checkpoint_data["optimizer"],
            config=checkpoint_data["config"],
            step=checkpoint_data["step"]
        )
    
    def save_checkpoint(
        self,
        tokenizer: Tokenizer,
        checkpoint_dir: str | Path = "checkpoints",
        checkpoint_name: Optional[str] = None
    ) -> Path:
        """Save current training state to a checkpoint.
        
        Saves model state, optimizer state, config, vocabulary, and step counter
        to disk for later resuming.
        
        Args:
            tokenizer: Tokenizer containing vocabulary to save.
            checkpoint_dir: Directory to save checkpoints in (default: "checkpoints").
            checkpoint_name: Optional name for checkpoint. If None, uses "checkpoint_step_{step}".
        
        Returns:
            Path to the saved checkpoint directory.
        
        Raises:
            IOError: If checkpoint directory cannot be created or files cannot be written.
        """
        return save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            config=self.config,
            tokenizer=tokenizer,
            step=self.step,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name
        )
    
    def _log_run_metadata(self) -> None:
        """Log run metadata including run_id, git_commit, config_hash, and hook_list.
        
        This is called once at the start of training to log reproducibility information.
        """
        if self._run_logged:
            return
        
        # Get git commit hash
        git_commit = self._get_git_commit()
        
        # Compute config hash
        config_hash = self._compute_config_hash()
        
        # Get active hooks list
        active_hooks = self.hook_registry.get_active_hooks()
        hook_list = {
            "forward": active_hooks.get("forward", []),
            "update": active_hooks.get("update", []),
        }
        
        # Log run metadata
        print(f"[Run Metadata]")
        print(f"  run_id: {self.run_id}")
        print(f"  git_commit: {git_commit}")
        print(f"  config_hash: {config_hash}")
        print(f"  hook_list: {hook_list}")
        
        self._run_logged = True
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash.
        
        Returns:
            Git commit hash, or "unknown" if not in a git repository.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return "unknown"
    
    def _compute_config_hash(self) -> str:
        """Compute hash of configuration for reproducibility.
        
        Returns:
            SHA256 hash of the configuration as a hex string.
        """
        config_dict = self.config.to_dict()
        # Sort keys for deterministic hashing
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def training_step(self, inputs: torch.Tensor) -> float:
        """Perform a single training step.
        
        Executes:
        1. Forward pass through the model
        2. Forward hooks (observe activations)
        3. Loss computation (cross-entropy for next-token prediction)
        4. Backward pass (gradient computation)
        5. Update hooks (transform gradients)
        6. Optimizer step (parameter update)
        
        For next-token prediction, targets are created by shifting inputs
        by one position: targets[:, i] = inputs[:, i+1] for i in [0, seq_len-2].
        The last position of inputs is used as the target for the second-to-last
        logit position.
        
        Args:
            inputs: Input token IDs of shape [B, seq_len] with dtype int64.
        
        Returns:
            Loss value as a float.
        
        Raises:
            ValueError: If inputs tensor has incorrect shape or dtype.
        """
        # Log run metadata on first step
        if not self._run_logged:
            self._log_run_metadata()
        
        # Validate inputs
        if inputs.dim() != 2:
            raise ValueError(
                f"Expected inputs to be 2D tensor [B, seq_len], "
                f"got shape {inputs.shape}"
            )
        if inputs.dtype != torch.int64:
            raise ValueError(
                f"Expected inputs dtype to be int64, got {inputs.dtype}"
            )
        
        batch_size, seq_len = inputs.shape
        
        # Create targets by shifting inputs by 1 position
        # For next-token prediction: predict token at position i+1 given tokens [0:i+1]
        # targets[:, i] should be inputs[:, i+1] for i in [0, seq_len-2]
        # For the last position, we use inputs[:, seq_len-1] as target for position seq_len-2
        targets = torch.zeros_like(inputs)
        targets[:, :-1] = inputs[:, 1:]  # Shift by 1 for positions [0, seq_len-2]
        targets[:, -1] = inputs[:, -1]   # Last position: predict same token (or could be ignored)
        
        # Forward pass
        logits = self.model(inputs)  # [B, seq_len, vocab_size]
        
        # Call forward hooks with activations (logits)
        self.hook_registry.call_forward_hooks(logits)
        
        # Compute loss
        loss = compute_loss(logits, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Collect gradients for update hooks
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        # Call update hooks to transform gradients
        transformed_gradients = self.hook_registry.call_update_hooks(gradients)
        
        # Apply transformed gradients back to model parameters
        for name, param in self.model.named_parameters():
            if name in transformed_gradients:
                param.grad = transformed_gradients[name]
        
        # Optimizer step
        self.optimizer.step()
        
        # Increment step counter
        self.step += 1
        
        # Log loss
        loss_value = loss.item()
        print(f"Step {self.step}: loss = {loss_value:.4f}")
        
        # Perform evaluation if cadence is configured
        if self.config.eval_cadence is not None and self.val_dataloader is not None:
            if self.step % self.config.eval_cadence == 0:
                # Import here to avoid circular import
                from src.evaluation import compute_val_loss
                val_loss = compute_val_loss(self.model, self.val_dataloader)
                print(f"Step {self.step}: val_loss = {val_loss:.4f}")
        
        # Perform sampling if cadence is configured
        if self.config.sampling_cadence is not None and self.tokenizer is not None:
            if self.step % self.config.sampling_cadence == 0:
                # Import here to avoid circular import
                from src.sampling import sample_text
                generated_text = sample_text(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=self.config.sampling_prompt,
                    max_length=self.config.sampling_max_length,
                    temperature=self.config.sampling_temperature,
                    seed=self.config.sampling_seed
                )
                print(f"Step {self.step}: sample = {generated_text!r}")
        
        return loss_value

