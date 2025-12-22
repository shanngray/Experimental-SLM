"""Trainer class for training transformer models.

This module provides the Trainer class that orchestrates the training process,
including forward pass, loss computation, backward pass, and optimizer updates.
"""

import torch
import torch.optim

from src.config import TrainingConfig
from src.training.loss import compute_loss


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
    and logs loss values.
    
    Attributes:
        model: The transformer model to train.
        optimizer: Optimizer for parameter updates.
        config: Training configuration with hyperparameters.
        step: Current training step counter.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig
    ):
        """Initialize the Trainer.
        
        Args:
            model: Transformer model to train (must be a PyTorch nn.Module).
            optimizer: Optimizer for parameter updates (e.g., AdamW).
            config: Training configuration with hyperparameters.
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.step = 0
    
    def training_step(self, inputs: torch.Tensor) -> float:
        """Perform a single training step.
        
        Executes:
        1. Forward pass through the model
        2. Loss computation (cross-entropy for next-token prediction)
        3. Backward pass (gradient computation)
        4. Optimizer step (parameter update)
        
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
        
        # Compute loss
        loss = compute_loss(logits, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Optimizer step
        self.optimizer.step()
        
        # Increment step counter
        self.step += 1
        
        # Log loss
        loss_value = loss.item()
        print(f"Step {self.step}: loss = {loss_value:.4f}")
        
        return loss_value

