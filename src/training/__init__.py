"""Training module for transformer model training.

This module provides training infrastructure including loss computation,
optimizer setup, and training step functionality.
"""

from src.training.loss import compute_loss
from src.training.trainer import Trainer, create_optimizer

__all__ = ["compute_loss", "Trainer", "create_optimizer"]

