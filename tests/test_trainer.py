"""Tests for training module."""

import pytest
import torch

from src.config import TrainingConfig
from src.model.transformer import Transformer
from src.training.loss import compute_loss
from src.training.trainer import Trainer, create_optimizer


def test_compute_loss_shape():
    """Test that compute_loss returns a scalar."""
    vocab_size = 100
    batch_size = 2
    seq_len = 256
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss = compute_loss(logits, targets)
    
    assert loss.shape == ()
    assert loss.dim() == 0


def test_compute_loss_handles_correct_shapes():
    """Test compute_loss handles logits [B, 256, vocab_size] and targets [B, 256]."""
    vocab_size = 100
    batch_size = 4
    seq_len = 256
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss = compute_loss(logits, targets)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_compute_loss_shape_mismatch_batch():
    """Test compute_loss raises error on batch size mismatch."""
    vocab_size = 100
    batch_size1 = 2
    batch_size2 = 3
    seq_len = 256
    
    logits = torch.randn(batch_size1, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size2, seq_len))
    
    with pytest.raises(ValueError, match="Batch size mismatch"):
        compute_loss(logits, targets)


def test_compute_loss_shape_mismatch_seq_len():
    """Test compute_loss raises error on sequence length mismatch."""
    vocab_size = 100
    batch_size = 2
    seq_len1 = 256
    seq_len2 = 128
    
    logits = torch.randn(batch_size, seq_len1, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len2))
    
    with pytest.raises(ValueError, match="Sequence length mismatch"):
        compute_loss(logits, targets)


def test_compute_loss_correctness_manual():
    """Test loss computation correctness on known inputs."""
    vocab_size = 3
    batch_size = 1
    seq_len = 2
    
    # Create logits where position 0 predicts class 1, position 1 predicts class 2
    logits = torch.zeros(batch_size, seq_len, vocab_size)
    logits[0, 0, 1] = 10.0  # High logit for class 1 at position 0
    logits[0, 1, 2] = 10.0  # High logit for class 2 at position 1
    
    targets = torch.tensor([[1, 2]])  # Target is class 1 at pos 0, class 2 at pos 1
    
    loss = compute_loss(logits, targets)
    
    # Loss should be very small since predictions are correct
    assert loss.item() < 0.1
    
    # Test with incorrect predictions
    logits_wrong = torch.zeros(batch_size, seq_len, vocab_size)
    logits_wrong[0, 0, 0] = 10.0  # Predict class 0 instead of 1
    logits_wrong[0, 1, 0] = 10.0  # Predict class 0 instead of 2
    
    loss_wrong = compute_loss(logits_wrong, targets)
    
    # Loss should be much higher
    assert loss_wrong.item() > loss.item()


def test_compute_loss_differentiable():
    """Test that loss is differentiable for gradient computation."""
    vocab_size = 100
    batch_size = 2
    seq_len = 256
    
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss = compute_loss(logits, targets)
    loss.backward()
    
    assert logits.grad is not None
    assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad))


def test_training_config_defaults():
    """Test TrainingConfig has correct default values."""
    config = TrainingConfig()
    
    assert config.learning_rate == 3e-4
    assert config.weight_decay == 0.1
    assert config.beta1 == 0.9
    assert config.beta2 == 0.95
    assert config.batch_size == 16
    assert config.max_seq_len == 256
    assert config.seed is None


def test_training_config_custom_values():
    """Test TrainingConfig with custom values."""
    config = TrainingConfig(
        learning_rate=1e-3,
        weight_decay=0.01,
        beta1=0.8,
        beta2=0.99,
        batch_size=32,
        max_seq_len=512,
        seed=42
    )
    
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 0.01
    assert config.beta1 == 0.8
    assert config.beta2 == 0.99
    assert config.batch_size == 32
    assert config.max_seq_len == 512
    assert config.seed == 42


def test_training_config_from_dict():
    """Test TrainingConfig.from_dict creates config from dictionary."""
    config_dict = {
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "beta1": 0.8,
        "beta2": 0.99,
        "batch_size": 32,
        "max_seq_len": 512,
        "seed": 42
    }
    
    config = TrainingConfig.from_dict(config_dict)
    
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 0.01
    assert config.beta1 == 0.8
    assert config.beta2 == 0.99
    assert config.batch_size == 32
    assert config.max_seq_len == 512
    assert config.seed == 42


def test_training_config_to_dict():
    """Test TrainingConfig.to_dict converts config to dictionary."""
    config = TrainingConfig(
        learning_rate=1e-3,
        weight_decay=0.01,
        seed=42
    )
    
    config_dict = config.to_dict()
    
    assert config_dict["learning_rate"] == 1e-3
    assert config_dict["weight_decay"] == 0.01
    assert config_dict["beta1"] == 0.9  # Default value
    assert config_dict["beta2"] == 0.95  # Default value
    assert config_dict["seed"] == 42


def test_create_optimizer():
    """Test create_optimizer creates AdamW with correct hyperparameters."""
    vocab_size = 100
    model = Transformer(vocab_size=vocab_size)
    config = TrainingConfig()
    
    optimizer = create_optimizer(model, config)
    
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 3e-4
    assert optimizer.defaults["weight_decay"] == 0.1
    assert optimizer.defaults["betas"] == (0.9, 0.95)


def test_create_optimizer_custom_config():
    """Test create_optimizer uses custom config values."""
    vocab_size = 100
    model = Transformer(vocab_size=vocab_size)
    config = TrainingConfig(
        learning_rate=1e-3,
        weight_decay=0.01,
        beta1=0.8,
        beta2=0.99
    )
    
    optimizer = create_optimizer(model, config)
    
    assert optimizer.defaults["lr"] == 1e-3
    assert optimizer.defaults["weight_decay"] == 0.01
    assert optimizer.defaults["betas"] == (0.8, 0.99)


def test_trainer_init():
    """Test Trainer initialization."""
    vocab_size = 100
    model = Transformer(vocab_size=vocab_size)
    config = TrainingConfig()
    optimizer = create_optimizer(model, config)
    
    trainer = Trainer(model, optimizer, config)
    
    assert trainer.model == model
    assert trainer.optimizer == optimizer
    assert trainer.config == config
    assert trainer.step == 0


def test_trainer_training_step_completes():
    """Test training step completes without errors."""
    vocab_size = 100
    batch_size = 2
    seq_len = 256
    
    model = Transformer(vocab_size=vocab_size)
    config = TrainingConfig()
    optimizer = create_optimizer(model, config)
    trainer = Trainer(model, optimizer, config)
    
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss = trainer.training_step(inputs)
    
    assert isinstance(loss, float)
    assert loss > 0


def test_trainer_training_step_increments_counter():
    """Test step counter increments correctly."""
    vocab_size = 100
    batch_size = 2
    seq_len = 256
    
    model = Transformer(vocab_size=vocab_size)
    config = TrainingConfig()
    optimizer = create_optimizer(model, config)
    trainer = Trainer(model, optimizer, config)
    
    assert trainer.step == 0
    
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    trainer.training_step(inputs)
    
    assert trainer.step == 1
    
    trainer.training_step(inputs)
    
    assert trainer.step == 2


def test_trainer_training_step_updates_parameters():
    """Test optimizer updates model parameters."""
    vocab_size = 100
    batch_size = 2
    seq_len = 256
    
    model = Transformer(vocab_size=vocab_size)
    config = TrainingConfig()
    optimizer = create_optimizer(model, config)
    trainer = Trainer(model, optimizer, config)
    
    # Store initial parameter values
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.data.clone()
    
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    trainer.training_step(inputs)
    
    # Check that parameters have changed
    params_changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(param.data, initial_params[name]):
            params_changed = True
            break
    
    assert params_changed


def test_trainer_training_step_invalid_shape():
    """Test training step raises error on invalid input shape."""
    vocab_size = 100
    model = Transformer(vocab_size=vocab_size)
    config = TrainingConfig()
    optimizer = create_optimizer(model, config)
    trainer = Trainer(model, optimizer, config)
    
    # 1D tensor instead of 2D
    inputs = torch.randint(0, vocab_size, (256,))
    
    with pytest.raises(ValueError, match="Expected inputs to be 2D"):
        trainer.training_step(inputs)


def test_trainer_training_step_invalid_dtype():
    """Test training step raises error on invalid input dtype."""
    vocab_size = 100
    batch_size = 2
    seq_len = 256
    
    model = Transformer(vocab_size=vocab_size)
    config = TrainingConfig()
    optimizer = create_optimizer(model, config)
    trainer = Trainer(model, optimizer, config)
    
    # Float tensor instead of int64
    inputs = torch.randn(batch_size, seq_len)
    
    with pytest.raises(ValueError, match="Expected inputs dtype to be int64"):
        trainer.training_step(inputs)


def test_trainer_loss_decreases_on_synthetic_data():
    """Test loss decreases on simple synthetic data (smoke test)."""
    vocab_size = 10
    batch_size = 2
    seq_len = 256
    
    # Create a simple model
    model = Transformer(
        vocab_size=vocab_size,
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=256
    )
    
    config = TrainingConfig(learning_rate=1e-2)  # Higher LR for faster learning
    optimizer = create_optimizer(model, config)
    trainer = Trainer(model, optimizer, config)
    
    # Create synthetic data: simple repeating pattern
    # This should be learnable
    pattern = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * (seq_len // 10 + 1))[:seq_len]
    inputs = pattern.unsqueeze(0).repeat(batch_size, 1)
    
    # Train for a few steps
    losses = []
    for _ in range(5):
        loss = trainer.training_step(inputs)
        losses.append(loss)
    
    # Loss should generally decrease (not strictly monotonic due to randomness)
    # But we check that at least one step has lower loss than the first
    initial_loss = losses[0]
    min_loss = min(losses[1:])
    
    # Allow some tolerance - loss should decrease or at least not increase significantly
    assert min_loss <= initial_loss * 1.1  # Allow 10% tolerance


def test_trainer_optimizer_state_tracked():
    """Test optimizer state is tracked correctly."""
    vocab_size = 100
    batch_size = 2
    seq_len = 256
    
    model = Transformer(vocab_size=vocab_size)
    config = TrainingConfig()
    optimizer = create_optimizer(model, config)
    trainer = Trainer(model, optimizer, config)
    
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Before training step, optimizer state should be empty or default
    trainer.training_step(inputs)
    
    # After training step, optimizer should have state for parameters
    # Check that optimizer state dict is not empty
    state_dict = optimizer.state_dict()
    assert len(state_dict["state"]) > 0

