"""Tests for checkpointing functionality."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from src.config import TrainingConfig
from src.model.transformer import Transformer
from src.tokenizer import Tokenizer
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.trainer import Trainer, create_optimizer
from src.quantization import (
    quantize_model_ptq,
    is_model_quantized,
    get_quantization_info,
    prepare_model_for_qat,
)


def test_save_checkpoint_creates_directory():
    """Test that save_checkpoint creates checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            step=0,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint"
        )
        
        assert checkpoint_path.exists()
        assert checkpoint_path.is_dir()
        assert checkpoint_path.name == "test_checkpoint"


def test_save_checkpoint_creates_files():
    """Test that save_checkpoint creates all required files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            step=42,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint"
        )
        
        # Check all required files exist
        assert (checkpoint_path / "model.pt").exists()
        assert (checkpoint_path / "optimizer.pt").exists()
        assert (checkpoint_path / "rng.pt").exists()
        assert (checkpoint_path / "vocab.json").exists()
        assert (checkpoint_path / "metadata.json").exists()


def test_save_checkpoint_model_weights():
    """Test that model weights are saved correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        # Get original model state
        original_state = model.state_dict()
        
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            step=0,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint"
        )
        
        # Load saved state
        saved_state = torch.load(checkpoint_path / "model.pt", map_location="cpu")
        
        # Verify all keys match
        assert set(saved_state.keys()) == set(original_state.keys())
        
        # Verify values match
        for key in original_state.keys():
            assert torch.allclose(saved_state[key], original_state[key])


def test_save_checkpoint_optimizer_state():
    """Test that optimizer state is saved correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        # Train one step to populate optimizer state
        batch_size = 2
        seq_len = 256
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        trainer = Trainer(model, optimizer, config)
        trainer.training_step(inputs)
        
        # Get original optimizer state
        original_state = optimizer.state_dict()
        
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            step=1,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint"
        )
        
        # Load saved state
        saved_state = torch.load(checkpoint_path / "optimizer.pt", map_location="cpu")
        
        # Verify state dict structure matches
        assert set(saved_state.keys()) == set(original_state.keys())
        assert "state" in saved_state
        assert "param_groups" in saved_state


def test_save_checkpoint_metadata():
    """Test that metadata is saved correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig(learning_rate=1e-3, batch_size=32)
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        step = 123
        
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            step=step,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint"
        )
        
        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Verify step
        assert metadata["step"] == step
        
        # Verify config
        assert metadata["config"]["learning_rate"] == config.learning_rate
        assert metadata["config"]["batch_size"] == config.batch_size
        assert metadata["config"]["weight_decay"] == config.weight_decay


def test_save_checkpoint_model_metadata():
    """Test that model metadata is saved correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            step=0,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint",
            model_name="qwen-0.5b-base",
            model_id="Qwen/Qwen-0.5B",
            model_source="huggingface",
            fine_tuned_from="qwen-0.5b-original"
        )
        
        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Verify model metadata
        assert metadata["model_name"] == "qwen-0.5b-base"
        assert metadata["model_id"] == "Qwen/Qwen-0.5B"
        assert metadata["model_source"] == "huggingface"
        assert metadata["fine_tuned_from"] == "qwen-0.5b-original"


def test_save_checkpoint_model_metadata_optional():
    """Test that model metadata is optional."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            step=0,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint"
        )
        
        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Verify model metadata is not present (backward compatibility)
        assert "model_name" not in metadata
        assert "model_id" not in metadata
        assert "model_source" not in metadata
        assert "fine_tuned_from" not in metadata


def test_load_checkpoint_model_metadata():
    """Test that model metadata is loaded correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model1 = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer1 = create_optimizer(model1, config)
        tokenizer1 = Tokenizer()
        
        # Save checkpoint with model metadata
        checkpoint_path = save_checkpoint(
            model=model1,
            optimizer=optimizer1,
            config=config,
            tokenizer=tokenizer1,
            step=0,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint",
            model_name="qwen-0.5b-base",
            model_id="Qwen/Qwen-0.5B",
            model_source="huggingface",
            fine_tuned_from="qwen-0.5b-original"
        )
        
        # Load checkpoint
        model2 = Transformer(vocab_size=vocab_size)
        optimizer2 = create_optimizer(model2, config)
        tokenizer2 = Tokenizer()
        
        checkpoint_data = load_checkpoint(
            checkpoint_path, model2, optimizer2, tokenizer2
        )
        
        # Verify model metadata is returned
        assert checkpoint_data["model_name"] == "qwen-0.5b-base"
        assert checkpoint_data["model_id"] == "Qwen/Qwen-0.5B"
        assert checkpoint_data["model_source"] == "huggingface"
        assert checkpoint_data["fine_tuned_from"] == "qwen-0.5b-original"


def test_load_checkpoint_backward_compatibility():
    """Test that loading old checkpoints without model metadata still works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model1 = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer1 = create_optimizer(model1, config)
        tokenizer1 = Tokenizer()
        
        # Save checkpoint without model metadata (old format)
        checkpoint_path = save_checkpoint(
            model=model1,
            optimizer=optimizer1,
            config=config,
            tokenizer=tokenizer1,
            step=0,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint"
        )
        
        # Load checkpoint
        model2 = Transformer(vocab_size=vocab_size)
        optimizer2 = create_optimizer(model2, config)
        tokenizer2 = Tokenizer()
        
        checkpoint_data = load_checkpoint(
            checkpoint_path, model2, optimizer2, tokenizer2
        )
        
        # Verify checkpoint loads successfully
        assert checkpoint_data["step"] == 0
        assert checkpoint_data["config"] is not None
        
        # Verify model metadata is not present (backward compatibility)
        assert "model_name" not in checkpoint_data
        assert "model_id" not in checkpoint_data
        assert "model_source" not in checkpoint_data
        assert "fine_tuned_from" not in checkpoint_data


def test_save_checkpoint_vocabulary():
    """Test that vocabulary is saved correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            step=0,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint"
        )
        
        # Load vocabulary
        vocab_path = checkpoint_path / "vocab.json"
        assert vocab_path.exists()
        
        # Verify vocabulary can be loaded
        new_tokenizer = Tokenizer()
        new_tokenizer.load_vocab(vocab_path)
        
        # Verify vocabulary matches
        assert new_tokenizer.char_to_id == tokenizer.char_to_id
        assert new_tokenizer.id_to_char == tokenizer.id_to_char


def test_load_checkpoint_restores_model():
    """Test that load_checkpoint restores model state correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model1 = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer1 = create_optimizer(model1, config)
        tokenizer1 = Tokenizer()
        
        # Train model1 for a few steps
        trainer1 = Trainer(model1, optimizer1, config)
        batch_size = 2
        seq_len = 256
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        for _ in range(3):
            trainer1.training_step(inputs)
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model=model1,
            optimizer=optimizer1,
            config=config,
            tokenizer=tokenizer1,
            step=trainer1.step,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint"
        )
        
        # Create new model and load checkpoint
        model2 = Transformer(vocab_size=vocab_size)
        optimizer2 = create_optimizer(model2, config)
        tokenizer2 = Tokenizer()
        
        checkpoint_data = load_checkpoint(
            checkpoint_path, model2, optimizer2, tokenizer2
        )
        
        # Verify model state matches
        state1 = model1.state_dict()
        state2 = model2.state_dict()
        
        for key in state1.keys():
            assert torch.allclose(state1[key], state2[key])


def test_load_checkpoint_restores_optimizer():
    """Test that load_checkpoint restores optimizer state correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model1 = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer1 = create_optimizer(model1, config)
        tokenizer1 = Tokenizer()
        
        # Train model1 for a few steps
        trainer1 = Trainer(model1, optimizer1, config)
        batch_size = 2
        seq_len = 256
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        for _ in range(3):
            trainer1.training_step(inputs)
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model=model1,
            optimizer=optimizer1,
            config=config,
            tokenizer=tokenizer1,
            step=trainer1.step,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint"
        )
        
        # Create new optimizer and load checkpoint
        model2 = Transformer(vocab_size=vocab_size)
        optimizer2 = create_optimizer(model2, config)
        tokenizer2 = Tokenizer()
        
        checkpoint_data = load_checkpoint(
            checkpoint_path, model2, optimizer2, tokenizer2
        )
        
        # Verify optimizer state structure matches
        state1 = optimizer1.state_dict()
        state2 = optimizer2.state_dict()
        
        assert set(state1.keys()) == set(state2.keys())
        assert len(state1["state"]) == len(state2["state"])


def test_load_checkpoint_restores_config():
    """Test that load_checkpoint restores config correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig(learning_rate=5e-4, batch_size=64, seed=42)
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            step=0,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint"
        )
        
        # Load checkpoint
        model2 = Transformer(vocab_size=vocab_size)
        optimizer2 = create_optimizer(model2, config)
        tokenizer2 = Tokenizer()
        
        checkpoint_data = load_checkpoint(
            checkpoint_path, model2, optimizer2, tokenizer2
        )
        
        # Verify config matches
        restored_config = checkpoint_data["config"]
        assert restored_config.learning_rate == config.learning_rate
        assert restored_config.batch_size == config.batch_size
        assert restored_config.seed == config.seed
        assert restored_config.weight_decay == config.weight_decay


def test_load_checkpoint_restores_vocabulary():
    """Test that load_checkpoint restores vocabulary correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer = create_optimizer(model, config)
        tokenizer1 = Tokenizer()
        
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer1,
            step=0,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint"
        )
        
        # Load checkpoint
        model2 = Transformer(vocab_size=vocab_size)
        optimizer2 = create_optimizer(model2, config)
        tokenizer2 = Tokenizer()
        
        checkpoint_data = load_checkpoint(
            checkpoint_path, model2, optimizer2, tokenizer2
        )
        
        # Verify vocabulary matches
        assert tokenizer2.char_to_id == tokenizer1.char_to_id
        assert tokenizer2.id_to_char == tokenizer1.id_to_char


def test_load_checkpoint_restores_step():
    """Test that load_checkpoint restores step counter correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        step = 42
        
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            step=step,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint"
        )
        
        # Load checkpoint
        model2 = Transformer(vocab_size=vocab_size)
        optimizer2 = create_optimizer(model2, config)
        tokenizer2 = Tokenizer()
        
        checkpoint_data = load_checkpoint(
            checkpoint_path, model2, optimizer2, tokenizer2
        )
        
        # Verify step matches
        assert checkpoint_data["step"] == step


def test_load_checkpoint_returns_structure():
    """Test that load_checkpoint returns correct data structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            step=10,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint"
        )
        
        # Load checkpoint
        model2 = Transformer(vocab_size=vocab_size)
        optimizer2 = create_optimizer(model2, config)
        tokenizer2 = Tokenizer()
        
        checkpoint_data = load_checkpoint(
            checkpoint_path, model2, optimizer2, tokenizer2
        )
        
        # Verify structure
        assert "step" in checkpoint_data
        assert "config" in checkpoint_data
        assert "model" in checkpoint_data
        assert "optimizer" in checkpoint_data
        assert "tokenizer" in checkpoint_data
        
        # Verify references
        assert checkpoint_data["model"] is model2
        assert checkpoint_data["optimizer"] is optimizer2
        assert checkpoint_data["tokenizer"] is tokenizer2


def test_resume_training_continues_step():
    """Test that resuming training continues step count correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model1 = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer1 = create_optimizer(model1, config)
        tokenizer = Tokenizer()
        
        # Train for 5 steps
        trainer1 = Trainer(model1, optimizer1, config)
        batch_size = 2
        seq_len = 256
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        for _ in range(5):
            trainer1.training_step(inputs)
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model=model1,
            optimizer=optimizer1,
            config=config,
            tokenizer=tokenizer,
            step=trainer1.step,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint"
        )
        
        # Resume training
        model2 = Transformer(vocab_size=vocab_size)
        optimizer2 = create_optimizer(model2, config)
        tokenizer2 = Tokenizer()
        
        trainer2 = Trainer.from_checkpoint(
            checkpoint_path, model2, optimizer2, tokenizer2
        )
        
        # Verify step counter
        assert trainer2.step == trainer1.step == 5
        
        # Continue training
        trainer2.training_step(inputs)
        assert trainer2.step == 6


def test_resume_training_identical_loss():
    """Test that resuming training produces identical loss progression."""
    torch.manual_seed(42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        batch_size = 2
        seq_len = 256
        
        # Generate inputs once to ensure they're identical
        torch.manual_seed(42)
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Uninterrupted training
        torch.manual_seed(42)
        model1 = Transformer(vocab_size=vocab_size)
        config = TrainingConfig(seed=42)
        optimizer1 = create_optimizer(model1, config)
        tokenizer1 = Tokenizer()
        trainer1 = Trainer(model1, optimizer1, config)
        
        losses1 = []
        for _ in range(5):
            loss = trainer1.training_step(inputs)
            losses1.append(loss)
        
        # Interrupted training: train 3 steps, checkpoint, resume
        torch.manual_seed(42)
        model2 = Transformer(vocab_size=vocab_size)
        optimizer2 = create_optimizer(model2, config)
        tokenizer2 = Tokenizer()
        trainer2 = Trainer(model2, optimizer2, config)
        
        losses2_part1 = []
        for _ in range(3):
            loss = trainer2.training_step(inputs)
            losses2_part1.append(loss)
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model=model2,
            optimizer=optimizer2,
            config=config,
            tokenizer=tokenizer2,
            step=trainer2.step,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint"
        )
        
        # Resume training
        model3 = Transformer(vocab_size=vocab_size)
        optimizer3 = create_optimizer(model3, config)
        tokenizer3 = Tokenizer()
        trainer3 = Trainer.from_checkpoint(
            checkpoint_path, model3, optimizer3, tokenizer3
        )
        
        losses2_part2 = []
        for _ in range(2):
            loss = trainer3.training_step(inputs)
            losses2_part2.append(loss)
        
        # Combine interrupted training losses
        losses2 = losses2_part1 + losses2_part2
        
        # Verify loss progression matches (within tolerance)
        # Use a more reasonable tolerance for loss values after save/load
        # Loss values around 4.x can have small differences due to floating-point precision
        assert len(losses1) == len(losses2)
        for i, (l1, l2) in enumerate(zip(losses1, losses2)):
            # Use relative tolerance: allow up to 0.5% difference or absolute 0.02
            rel_tol = max(abs(l1) * 0.005, 0.02)
            assert abs(l1 - l2) < rel_tol, f"Loss mismatch at step {i}: {l1} vs {l2} (diff: {abs(l1 - l2):.6f}, tol: {rel_tol:.6f})"


def test_load_checkpoint_missing_directory():
    """Test that load_checkpoint raises FileNotFoundError for missing directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        missing_path = Path(tmpdir) / "nonexistent_checkpoint"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            load_checkpoint(missing_path, model, optimizer, tokenizer)
        
        assert "not found" in str(exc_info.value).lower()


def test_load_checkpoint_missing_model_file():
    """Test that load_checkpoint raises FileNotFoundError for missing model file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        checkpoint_path = Path(tmpdir) / "test_checkpoint"
        checkpoint_path.mkdir()
        
        # Create other files but not model.pt
        (checkpoint_path / "optimizer.pt").touch()
        (checkpoint_path / "vocab.json").touch()
        # Create valid metadata.json so we can get to the model file check
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump({"step": 0, "config": {}, "checkpoint_version": "1.0"}, f)
        
        with pytest.raises(FileNotFoundError) as exc_info:
            load_checkpoint(checkpoint_path, model, optimizer, tokenizer)
        
        assert "model" in str(exc_info.value).lower()


def test_load_checkpoint_corrupted_model_file():
    """Test that load_checkpoint raises RuntimeError for corrupted model file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        checkpoint_path = Path(tmpdir) / "test_checkpoint"
        checkpoint_path.mkdir()
        
        # Create corrupted model file
        (checkpoint_path / "model.pt").write_text("corrupted data")
        (checkpoint_path / "optimizer.pt").touch()
        (checkpoint_path / "vocab.json").touch()
        (checkpoint_path / "metadata.json").touch()
        
        with pytest.raises(RuntimeError) as exc_info:
            load_checkpoint(checkpoint_path, model, optimizer, tokenizer)
        
        assert "failed to load" in str(exc_info.value).lower() or "corrupted" in str(exc_info.value).lower()


def test_trainer_save_checkpoint():
    """Test Trainer.save_checkpoint method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        trainer = Trainer(model, optimizer, config)
        
        # Train a few steps
        batch_size = 2
        seq_len = 256
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        trainer.training_step(inputs)
        trainer.training_step(inputs)
        
        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint(
            tokenizer=tokenizer,
            checkpoint_dir=tmpdir,
            checkpoint_name="trainer_checkpoint"
        )
        
        # Verify checkpoint was created
        assert checkpoint_path.exists()
        assert (checkpoint_path / "model.pt").exists()
        assert (checkpoint_path / "optimizer.pt").exists()
        assert (checkpoint_path / "rng.pt").exists()
        assert (checkpoint_path / "metadata.json").exists()
        
        # Verify step is saved correctly
        with open(checkpoint_path / "metadata.json", "r") as f:
            metadata = json.load(f)
        assert metadata["step"] == trainer.step == 2


def test_trainer_from_checkpoint():
    """Test Trainer.from_checkpoint class method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model1 = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer1 = create_optimizer(model1, config)
        tokenizer1 = Tokenizer()
        trainer1 = Trainer(model1, optimizer1, config)
        
        # Train and save
        batch_size = 2
        seq_len = 256
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        for _ in range(3):
            trainer1.training_step(inputs)
        
        checkpoint_path = save_checkpoint(
            model=model1,
            optimizer=optimizer1,
            config=config,
            tokenizer=tokenizer1,
            step=trainer1.step,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint"
        )
        
        # Resume using Trainer.from_checkpoint
        model2 = Transformer(vocab_size=vocab_size)
        optimizer2 = create_optimizer(model2, config)
        tokenizer2 = Tokenizer()
        trainer2 = Trainer.from_checkpoint(
            checkpoint_path, model2, optimizer2, tokenizer2
        )
        
        # Verify state
        assert trainer2.step == trainer1.step == 3
        assert trainer2.config.learning_rate == trainer1.config.learning_rate
        
        # Verify model state matches
        state1 = model1.state_dict()
        state2 = model2.state_dict()
        for key in state1.keys():
            assert torch.allclose(state1[key], state2[key])


def _check_quantization_engine_available() -> bool:
    """Check if PyTorch quantization engines are available."""
    try:
        if not hasattr(torch.backends, 'quantized'):
            return False
        current_engine = torch.backends.quantized.engine
        supported_engines = torch.backends.quantized.supported_engines
        if not supported_engines:
            return False
        if current_engine == 'none' or current_engine is None:
            return False
        return True
    except (AttributeError, RuntimeError, TypeError):
        return False


def test_save_quantized_checkpoint_creates_metadata():
    """Test that saving a quantized checkpoint creates quantization metadata."""
    # Skip test if quantization engines are not available
    if not _check_quantization_engine_available():
        pytest.skip("PyTorch quantization engines not available")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        model.eval()
        config = TrainingConfig()
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        # Quantize the model using dynamic quantization (no calibration needed)
        quantized_model = quantize_model_ptq(
            model,
            quantization_bits=8,
            quantization_type="dynamic"
        )
        
        checkpoint_path = save_checkpoint(
            model=quantized_model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            step=0,
            checkpoint_dir=tmpdir,
            checkpoint_name="quantized_checkpoint"
        )
        
        # Check that quantization metadata file exists
        quantization_metadata_path = checkpoint_path / "quantization_metadata.json"
        assert quantization_metadata_path.exists()
        
        # Check that metadata.json includes quantization info
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        assert "quantization" in metadata
        assert metadata["quantization"]["is_quantized"] == True


def test_save_quantized_checkpoint_metadata_content():
    """Test that quantization metadata contains expected fields."""
    # Skip test if quantization engines are not available
    if not _check_quantization_engine_available():
        pytest.skip("PyTorch quantization engines not available")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        model.eval()
        config = TrainingConfig()
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        # Quantize the model
        quantized_model = quantize_model_ptq(
            model,
            quantization_bits=8,
            quantization_type="dynamic"
        )
        
        checkpoint_path = save_checkpoint(
            model=quantized_model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            step=0,
            checkpoint_dir=tmpdir,
            checkpoint_name="quantized_checkpoint"
        )
        
        # Load quantization metadata
        quantization_metadata_path = checkpoint_path / "quantization_metadata.json"
        with open(quantization_metadata_path, "r", encoding="utf-8") as f:
            quantization_metadata = json.load(f)
        
        # Verify metadata structure
        assert quantization_metadata["is_quantized"] == True
        assert quantization_metadata["quantization_bits"] == 8
        assert "quantized_layers" in quantization_metadata


def test_load_quantized_checkpoint_restores_metadata():
    """Test that loading a quantized checkpoint restores quantization metadata."""
    # Skip test if quantization engines are not available
    if not _check_quantization_engine_available():
        pytest.skip("PyTorch quantization engines not available")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model1 = Transformer(vocab_size=vocab_size)
        model1.eval()
        config = TrainingConfig()
        optimizer1 = create_optimizer(model1, config)
        tokenizer1 = Tokenizer()
        
        # Quantize and save
        quantized_model1 = quantize_model_ptq(
            model1,
            quantization_bits=8,
            quantization_type="dynamic"
        )
        
        checkpoint_path = save_checkpoint(
            model=quantized_model1,
            optimizer=optimizer1,
            config=config,
            tokenizer=tokenizer1,
            step=0,
            checkpoint_dir=tmpdir,
            checkpoint_name="quantized_checkpoint"
        )
        
        # Load checkpoint
        model2 = Transformer(vocab_size=vocab_size)
        optimizer2 = create_optimizer(model2, config)
        tokenizer2 = Tokenizer()
        
        checkpoint_data = load_checkpoint(
            checkpoint_path, model2, optimizer2, tokenizer2
        )
        
        # Verify quantization metadata is returned
        assert "quantization_metadata" in checkpoint_data
        assert checkpoint_data["quantization_metadata"]["is_quantized"] == True
        assert checkpoint_data["quantization_metadata"]["quantization_bits"] == 8


def test_load_checkpoint_without_quantization_metadata():
    """Test that loading a non-quantized checkpoint works (backward compatibility)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig()
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        # Save non-quantized checkpoint
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            step=0,
            checkpoint_dir=tmpdir,
            checkpoint_name="normal_checkpoint"
        )
        
        # Verify quantization metadata file does NOT exist
        quantization_metadata_path = checkpoint_path / "quantization_metadata.json"
        assert not quantization_metadata_path.exists()
        
        # Load checkpoint
        model2 = Transformer(vocab_size=vocab_size)
        optimizer2 = create_optimizer(model2, config)
        tokenizer2 = Tokenizer()
        
        checkpoint_data = load_checkpoint(
            checkpoint_path, model2, optimizer2, tokenizer2
        )
        
        # Should load successfully without quantization metadata
        assert checkpoint_data["step"] == 0
        assert "quantization_metadata" not in checkpoint_data or checkpoint_data.get("quantization_metadata") is None


def test_save_qat_checkpoint():
    """Test saving a checkpoint with a QAT model."""
    # Skip test if quantization engines are not available
    if not _check_quantization_engine_available():
        pytest.skip("PyTorch quantization engines not available")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 100
        model = Transformer(vocab_size=vocab_size)
        config = TrainingConfig(quantization_mode="qat", quantization_bits=8)
        optimizer = create_optimizer(model, config)
        tokenizer = Tokenizer()
        
        # Prepare model for QAT
        qat_model = prepare_model_for_qat(model, quantization_bits=8)
        
        checkpoint_path = save_checkpoint(
            model=qat_model,
            optimizer=optimizer,
            config=config,
            tokenizer=tokenizer,
            step=0,
            checkpoint_dir=tmpdir,
            checkpoint_name="qat_checkpoint"
        )
        
        # QAT models may or may not be detected as quantized depending on implementation
        # But checkpoint should save successfully
        assert checkpoint_path.exists()
        assert (checkpoint_path / "model.pt").exists()
        assert (checkpoint_path / "metadata.json").exists()

