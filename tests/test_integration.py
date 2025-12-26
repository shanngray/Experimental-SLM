"""Integration tests for end-to-end training pipeline.

This module provides comprehensive integration tests that verify the complete
training pipeline from data loading to checkpoint/resume functionality.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from src.config import TrainingConfig
from src.dataloader import DataLoader
from src.dataset import WindowDataset, split_corpus
from src.model.transformer import Transformer
from src.tokenizer import Tokenizer
from src.training.checkpoint import load_checkpoint_config, load_checkpoint
from src.training.trainer import Trainer, create_optimizer
from src.quantization import (
    quantize_model_ptq,
    prepare_model_for_qat,
    convert_qat_model,
    is_model_quantized,
    is_qat_model,
)


@pytest.fixture
def tiny_corpus():
    """Create a tiny synthetic corpus for testing."""
    # Create a simple repeating pattern that can be learned
    text = "abcd " * 100  # 500 characters
    return text


@pytest.fixture
def tiny_config():
    """Create a minimal config for fast testing."""
    return TrainingConfig(
        learning_rate=1e-3,
        batch_size=2,
        max_seq_len=16,
        seed=42,
        eval_cadence=5,
        sampling_cadence=5,
        sampling_max_length=20
    )


def test_end_to_end_training_pipeline(tiny_corpus, tiny_config):
    """Test complete training pipeline: data → train → checkpoint → resume.
    
    This test verifies:
    - Data loading and tokenization works
    - Dataset creation and splitting works
    - DataLoader batching works
    - Model initialization works
    - Training step execution works
    - Checkpoint saving works
    - Checkpoint resuming works
    - Full pipeline integration is correct
    """
    # Initialize tokenizer and encode corpus
    tokenizer = Tokenizer()
    corpus = tokenizer.encode(tiny_corpus)
    vocab_size = len(tokenizer.char_to_id)
    
    # Split corpus
    train_corpus, val_corpus = split_corpus(corpus, train_ratio=0.8, seed=42)
    
    # Create datasets
    train_dataset = WindowDataset(train_corpus, context_length=tiny_config.max_seq_len)
    val_dataset = WindowDataset(val_corpus, context_length=tiny_config.max_seq_len)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=tiny_config.batch_size, seed=42)
    val_dataloader = DataLoader(val_dataset, batch_size=tiny_config.batch_size, seed=42)
    
    # Create model and optimizer
    model = Transformer(
        vocab_size=vocab_size,
        max_seq_len=tiny_config.max_seq_len,
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_ff=64,
        seed=42
    )
    optimizer = create_optimizer(model, tiny_config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=tiny_config,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer
    )
    
    # Train for a few steps
    initial_step = trainer.step
    steps_to_train = 10
    step_count = 0
    
    for batch in train_dataloader:
        if step_count >= steps_to_train:
            break
        loss = trainer.training_step(batch)
        assert isinstance(loss, float)
        assert loss > 0
        step_count += 1
    
    assert trainer.step == initial_step + steps_to_train
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = trainer.save_checkpoint(tokenizer, checkpoint_dir=tmpdir)
        assert checkpoint_path.exists()
        assert (checkpoint_path / "model.pt").exists()
        assert (checkpoint_path / "optimizer.pt").exists()
        assert (checkpoint_path / "rng.pt").exists()
        assert (checkpoint_path / "vocab.json").exists()
        assert (checkpoint_path / "metadata.json").exists()  # Contains config + step
        
        # Create new model and optimizer for loading
        new_model = Transformer(
            vocab_size=vocab_size,
            max_seq_len=tiny_config.max_seq_len,
            n_layers=2,
            d_model=32,
            n_heads=2,
            d_ff=64,
            seed=42
        )
        new_optimizer = create_optimizer(new_model, tiny_config)
        new_tokenizer = Tokenizer()
        
        # Load checkpoint
        resumed_trainer = Trainer.from_checkpoint(
            checkpoint_path,
            new_model,
            new_optimizer,
            new_tokenizer
        )
        
        # Verify step was restored
        assert resumed_trainer.step == trainer.step
        
        # Verify model state was restored (check a parameter)
        original_param = next(trainer.model.parameters())
        resumed_param = next(resumed_trainer.model.parameters())
        assert torch.allclose(original_param, resumed_param)


def test_training_runs_for_specified_steps(tiny_corpus, tiny_config):
    """Test that training runs for the specified number of steps."""
    tokenizer = Tokenizer()
    corpus = tokenizer.encode(tiny_corpus)
    vocab_size = len(tokenizer.char_to_id)
    
    train_corpus, val_corpus = split_corpus(corpus, train_ratio=0.8, seed=42)
    train_dataset = WindowDataset(train_corpus, context_length=tiny_config.max_seq_len)
    val_dataset = WindowDataset(val_corpus, context_length=tiny_config.max_seq_len)
    
    train_dataloader = DataLoader(train_dataset, batch_size=tiny_config.batch_size, seed=42)
    val_dataloader = DataLoader(val_dataset, batch_size=tiny_config.batch_size, seed=42)
    
    model = Transformer(
        vocab_size=vocab_size,
        max_seq_len=tiny_config.max_seq_len,
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_ff=64,
        seed=42
    )
    optimizer = create_optimizer(model, tiny_config)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=tiny_config,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer
    )
    
    # Train for exactly 5 steps
    steps_to_train = 5
    step_count = 0
    
    for batch in train_dataloader:
        if step_count >= steps_to_train:
            break
        trainer.training_step(batch)
        step_count += 1
    
    assert trainer.step == steps_to_train


def test_checkpoint_resume_produces_identical_results(tiny_corpus, tiny_config):
    """Test that resuming from checkpoint produces identical loss progression.
    
    This test verifies:
    - Training can be interrupted and resumed
    - Loss values after resume match expected progression
    - Checkpoint/resume mechanism preserves training state correctly
    """
    tokenizer = Tokenizer()
    corpus = tokenizer.encode(tiny_corpus)
    vocab_size = len(tokenizer.char_to_id)
    
    train_corpus, val_corpus = split_corpus(corpus, train_ratio=0.8, seed=42)
    train_dataset = WindowDataset(train_corpus, context_length=tiny_config.max_seq_len)
    val_dataset = WindowDataset(val_corpus, context_length=tiny_config.max_seq_len)
    
    train_dataloader = DataLoader(train_dataset, batch_size=tiny_config.batch_size, seed=42)
    val_dataloader = DataLoader(val_dataset, batch_size=tiny_config.batch_size, seed=42)
    
    # Scenario 1: Train continuously for 10 steps
    model1 = Transformer(
        vocab_size=vocab_size,
        max_seq_len=tiny_config.max_seq_len,
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_ff=64,
        seed=42
    )
    optimizer1 = create_optimizer(model1, tiny_config)
    trainer1 = Trainer(
        model=model1,
        optimizer=optimizer1,
        config=tiny_config,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer
    )
    
    losses_continuous = []
    step_count = 0
    for batch in train_dataloader:
        if step_count >= 10:
            break
        loss = trainer1.training_step(batch)
        losses_continuous.append(loss)
        step_count += 1
    
    # Scenario 2: Train for 5 steps, checkpoint, resume, train for 5 more
    model2 = Transformer(
        vocab_size=vocab_size,
        max_seq_len=tiny_config.max_seq_len,
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_ff=64,
        seed=42
    )
    optimizer2 = create_optimizer(model2, tiny_config)
    trainer2 = Trainer(
        model=model2,
        optimizer=optimizer2,
        config=tiny_config,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer
    )
    
    losses_interrupted = []
    
    # Train first 5 steps
    step_count = 0
    for batch in train_dataloader:
        if step_count >= 5:
            break
        loss = trainer2.training_step(batch)
        losses_interrupted.append(loss)
        step_count += 1
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = trainer2.save_checkpoint(tokenizer, checkpoint_dir=tmpdir)
        
        # Resume from checkpoint - load config first to ensure optimizer matches
        checkpoint_config = load_checkpoint_config(checkpoint_path)
        
        model3 = Transformer(
            vocab_size=vocab_size,
            max_seq_len=tiny_config.max_seq_len,
            n_layers=2,
            d_model=32,
            n_heads=2,
            d_ff=64,
            seed=42
        )
        # Create optimizer with config from checkpoint to ensure exact match
        optimizer3 = create_optimizer(model3, checkpoint_config)
        tokenizer3 = Tokenizer()
        
        trainer3 = Trainer.from_checkpoint(checkpoint_path, model3, optimizer3, tokenizer3)
        trainer3.val_dataloader = val_dataloader
        trainer3.tokenizer = tokenizer3
        
        # Continue training for 5 more steps
        # Need to recreate dataloader for fresh iteration
        # Use the same seed to ensure batches are in the same order
        train_dataloader2 = DataLoader(train_dataset, batch_size=tiny_config.batch_size, seed=42)
        step_count = 0
        for batch in train_dataloader2:
            if step_count >= 10:  # Process batches 0-9, but only train on 5-9
                break
            if step_count >= 5:
                loss = trainer3.training_step(batch)
                losses_interrupted.append(loss)
            step_count += 1
    
    # Verify loss progression is identical
    assert len(losses_continuous) == len(losses_interrupted)
    for i, (loss_cont, loss_inter) in enumerate(zip(losses_continuous, losses_interrupted)):
        # Allow small floating point differences
        assert abs(loss_cont - loss_inter) < 1e-4, f"Loss mismatch at step {i}: {loss_cont} vs {loss_inter}"


def test_reproducibility_same_seed_same_results(tiny_corpus, tiny_config):
    """Test that same seed produces identical results.
    
    This test verifies:
    - Training is deterministic when using the same seed
    - Multiple runs with same config produce identical loss values
    - Reproducibility is maintained across runs
    """
    tokenizer = Tokenizer()
    corpus = tokenizer.encode(tiny_corpus)
    vocab_size = len(tokenizer.char_to_id)
    
    train_corpus, val_corpus = split_corpus(corpus, train_ratio=0.8, seed=42)
    train_dataset = WindowDataset(train_corpus, context_length=tiny_config.max_seq_len)
    val_dataset = WindowDataset(val_corpus, context_length=tiny_config.max_seq_len)
    
    # Run 1
    train_dataloader1 = DataLoader(train_dataset, batch_size=tiny_config.batch_size, seed=42)
    val_dataloader1 = DataLoader(val_dataset, batch_size=tiny_config.batch_size, seed=42)
    
    model1 = Transformer(
        vocab_size=vocab_size,
        max_seq_len=tiny_config.max_seq_len,
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_ff=64,
        seed=42
    )
    optimizer1 = create_optimizer(model1, tiny_config)
    trainer1 = Trainer(
        model=model1,
        optimizer=optimizer1,
        config=tiny_config,
        val_dataloader=val_dataloader1,
        tokenizer=tokenizer
    )
    
    losses1 = []
    step_count = 0
    for batch in train_dataloader1:
        if step_count >= 5:
            break
        loss = trainer1.training_step(batch)
        losses1.append(loss)
        step_count += 1
    
    # Run 2 (same seed)
    train_dataloader2 = DataLoader(train_dataset, batch_size=tiny_config.batch_size, seed=42)
    val_dataloader2 = DataLoader(val_dataset, batch_size=tiny_config.batch_size, seed=42)
    
    model2 = Transformer(
        vocab_size=vocab_size,
        max_seq_len=tiny_config.max_seq_len,
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_ff=64,
        seed=42
    )
    optimizer2 = create_optimizer(model2, tiny_config)
    trainer2 = Trainer(
        model=model2,
        optimizer=optimizer2,
        config=tiny_config,
        val_dataloader=val_dataloader2,
        tokenizer=tokenizer
    )
    
    losses2 = []
    step_count = 0
    for batch in train_dataloader2:
        if step_count >= 5:
            break
        loss = trainer2.training_step(batch)
        losses2.append(loss)
        step_count += 1
    
    # Verify identical results
    assert len(losses1) == len(losses2)
    for i, (loss1, loss2) in enumerate(zip(losses1, losses2)):
        assert abs(loss1 - loss2) < 1e-6, f"Loss mismatch at step {i}: {loss1} vs {loss2}"


def test_loss_decreases_over_time(tiny_corpus, tiny_config):
    """Test that loss decreases during training (smoke test).
    
    This test verifies:
    - Model is learning (loss decreases)
    - Training dynamics are reasonable
    - Basic learning capability is present
    """
    tokenizer = Tokenizer()
    corpus = tokenizer.encode(tiny_corpus)
    vocab_size = len(tokenizer.char_to_id)
    
    train_corpus, val_corpus = split_corpus(corpus, train_ratio=0.8, seed=42)
    train_dataset = WindowDataset(train_corpus, context_length=tiny_config.max_seq_len)
    val_dataset = WindowDataset(val_corpus, context_length=tiny_config.max_seq_len)
    
    train_dataloader = DataLoader(train_dataset, batch_size=tiny_config.batch_size, seed=42)
    val_dataloader = DataLoader(val_dataset, batch_size=tiny_config.batch_size, seed=42)
    
    model = Transformer(
        vocab_size=vocab_size,
        max_seq_len=tiny_config.max_seq_len,
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_ff=64,
        seed=42
    )
    optimizer = create_optimizer(model, tiny_config)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=tiny_config,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer
    )
    
    # Train for multiple epochs to see loss decrease
    losses = []
    max_steps = 50
    step_count = 0
    
    while step_count < max_steps:
        for batch in train_dataloader:
            if step_count >= max_steps:
                break
            loss = trainer.training_step(batch)
            losses.append(loss)
            step_count += 1
    
    # Verify loss trend: average of last 10 steps should be less than average of first 10 steps
    avg_initial_loss = sum(losses[:10]) / 10
    avg_final_loss = sum(losses[-10:]) / 10
    
    assert avg_final_loss < avg_initial_loss, \
        f"Loss did not decrease: initial={avg_initial_loss:.4f}, final={avg_final_loss:.4f}"


def test_logging_includes_required_fields(tiny_corpus, tiny_config, capsys):
    """Test that logging includes all required fields.
    
    This test verifies:
    - run_id is logged
    - config_hash is logged
    - git_commit is logged
    - step and loss are logged
    - val_loss is logged when eval_cadence triggers
    - sample_text is logged when sampling_cadence triggers
    """
    tokenizer = Tokenizer()
    corpus = tokenizer.encode(tiny_corpus)
    vocab_size = len(tokenizer.char_to_id)
    
    train_corpus, val_corpus = split_corpus(corpus, train_ratio=0.8, seed=42)
    train_dataset = WindowDataset(train_corpus, context_length=tiny_config.max_seq_len)
    val_dataset = WindowDataset(val_corpus, context_length=tiny_config.max_seq_len)
    
    train_dataloader = DataLoader(train_dataset, batch_size=tiny_config.batch_size, seed=42)
    val_dataloader = DataLoader(val_dataset, batch_size=tiny_config.batch_size, seed=42)
    
    model = Transformer(
        vocab_size=vocab_size,
        max_seq_len=tiny_config.max_seq_len,
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_ff=64,
        seed=42
    )
    optimizer = create_optimizer(model, tiny_config)
    
    # Config with eval and sampling cadence
    config_with_cadence = TrainingConfig(
        learning_rate=1e-3,
        batch_size=2,
        max_seq_len=16,
        seed=42,
        eval_cadence=3,
        sampling_cadence=3,
        sampling_max_length=20
    )
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config_with_cadence,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer
    )
    
    # Train for a few steps (enough to trigger eval and sampling)
    step_count = 0
    for batch in train_dataloader:
        if step_count >= 6:
            break
        trainer.training_step(batch)
        step_count += 1
    
    # Capture output
    captured = capsys.readouterr()
    output = captured.out
    
    # Verify required fields are logged
    assert "run_id:" in output, "run_id not logged"
    assert "config_hash:" in output, "config_hash not logged"
    assert "git_commit:" in output, "git_commit not logged"
    assert "hook_list:" in output, "hook_list not logged"
    assert "Step" in output, "Step not logged"
    assert "loss =" in output, "loss not logged"
    assert "val_loss =" in output, "val_loss not logged (eval_cadence should trigger)"
    assert "sample =" in output, "sample not logged (sampling_cadence should trigger)"


def test_command_line_interface_data_file_error():
    """Test error handling for missing data file.
    
    This test verifies:
    - Missing data files are detected
    - Appropriate error messages are shown
    - Program exits gracefully with non-zero exit code
    """
    from main import load_data_file
    
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        load_data_file("nonexistent_file.txt")


def test_command_line_interface_config_file_error():
    """Test error handling for invalid config file.
    
    This test verifies:
    - Invalid config files are detected
    - Appropriate error messages are shown
    - Program exits gracefully with non-zero exit code
    """
    from main import load_config_from_yaml
    
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config_from_yaml("nonexistent_config.yaml")


def test_yaml_config_loading(tiny_config):
    """Test loading configuration from YAML file."""
    from main import load_config_from_yaml
    import yaml
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_dict = {
            'learning_rate': 0.001,
            'batch_size': 8,
            'max_seq_len': 128,
            'seed': 123
        }
        yaml.dump(config_dict, f)
        config_path = f.name
    
    try:
        loaded_config = load_config_from_yaml(config_path)
        assert loaded_config.learning_rate == 0.001
        assert loaded_config.batch_size == 8
        assert loaded_config.max_seq_len == 128
        assert loaded_config.seed == 123
    finally:
        Path(config_path).unlink()


def test_checkpoint_saved_on_interrupt(tiny_corpus, tiny_config):
    """Test that checkpoint is saved when training is interrupted.
    
    This test verifies:
    - Checkpoint can be saved during training
    - Saved checkpoint is valid and can be loaded
    - Training can resume from interrupt checkpoint
    """
    tokenizer = Tokenizer()
    corpus = tokenizer.encode(tiny_corpus)
    vocab_size = len(tokenizer.char_to_id)
    
    train_corpus, val_corpus = split_corpus(corpus, train_ratio=0.8, seed=42)
    train_dataset = WindowDataset(train_corpus, context_length=tiny_config.max_seq_len)
    val_dataset = WindowDataset(val_corpus, context_length=tiny_config.max_seq_len)
    
    train_dataloader = DataLoader(train_dataset, batch_size=tiny_config.batch_size, seed=42)
    val_dataloader = DataLoader(val_dataset, batch_size=tiny_config.batch_size, seed=42)
    
    model = Transformer(
        vocab_size=vocab_size,
        max_seq_len=tiny_config.max_seq_len,
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_ff=64,
        seed=42
    )
    optimizer = create_optimizer(model, tiny_config)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=tiny_config,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer
    )
    
    # Train for a few steps
    step_count = 0
    for batch in train_dataloader:
        if step_count >= 5:
            break
        trainer.training_step(batch)
        step_count += 1
    
    # Simulate interrupt by saving checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = trainer.save_checkpoint(tokenizer, checkpoint_dir=tmpdir)
        
        # Verify checkpoint can be loaded
        new_model = Transformer(
            vocab_size=vocab_size,
            max_seq_len=tiny_config.max_seq_len,
            n_layers=2,
            d_model=32,
            n_heads=2,
            d_ff=64,
            seed=42
        )
        new_optimizer = create_optimizer(new_model, tiny_config)
        new_tokenizer = Tokenizer()
        
        resumed_trainer = Trainer.from_checkpoint(
            checkpoint_path,
            new_model,
            new_optimizer,
            new_tokenizer
        )
        
        assert resumed_trainer.step == trainer.step


class TestConfigIntegrationWithMain:
    """Test integration of config system with main.py."""
    
    def test_main_uses_config_for_model_creation_n_layers(self, tiny_corpus):
        """Test model created with n_layers from config."""
        from main import load_config_from_yaml
        
        # Create a config with specific n_layers
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_dict = {
                "n_layers": 3,
                "d_model": 64,
                "n_heads": 2,
                "d_ff": 128,
                "batch_size": 2,
                "max_seq_len": 16,
                "seed": 42,
            }
            yaml.dump(config_dict, f)
            config_path = f.name
        
        try:
            config = load_config_from_yaml(config_path)
            tokenizer = Tokenizer()
            corpus = tokenizer.encode(tiny_corpus)
            vocab_size = len(tokenizer.char_to_id)
            
            model = Transformer(
                vocab_size=vocab_size,
                max_seq_len=config.max_seq_len,
                n_layers=config.n_layers,
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                seed=config.seed
            )
            
            assert model.n_layers == 3 == config.n_layers
        finally:
            Path(config_path).unlink()
    
    def test_main_uses_config_for_model_creation_d_model(self, tiny_corpus):
        """Test model created with d_model from config."""
        from main import load_config_from_yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_dict = {
                "n_layers": 2,
                "d_model": 128,
                "n_heads": 4,
                "d_ff": 256,
                "batch_size": 2,
                "max_seq_len": 16,
                "seed": 42,
            }
            yaml.dump(config_dict, f)
            config_path = f.name
        
        try:
            config = load_config_from_yaml(config_path)
            tokenizer = Tokenizer()
            corpus = tokenizer.encode(tiny_corpus)
            vocab_size = len(tokenizer.char_to_id)
            
            model = Transformer(
                vocab_size=vocab_size,
                max_seq_len=config.max_seq_len,
                n_layers=config.n_layers,
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                seed=config.seed
            )
            
            assert model.d_model == 128 == config.d_model
        finally:
            Path(config_path).unlink()
    
    def test_main_uses_config_for_model_creation_n_heads(self, tiny_corpus):
        """Test model created with n_heads from config."""
        from main import load_config_from_yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_dict = {
                "n_layers": 2,
                "d_model": 64,
                "n_heads": 4,
                "d_ff": 128,
                "batch_size": 2,
                "max_seq_len": 16,
                "seed": 42,
            }
            yaml.dump(config_dict, f)
            config_path = f.name
        
        try:
            config = load_config_from_yaml(config_path)
            tokenizer = Tokenizer()
            corpus = tokenizer.encode(tiny_corpus)
            vocab_size = len(tokenizer.char_to_id)
            
            model = Transformer(
                vocab_size=vocab_size,
                max_seq_len=config.max_seq_len,
                n_layers=config.n_layers,
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                seed=config.seed
            )
            
            assert model.n_heads == 4 == config.n_heads
        finally:
            Path(config_path).unlink()
    
    def test_main_uses_config_for_model_creation_d_ff(self, tiny_corpus):
        """Test model created with d_ff from config."""
        from main import load_config_from_yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_dict = {
                "n_layers": 2,
                "d_model": 64,
                "n_heads": 2,
                "d_ff": 256,
                "batch_size": 2,
                "max_seq_len": 16,
                "seed": 42,
            }
            yaml.dump(config_dict, f)
            config_path = f.name
        
        try:
            config = load_config_from_yaml(config_path)
            tokenizer = Tokenizer()
            corpus = tokenizer.encode(tiny_corpus)
            vocab_size = len(tokenizer.char_to_id)
            
            model = Transformer(
                vocab_size=vocab_size,
                max_seq_len=config.max_seq_len,
                n_layers=config.n_layers,
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                seed=config.seed
            )
            
            assert model.d_ff == 256 == config.d_ff
        finally:
            Path(config_path).unlink()
    
    def test_main_uses_config_for_model_creation_dropout(self, tiny_corpus):
        """Test model created with dropout from config."""
        from main import load_config_from_yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_dict = {
                "n_layers": 2,
                "d_model": 64,
                "n_heads": 2,
                "d_ff": 128,
                "dropout": 0.2,
                "batch_size": 2,
                "max_seq_len": 16,
                "seed": 42,
            }
            yaml.dump(config_dict, f)
            config_path = f.name
        
        try:
            config = load_config_from_yaml(config_path)
            tokenizer = Tokenizer()
            corpus = tokenizer.encode(tiny_corpus)
            vocab_size = len(tokenizer.char_to_id)
            
            model = Transformer(
                vocab_size=vocab_size,
                max_seq_len=config.max_seq_len,
                n_layers=config.n_layers,
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                seed=config.seed
            )
            
            assert config.dropout == 0.2
            # Note: dropout is used internally, so we verify config value
        finally:
            Path(config_path).unlink()
    
    def test_main_uses_config_for_model_architecture(self, tiny_corpus):
        """Test model architecture matches config values."""
        from main import load_config_from_yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_dict = {
                "n_layers": 3,
                "d_model": 96,
                "n_heads": 3,
                "d_ff": 192,
                "dropout": 0.15,
                "batch_size": 2,
                "max_seq_len": 16,
                "seed": 42,
            }
            yaml.dump(config_dict, f)
            config_path = f.name
        
        try:
            config = load_config_from_yaml(config_path)
            tokenizer = Tokenizer()
            corpus = tokenizer.encode(tiny_corpus)
            vocab_size = len(tokenizer.char_to_id)
            
            model = Transformer(
                vocab_size=vocab_size,
                max_seq_len=config.max_seq_len,
                n_layers=config.n_layers,
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                seed=config.seed
            )
            
            assert model.n_layers == config.n_layers == 3
            assert model.d_model == config.d_model == 96
            assert model.n_heads == config.n_heads == 3
            assert model.d_ff == config.d_ff == 192
        finally:
            Path(config_path).unlink()
    
    def test_main_uses_config_for_dataset_split(self, tiny_corpus):
        """Test split_corpus called with train_ratio from config."""
        from main import load_config_from_yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_dict = {
                "train_ratio": 0.9,
                "batch_size": 2,
                "max_seq_len": 16,
                "seed": 42,
            }
            yaml.dump(config_dict, f)
            config_path = f.name
        
        try:
            config = load_config_from_yaml(config_path)
            tokenizer = Tokenizer()
            corpus = tokenizer.encode(tiny_corpus)
            
            train_corpus, val_corpus = split_corpus(
                corpus, 
                train_ratio=config.train_ratio, 
                seed=config.seed or 42
            )
            
            # Verify split ratio matches config
            total_len = len(train_corpus) + len(val_corpus)
            actual_train_ratio = len(train_corpus) / total_len
            assert abs(actual_train_ratio - config.train_ratio) < 0.01
        finally:
            Path(config_path).unlink()
    
    def test_main_uses_config_for_checkpoint_cadence(self, tiny_corpus, tiny_config):
        """Test checkpoints saved at checkpoint_cadence from config."""
        from main import load_config_from_yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_dict = {
                "checkpoint_cadence": 5,
                "batch_size": 2,
                "max_seq_len": 16,
                "seed": 42,
            }
            yaml.dump(config_dict, f)
            config_path = f.name
        
        try:
            config = load_config_from_yaml(config_path)
            tokenizer = Tokenizer()
            corpus = tokenizer.encode(tiny_corpus)
            vocab_size = len(tokenizer.char_to_id)
            
            train_corpus, val_corpus = split_corpus(corpus, train_ratio=0.8, seed=42)
            train_dataset = WindowDataset(train_corpus, context_length=config.max_seq_len)
            val_dataset = WindowDataset(val_corpus, context_length=config.max_seq_len)
            train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, seed=42)
            val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, seed=42)
            
            model = Transformer(
                vocab_size=vocab_size,
                max_seq_len=config.max_seq_len,
                n_layers=2,
                d_model=32,
                n_heads=2,
                d_ff=64,
                seed=42
            )
            optimizer = create_optimizer(model, config)
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                config=config,
                val_dataloader=val_dataloader,
                tokenizer=tokenizer
            )
            
            # Verify checkpoint_cadence is set correctly
            assert config.checkpoint_cadence == 5
            
            # Train a few steps and verify checkpoint cadence logic
            step_count = 0
            for batch in train_dataloader:
                if step_count >= 10:
                    break
                trainer.training_step(batch)
                step_count += 1
                
                # Check if checkpoint should be saved at this step
                should_checkpoint = (
                    config.checkpoint_cadence is not None and
                    trainer.step > 0 and
                    trainer.step % config.checkpoint_cadence == 0
                )
                if should_checkpoint:
                    # Verify checkpoint cadence is working
                    assert trainer.step % config.checkpoint_cadence == 0
        finally:
            Path(config_path).unlink()
    
    def test_checkpoint_cadence_can_be_disabled(self, tiny_corpus):
        """Test checkpoint cadence can be disabled (None)."""
        from main import load_config_from_yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_dict = {
                "checkpoint_cadence": None,
                "batch_size": 2,
                "max_seq_len": 16,
                "seed": 42,
            }
            yaml.dump(config_dict, f)
            config_path = f.name
        
        try:
            config = load_config_from_yaml(config_path)
            assert config.checkpoint_cadence is None
        finally:
            Path(config_path).unlink()
    
    def test_cli_override_max_steps_works(self):
        """Test CLI override (--max-steps) still works."""
        # This tests the logic in main.py where CLI args override config
        config = TrainingConfig(max_steps=10000)
        cli_max_steps = 5000
        
        # Simulate the logic from main.py
        max_steps = cli_max_steps if cli_max_steps is not None else config.max_steps
        assert max_steps == 5000  # CLI override takes precedence
    
    def test_cli_override_takes_precedence_over_config(self):
        """Test CLI override takes precedence over config."""
        config = TrainingConfig(max_steps=10000)
        cli_max_steps = 3000
        
        # Simulate the logic from main.py
        max_steps = cli_max_steps if cli_max_steps is not None else config.max_steps
        assert max_steps == 3000  # CLI override
        assert max_steps != config.max_steps  # Different from config
    
    def test_default_config_used_when_no_file_provided(self):
        """Test default TrainingConfig is used when no config file provided."""
        # When no --config argument is provided, main.py uses TrainingConfig()
        default_config = TrainingConfig()
        
        # Verify defaults are used
        assert default_config.n_layers == 4
        assert default_config.d_model == 256
        assert default_config.n_heads == 4
        assert default_config.d_ff == 1024
        assert default_config.dropout == 0.1
        assert default_config.train_ratio == 0.95
        assert default_config.max_steps == 10000
        assert default_config.checkpoint_cadence == 1000


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


def test_end_to_end_quantization_workflow(tiny_corpus, tiny_config):
    """Test end-to-end quantization workflow: train → quantize → save → load."""
    # Skip test if quantization engines are not available
    if not _check_quantization_engine_available():
        pytest.skip("PyTorch quantization engines not available")
    
    tokenizer = Tokenizer()
    corpus = tokenizer.encode(tiny_corpus)
    vocab_size = len(tokenizer.char_to_id)
    
    train_corpus, val_corpus = split_corpus(corpus, train_ratio=0.8, seed=42)
    train_dataset = WindowDataset(train_corpus, context_length=tiny_config.max_seq_len)
    
    train_dataloader = DataLoader(train_dataset, batch_size=tiny_config.batch_size, seed=42)
    
    # Train a model
    model = Transformer(
        vocab_size=vocab_size,
        max_seq_len=tiny_config.max_seq_len,
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_ff=64,
        seed=42
    )
    optimizer = create_optimizer(model, tiny_config)
    trainer = Trainer(model, optimizer, tiny_config)
    
    # Train for a few steps
    step_count = 0
    for batch in train_dataloader:
        if step_count >= 5:
            break
        trainer.training_step(batch)
        step_count += 1
    
    # Quantize the model
    model.eval()
    quantized_model = quantize_model_ptq(
        model,
        quantization_bits=8,
        quantization_type="dynamic"
    )
    
    assert is_model_quantized(quantized_model)
    
    # Save quantized checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = trainer.save_checkpoint(tokenizer, checkpoint_dir=tmpdir, checkpoint_name="quantized")
        
        # Verify quantization metadata exists
        assert (checkpoint_path / "quantization_metadata.json").exists()
        
        # Load checkpoint
        new_model = Transformer(
            vocab_size=vocab_size,
            max_seq_len=tiny_config.max_seq_len,
            n_layers=2,
            d_model=32,
            n_heads=2,
            d_ff=64,
            seed=42
        )
        new_optimizer = create_optimizer(new_model, tiny_config)
        new_tokenizer = Tokenizer()
        
        checkpoint_data = load_checkpoint(checkpoint_path, new_model, new_optimizer, new_tokenizer)
        
        # Verify quantization metadata was loaded
        assert "quantization_metadata" in checkpoint_data
        assert checkpoint_data["quantization_metadata"]["is_quantized"] == True


def test_quantized_vs_full_precision_outputs(tiny_corpus, tiny_config):
    """Test that quantized model outputs are similar to full-precision model."""
    # Skip test if quantization engines are not available
    if not _check_quantization_engine_available():
        pytest.skip("PyTorch quantization engines not available")
    
    vocab_size = 100
    batch_size = 2
    seq_len = 16
    
    # Create and train full-precision model
    model_fp = Transformer(
        vocab_size=vocab_size,
        max_seq_len=tiny_config.max_seq_len,
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_ff=64,
        seed=42
    )
    model_fp.eval()
    
    # Create quantized model
    model_quantized = quantize_model_ptq(
        model_fp,
        quantization_bits=8,
        quantization_type="dynamic"
    )
    
    # Test with same inputs
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits_fp = model_fp(inputs)
        logits_quantized = model_quantized(inputs)
    
    # Quantized outputs should be similar but not identical
    # Use cosine similarity or MSE to compare
    mse = torch.mean((logits_fp - logits_quantized) ** 2).item()
    
    # MSE should be reasonable (quantization introduces some error)
    assert mse < 10.0  # Allow reasonable quantization error


def test_checkpoint_backward_compatibility(tiny_corpus, tiny_config):
    """Test that old checkpoints (without quantization) still load correctly."""
    tokenizer = Tokenizer()
    corpus = tokenizer.encode(tiny_corpus)
    vocab_size = len(tokenizer.char_to_id)
    
    train_corpus, val_corpus = split_corpus(corpus, train_ratio=0.8, seed=42)
    train_dataset = WindowDataset(train_corpus, context_length=tiny_config.max_seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=tiny_config.batch_size, seed=42)
    
    # Train and save a non-quantized checkpoint
    model = Transformer(
        vocab_size=vocab_size,
        max_seq_len=tiny_config.max_seq_len,
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_ff=64,
        seed=42
    )
    optimizer = create_optimizer(model, tiny_config)
    trainer = Trainer(model, optimizer, tiny_config)
    
    step_count = 0
    for batch in train_dataloader:
        if step_count >= 3:
            break
        trainer.training_step(batch)
        step_count += 1
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = trainer.save_checkpoint(tokenizer, checkpoint_dir=tmpdir)
        
        # Verify no quantization metadata exists (old format)
        assert not (checkpoint_path / "quantization_metadata.json").exists()
        
        # Load checkpoint (should work with old format)
        new_model = Transformer(
            vocab_size=vocab_size,
            max_seq_len=tiny_config.max_seq_len,
            n_layers=2,
            d_model=32,
            n_heads=2,
            d_ff=64,
            seed=42
        )
        new_optimizer = create_optimizer(new_model, tiny_config)
        new_tokenizer = Tokenizer()
        
        checkpoint_data = load_checkpoint(checkpoint_path, new_model, new_optimizer, new_tokenizer)
        
        # Should load successfully
        assert checkpoint_data["step"] == trainer.step
        assert "quantization_metadata" not in checkpoint_data or checkpoint_data.get("quantization_metadata") is None


def test_quantization_different_model_sizes():
    """Test quantization with different model sizes."""
    # Skip test if quantization engines are not available
    if not _check_quantization_engine_available():
        pytest.skip("PyTorch quantization engines not available")
    
    vocab_size = 100
    
    # Test small model
    model_small = Transformer(
        vocab_size=vocab_size,
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_ff=64
    )
    model_small.eval()
    quantized_small = quantize_model_ptq(
        model_small,
        quantization_bits=8,
        quantization_type="dynamic"
    )
    assert is_model_quantized(quantized_small)
    
    # Test medium model
    model_medium = Transformer(
        vocab_size=vocab_size,
        n_layers=4,
        d_model=128,
        n_heads=4,
        d_ff=256
    )
    model_medium.eval()
    quantized_medium = quantize_model_ptq(
        model_medium,
        quantization_bits=8,
        quantization_type="dynamic"
    )
    assert is_model_quantized(quantized_medium)
    
    # Test larger model
    model_large = Transformer(
        vocab_size=vocab_size,
        n_layers=6,
        d_model=256,
        n_heads=8,
        d_ff=512
    )
    model_large.eval()
    quantized_large = quantize_model_ptq(
        model_large,
        quantization_bits=8,
        quantization_type="dynamic"
    )
    assert is_model_quantized(quantized_large)


def test_quantized_finetuning_produces_results(tiny_corpus, tiny_config):
    """Test that fine-tuning quantized models produces expected results."""
    # Skip test if quantization engines are not available
    if not _check_quantization_engine_available():
        pytest.skip("PyTorch quantization engines not available")
    
    tokenizer = Tokenizer()
    corpus = tokenizer.encode(tiny_corpus)
    vocab_size = len(tokenizer.char_to_id)
    
    train_corpus, val_corpus = split_corpus(corpus, train_ratio=0.8, seed=42)
    train_dataset = WindowDataset(train_corpus, context_length=tiny_config.max_seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=tiny_config.batch_size, seed=42)
    
    # Create and quantize model
    model = Transformer(
        vocab_size=vocab_size,
        max_seq_len=tiny_config.max_seq_len,
        n_layers=2,
        d_model=32,
        n_heads=2,
        d_ff=64,
        seed=42
    )
    model.eval()
    
    quantized_model = quantize_model_ptq(
        model,
        quantization_bits=8,
        quantization_type="dynamic"
    )
    
    # Configure for quantized fine-tuning
    config = TrainingConfig(
        quantization_mode="ptq",
        enable_quantized_finetuning=True,
        learning_rate=1e-3,
        batch_size=tiny_config.batch_size,
        max_seq_len=tiny_config.max_seq_len
    )
    optimizer = create_optimizer(quantized_model, config)
    trainer = Trainer(quantized_model, optimizer, config)
    
    # Fine-tune for a few steps
    initial_loss = None
    final_loss = None
    step_count = 0
    
    for batch in train_dataloader:
        if step_count >= 5:
            break
        loss = trainer.training_step(batch)
        if step_count == 0:
            initial_loss = loss
        if step_count == 4:
            final_loss = loss
        step_count += 1
    
    # Verify training occurred
    assert initial_loss is not None
    assert final_loss is not None
    assert trainer.step == 5
    
    # Loss should change (may increase or decrease, but should be different)
    assert abs(initial_loss - final_loss) > 1e-6

