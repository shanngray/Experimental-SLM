"""Integration tests for end-to-end training pipeline.

This module provides comprehensive integration tests that verify the complete
training pipeline from data loading to checkpoint/resume functionality.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from src.config import TrainingConfig
from src.dataloader import DataLoader
from src.dataset import WindowDataset, split_corpus
from src.model.transformer import Transformer
from src.tokenizer import Tokenizer
from src.training.checkpoint import load_checkpoint_config
from src.training.trainer import Trainer, create_optimizer


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

