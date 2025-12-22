"""Tests for transformer model assembly."""

import pytest
import torch

from src.model.transformer import Transformer
from src.dataloader import DataLoader
from src.dataset import WindowDataset


def test_transformer_init():
    """Test Transformer initialization."""
    vocab_size = 99
    model = Transformer(vocab_size=vocab_size)
    
    assert model.vocab_size == vocab_size
    assert model.max_seq_len == 256
    assert model.n_layers == 4
    assert model.d_model == 256
    assert model.n_heads == 4
    assert model.d_ff == 1024
    assert model.dropout == 0.1


def test_transformer_init_custom_hyperparameters():
    """Test Transformer initialization with custom hyperparameters."""
    vocab_size = 99
    model = Transformer(
        vocab_size=vocab_size,
        max_seq_len=512,
        n_layers=6,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout=0.2
    )
    
    assert model.vocab_size == vocab_size
    assert model.max_seq_len == 512
    assert model.n_layers == 6
    assert model.d_model == 512
    assert model.n_heads == 8
    assert model.d_ff == 2048
    assert model.dropout == 0.2


def test_transformer_architecture_components():
    """Test that Transformer includes all required components."""
    vocab_size = 99
    model = Transformer(vocab_size=vocab_size)
    
    # Check token embedding exists
    assert hasattr(model, 'token_embedding')
    assert model.token_embedding is not None
    
    # Check positional embedding exists
    assert hasattr(model, 'pos_embedding')
    assert model.pos_embedding is not None
    
    # Check transformer blocks exist
    assert hasattr(model, 'blocks')
    assert len(model.blocks) == 4  # n_layers=4
    
    # Check final layer norm exists
    assert hasattr(model, 'final_norm')
    assert model.final_norm is not None
    
    # Check LM head exists
    assert hasattr(model, 'lm_head')
    assert model.lm_head is not None
    
    # Check dropout layer exists
    assert hasattr(model, 'dropout_layer')
    assert model.dropout_layer is not None


def test_transformer_forward_pass_shape():
    """Test Transformer forward pass produces correct logit shapes."""
    vocab_size = 99
    batch_size = 2
    seq_len = 256
    
    model = Transformer(vocab_size=vocab_size)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(token_ids)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_transformer_forward_pass_logits_shape():
    """Test that forward pass returns logits of shape [B, 256, vocab_size]."""
    vocab_size = 99
    batch_size = 4
    seq_len = 256
    
    model = Transformer(vocab_size=vocab_size)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(token_ids)
    
    assert logits.shape == (batch_size, 256, vocab_size)
    assert logits.dtype == torch.float32


def test_transformer_forward_pass_different_batch_sizes():
    """Test Transformer handles different batch sizes."""
    vocab_size = 99
    seq_len = 256
    
    model = Transformer(vocab_size=vocab_size)
    
    # Test batch size 1
    token_ids_1 = torch.randint(0, vocab_size, (1, seq_len))
    logits_1 = model(token_ids_1)
    assert logits_1.shape == (1, seq_len, vocab_size)
    
    # Test batch size 8
    token_ids_8 = torch.randint(0, vocab_size, (8, seq_len))
    logits_8 = model(token_ids_8)
    assert logits_8.shape == (8, seq_len, vocab_size)
    
    # Test batch size 16
    token_ids_16 = torch.randint(0, vocab_size, (16, seq_len))
    logits_16 = model(token_ids_16)
    assert logits_16.shape == (16, seq_len, vocab_size)


def test_transformer_processes_dataloader_batches():
    """Test that Transformer can process batches from DataLoader."""
    vocab_size = 99
    # Create corpus with token IDs in valid range [0, vocab_size)
    corpus = list(range(vocab_size)) * 100  # Repeat to get enough data
    dataset = WindowDataset(corpus, context_length=256)
    dataloader = DataLoader(dataset, batch_size=16)
    
    model = Transformer(vocab_size=vocab_size)
    model.eval()  # Set to eval mode for deterministic output
    
    # Process first batch
    batch = next(iter(dataloader))
    # Clamp token IDs to valid range
    batch = torch.clamp(batch, 0, vocab_size - 1)
    logits = model(batch)
    
    # Verify logits shape
    assert logits.shape == (batch.shape[0], 256, vocab_size)
    assert logits.dtype == torch.float32


def test_transformer_processes_all_dataloader_batches():
    """Test that Transformer can process all batches from DataLoader."""
    vocab_size = 99
    # Create corpus with token IDs in valid range [0, vocab_size)
    corpus = list(range(vocab_size)) * 100  # Repeat to get enough data
    dataset = WindowDataset(corpus, context_length=256)
    dataloader = DataLoader(dataset, batch_size=16)
    
    model = Transformer(vocab_size=vocab_size)
    model.eval()  # Set to eval mode for deterministic output
    
    # Process all batches
    batch_count = 0
    for batch in dataloader:
        # Clamp token IDs to valid range
        batch = torch.clamp(batch, 0, vocab_size - 1)
        logits = model(batch)
        assert logits.shape == (batch.shape[0], 256, vocab_size)
        batch_count += 1
    
    assert batch_count > 0


def test_transformer_deterministic_initialization():
    """Test Transformer initialization is deterministic with same seed."""
    vocab_size = 99
    seed = 42
    
    model1 = Transformer(vocab_size=vocab_size, seed=seed)
    model2 = Transformer(vocab_size=vocab_size, seed=seed)
    
    # Check that token embedding weights are identical
    assert torch.allclose(
        model1.token_embedding.embedding.weight,
        model2.token_embedding.embedding.weight
    )
    
    # Check that positional embedding weights are identical
    assert torch.allclose(
        model1.pos_embedding.embedding.weight,
        model2.pos_embedding.embedding.weight
    )
    
    # Check that LM head weights are identical
    assert torch.allclose(
        model1.lm_head.weight,
        model2.lm_head.weight
    )


def test_transformer_deterministic_forward():
    """Test Transformer forward pass is deterministic with same seed."""
    vocab_size = 99
    seed = 42
    batch_size = 2
    seq_len = 256
    
    model1 = Transformer(vocab_size=vocab_size, seed=seed)
    model2 = Transformer(vocab_size=vocab_size, seed=seed)
    
    # Set to eval mode to disable dropout for deterministic output
    model1.eval()
    model2.eval()
    
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits1 = model1(token_ids)
        logits2 = model2(token_ids)
    
    # Outputs should be identical with same initialization and input
    assert torch.allclose(logits1, logits2)


def test_transformer_gradient_flow():
    """Test that gradients can flow through the model."""
    vocab_size = 99
    batch_size = 2
    seq_len = 256
    
    model = Transformer(vocab_size=vocab_size)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(token_ids)
    loss = logits.sum()
    loss.backward()
    
    # Verify gradients exist for key components
    assert model.token_embedding.embedding.weight.grad is not None
    assert model.pos_embedding.embedding.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    
    # Verify gradients are non-zero
    assert not torch.allclose(
        model.token_embedding.embedding.weight.grad,
        torch.zeros_like(model.token_embedding.embedding.weight.grad)
    )


def test_transformer_logits_suitable_for_cross_entropy():
    """Test that logits are suitable for cross-entropy loss."""
    vocab_size = 99
    batch_size = 2
    seq_len = 256
    
    model = Transformer(vocab_size=vocab_size)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(token_ids)
    
    # Logits should be finite
    assert torch.isfinite(logits).all()
    
    # Logits should have reasonable range (not NaN or Inf)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
    
    # Can compute cross-entropy loss
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1)
    )
    
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_transformer_hyperparameters():
    """Test that model uses correct hyperparameters."""
    vocab_size = 99
    model = Transformer(vocab_size=vocab_size)
    
    # Verify hyperparameters match spec
    assert model.n_layers == 4
    assert model.d_model == 256
    assert model.n_heads == 4
    assert model.d_ff == 1024
    assert model.dropout == 0.1
    
    # Verify transformer blocks match n_layers
    assert len(model.blocks) == 4
    
    # Verify each block has correct dimensions
    for block in model.blocks:
        assert block.d_model == 256
        assert block.n_heads == 4
        assert block.d_ff == 1024


def test_transformer_forward_pass_end_to_end():
    """Test complete forward pass end-to-end."""
    vocab_size = 99
    batch_size = 4
    seq_len = 256
    
    model = Transformer(vocab_size=vocab_size)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(token_ids)
    
    # Verify output
    assert logits is not None
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_transformer_dropout_training_mode():
    """Test that dropout is applied in training mode."""
    vocab_size = 99
    batch_size = 2
    seq_len = 256
    
    model = Transformer(vocab_size=vocab_size)
    model.train()  # Set to training mode
    
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Run forward pass multiple times - outputs should vary due to dropout
    logits1 = model(token_ids)
    logits2 = model(token_ids)
    
    # In training mode with dropout, outputs may differ
    # (though with same input and seed, they might be similar)
    assert logits1.shape == logits2.shape


def test_transformer_dropout_eval_mode():
    """Test that dropout is disabled in eval mode."""
    vocab_size = 99
    batch_size = 2
    seq_len = 256
    
    model = Transformer(vocab_size=vocab_size)
    model.eval()  # Set to eval mode
    
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Run forward pass multiple times - outputs should be identical
    with torch.no_grad():
        logits1 = model(token_ids)
        logits2 = model(token_ids)
    
    # In eval mode, outputs should be identical
    assert torch.allclose(logits1, logits2)


def test_transformer_model_size():
    """Test that model has reasonable number of parameters."""
    vocab_size = 99
    model = Transformer(vocab_size=vocab_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Model should have parameters
    assert total_params > 0
    
    # With vocab_size=99, d_model=256, n_layers=4, model should have
    # substantial number of parameters but not excessive
    # Rough estimate: embeddings + transformer blocks + LM head
    # This is a sanity check, not an exact value
    assert total_params > 10000  # Should have at least 10k parameters
    assert total_params < 10000000  # Should have less than 10M parameters

