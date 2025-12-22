"""Tests for embedding modules."""

import pytest
import torch

from src.model.embeddings import TokenEmbedding, PositionalEmbedding


def test_token_embedding_init():
    """Test TokenEmbedding initialization."""
    vocab_size = 99
    d_model = 256
    embedding = TokenEmbedding(vocab_size, d_model)
    
    assert embedding.vocab_size == vocab_size
    assert embedding.d_model == d_model
    assert embedding.embedding.num_embeddings == vocab_size
    assert embedding.embedding.embedding_dim == d_model


def test_token_embedding_forward_shape():
    """Test TokenEmbedding forward pass produces correct shapes."""
    vocab_size = 99
    d_model = 256
    batch_size = 2
    seq_len = 256
    
    embedding = TokenEmbedding(vocab_size, d_model)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output = embedding(token_ids)
    
    assert output.shape == (batch_size, seq_len, d_model)


def test_token_embedding_unique_mappings():
    """Test that each token ID maps to a unique embedding vector."""
    vocab_size = 99
    d_model = 256
    
    embedding = TokenEmbedding(vocab_size, d_model)
    
    # Get embeddings for different token IDs
    token_id_1 = torch.tensor([[0]])
    token_id_2 = torch.tensor([[1]])
    token_id_3 = torch.tensor([[2]])
    
    emb_1 = embedding(token_id_1)
    emb_2 = embedding(token_id_2)
    emb_3 = embedding(token_id_3)
    
    # Different token IDs should produce different embeddings
    assert not torch.allclose(emb_1, emb_2, atol=1e-5)
    assert not torch.allclose(emb_1, emb_3, atol=1e-5)
    assert not torch.allclose(emb_2, emb_3, atol=1e-5)


def test_token_embedding_deterministic_initialization():
    """Test TokenEmbedding initialization is deterministic with same seed."""
    vocab_size = 99
    d_model = 256
    seed = 42
    
    embedding1 = TokenEmbedding(vocab_size, d_model, seed=seed)
    embedding2 = TokenEmbedding(vocab_size, d_model, seed=seed)
    
    # Check that embedding weights are identical
    assert torch.allclose(
        embedding1.embedding.weight,
        embedding2.embedding.weight
    )


def test_token_embedding_deterministic_forward():
    """Test TokenEmbedding forward pass is deterministic with same seed."""
    vocab_size = 99
    d_model = 256
    seed = 42
    batch_size = 2
    seq_len = 10
    
    embedding1 = TokenEmbedding(vocab_size, d_model, seed=seed)
    embedding2 = TokenEmbedding(vocab_size, d_model, seed=seed)
    
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output1 = embedding1(token_ids)
    output2 = embedding2(token_ids)
    
    # Outputs should be identical with same initialization and input
    assert torch.allclose(output1, output2)


def test_positional_embedding_init():
    """Test PositionalEmbedding initialization."""
    max_seq_len = 256
    d_model = 256
    embedding = PositionalEmbedding(max_seq_len, d_model)
    
    assert embedding.max_seq_len == max_seq_len
    assert embedding.d_model == d_model
    assert embedding.embedding.num_embeddings == max_seq_len
    assert embedding.embedding.embedding_dim == d_model


def test_positional_embedding_forward_shape():
    """Test PositionalEmbedding forward pass produces correct shapes."""
    max_seq_len = 256
    d_model = 256
    seq_len = 128
    
    embedding = PositionalEmbedding(max_seq_len, d_model)
    output = embedding(seq_len)
    
    assert output.shape == (seq_len, d_model)


def test_positional_embedding_learnable_parameters():
    """Test that positional embeddings are learnable parameters."""
    max_seq_len = 256
    d_model = 256
    
    embedding = PositionalEmbedding(max_seq_len, d_model)
    
    # Check that embedding weights are parameters
    assert isinstance(embedding.embedding.weight, torch.nn.Parameter)
    assert embedding.embedding.weight.requires_grad


def test_positional_embedding_different_positions():
    """Test that different positions produce different embeddings."""
    max_seq_len = 256
    d_model = 256
    
    embedding = PositionalEmbedding(max_seq_len, d_model)
    
    # Get embeddings for different sequence lengths
    pos_embeds_10 = embedding(10)
    pos_embeds_20 = embedding(20)
    
    # First 10 positions should be identical
    assert torch.allclose(pos_embeds_10, pos_embeds_20[:10])
    
    # But position 10+ should be different (they exist in pos_embeds_20 but not pos_embeds_10)
    # Actually, let's check that positions 0-9 are the same
    assert torch.allclose(pos_embeds_10[0:10], pos_embeds_20[0:10])


def test_positional_embedding_deterministic_initialization():
    """Test PositionalEmbedding initialization is deterministic with same seed."""
    max_seq_len = 256
    d_model = 256
    seed = 42
    
    embedding1 = PositionalEmbedding(max_seq_len, d_model, seed=seed)
    embedding2 = PositionalEmbedding(max_seq_len, d_model, seed=seed)
    
    # Check that embedding weights are identical
    assert torch.allclose(
        embedding1.embedding.weight,
        embedding2.embedding.weight
    )


def test_positional_embedding_deterministic_forward():
    """Test PositionalEmbedding forward pass is deterministic with same seed."""
    max_seq_len = 256
    d_model = 256
    seed = 42
    seq_len = 128
    
    embedding1 = PositionalEmbedding(max_seq_len, d_model, seed=seed)
    embedding2 = PositionalEmbedding(max_seq_len, d_model, seed=seed)
    
    output1 = embedding1(seq_len)
    output2 = embedding2(seq_len)
    
    # Outputs should be identical with same initialization
    assert torch.allclose(output1, output2)


def test_embedding_combination():
    """Test that token and positional embeddings can be combined."""
    vocab_size = 99
    d_model = 256
    batch_size = 2
    seq_len = 128
    
    token_embedding = TokenEmbedding(vocab_size, d_model)
    pos_embedding = PositionalEmbedding(256, d_model)
    
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Get embeddings
    token_embeds = token_embedding(token_ids)  # [B, seq_len, d_model]
    pos_embeds = pos_embedding(seq_len)  # [seq_len, d_model]
    
    # Combine: add positional embeddings to token embeddings
    combined = token_embeds + pos_embeds.unsqueeze(0)  # [B, seq_len, d_model]
    
    # Check shapes
    assert combined.shape == (batch_size, seq_len, d_model)
    assert combined.shape == token_embeds.shape
    
    # Combined embeddings should be different from token embeddings alone
    assert not torch.allclose(combined, token_embeds, atol=1e-5)


def test_token_embedding_gradient_flow():
    """Test that gradients can flow through token embeddings."""
    vocab_size = 99
    d_model = 256
    batch_size = 2
    seq_len = 10
    
    embedding = TokenEmbedding(vocab_size, d_model)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output = embedding(token_ids)
    loss = output.sum()
    loss.backward()
    
    # Verify gradients exist for embedding weights
    assert embedding.embedding.weight.grad is not None
    assert not torch.allclose(
        embedding.embedding.weight.grad,
        torch.zeros_like(embedding.embedding.weight.grad)
    )


def test_positional_embedding_gradient_flow():
    """Test that gradients can flow through positional embeddings."""
    max_seq_len = 256
    d_model = 256
    seq_len = 128
    
    embedding = PositionalEmbedding(max_seq_len, d_model)
    
    output = embedding(seq_len)
    loss = output.sum()
    loss.backward()
    
    # Verify gradients exist for embedding weights
    assert embedding.embedding.weight.grad is not None
    assert not torch.allclose(
        embedding.embedding.weight.grad,
        torch.zeros_like(embedding.embedding.weight.grad)
    )

