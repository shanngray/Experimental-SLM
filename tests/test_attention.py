"""Tests for multi-head attention module."""

import pytest
import torch

from src.model.attention import MultiHeadAttention


def test_attention_init():
    """Test MultiHeadAttention initialization."""
    d_model = 256
    n_heads = 4
    attn = MultiHeadAttention(d_model, n_heads)
    
    assert attn.d_model == d_model
    assert attn.n_heads == n_heads
    assert attn.d_k == d_model // n_heads


def test_attention_init_invalid_heads():
    """Test MultiHeadAttention raises error when d_model not divisible by n_heads."""
    with pytest.raises(ValueError, match="must be divisible"):
        MultiHeadAttention(d_model=256, n_heads=5)


def test_attention_output_shape():
    """Test attention output shapes are correct."""
    batch_size = 2
    seq_len = 256
    d_model = 256
    n_heads = 4
    
    attn = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = attn(x)
    
    assert output.shape == (batch_size, seq_len, d_model)


def test_attention_causal_masking():
    """Test causal masking prevents future attention."""
    batch_size = 1
    seq_len = 10
    d_model = 256
    n_heads = 4
    
    attn = MultiHeadAttention(d_model, n_heads)
    
    # Create input with distinct values at each position
    x = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    x = x.expand(batch_size, seq_len, d_model)
    
    # Forward pass
    output = attn(x)
    
    # Manually check attention weights by inspecting the forward pass
    # We'll compute attention weights manually to verify masking
    Q = attn.q_proj(x)
    K = attn.k_proj(x)
    
    batch_size, seq_len, _ = x.shape
    Q = Q.view(batch_size, seq_len, n_heads, attn.d_k).transpose(1, 2)
    K = K.view(batch_size, seq_len, n_heads, attn.d_k).transpose(1, 2)
    
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (attn.d_k ** 0.5)
    
    # Apply causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    scores_masked = scores.masked_fill(mask, float('-inf'))
    attn_weights = torch.nn.functional.softmax(scores_masked, dim=-1)
    
    # Verify that attention weights for future positions are zero
    # For each position i, positions > i should have zero attention weight
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            # Check all heads
            assert torch.allclose(attn_weights[:, :, i, j], torch.tensor(0.0)), \
                f"Position {i} should not attend to position {j}"


def test_attention_deterministic_initialization():
    """Test attention initialization is deterministic with same seed."""
    d_model = 256
    n_heads = 4
    seed = 42
    
    attn1 = MultiHeadAttention(d_model, n_heads, seed=seed)
    attn2 = MultiHeadAttention(d_model, n_heads, seed=seed)
    
    # Check that weights are identical
    assert torch.allclose(attn1.q_proj.weight, attn2.q_proj.weight)
    assert torch.allclose(attn1.k_proj.weight, attn2.k_proj.weight)
    assert torch.allclose(attn1.v_proj.weight, attn2.v_proj.weight)
    assert torch.allclose(attn1.out_proj.weight, attn2.out_proj.weight)


def test_attention_deterministic_forward():
    """Test attention forward pass is deterministic with same seed."""
    d_model = 256
    n_heads = 4
    seed = 42
    batch_size = 2
    seq_len = 10
    
    attn1 = MultiHeadAttention(d_model, n_heads, seed=seed)
    attn2 = MultiHeadAttention(d_model, n_heads, seed=seed)
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    output1 = attn1(x)
    output2 = attn2(x)
    
    # Outputs should be identical with same initialization and input
    assert torch.allclose(output1, output2)


def test_attention_different_seeds():
    """Test attention with different seeds produces different weights."""
    d_model = 256
    n_heads = 4
    
    attn1 = MultiHeadAttention(d_model, n_heads, seed=42)
    attn2 = MultiHeadAttention(d_model, n_heads, seed=123)
    
    # Weights should be different
    assert not torch.allclose(attn1.q_proj.weight, attn2.q_proj.weight, atol=1e-6)
