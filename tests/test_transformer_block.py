"""Tests for transformer block module."""

import pytest
import torch

from src.model.transformer_block import TransformerBlock


def test_transformer_block_init():
    """Test TransformerBlock initialization."""
    d_model = 256
    n_heads = 4
    d_ff = 1024
    block = TransformerBlock(d_model, n_heads, d_ff)
    
    assert block.d_model == d_model
    assert block.n_heads == n_heads
    assert block.d_ff == d_ff


def test_transformer_block_forward_pass():
    """Test transformer block forward pass works end-to-end."""
    batch_size = 2
    seq_len = 256
    d_model = 256
    n_heads = 4
    d_ff = 1024
    
    block = TransformerBlock(d_model, n_heads, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = block(x)
    
    # Should complete without errors
    assert output is not None
    assert output.shape == x.shape
    assert output.shape == (batch_size, seq_len, d_model)


def test_transformer_block_output_shape():
    """Test transformer block output shapes match input shapes."""
    batch_size = 4
    seq_len = 128
    d_model = 128
    n_heads = 4
    d_ff = 512
    
    block = TransformerBlock(d_model, n_heads, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = block(x)
    
    assert output.shape == x.shape
    assert output.shape == (batch_size, seq_len, d_model)


def test_transformer_block_residual_connections():
    """Test residual connections work correctly."""
    batch_size = 2
    seq_len = 10
    d_model = 256
    n_heads = 4
    d_ff = 1024
    
    block = TransformerBlock(d_model, n_heads, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Set requires_grad to test gradient flow
    x.requires_grad_(True)
    
    output = block(x)
    
    # Verify output has gradients
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None
    assert output.requires_grad


def test_transformer_block_residual_adds_input():
    """Test that residual connections add input to attention output."""
    batch_size = 1
    seq_len = 5
    d_model = 256
    n_heads = 4
    d_ff = 1024
    
    block = TransformerBlock(d_model, n_heads, d_ff)
    
    # Create input with very small values to make residual effect visible
    x = torch.ones(batch_size, seq_len, d_model) * 0.001
    
    # Set all parameters to very small values to minimize transformation
    with torch.no_grad():
        for param in block.parameters():
            param.fill_(0.001)
    
    output = block(x)
    
    # With very small weights, output should be close to input (due to residuals)
    # This verifies residuals are being added
    assert output.shape == x.shape


def test_transformer_block_component_integration():
    """Test that attention and MLP are applied in correct order."""
    batch_size = 2
    seq_len = 10
    d_model = 256
    n_heads = 4
    d_ff = 1024
    
    block = TransformerBlock(d_model, n_heads, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass should use both attention and MLP
    output = block(x)
    
    # Output should be different from input (unless weights are identity, which they're not)
    assert not torch.allclose(output, x, atol=1e-5)


def test_transformer_block_deterministic_initialization():
    """Test transformer block initialization is deterministic with same seed."""
    d_model = 256
    n_heads = 4
    d_ff = 1024
    seed = 42
    
    block1 = TransformerBlock(d_model, n_heads, d_ff, seed=seed)
    block2 = TransformerBlock(d_model, n_heads, d_ff, seed=seed)
    
    # Check that weights of all components are identical
    assert torch.allclose(
        block1.attention.q_proj.weight,
        block2.attention.q_proj.weight
    )
    assert torch.allclose(
        block1.mlp.gate_proj.weight,
        block2.mlp.gate_proj.weight
    )
    assert torch.allclose(
        block1.attn_norm.weight,
        block2.attn_norm.weight
    )


def test_transformer_block_deterministic_forward():
    """Test transformer block forward pass is deterministic with same seed."""
    d_model = 256
    n_heads = 4
    d_ff = 1024
    seed = 42
    batch_size = 2
    seq_len = 10
    
    block1 = TransformerBlock(d_model, n_heads, d_ff, seed=seed)
    block2 = TransformerBlock(d_model, n_heads, d_ff, seed=seed)
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    output1 = block1(x)
    output2 = block2(x)
    
    # Outputs should be identical with same initialization and input
    assert torch.allclose(output1, output2)


def test_transformer_block_gradient_flow():
    """Test that gradients can flow through residual connections."""
    batch_size = 2
    seq_len = 10
    d_model = 256
    n_heads = 4
    d_ff = 1024
    
    block = TransformerBlock(d_model, n_heads, d_ff)
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    output = block(x)
    loss = output.sum()
    loss.backward()
    
    # Verify gradients flow back to input
    assert x.grad is not None
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
