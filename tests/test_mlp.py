"""Tests for MLP (feed-forward network) module."""

import pytest
import torch

from src.model.mlp import MLP


def test_mlp_init():
    """Test MLP initialization."""
    d_model = 256
    d_ff = 1024
    mlp = MLP(d_model, d_ff)
    
    assert mlp.d_model == d_model
    assert mlp.d_ff == d_ff


def test_mlp_output_shape():
    """Test MLP output shapes match input shapes."""
    batch_size = 2
    seq_len = 256
    d_model = 256
    d_ff = 1024
    
    mlp = MLP(d_model, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = mlp(x)
    
    assert output.shape == x.shape
    assert output.shape == (batch_size, seq_len, d_model)


def test_mlp_preserves_batch_and_sequence_dims():
    """Test MLP preserves batch and sequence dimensions."""
    batch_size = 4
    seq_len = 128
    d_model = 128
    d_ff = 512
    
    mlp = MLP(d_model, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = mlp(x)
    
    assert output.shape[0] == batch_size
    assert output.shape[1] == seq_len
    assert output.shape[2] == d_model


def test_mlp_deterministic_initialization():
    """Test MLP initialization is deterministic with same seed."""
    d_model = 256
    d_ff = 1024
    seed = 42
    
    mlp1 = MLP(d_model, d_ff, seed=seed)
    mlp2 = MLP(d_model, d_ff, seed=seed)
    
    # Check that weights are identical
    assert torch.allclose(mlp1.gate_proj.weight, mlp2.gate_proj.weight)
    assert torch.allclose(mlp1.up_proj.weight, mlp2.up_proj.weight)
    assert torch.allclose(mlp1.down_proj.weight, mlp2.down_proj.weight)


def test_mlp_deterministic_forward():
    """Test MLP forward pass is deterministic with same seed."""
    d_model = 256
    d_ff = 1024
    seed = 42
    batch_size = 2
    seq_len = 10
    
    mlp1 = MLP(d_model, d_ff, seed=seed)
    mlp2 = MLP(d_model, d_ff, seed=seed)
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    output1 = mlp1(x)
    output2 = mlp2(x)
    
    # Outputs should be identical with same initialization and input
    assert torch.allclose(output1, output2)


def test_mlp_different_seeds():
    """Test MLP with different seeds produces different weights."""
    d_model = 256
    d_ff = 1024
    
    mlp1 = MLP(d_model, d_ff, seed=42)
    mlp2 = MLP(d_model, d_ff, seed=123)
    
    # Weights should be different
    assert not torch.allclose(mlp1.gate_proj.weight, mlp2.gate_proj.weight, atol=1e-6)
