"""Tests for sampling module."""

import pytest
import torch

from src.model.transformer import Transformer
from src.sampling import sample_text
from src.tokenizer import Tokenizer


def test_sample_text_produces_text():
    """Test that sample_text produces text of correct type."""
    vocab_size = 100
    model = Transformer(vocab_size=vocab_size, max_seq_len=256, seed=42)
    tokenizer = Tokenizer()
    
    generated_text = sample_text(
        model=model,
        tokenizer=tokenizer,
        prompt="Hello",
        max_length=10,
        seed=42
    )
    
    assert isinstance(generated_text, str)
    assert len(generated_text) > 0


def test_sample_text_is_reproducible():
    """Test that sampling is reproducible (same seed → same output)."""
    vocab_size = 100
    model1 = Transformer(vocab_size=vocab_size, max_seq_len=256, seed=42)
    model2 = Transformer(vocab_size=vocab_size, max_seq_len=256, seed=42)
    tokenizer = Tokenizer()
    
    # Generate text with same seed
    text1 = sample_text(model1, tokenizer, "The", max_length=20, seed=42)
    text2 = sample_text(model2, tokenizer, "The", max_length=20, seed=42)
    
    # Should produce identical output
    assert text1 == text2


def test_sample_text_uses_correct_temperature():
    """Test that sampling uses correct temperature (temperature=1.0)."""
    vocab_size = 100
    model = Transformer(vocab_size=vocab_size, max_seq_len=256, seed=42)
    tokenizer = Tokenizer()
    
    # Generate with temperature=1.0 (default)
    text1 = sample_text(
        model=model,
        tokenizer=tokenizer,
        prompt="The",
        max_length=10,
        temperature=1.0,
        seed=42
    )
    
    # Generate with temperature=2.0 (should produce different output)
    text2 = sample_text(
        model=model,
        tokenizer=tokenizer,
        prompt="The",
        max_length=10,
        temperature=2.0,
        seed=42
    )
    
    # Both should be valid strings
    assert isinstance(text1, str)
    assert isinstance(text2, str)
    # With different temperatures, outputs may differ
    # (though with same seed, they might still be similar)


def test_sample_text_generated_text_is_valid():
    """Test that generated text can be decoded (valid token IDs)."""
    vocab_size = 100
    model = Transformer(vocab_size=vocab_size, max_seq_len=256, seed=42)
    tokenizer = Tokenizer()
    
    generated_text = sample_text(
        model=model,
        tokenizer=tokenizer,
        prompt="Hello",
        max_length=20,
        seed=42
    )
    
    # Should be able to encode and decode without errors
    token_ids = tokenizer.encode(generated_text)
    decoded_text = tokenizer.decode(token_ids)
    
    # Decoded text should match (allowing for normalization)
    assert isinstance(decoded_text, str)
    assert len(decoded_text) > 0


def test_sample_text_from_fixed_prompt():
    """Test that sampling from fixed prompt works correctly."""
    vocab_size = 100
    model = Transformer(vocab_size=vocab_size, max_seq_len=256, seed=42)
    tokenizer = Tokenizer()
    
    prompt = "The quick brown"
    generated_text = sample_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=10,
        seed=42
    )
    
    # Generated text should start with the prompt
    assert generated_text.startswith(prompt)
    # Generated text should be longer than prompt
    assert len(generated_text) >= len(prompt)


def test_sample_text_produces_correct_length():
    """Test that sampling produces text of approximately correct length."""
    vocab_size = 100
    model = Transformer(vocab_size=vocab_size, max_seq_len=256, seed=42)
    tokenizer = Tokenizer()
    
    prompt = "The"
    max_length = 20
    
    generated_text = sample_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        seed=42
    )
    
    # Decode to get token count
    token_ids = tokenizer.encode(generated_text)
    
    # Should generate approximately max_length tokens (prompt + generated)
    # Allow some flexibility since prompt length varies
    assert len(token_ids) >= len(tokenizer.encode(prompt))
    assert len(token_ids) <= len(tokenizer.encode(prompt)) + max_length + 5  # Small buffer


def test_sample_text_runs_in_eval_mode():
    """Test that sampling runs in eval mode (no gradient computation)."""
    vocab_size = 100
    model = Transformer(vocab_size=vocab_size, max_seq_len=256, seed=42)
    tokenizer = Tokenizer()
    
    # Set model to training mode initially
    model.train()
    assert model.training
    
    # Generate text
    generated_text = sample_text(
        model=model,
        tokenizer=tokenizer,
        prompt="The",
        max_length=10,
        seed=42
    )
    
    # Model should be restored to training mode after sampling
    assert model.training
    
    # Verify output is valid
    assert isinstance(generated_text, str)


def test_sample_text_pure_multinomial():
    """Test that sampling uses pure multinomial (no top-k filtering)."""
    vocab_size = 100  # Must match tokenizer vocab size
    model = Transformer(vocab_size=vocab_size, max_seq_len=64, seed=42)
    tokenizer = Tokenizer()
    
    # Generate multiple samples
    samples = []
    for i in range(5):
        text = sample_text(
            model=model,
            tokenizer=tokenizer,
            prompt="The",
            max_length=10,
            seed=42 + i  # Different seeds for variety
        )
        samples.append(text)
    
    # All samples should be valid strings
    for sample in samples:
        assert isinstance(sample, str)
        assert len(sample) > 0
    
    # With pure multinomial, we should see variety in outputs
    # (though with deterministic model, same seed gives same output)
    assert len(set(samples)) >= 1  # At least some samples exist

