"""Sampler for generating text from model outputs.

This module provides functionality to generate text by sampling tokens
from model logits using temperature-based multinomial sampling.
"""

import torch

from src.tokenizer import Tokenizer


def sample_text(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    seed: int = 42
) -> str:
    """Generate text by sampling tokens from model logits.
    
    Generates text by autoregressively sampling tokens from the model's
    logit distribution. Uses temperature scaling (temperature=1.0 applies
    no scaling) and pure multinomial sampling (no top-k filtering).
    Uses a fixed seed for reproducibility.
    
    Args:
        model: Transformer model to sample from (must be a PyTorch nn.Module).
        tokenizer: Tokenizer for encoding prompt and decoding generated tokens.
        prompt: Initial prompt text to start generation from.
        max_length: Maximum number of tokens to generate (default: 100).
            Total output length will be prompt_length + max_length.
        temperature: Temperature for sampling (default: 1.0).
            temperature=1.0 applies no scaling (logits used as-is).
        seed: Random seed for reproducibility (default: 42).
    
    Returns:
        Generated text string (prompt + generated tokens).
    
    Example:
        >>> model = Transformer(vocab_size=100, max_seq_len=256)
        >>> tokenizer = Tokenizer()
        >>> text = sample_text(model, tokenizer, "Hello", max_length=10)
        >>> isinstance(text, str)
        True
    """
    # Set model to evaluation mode
    model.eval()
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    
    # Convert to tensor: [1, prompt_len] (batch_size=1)
    current_tokens = torch.tensor([prompt_tokens], dtype=torch.int64)
    
    # Generate tokens autoregressively
    generated_tokens = prompt_tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get current sequence length
            seq_len = current_tokens.shape[1]
            
            # Truncate if exceeds max_seq_len (use last max_seq_len tokens)
            if seq_len > model.max_seq_len:
                current_tokens = current_tokens[:, -model.max_seq_len:]
            
            # Forward pass: [1, seq_len] -> [1, seq_len, vocab_size]
            logits = model(current_tokens)
            
            # Get logits for the last position: [1, vocab_size]
            last_logits = logits[0, -1, :]  # [vocab_size]
            
            # Apply temperature scaling
            if temperature != 1.0:
                scaled_logits = last_logits / temperature
            else:
                scaled_logits = last_logits
            
            # Convert to probabilities using softmax
            probs = torch.softmax(scaled_logits, dim=-1)
            
            # Sample from multinomial distribution (pure multinomial, no top-k)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # Append to generated tokens
            generated_tokens.append(next_token_id)
            
            # Update current_tokens for next iteration
            # Append new token: [1, seq_len] -> [1, seq_len+1]
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.int64)
            current_tokens = torch.cat([current_tokens, next_token_tensor], dim=1)
    
    # Restore model to training mode
    model.train()
    
    # Decode generated tokens to text
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text

