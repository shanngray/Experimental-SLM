"""Tests for tokenizer module."""

import json
import tempfile
from pathlib import Path

import pytest

from src.tokenizer import Tokenizer


def test_tokenizer_init():
    """Test that tokenizer initializes with correct vocabulary."""
    tokenizer = Tokenizer()
    
    # Check special tokens
    assert tokenizer.char_to_id["<PAD>"] == 0
    assert tokenizer.char_to_id["<UNK>"] == 1
    
    # Check reverse mapping
    assert tokenizer.id_to_char[0] == "<PAD>"
    assert tokenizer.id_to_char[1] == "<UNK>"


def test_encode_basic():
    """Test basic encoding of text."""
    tokenizer = Tokenizer()
    text = "Hello"
    token_ids = tokenizer.encode(text)
    
    assert isinstance(token_ids, list)
    assert all(isinstance(id_, int) for id_ in token_ids)
    assert len(token_ids) == len(text)


def test_decode_basic():
    """Test basic decoding of token IDs."""
    tokenizer = Tokenizer()
    token_ids = tokenizer.encode("Hello")
    decoded = tokenizer.decode(token_ids)
    
    assert decoded == "Hello"


def test_round_trip():
    """Test that encode then decode produces normalized original text."""
    tokenizer = Tokenizer()
    
    test_cases = [
        "Hello, world!",
        "Test\nwith\nnewlines",
        "Tab\tseparated\tvalues",
        "Mixed\ncontent\there!",
        "1234567890",
        "!@#$%^&*()",
    ]
    
    for text in test_cases:
        normalized = tokenizer.decode(tokenizer.encode(text))
        # After normalization, the text should match
        expected = tokenizer.encode(text)
        re_encoded = tokenizer.encode(normalized)
        assert re_encoded == expected


def test_encode_unknown_char():
    """Test that unknown characters map to <UNK> token ID."""
    tokenizer = Tokenizer()
    # Text with unicode character that will be normalized to <UNK>
    text = "Hello é"
    token_ids = tokenizer.encode(text)
    
    # Should contain <UNK> token ID (1)
    assert 1 in token_ids


def test_decode_unknown_token_id():
    """Test that unknown token IDs decode to placeholder."""
    tokenizer = Tokenizer()
    # Use a token ID that doesn't exist in vocab
    token_ids = [99999]
    decoded = tokenizer.decode(token_ids)
    
    # Should decode to \x00 placeholder
    assert decoded == "\x00"


def test_special_tokens():
    """Test handling of special tokens."""
    tokenizer = Tokenizer()
    
    # <PAD> should be token ID 0
    assert tokenizer.char_to_id["<PAD>"] == 0
    
    # <UNK> should be token ID 1
    assert tokenizer.char_to_id["<UNK>"] == 1
    
    # Encoding <UNK> character should produce token ID 1
    # But <UNK> is not a normal character, so we test via unknown input
    text = "é"  # Will normalize to <UNK>
    token_ids = tokenizer.encode(text)
    assert token_ids == [1]


def test_save_vocab():
    """Test saving vocabulary to JSON file."""
    tokenizer = Tokenizer()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = Path(tmpdir) / "vocab.json"
        tokenizer.save_vocab(vocab_path)
        
        assert vocab_path.exists()
        
        # Verify JSON is valid and readable
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        
        assert "char_to_id" in vocab_data
        assert "id_to_char" in vocab_data
        assert vocab_data["char_to_id"]["<PAD>"] == 0
        assert vocab_data["char_to_id"]["<UNK>"] == 1


def test_load_vocab():
    """Test loading vocabulary from JSON file."""
    tokenizer1 = Tokenizer()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = Path(tmpdir) / "vocab.json"
        tokenizer1.save_vocab(vocab_path)
        
        # Create new tokenizer and load vocab
        tokenizer2 = Tokenizer()
        tokenizer2.load_vocab(vocab_path)
        
        # Verify vocabularies match
        assert tokenizer2.char_to_id == tokenizer1.char_to_id
        assert tokenizer2.id_to_char == tokenizer1.id_to_char


def test_save_load_round_trip():
    """Test that save/load produces identical tokenization."""
    tokenizer1 = Tokenizer()
    test_text = "Hello, world!\nTest\t123"
    
    original_token_ids = tokenizer1.encode(test_text)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = Path(tmpdir) / "vocab.json"
        tokenizer1.save_vocab(vocab_path)
        
        tokenizer2 = Tokenizer()
        tokenizer2.load_vocab(vocab_path)
        
        # Encode same text with loaded tokenizer
        loaded_token_ids = tokenizer2.encode(test_text)
        
        assert loaded_token_ids == original_token_ids


def test_decode_padding():
    """Test that padding tokens are skipped in decode output."""
    tokenizer = Tokenizer()
    
    # Create token IDs with padding
    token_ids = [tokenizer.char_to_id["<PAD>"], 
                 tokenizer.char_to_id["H"],
                 tokenizer.char_to_id["e"],
                 tokenizer.char_to_id["<PAD>"],
                 tokenizer.char_to_id["l"],
                 tokenizer.char_to_id["l"],
                 tokenizer.char_to_id["o"]]
    
    decoded = tokenizer.decode(token_ids)
    # Should not contain <PAD> in output
    assert decoded == "Hello"
    assert "<PAD>" not in decoded
    # Should not contain \x00 (UNK placeholder) either
    assert "\x00" not in decoded


def test_vocab_size():
    """Test that vocabulary has expected size."""
    tokenizer = Tokenizer()
    
    # char_to_id: 2 special tokens (<PAD>, <UNK>) + 95 printable ASCII (32-126) + 2 (\n, \t) + 1 (\x00 placeholder) = 100
    # Note: \x00 is mapped to <UNK> token ID 1, but it's still a separate entry in char_to_id
    expected_char_to_id_size = 2 + (127 - 32) + 2 + 1  # 100
    
    # id_to_char: 2 special tokens + 95 printable ASCII + 2 (\n, \t) = 99
    # Note: \x00 and <UNK> both map to token ID 1, so id_to_char[1] only has one entry
    expected_id_to_char_size = 2 + (127 - 32) + 2  # 99
    
    assert len(tokenizer.char_to_id) == expected_char_to_id_size
    assert len(tokenizer.id_to_char) == expected_id_to_char_size


def test_encode_empty_string():
    """Test encoding empty string."""
    tokenizer = Tokenizer()
    token_ids = tokenizer.encode("")
    assert token_ids == []


def test_decode_empty_list():
    """Test decoding empty token list."""
    tokenizer = Tokenizer()
    decoded = tokenizer.decode([])
    assert decoded == ""


def test_ascii_character_order():
    """Test that ASCII characters are in correct order in vocabulary."""
    tokenizer = Tokenizer()
    
    # Check that space (32) comes before other printable chars
    space_id = tokenizer.char_to_id[" "]
    assert space_id == 2  # After <PAD>=0, <UNK>=1
    
    # Check that '!' (33) comes after space
    exclamation_id = tokenizer.char_to_id["!"]
    assert exclamation_id == space_id + 1
    
    # Check that '~' (126) is the last printable ASCII
    tilde_id = tokenizer.char_to_id["~"]
    assert tilde_id == 2 + (126 - 32)  # 2 + 94 = 96
