"""Tests for text normalization module."""

import pytest

from src.normalize import normalize_text


def test_normalize_printable_ascii():
    """Test that printable ASCII characters are preserved."""
    text = "Hello, world! 123"
    assert normalize_text(text) == text


def test_normalize_whitespace():
    """Test that newline and tab are preserved."""
    text = "Hello\nWorld\tTab"
    assert normalize_text(text) == text


def test_normalize_empty_string():
    """Test that empty string returns empty string."""
    assert normalize_text("") == ""


def test_normalize_unicode():
    """Test that unicode characters are replaced with placeholder."""
    text = "Hello é"
    # Unicode char should be replaced with \x00 placeholder
    # Note: space before é is preserved (it's printable ASCII)
    result = normalize_text(text)
    assert result == "Hello \x00"
    assert "\x00" in result


def test_normalize_control_chars():
    """Test that control characters (except \\n and \\t) are replaced."""
    # Bell character (ASCII 7)
    text = "Hello\aWorld"
    result = normalize_text(text)
    assert result == "Hello\x00World"
    assert "\x00" in result


def test_normalize_mixed_content():
    """Test normalization with mixed allowed and disallowed characters."""
    text = "Hello\né\tWorld"
    result = normalize_text(text)
    assert result == "Hello\n\x00\tWorld"
    assert "\x00" in result


def test_normalize_all_printable_ascii():
    """Test that all printable ASCII characters (32-126) are preserved."""
    # Test all printable ASCII characters
    for i in range(32, 127):
        char = chr(i)
        assert normalize_text(char) == char


def test_normalize_multiple_unk():
    """Test that multiple unknown characters each become placeholder."""
    text = "aébç"
    result = normalize_text(text)
    assert result == "a\x00b\x00"
    assert result.count("\x00") == 2


def test_normalize_newline_tab_only():
    """Test that newline and tab alone are preserved."""
    assert normalize_text("\n") == "\n"
    assert normalize_text("\t") == "\t"


def test_normalize_emoji():
    """Test that emoji characters are replaced with placeholder."""
    text = "Hello 😀"
    result = normalize_text(text)
    assert result == "Hello \x00"
    assert "\x00" in result
