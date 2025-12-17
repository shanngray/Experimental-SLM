"""Text normalization module for ASCII policy tokenization.

This module provides text normalization functionality that converts input text
to a normalized form containing only allowed ASCII characters. Characters outside
the allowed set are replaced with a special placeholder character that maps to
the `<UNK>` token in the tokenizer vocabulary.
"""

# Special character used as placeholder for unknown characters
# This will be mapped to <UNK> token ID in the tokenizer
_UNK_PLACEHOLDER = "\x00"


def normalize_text(text: str) -> str:
    """Normalize text according to ASCII policy.
    
    The ASCII policy allows:
    - Printable ASCII characters (32-126)
    - Newline character (`\n`)
    - Tab character (`\t`)
    
    All other characters (non-ASCII, control chars except `\n` and `\t`) are
    replaced with a placeholder character that maps to `<UNK>` token.
    
    Args:
        text: Input text string to normalize.
        
    Returns:
        Normalized text string with only allowed characters or placeholder.
        
    Examples:
        >>> normalize_text("Hello, world!")
        'Hello, world!'
        >>> normalize_text("Hello\\nWorld")
        'Hello\\nWorld'
        >>> normalize_text("Hello\\u00e9")  # é character
        'Hello\\x00'
    """
    if not text:
        return ""
    
    normalized = []
    for char in text:
        # Check if character is in allowed set
        if _is_allowed_char(char):
            normalized.append(char)
        else:
            normalized.append(_UNK_PLACEHOLDER)
    
    return "".join(normalized)


def _is_allowed_char(char: str) -> bool:
    """Check if a character is allowed according to ASCII policy.
    
    Args:
        char: Single character string to check.
        
    Returns:
        True if character is allowed, False otherwise.
    """
    # Allow printable ASCII (32-126)
    if 32 <= ord(char) <= 126:
        return True
    
    # Allow newline and tab
    if char in ("\n", "\t"):
        return True
    
    return False
