"""Tokenizer module for character-level ASCII tokenization.

This module provides a Tokenizer class that converts normalized text to token IDs
and vice versa, using a vocabulary that maps characters to integer IDs.
"""

import json
from pathlib import Path
from typing import List

from .normalize import normalize_text


class Tokenizer:
    """Character-level tokenizer with ASCII vocabulary.
    
    The vocabulary consists of:
    - `<PAD>` = 0 (padding token)
    - `<UNK>` = 1 (unknown token)
    - ASCII characters (32-126, plus `\n` and `\t`) = 2 onwards
    
    Attributes:
        char_to_id: Dictionary mapping characters to token IDs.
        id_to_char: Dictionary mapping token IDs to characters.
    """
    
    def __init__(self):
        """Initialize the tokenizer with default vocabulary."""
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        self._build_vocab()
    
    def _build_vocab(self) -> None:
        """Build the vocabulary mapping."""
        # Special tokens
        self.char_to_id["<PAD>"] = 0
        self.char_to_id["<UNK>"] = 1
        
        # Map the UNK placeholder character (\x00) to token ID 1
        # This is the character used by normalization for unknown chars
        from .normalize import _UNK_PLACEHOLDER
        self.char_to_id[_UNK_PLACEHOLDER] = 1
        
        # ASCII characters in order: printable (32-126), then \n, then \t
        token_id = 2
        
        # Printable ASCII (32-126)
        for i in range(32, 127):
            char = chr(i)
            self.char_to_id[char] = token_id
            token_id += 1
        
        # Newline and tab
        self.char_to_id["\n"] = token_id
        token_id += 1
        self.char_to_id["\t"] = token_id
        
        # Build reverse mapping
        self.id_to_char = {id_: char for char, id_ in self.char_to_id.items()}
        # For token ID 1, use "<UNK>" as the display representation
        self.id_to_char[1] = "<UNK>"
    
    def encode(self, text: str) -> List[int]:
        """Encode text to a list of token IDs.
        
        The text is first normalized according to ASCII policy, then each
        character is mapped to its corresponding token ID.
        
        Args:
            text: Input text string to encode.
            
        Returns:
            List of integer token IDs.
            
        Examples:
            >>> tokenizer = Tokenizer()
            >>> tokenizer.encode("Hello")
            [40, 69, 76, 76, 79]  # H=40, e=69, l=76, l=76, o=79
        """
        normalized = normalize_text(text)
        token_ids = []
        
        for char in normalized:
            if char in self.char_to_id:
                token_ids.append(self.char_to_id[char])
            else:
                # Should not happen after normalization, but handle gracefully
                token_ids.append(self.char_to_id["<UNK>"])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs to text.
        
        Args:
            token_ids: List of integer token IDs to decode.
            
        Returns:
            Decoded text string.
            
        Examples:
            >>> tokenizer = Tokenizer()
            >>> tokenizer.decode([40, 69, 76, 76, 79])
            'Hello'
        """
        chars = []
        for token_id in token_ids:
            if token_id in self.id_to_char:
                char = self.id_to_char[token_id]
                # Skip padding tokens in output
                if char != "<PAD>":
                    # Replace <UNK> token with placeholder character for normalization compatibility
                    if char == "<UNK>":
                        from .normalize import _UNK_PLACEHOLDER
                        chars.append(_UNK_PLACEHOLDER)
                    else:
                        chars.append(char)
            else:
                # Unknown token ID, use placeholder character
                from .normalize import _UNK_PLACEHOLDER
                chars.append(_UNK_PLACEHOLDER)
        
        return "".join(chars)
    
    def save_vocab(self, filepath: str | Path) -> None:
        """Save vocabulary to a JSON file.
        
        Args:
            filepath: Path to save the vocabulary JSON file.
            
        Raises:
            IOError: If the file cannot be written.
        """
        filepath = Path(filepath)
        vocab_data = {
            "char_to_id": self.char_to_id,
            "id_to_char": {str(k): v for k, v in self.id_to_char.items()}
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    
    def load_vocab(self, filepath: str | Path) -> None:
        """Load vocabulary from a JSON file.
        
        Args:
            filepath: Path to load the vocabulary JSON file from.
            
        Raises:
            IOError: If the file cannot be read.
            ValueError: If the vocabulary format is invalid.
        """
        filepath = Path(filepath)
        
        with open(filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        
        self.char_to_id = vocab_data["char_to_id"]
        # Convert string keys back to integers for id_to_char
        self.id_to_char = {int(k): v for k, v in vocab_data["id_to_char"].items()}
