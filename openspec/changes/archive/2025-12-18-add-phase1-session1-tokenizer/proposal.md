# Change: Add Phase 1 Session 1 - Tokenizer Foundation

## Why
Phase 1 Session 1 establishes the foundational tokenization capability required for all subsequent model training. The tokenizer converts raw text into token IDs that the model can process, and handles text normalization according to an ASCII policy. This is the first critical component needed before any model training can begin.

## What Changes
- Add tokenizer capability with character-level ASCII tokenization
- Implement text normalization with ASCII policy (printable 32-126 + `\n` + `\t`)
- Unknown characters map to `<UNK>` placeholder
- Vocab definition: `<PAD>=0`, `<UNK>=1`, then allowed ASCII characters
- Vocab persistence: save/load vocab to disk in JSON format
- Encode/decode methods for text ↔ token ID conversion
- Comprehensive test coverage for normalization and tokenization

## Impact
- Affected specs: New `tokenizer` capability specification
- Affected code: 
  - `src/tokenizer.py` (new)
  - `src/normalize.py` (new)
  - `tests/test_tokenizer.py` (new)
  - `tests/test_normalize.py` (new)
- Dependencies: None (foundational component)
- Future impact: All subsequent sessions depend on this tokenizer for data processing
