# tokenizer Specification

## Purpose
TBD - created by archiving change add-phase1-session1-tokenizer. Update Purpose after archive.
## Requirements
### Requirement: Text Normalization
The system SHALL normalize text according to an ASCII policy before tokenization.

#### Scenario: Normalize printable ASCII characters
- **WHEN** text contains printable ASCII characters (32-126)
- **THEN** those characters are preserved unchanged

#### Scenario: Normalize whitespace characters
- **WHEN** text contains newline (`\n`) or tab (`\t`) characters
- **THEN** those characters are preserved unchanged

#### Scenario: Handle unknown characters
- **WHEN** text contains characters outside the allowed set (non-ASCII, control chars except `\n` and `\t`)
- **THEN** those characters are replaced with `<UNK>` placeholder

#### Scenario: Handle empty strings
- **WHEN** normalization receives an empty string
- **THEN** it returns an empty string without error

#### Scenario: Handle unicode characters
- **WHEN** text contains unicode characters outside ASCII range
- **THEN** those characters are replaced with `<UNK>` placeholder

### Requirement: Character-Level Tokenization
The system SHALL provide a tokenizer that converts normalized text to token IDs and vice versa.

#### Scenario: Encode text to token IDs
- **WHEN** text is provided to the `encode()` method
- **THEN** it returns a list of integer token IDs corresponding to each character in the normalized text

#### Scenario: Decode token IDs to text
- **WHEN** a list of token IDs is provided to the `decode()` method
- **THEN** it returns the corresponding text string

#### Scenario: Round-trip preservation
- **WHEN** text is encoded and then decoded
- **THEN** the decoded text matches the original normalized text (where normalization is reversible)

### Requirement: Vocabulary Definition
The system SHALL define a vocabulary mapping characters to token IDs with special tokens.

#### Scenario: Special token mapping
- **WHEN** the vocabulary is initialized
- **THEN** `<PAD>` maps to token ID 0
- **AND** `<UNK>` maps to token ID 1
- **AND** allowed ASCII characters map to sequential IDs starting from 2

#### Scenario: Character to ID mapping
- **WHEN** encoding a known ASCII character
- **THEN** the tokenizer returns the correct token ID from the vocabulary

#### Scenario: Unknown character handling
- **WHEN** encoding text containing an unknown character (after normalization)
- **THEN** the tokenizer returns token ID 1 (`<UNK>`)

### Requirement: Vocabulary Persistence
The system SHALL save and load vocabulary definitions to disk.

#### Scenario: Save vocabulary to disk
- **WHEN** `save_vocab()` is called with a file path
- **THEN** the vocabulary is saved to disk in JSON format
- **AND** the JSON file is human-readable

#### Scenario: Load vocabulary from disk
- **WHEN** `load_vocab()` is called with a file path
- **THEN** the vocabulary is loaded from the JSON file
- **AND** the loaded vocabulary produces identical tokenization as before saving

#### Scenario: Save/load round-trip
- **WHEN** a vocabulary is saved and then loaded
- **THEN** encoding with the loaded vocabulary produces identical token IDs as before saving

