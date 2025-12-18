# dataset Specification

## Purpose
TBD - created by archiving change add-phase1-session2-dataset. Update Purpose after archive.
## Requirements
### Requirement: Corpus Train/Val Split
The system SHALL split a tokenized corpus into training and validation sets using a contiguous split strategy.

#### Scenario: Split corpus into train/val sets
- **WHEN** a tokenized corpus and split ratio are provided to `split_corpus()`
- **THEN** the corpus is split into contiguous train and validation portions
- **AND** the split ratio matches the requested ratio (e.g., 95%/5%)

#### Scenario: Deterministic split with seed
- **WHEN** `split_corpus()` is called with the same seed and corpus
- **THEN** the split produces identical train/val boundaries

#### Scenario: Handle configurable split ratio
- **WHEN** a custom split ratio is provided (e.g., 90%/10%)
- **THEN** the corpus is split according to the specified ratio

### Requirement: Sequence Windowing
The system SHALL create (x, y) sequence pairs from a tokenized corpus using sliding windows.

#### Scenario: Create windows of correct shape
- **WHEN** a tokenized corpus is provided to `WindowDataset`
- **THEN** each window has shape [256] (context length)

#### Scenario: Non-overlapping windows
- **WHEN** windows are created from corpus
- **THEN** windows use stride 256 (non-overlapping)
- **AND** each token appears in exactly one window (except boundary tokens)

#### Scenario: Next-token prediction pairs
- **WHEN** a window x is created
- **THEN** the corresponding y is x shifted by 1 position
- **AND** y[i] == x[i+1] for all valid indices

#### Scenario: Handle corpus boundaries
- **WHEN** the corpus length is not evenly divisible by 256
- **THEN** the incomplete last window is dropped
- **AND** no windows shorter than 256 are created

#### Scenario: Iterate through dataset
- **WHEN** iterating through `WindowDataset`
- **THEN** each iteration yields a valid (x, y) pair
- **AND** iteration completes without errors

#### Scenario: Handle empty or small corpus
- **WHEN** corpus length is less than 256 tokens
- **THEN** the dataset is empty (no windows created)
- **AND** iteration completes without errors

