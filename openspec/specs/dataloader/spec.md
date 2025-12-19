# dataloader Specification

## Purpose
TBD - created by archiving change add-phase1-session3-dataloader. Update Purpose after archive.
## Requirements
### Requirement: Batch Creation
The system SHALL create batches of sequences from a dataset for efficient model training.

#### Scenario: Create batches of correct shape
- **WHEN** a dataset and batch size are provided to `DataLoader`
- **THEN** each batch has shape [B, 256] where B is the batch size
- **AND** B is configurable (default: 16)

#### Scenario: Correct tensor dtype
- **WHEN** batches are created from dataset
- **THEN** all tensors are dtype `int64`

#### Scenario: Handle incomplete last batch
- **WHEN** the dataset size is not evenly divisible by batch size
- **THEN** the incomplete last batch is dropped deterministically
- **AND** no batches smaller than the specified batch size are created

#### Scenario: Iterate through batches
- **WHEN** iterating through `DataLoader`
- **THEN** each iteration yields a valid batch tensor
- **AND** iteration completes without errors
- **AND** all complete batches are yielded

### Requirement: Deterministic Batching
The system SHALL produce deterministic batches when given the same seed and dataset.

#### Scenario: Same seed produces same batches
- **WHEN** `DataLoader` is created with the same seed and dataset
- **THEN** batches are produced in the same order
- **AND** batch contents are identical

#### Scenario: Handle empty dataset
- **WHEN** an empty dataset is provided to `DataLoader`
- **THEN** iteration completes without errors
- **AND** no batches are yielded

### Requirement: Model Input Readiness
The system SHALL produce batches that are ready for direct model input.

#### Scenario: Batch format compatibility
- **WHEN** batches are created from `DataLoader`
- **THEN** batches can be passed directly to model forward pass
- **AND** batch shape and dtype are compatible with model expectations

