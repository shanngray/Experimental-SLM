# model-import Specification

## Purpose
TBD - created by archiving change add-multi-model-support. Update Purpose after archive.
## Requirements
### Requirement: HuggingFace Model Import
The system SHALL provide a CLI tool to download and import pretrained models from HuggingFace, converting them to local format and registering them in the model registry.

#### Scenario: Import model from HuggingFace
- **WHEN** user runs `python main.py import-model Qwen/Qwen-0.5B`
- **THEN** the CLI downloads the model from HuggingFace Hub
- **AND** converts model weights to local PyTorch format
- **AND** saves model config, weights, and metadata to `models/{sanitized-name}/` directory
- **AND** registers model in `models/registry.json` with auto-generated model_name
- **AND** displays success message with model_name for use in configs

#### Scenario: Import with custom name
- **WHEN** user runs `python main.py import-model Qwen/Qwen-0.5B --name my-qwen-base`
- **THEN** model is saved as `models/my-qwen-base/`
- **AND** registry entry uses `my-qwen-base` as model_name
- **AND** original HuggingFace model ID is preserved in model_id field in metadata

#### Scenario: Import with authentication
- **WHEN** user imports a gated model requiring HuggingFace authentication
- **THEN** CLI prompts for HuggingFace token or reads from environment variable
- **AND** authenticates with HuggingFace Hub
- **AND** proceeds with download after successful authentication

#### Scenario: Display license during import
- **WHEN** model is being imported
- **THEN** CLI displays model license from HuggingFace metadata
- **AND** prompts user to acknowledge license terms
- **AND** proceeds with import only after user acknowledgment
- **AND** saves license information in model metadata

#### Scenario: Validate model architecture support
- **WHEN** user attempts to import a model
- **THEN** CLI checks if model architecture is supported (Qwen family initially)
- **AND** if architecture is unsupported, displays clear error message listing supported architectures
- **AND** exits without downloading or importing

#### Scenario: Handle duplicate model imports
- **WHEN** user attempts to import a model with a model_name that already exists in registry
- **THEN** CLI prompts user to choose: overwrite, rename, or cancel
- **AND** proceeds according to user choice
- **AND** updates registry appropriately

### Requirement: Model Import Progress and Validation
The system SHALL provide progress feedback during model import and validate imported models for integrity.

#### Scenario: Display download progress
- **WHEN** model is being downloaded from HuggingFace
- **THEN** CLI displays progress bar showing download percentage and speed
- **AND** displays estimated time remaining
- **AND** displays total model size

#### Scenario: Display conversion progress
- **WHEN** model is being converted to local format
- **THEN** CLI displays conversion status for each component (weights, config, tokenizer)
- **AND** displays any warnings or issues encountered during conversion

#### Scenario: Validate model after import
- **WHEN** model import completes
- **THEN** CLI validates model files exist and are not corrupted
- **AND** validates model can be loaded successfully
- **AND** performs basic forward pass test
- **AND** displays validation results

#### Scenario: Handle network errors during download
- **WHEN** network error occurs during model download
- **THEN** CLI displays clear error message
- **AND** cleans up partial downloads
- **AND** exits gracefully without registering incomplete model

#### Scenario: Handle insufficient disk space
- **WHEN** insufficient disk space for model import
- **THEN** CLI checks available disk space before starting download
- **AND** displays clear error message if insufficient space
- **AND** does not proceed with download

### Requirement: Import Model Metadata Tracking
The system SHALL track comprehensive metadata for imported models including source, architecture, creation date, and license information.

#### Scenario: Save complete model metadata
- **WHEN** model is successfully imported
- **THEN** metadata file is saved in model directory containing:
  - model_name: user-friendly name for referencing model
  - model_id: original HuggingFace repository ID
  - source: "huggingface"
  - architecture_type: model architecture family (e.g., "qwen")
  - created_at: import timestamp
  - license: license information from HuggingFace
  - model_size: size in parameters
  - file_size: total size on disk
- **AND** metadata is also saved in registry entry

#### Scenario: Metadata is human-readable
- **WHEN** metadata file is created
- **THEN** file is saved in JSON format
- **AND** JSON is formatted with indentation for readability
- **AND** includes comments (via special fields) explaining each field

