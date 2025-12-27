# model-registry Specification

## Purpose
TBD - created by archiving change add-multi-model-support. Update Purpose after archive.
## Requirements
### Requirement: Model Registry Management
The system SHALL provide a model registry to track available models, their metadata, and enable model discovery and selection.

#### Scenario: Registry initialization
- **WHEN** system first runs or `models/` directory is created
- **THEN** registry file `models/registry.json` is created if it doesn't exist
- **AND** registry is initialized with empty model list
- **AND** registry includes schema version for future compatibility

#### Scenario: Add model to registry
- **WHEN** a model is imported or registered
- **THEN** registry is updated with new entry containing:
  - model_name: unique user-friendly identifier for the model
  - model_id: original identifier (e.g., HuggingFace repo ID like "Qwen/Qwen-0.5B")
  - architecture_type: model architecture family (e.g., "qwen", "custom-transformer")
  - local_path: relative path to model directory
  - source: source of model ("huggingface", "custom", "finetuned")
  - created_at: registration timestamp
  - metadata: additional model-specific metadata
- **AND** registry file is saved atomically to prevent corruption

#### Scenario: Get model from registry
- **WHEN** code requests a model by model_name
- **THEN** registry returns model entry if it exists
- **AND** validates that local_path exists and is accessible
- **AND** returns None or raises error if model_name not found

#### Scenario: List all models in registry
- **WHEN** user requests list of available models
- **THEN** registry returns all registered models
- **AND** includes summary information (model_name, model_id, architecture_type, source)
- **AND** orders models by created_at (most recent first)

#### Scenario: Delete model from registry
- **WHEN** user deletes a model
- **THEN** model entry is removed from registry
- **AND** optionally deletes model files from disk if requested
- **AND** registry is saved after update

#### Scenario: Validate registry integrity
- **WHEN** registry is loaded
- **THEN** validates JSON schema is correct
- **AND** validates all required fields are present
- **AND** checks for duplicate model_names
- **AND** warns about models with missing local_path directories
- **AND** attempts to fix or reports errors clearly

### Requirement: Model Metadata Querying
The system SHALL provide functionality to query and display detailed model metadata.

#### Scenario: Get detailed model info
- **WHEN** user requests detailed information for a model
- **THEN** system returns complete metadata including:
  - All registry fields (model_name, model_id, architecture_type, local_path, source, created_at)
  - Model-specific metadata from model's metadata.json file
  - File size information
  - Fine-tuning lineage if applicable
- **AND** metadata is formatted for human readability

#### Scenario: Query models by architecture type
- **WHEN** user queries for models of a specific architecture (e.g., "qwen")
- **THEN** registry returns all models matching that architecture_type
- **AND** results are ordered by created_at

#### Scenario: Query models by source
- **WHEN** user queries for models from a specific source (e.g., "huggingface")
- **THEN** registry returns all models matching that source
- **AND** results are ordered by created_at

### Requirement: Fine-Tuning Lineage Tracking
The system SHALL track fine-tuning lineage, recording which models were fine-tuned from which base models.

#### Scenario: Record fine-tuning parent
- **WHEN** a model is fine-tuned from a base model
- **THEN** fine-tuned model's metadata includes `fine_tuned_from` field with parent model_name
- **AND** registry entry includes fine-tuning timestamp
- **AND** parent model remains in registry unchanged

#### Scenario: Query fine-tuning children
- **WHEN** user requests models fine-tuned from a specific base model
- **THEN** registry returns all models with matching `fine_tuned_from` field
- **AND** results include fine-tuning timestamps

#### Scenario: Display fine-tuning chain
- **WHEN** user requests lineage for a fine-tuned model
- **THEN** system displays full fine-tuning chain: base → fine-tune1 → fine-tune2 → current
- **AND** includes timestamps for each fine-tuning step
- **AND** indicates if any parent models are missing from registry

### Requirement: Registry CLI Commands
The system SHALL provide CLI commands for managing the model registry.

#### Scenario: List models command
- **WHEN** user runs `python main.py list-models`
- **THEN** displays table of all models with columns: model_name, model_id, architecture, source, created_at
- **AND** includes summary count of total models
- **AND** supports filtering by architecture type with `--architecture` flag
- **AND** supports filtering by source with `--source` flag

#### Scenario: Model info command
- **WHEN** user runs `python main.py model-info <model-name>`
- **THEN** displays detailed information for specified model
- **AND** includes all metadata fields in readable format
- **AND** includes file size and disk usage
- **AND** includes fine-tuning lineage if applicable
- **AND** displays error if model_name not found

#### Scenario: Delete model command
- **WHEN** user runs `python main.py delete-model <model-name>`
- **THEN** prompts user for confirmation
- **AND** removes model from registry
- **AND** optionally deletes model files with `--delete-files` flag
- **AND** warns if model is parent of fine-tuned models
- **AND** displays success message after deletion

#### Scenario: Validate registry command
- **WHEN** user runs `python main.py validate-registry`
- **THEN** checks registry integrity
- **AND** validates all model paths exist
- **AND** checks for orphaned model directories
- **AND** reports any issues found
- **AND** offers to fix issues automatically with `--fix` flag

