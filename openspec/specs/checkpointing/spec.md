# checkpointing Specification

## Purpose
TBD - created by archiving change add-phase1-session7-checkpointing. Update Purpose after archive.
## Requirements
### Requirement: Checkpoint Save
The system SHALL provide functionality to save complete training state to disk, including model weights, optimizer state, training configuration, vocabulary, step counter, model metadata (model_name, model_id, architecture_type, source), and fine-tuning lineage.

#### Scenario: Save checkpoint creates files
- **WHEN** `save_checkpoint()` is called with model, optimizer, config, vocab, and step
- **THEN** checkpoint files are created in the checkpoints directory
- **AND** model state_dict is saved in binary format (PyTorch format)
- **AND** optimizer state_dict is saved
- **AND** training config is saved in JSON-compatible format
- **AND** vocabulary is saved
- **AND** step counter is saved
- **AND** metadata file contains checkpoint information including model metadata

#### Scenario: Checkpoint format
- **WHEN** checkpoint is saved
- **THEN** checkpoint uses PyTorch format (torch.save) for binary weights
- **AND** metadata is stored in JSON format for human readability
- **AND** checkpoint files are organized in checkpoints/ directory

#### Scenario: Save model metadata in checkpoint
- **WHEN** checkpoint is saved for a model loaded from registry
- **THEN** checkpoint metadata includes model_name from registry
- **AND** metadata includes model_id (original identifier, e.g., HuggingFace repo)
- **AND** metadata includes architecture_type (e.g., "qwen", "custom-transformer")
- **AND** metadata includes source (e.g., "huggingface", "custom", "finetuned")

#### Scenario: Save fine-tuning lineage in checkpoint
- **WHEN** checkpoint is saved for a fine-tuned model
- **THEN** checkpoint metadata includes fine_tuned_from field with parent model_name
- **AND** metadata includes fine-tuning start timestamp
- **AND** metadata preserves full fine-tuning chain if parent was also fine-tuned

#### Scenario: Save checkpoint for custom Transformer
- **WHEN** checkpoint is saved for custom Transformer (no model_id specified)
- **THEN** checkpoint metadata indicates architecture_type is "custom-transformer"
- **AND** metadata includes model architecture params (n_layers, d_model, etc.)
- **AND** checkpoint is compatible with existing checkpoint format for backward compatibility

### Requirement: Checkpoint Load
The system SHALL provide functionality to load complete training state from disk, restoring model weights, optimizer state, training configuration, vocabulary, step counter, model metadata (model_name, model_id), and fine-tuning lineage.

#### Scenario: Load checkpoint restores model state
- **WHEN** `load_checkpoint()` is called with a checkpoint path
- **THEN** model state_dict is loaded and restored to the model
- **AND** model parameters match the saved checkpoint exactly

#### Scenario: Load checkpoint restores optimizer state
- **WHEN** `load_checkpoint()` is called with a checkpoint path
- **THEN** optimizer state_dict is loaded and restored to the optimizer
- **AND** optimizer state (momentum, etc.) matches the saved checkpoint

#### Scenario: Load checkpoint restores training state
- **WHEN** `load_checkpoint()` is called with a checkpoint path
- **THEN** training config is loaded and restored
- **AND** vocabulary is loaded and restored
- **AND** step counter is loaded and restored
- **AND** all checkpoint data is returned in a structured format

#### Scenario: Load checkpoint restores model metadata
- **WHEN** checkpoint with model metadata is loaded
- **THEN** model_name is restored if present
- **AND** model_id is restored if present
- **AND** architecture_type is restored
- **AND** source information is restored
- **AND** model metadata is accessible to caller

#### Scenario: Load checkpoint restores fine-tuning lineage
- **WHEN** checkpoint for fine-tuned model is loaded
- **THEN** fine_tuned_from field is restored
- **AND** full fine-tuning chain is accessible
- **AND** fine-tuning timestamps are preserved

#### Scenario: Load checkpoint with architecture adapter
- **WHEN** checkpoint is loaded for a specific architecture
- **THEN** appropriate adapter is selected based on architecture_type
- **AND** adapter loads weights correctly for that architecture
- **AND** model is ready for training or inference with correct architecture

#### Scenario: Handle checkpoints without model metadata (backward compatibility)
- **WHEN** older checkpoint without model metadata is loaded
- **THEN** system assumes custom-transformer architecture
- **AND** loads checkpoint successfully with defaults
- **AND** logs warning about missing metadata

### Requirement: Resume Training
The system SHALL provide functionality to resume training from a saved checkpoint, continuing training from the saved step with identical loss progression, preserving model metadata (model_name, model_id) and fine-tuning lineage.

#### Scenario: Resume continues step count
- **WHEN** training is resumed from a checkpoint
- **THEN** step counter continues from the saved step number
- **AND** subsequent steps increment correctly from the resumed step

#### Scenario: Resume produces identical loss progression
- **WHEN** training is interrupted, checkpointed, and resumed
- **THEN** loss values after resume match loss values from uninterrupted training (within numerical tolerance)
- **AND** model parameters after resume match parameters from uninterrupted training (within numerical tolerance)
- **AND** training behavior is identical to uninterrupted training

#### Scenario: Resume restores complete state
- **WHEN** training is resumed from a checkpoint
- **THEN** model is restored to saved state
- **AND** optimizer is restored to saved state
- **AND** training configuration matches saved configuration
- **AND** vocabulary matches saved vocabulary
- **AND** step counter matches saved step

#### Scenario: Resume preserves model metadata
- **WHEN** training is resumed from checkpoint with model metadata
- **THEN** model_name is preserved
- **AND** model_id is preserved
- **AND** architecture_type is preserved
- **AND** fine-tuning lineage is preserved
- **AND** subsequent checkpoints maintain metadata chain

#### Scenario: Resume as new fine-tuning run
- **WHEN** user resumes from checkpoint but specifies new model_name
- **THEN** system creates new fine-tuning variant
- **AND** new model_name is registered in registry
- **AND** fine_tuned_from points to original checkpoint's model_name
- **AND** new fine-tuning chain is started

### Requirement: Checkpoint Error Handling
The system SHALL handle checkpoint errors gracefully, providing clear error messages for missing, corrupted, or incompatible checkpoints.

#### Scenario: Missing checkpoint file
- **WHEN** `load_checkpoint()` is called with a non-existent checkpoint path
- **THEN** an appropriate error is raised (e.g., FileNotFoundError)
- **AND** error message clearly indicates the checkpoint file is missing

#### Scenario: Corrupted checkpoint file
- **WHEN** `load_checkpoint()` is called with a corrupted checkpoint file
- **THEN** an appropriate error is raised (e.g., RuntimeError or ValueError)
- **AND** error message clearly indicates the checkpoint file is corrupted or invalid

#### Scenario: Incompatible architecture checkpoint
- **WHEN** checkpoint is loaded with mismatched architecture_type
- **THEN** system detects incompatibility
- **AND** displays clear error explaining architecture mismatch
- **AND** suggests correct model_id or architecture to use
- **AND** does not proceed with loading

