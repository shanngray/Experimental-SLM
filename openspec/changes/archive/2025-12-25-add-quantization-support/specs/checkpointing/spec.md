## MODIFIED Requirements

### Requirement: Checkpoint Save
The system SHALL provide functionality to save complete training state to disk, including model weights, optimizer state, training configuration, vocabulary, and step counter. For quantized models, quantization metadata and quantized state_dict SHALL also be saved.

#### Scenario: Save checkpoint creates files
- **WHEN** `save_checkpoint()` is called with model, optimizer, config, vocab, and step
- **THEN** checkpoint files are created in the checkpoints directory
- **AND** model state_dict is saved in binary format (PyTorch format)
- **AND** optimizer state_dict is saved
- **AND** training config is saved in JSON-compatible format
- **AND** vocabulary is saved
- **AND** step counter is saved
- **AND** metadata file contains checkpoint information

#### Scenario: Save quantized checkpoint
- **WHEN** `save_checkpoint()` is called with a quantized model
- **THEN** quantization metadata is saved in `quantization_metadata.json`
- **AND** quantized model state_dict is saved
- **AND** quantization parameters (scales, zero-points) are included
- **AND** checkpoint metadata indicates quantization mode and bits
- **AND** checkpoint can be distinguished from full-precision checkpoints

#### Scenario: Checkpoint format
- **WHEN** checkpoint is saved
- **THEN** checkpoint uses PyTorch format (torch.save) for binary weights
- **AND** metadata is stored in JSON format for human readability
- **AND** checkpoint files are organized in checkpoints/ directory
- **AND** checkpoint format version is included in metadata

### Requirement: Checkpoint Load
The system SHALL provide functionality to load complete training state from disk, restoring model weights, optimizer state, training configuration, vocabulary, and step counter. For quantized checkpoints, quantization metadata and quantized state_dict SHALL also be restored.

#### Scenario: Load checkpoint restores model state
- **WHEN** `load_checkpoint()` is called with a checkpoint path
- **THEN** model state_dict is loaded and restored to the model
- **AND** model parameters match the saved checkpoint exactly

#### Scenario: Load quantized checkpoint
- **WHEN** `load_checkpoint()` is called with a quantized checkpoint path
- **THEN** quantization metadata is loaded from `quantization_metadata.json`
- **AND** quantized model state_dict is restored
- **AND** quantization parameters are restored
- **AND** model is restored in quantized format
- **AND** model can be used for inference or fine-tuning

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

#### Scenario: Backward compatibility with old checkpoints
- **WHEN** `load_checkpoint()` is called with an old checkpoint (no quantization metadata)
- **THEN** checkpoint loads successfully as full-precision model
- **AND** no errors are raised due to missing quantization metadata
- **AND** model is restored in FP32 format

### Requirement: Resume Training
The system SHALL provide functionality to resume training from a saved checkpoint, continuing training from the saved step with identical loss progression. For quantized checkpoints, fine-tuning SHALL be supported if enabled.

#### Scenario: Resume continues step count
- **WHEN** training is resumed from a checkpoint
- **THEN** step counter continues from the saved step number
- **AND** subsequent steps increment correctly from the resumed step

#### Scenario: Resume quantized model for fine-tuning
- **WHEN** training is resumed from a quantized checkpoint with fine-tuning enabled
- **THEN** model is restored in quantized format
- **AND** training can continue with quantized weights
- **AND** quantization is maintained throughout fine-tuning

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
- **AND** quantization state is restored if present

### Requirement: Checkpoint Error Handling
The system SHALL handle checkpoint errors gracefully, providing clear error messages for missing or corrupted checkpoints. Errors SHALL also handle quantization-related issues.

#### Scenario: Missing checkpoint file
- **WHEN** `load_checkpoint()` is called with a non-existent checkpoint path
- **THEN** an appropriate error is raised (e.g., FileNotFoundError)
- **AND** error message clearly indicates the checkpoint file is missing

#### Scenario: Corrupted checkpoint file
- **WHEN** `load_checkpoint()` is called with a corrupted checkpoint file
- **THEN** an appropriate error is raised (e.g., RuntimeError or ValueError)
- **AND** error message clearly indicates the checkpoint file is corrupted or invalid

#### Scenario: Missing quantization metadata
- **WHEN** `load_checkpoint()` is called on a checkpoint that claims to be quantized but lacks quantization metadata
- **THEN** an appropriate error is raised with clear message
- **AND** error message indicates missing quantization information

