## MODIFIED Requirements

### Requirement: Configuration Loading
The system SHALL load training configuration, supporting YAML files or defaults, with command-line overrides. TrainingConfig SHALL include model selection field (model_name) to specify which model from the registry to load. The system SHALL provide example YAML configuration files demonstrating model selection for different architectures.

#### Scenario: Load configuration from YAML file
- **WHEN** `main.py` is invoked with `--config <path>` argument
- **THEN** configuration is loaded from YAML file and converted to TrainingConfig
- **AND** loaded values override TrainingConfig defaults
- **AND** TrainingConfig includes model_name field for model selection
- **AND** TrainingConfig includes all existing hyperparameters

#### Scenario: Use default configuration
- **WHEN** `main.py` is invoked without `--config` argument
- **THEN** default TrainingConfig is used
- **AND** model_name defaults to None (uses custom Transformer)
- **AND** training proceeds with sensible defaults for all hyperparameters

#### Scenario: Load configuration with model_name
- **WHEN** YAML config includes `model_name: "qwen-0.5b-base"`
- **THEN** TrainingConfig.model_name is set to "qwen-0.5b-base"
- **AND** system will load specified model from registry
- **AND** model's architecture config takes precedence over TrainingConfig architecture params

#### Scenario: Command-line argument overrides
- **WHEN** `main.py` is invoked with `--max-steps <N>`
- **THEN** max_steps override is applied to configuration
- **AND** training runs for exactly N steps

#### Scenario: TrainingConfig serialization includes model_name
- **WHEN** TrainingConfig.to_dict() is called
- **THEN** model_name field is included in serialized dictionary
- **AND** model_name can be None for custom Transformer

#### Scenario: Example configs for different models
- **WHEN** users examine the `configs/` directory
- **THEN** example configs exist for different model types:
  - `configs/custom-transformer.yaml` - using custom architecture
  - `configs/qwen-base.yaml` - using imported Qwen base model
  - `configs/qwen-finetuned.yaml` - using fine-tuned Qwen variant
- **AND** each config includes comments explaining model_id usage

### Requirement: Model Loading and Initialization
The system SHALL initialize models based on config.model_name, loading from registry for imported models or creating custom Transformer when model_name is None. The system SHALL use appropriate architecture adapter for each model type.

#### Scenario: Load custom Transformer when model_name is None
- **WHEN** config.model_name is None or not specified
- **THEN** system creates custom Transformer using architecture params from config
- **AND** uses CustomTransformerAdapter to wrap model
- **AND** model is initialized with config.n_layers, config.d_model, etc.
- **AND** training proceeds as before (backward compatible behavior)

#### Scenario: Load model from registry by model_name
- **WHEN** config.model_name is specified (e.g., "qwen-0.5b-base")
- **THEN** system queries model registry for model_name
- **AND** retrieves model metadata (model_id, architecture_type, local_path, etc.)
- **AND** loads model weights from local_path
- **AND** selects appropriate adapter based on architecture_type
- **AND** initializes adapter with loaded model

#### Scenario: Model registry lookup fails
- **WHEN** config.model_name references non-existent model in registry
- **THEN** system displays clear error message indicating model not found
- **AND** lists available model_names from registry
- **AND** exits gracefully without proceeding to training

#### Scenario: Model files missing for registry entry
- **WHEN** registry entry exists but local model files are missing
- **THEN** system displays clear error message indicating missing files
- **AND** suggests re-importing model or running validate-registry
- **AND** exits gracefully

#### Scenario: Initialize optimizer with adapter model
- **WHEN** model is loaded via adapter
- **THEN** optimizer is created with adapter's trainable parameters
- **AND** optimizer works correctly regardless of architecture
- **AND** gradients flow correctly during training

### Requirement: Model Metadata Logging
The system SHALL log comprehensive model metadata at training start, including model source, architecture type, and fine-tuning lineage.

#### Scenario: Log custom Transformer metadata
- **WHEN** training starts with custom Transformer (model_name is None)
- **THEN** logs indicate architecture_type is "custom-transformer"
- **AND** logs include model architecture params (n_layers, d_model, n_heads, etc.)
- **AND** logs indicate model is trained from scratch

#### Scenario: Log imported model metadata
- **WHEN** training starts with imported model from registry
- **THEN** logs include model_name
- **AND** logs include model_id (original HuggingFace repo ID)
- **AND** logs include architecture_type (e.g., "qwen")
- **AND** logs include source (e.g., "huggingface")
- **AND** logs include model size (parameters, disk space)

#### Scenario: Log fine-tuning lineage
- **WHEN** training starts with a fine-tuned model or creating new fine-tune
- **THEN** logs include fine_tuned_from field showing parent model
- **AND** logs include full fine-tuning chain if applicable
- **AND** logs distinguish between resuming existing fine-tune vs starting new fine-tune

### Requirement: Fine-Tuning Workflow
The system SHALL support fine-tuning imported models and automatically track fine-tuning lineage in checkpoints and registry.

#### Scenario: Start fine-tuning from base model
- **WHEN** training starts with base model from registry (e.g., config.model_name = "qwen-0.5b-base")
- **THEN** system loads base model weights
- **AND** training proceeds with loaded model
- **AND** saved checkpoints include fine_tuned_from = "qwen-0.5b-base"
- **AND** checkpoints track fine-tuning start timestamp

#### Scenario: Register fine-tuned model after training
- **WHEN** training completes or checkpoint is saved
- **THEN** user can register fine-tuned checkpoint as new model in registry
- **AND** new model entry includes fine_tuned_from lineage
- **AND** new model can be used as model_name in subsequent configs

#### Scenario: Continue fine-tuning from checkpoint
- **WHEN** training resumes from checkpoint of fine-tuned model
- **THEN** fine-tuning lineage is preserved from checkpoint
- **AND** subsequent checkpoints maintain lineage chain
- **AND** logs indicate continuing existing fine-tuning run

#### Scenario: Start new fine-tuning branch from checkpoint
- **WHEN** user loads checkpoint but specifies new model_name
- **THEN** system creates new fine-tuning branch
- **AND** new branch's fine_tuned_from points to checkpoint's model_name
- **AND** new registry entry created for the branch
- **AND** original model remains unchanged

### Requirement: Model Management CLI Commands
The system SHALL provide CLI commands for managing models: importing, listing, viewing details, and deleting.

#### Scenario: Import model command
- **WHEN** user runs `python main.py import-model Qwen/Qwen-0.5B`
- **THEN** model is downloaded and imported as described in model-import spec
- **AND** command is integrated into main.py as subcommand

#### Scenario: List models command
- **WHEN** user runs `python main.py list-models`
- **THEN** displays all models from registry as described in model-registry spec
- **AND** command is integrated into main.py as subcommand

#### Scenario: Model info command
- **WHEN** user runs `python main.py model-info <model-name>`
- **THEN** displays detailed model information as described in model-registry spec
- **AND** command is integrated into main.py as subcommand

#### Scenario: Delete model command
- **WHEN** user runs `python main.py delete-model <model-name>`
- **THEN** deletes model as described in model-registry spec
- **AND** command is integrated into main.py as subcommand

#### Scenario: Main.py subcommand routing
- **WHEN** main.py is invoked with subcommand (e.g., `python main.py import-model`, `python main.py list-models`)
- **THEN** main.py routes to appropriate handler function
- **AND** default behavior (no subcommand) runs training as before
- **AND** `--help` displays all available subcommands

### Requirement: Tokenizer Handling for Different Architectures
The system SHALL support different tokenizers for different model architectures, using model's native tokenizer by default with optional override capability.

#### Scenario: Use model's native tokenizer
- **WHEN** model is loaded from registry with native tokenizer
- **THEN** system uses model's tokenizer for encoding/decoding
- **AND** tokenizer is loaded from model's directory
- **AND** tokenizer is used consistently throughout training and inference

#### Scenario: Use char-level tokenizer for custom Transformer
- **WHEN** custom Transformer is used (model_name is None)
- **THEN** system uses char-level tokenizer as before
- **AND** behavior is backward compatible with existing implementation

#### Scenario: Override tokenizer in config
- **WHEN** config includes tokenizer override option
- **THEN** system uses specified tokenizer instead of model's native tokenizer
- **AND** override is documented in training metadata
- **AND** warnings are logged about potential compatibility issues

#### Scenario: Tokenizer in checkpoint
- **WHEN** checkpoint is saved
- **THEN** checkpoint includes tokenizer identifier
- **AND** checkpoint can be loaded with correct tokenizer
- **AND** generated samples use correct tokenizer for decoding

### Requirement: Integration Testing for Multi-Model Support
The system SHALL provide comprehensive integration tests verifying multi-model support end-to-end, including model import, loading, fine-tuning, and checkpoint resume for different architectures.

#### Scenario: End-to-end custom Transformer (backward compatibility)
- **WHEN** integration test runs without model_name
- **THEN** custom Transformer is created and trained
- **AND** behavior matches pre-multi-model implementation
- **AND** checkpoints save and load correctly

#### Scenario: End-to-end Qwen model import and inference
- **WHEN** integration test imports Qwen model and runs inference
- **THEN** model imports successfully
- **AND** model loads correctly from registry
- **AND** forward pass produces logits
- **AND** text generation works

#### Scenario: End-to-end Qwen model fine-tuning
- **WHEN** integration test fine-tunes imported Qwen model
- **THEN** model trains successfully
- **AND** loss decreases over training
- **AND** checkpoints save correctly with metadata
- **AND** fine-tuning lineage is tracked

#### Scenario: Checkpoint resume with Qwen model
- **WHEN** integration test interrupts and resumes Qwen fine-tuning
- **THEN** training resumes correctly from checkpoint
- **AND** model metadata is preserved
- **AND** fine-tuning lineage is maintained
- **AND** loss progression continues identically

#### Scenario: Switch between model architectures
- **WHEN** integration test trains custom Transformer, then switches to Qwen model
- **THEN** both models train successfully in separate runs
- **AND** checkpoints for each are isolated and correct
- **AND** configs correctly specify model_name for each

#### Scenario: Model registry CLI integration test
- **WHEN** integration test exercises registry CLI commands
- **THEN** list-models works correctly
- **AND** model-info displays accurate information
- **AND** delete-model removes models correctly
- **AND** registry remains consistent after operations

