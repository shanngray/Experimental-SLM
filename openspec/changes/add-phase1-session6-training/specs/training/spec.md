## ADDED Requirements

### Requirement: Loss Computation
The system SHALL provide loss computation for next-token prediction using cross-entropy loss over all sequence positions.

#### Scenario: Loss computation on model outputs
- **WHEN** `compute_loss()` receives logits of shape [B, 256, vocab_size] and targets of shape [B, 256]
- **THEN** it computes cross-entropy loss for next-token prediction over all positions
- **AND** returns a scalar loss value suitable for backward pass

#### Scenario: Loss correctness
- **WHEN** `compute_loss()` is called with known inputs
- **THEN** the computed loss matches expected values (within numerical tolerance)
- **AND** loss is differentiable for gradient computation

### Requirement: Training Step
The system SHALL provide a training step that performs forward pass, loss computation, backward pass, and optimizer update.

#### Scenario: Complete training step
- **WHEN** `Trainer.training_step()` is called with a batch of input data
- **THEN** it performs:
  - **AND** forward pass through the model
  - **AND** loss computation
  - **AND** backward pass (gradient computation)
  - **AND** optimizer step (parameter update)
- **AND** the step completes without errors

#### Scenario: Step counter increments
- **WHEN** `Trainer.training_step()` is called
- **THEN** the step counter increments by one
- **AND** the step number is tracked correctly across multiple steps

#### Scenario: Loss logging
- **WHEN** `Trainer.training_step()` completes
- **THEN** the loss value is logged (e.g., printed or written to log file)
- **AND** the logged loss corresponds to the computed loss for that step

### Requirement: Optimizer Configuration
The system SHALL provide AdamW optimizer configured with specific hyperparameters for training.

#### Scenario: AdamW optimizer setup
- **WHEN** `Trainer` is instantiated with a model
- **THEN** AdamW optimizer is configured with:
  - **AND** learning_rate=3e-4
  - **AND** weight_decay=0.1
  - **AND** betas=(0.9, 0.95)
- **AND** optimizer state is tracked correctly

#### Scenario: Optimizer updates parameters
- **WHEN** `Trainer.training_step()` completes an optimizer step
- **THEN** model parameters are updated according to computed gradients
- **AND** optimizer state (momentum, etc.) is maintained correctly

### Requirement: Configuration Management
The system SHALL provide a configuration system for managing training hyperparameters.

#### Scenario: Config loading
- **WHEN** training is initiated
- **THEN** hyperparameters can be loaded from configuration file or passed as parameters
- **AND** config includes learning rate, weight decay, betas, and other training parameters

#### Scenario: Config documentation
- **WHEN** config system is used
- **THEN** all config parameters are documented
- **AND** default values are clearly specified

### Requirement: Training Loop Integration
The system SHALL integrate training components to enable end-to-end training on datasets.

#### Scenario: Training on tiny dataset
- **WHEN** `Trainer` is used to train for a few steps on a tiny dataset
- **THEN** training completes without errors
- **AND** loss is computed and logged for each step
- **AND** model parameters are updated

#### Scenario: Loss decreases on synthetic data
- **WHEN** `Trainer` trains on simple synthetic data
- **THEN** loss decreases over multiple steps (smoke test)
- **AND** training demonstrates learning capability

