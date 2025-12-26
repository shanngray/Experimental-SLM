## MODIFIED Requirements
### Requirement: Integration Testing
The system SHALL provide comprehensive integration tests that verify the full training pipeline end-to-end, including thorough testing and validation of the configuration system to ensure all hyperparameters are correctly loaded, config system works end-to-end with all components, backward compatibility is maintained, and error handling is robust.

#### Scenario: End-to-end training pipeline test
- **WHEN** integration test runs with tiny corpus
- **THEN** test executes: load data → tokenize → create datasets → train → checkpoint → resume → verify
- **AND** training completes without errors
- **AND** full pipeline works correctly

#### Scenario: Checkpoint resume produces identical results
- **WHEN** training is interrupted and resumed from checkpoint
- **THEN** resumed training produces identical loss progression
- **AND** step counter continues correctly
- **AND** results are reproducible

#### Scenario: Reproducibility verification
- **WHEN** multiple training runs use same seed and configuration
- **THEN** all runs produce identical loss values at each step
- **AND** generated samples are identical (when using fixed sampling seed)
- **AND** results are deterministic

#### Scenario: Training dynamics verification (smoke test)
- **WHEN** training runs on simple synthetic or real data
- **THEN** loss decreases over time
- **AND** model demonstrates learning capability
- **AND** generated text samples improve qualitatively

#### Scenario: Logging completeness verification
- **WHEN** training runs and logs are captured
- **THEN** logs include: run_id, config_hash, git_commit, step, loss
- **AND** val_loss is logged when eval_cadence triggers
- **AND** sample_text is logged when sampling_cadence triggers
- **AND** log format is parseable (structured text or JSON-like format)
- **AND** all required metadata is present for reproducibility

#### Scenario: TrainingConfig includes all new hyperparameters
- **WHEN** TrainingConfig tests run
- **THEN** tests verify all new hyperparameters are included: n_layers (default: 4), d_model (default: 256), n_heads (default: 4), d_ff (default: 1024), dropout (default: 0.1), train_ratio (default: 0.95), max_steps (default: 10000), checkpoint_cadence (default: 1000)
- **AND** tests verify defaults match expected values
- **AND** tests verify `from_dict` correctly loads new fields
- **AND** tests verify `to_dict` includes new fields

#### Scenario: Config loading from YAML works correctly
- **WHEN** config loading tests run
- **THEN** tests verify default config (`configs/default.yaml`) loads successfully
- **AND** tests verify custom config files load successfully
- **AND** tests verify missing fields use TrainingConfig defaults
- **AND** tests verify invalid YAML syntax raises clear error messages
- **AND** tests verify invalid value types raise appropriate errors

#### Scenario: Config integration with main.py works correctly
- **WHEN** integration tests run with config files
- **THEN** tests verify `main.py` uses config for model creation (all architecture hyperparameters: n_layers, d_model, n_heads, d_ff, dropout)
- **AND** tests verify `main.py` uses config for dataset split (train_ratio)
- **AND** tests verify `main.py` uses config for checkpoint cadence (checkpoint_cadence)
- **AND** tests verify CLI override (`--max-steps`) still works and takes precedence
- **AND** tests verify default config is used when no config file provided

#### Scenario: Backward compatibility with old configs
- **WHEN** old config files (missing new hyperparameters) are loaded
- **THEN** configs load successfully without errors
- **AND** missing fields use TrainingConfig defaults
- **AND** training proceeds normally with defaults
- **AND** default behavior matches previous hardcoded values when no config provided

#### Scenario: Example config files work correctly
- **WHEN** each example config file is loaded and used
- **THEN** `configs/default.yaml` loads and works correctly
- **AND** `configs/small-model.yaml` loads and creates smaller model with correct architecture
- **AND** `configs/large-model.yaml` loads and creates larger model with correct architecture
- **AND** `configs/fast-training.yaml` loads and uses faster training settings
- **AND** `configs/detailed-eval.yaml` loads and uses more frequent evaluation/sampling
- **AND** model architecture matches config values for each example
- **AND** training uses correct hyperparameters from config
- **AND** checkpointing uses correct cadence from config
- **AND** evaluation uses correct settings from config

