# Change: Phase 2 Session 1 - Hyperparameter Audit

## Why
Currently, many hyperparameters are hardcoded throughout the codebase (e.g., model architecture parameters in `main.py`, dataset split ratio, checkpoint cadence), making experimentation difficult and the configuration system incomplete. Before extending the configuration system to centralize all hyperparameters, we need to comprehensively identify every hyperparameter in the codebase, document their current locations and default values, and determine which ones should be configurable. This audit will provide the foundation for Phase 2 Sessions 2-6, which will extend the config system, create YAML configs, and update documentation.

## What Changes
- Create comprehensive inventory document listing all hyperparameters in the codebase
- Document current location (file/line) and default value for each hyperparameter
- Categorize hyperparameters (Model Architecture, Training, Dataset, Evaluation, Sampling, Checkpointing)
- Identify which hyperparameters should be configurable vs remain hardcoded (with justification)
- Document reasonable value ranges/constraints for each hyperparameter
- Create summary document that will guide Phase 2 Sessions 2-6 implementation
- Review all relevant source files to identify hardcoded values:
  - `src/config.py` - Current config structure
  - `src/model/transformer.py` - Model architecture defaults
  - `main.py` - Training loop and hardcoded values
  - `src/dataset.py` - Dataset split defaults
  - `src/training/trainer.py` - Training-related defaults
  - `src/training/checkpoint.py` - Checkpoint-related defaults
  - `src/evaluation/evaluator.py` - Evaluation defaults
  - `src/sampling/sampler.py` - Sampling defaults

## Impact
- Affected specs: New `hyperparameter-audit` capability specification
- Affected code:
  - No code changes in this session (documentation/analysis only)
  - Will create audit document (markdown) listing all findings
- Dependencies:
  - Requires understanding of all Phase 1 components (tokenizer, dataset, dataloader, model, trainer, checkpointing, evaluation, sampling)
  - Uses existing codebase as reference
- Future impact:
  - Audit results will inform Phase 2 Sessions 2-6 (config extension, YAML creation, main.py updates, README updates, testing)
  - Ensures no hyperparameters are missed when centralizing configuration
  - Provides foundation for comprehensive hyperparameter documentation

