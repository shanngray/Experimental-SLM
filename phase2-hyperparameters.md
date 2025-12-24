# Phase 2: Hyperparameters

## Overview

Phase 2 focuses on centralizing all hyperparameter control to YAML configuration files in `/configs` and updating documentation to make hyperparameter management clear and comprehensive. Currently, many hyperparameters are hardcoded in the codebase, making experimentation difficult and the configuration system incomplete.

## Goals

1. **Audit all hyperparameters** - Identify every hardcoded value that should be configurable
2. **Extend configuration system** - Add missing hyperparameters to `TrainingConfig`
3. **Create YAML config structure** - Establish standard config file format with examples
4. **Update main.py** - Ensure all hyperparameters flow from config to components
5. **Comprehensive documentation** - Update README with full hyperparameter reference
6. **Validation and testing** - Ensure config system works end-to-end

## Current State Analysis

### Hyperparameters Already in Config (`TrainingConfig`)
- ✅ `learning_rate`: 3e-4
- ✅ `weight_decay`: 0.1
- ✅ `beta1`: 0.9
- ✅ `beta2`: 0.95
- ✅ `batch_size`: 16
- ✅ `max_seq_len`: 256
- ✅ `seed`: None (optional)
- ✅ `hooks`: Hook configuration dictionary
- ✅ `eval_cadence`: None (optional)
- ✅ `sampling_cadence`: None (optional)
- ✅ `sampling_temperature`: 1.0
- ✅ `sampling_prompt`: "The"
- ✅ `sampling_max_length`: 100
- ✅ `sampling_seed`: 42

### Hyperparameters Currently Hardcoded (Need to Add)

**Model Architecture** (hardcoded in `main.py` when creating `Transformer`):
- ❌ `n_layers`: 4
- ❌ `d_model`: 256
- ❌ `n_heads`: 4
- ❌ `d_ff`: 1024
- ❌ `dropout`: 0.1

**Dataset** (hardcoded in `main.py`):
- ❌ `train_ratio`: 0.95

**Training Loop** (hardcoded in `main.py`):
- ❌ `max_steps`: 10000 (default, can be overridden via CLI)
- ❌ `checkpoint_cadence`: 1000 (hardcoded checkpoint save frequency)

**Tokenizer** (hardcoded in `Tokenizer` class):
- ❌ Tokenizer type/policy (currently character-level ASCII, but should be documented)
- Note: Tokenizer hyperparameters may be Phase 3+ material, but should be documented

## Session Breakdown

| Session | Focus | Dependencies | Estimated Complexity |
|---------|-------|--------------|---------------------|
| 1 | Hyperparameter Audit | None | Low |
| 2 | Extend Config System | Session 1 | Medium |
| 3 | Create YAML Config Structure | Session 2 | Low |
| 4 | Update main.py Integration | Session 2 | Medium |
| 5 | Update README Documentation | Session 3 | Low |
| 6 | Testing & Validation | Sessions 2-4 | Medium |

---

## Session 1: Hyperparameter Audit

**Goal:** Comprehensively identify all hyperparameters in the codebase and document their current locations, default values, and intended usage.

**Deliverables:**
- Complete inventory of all hyperparameters
- Documentation of where each is currently defined/used
- Identification of any missing hyperparameters that should be configurable
- Notes on hyperparameters that should remain hardcoded (if any)

**Tasks:**
- [x] Search codebase for all numeric constants that represent hyperparameters
- [x] Review `Transformer.__init__` to identify model architecture parameters
- [x] Review `main.py` to identify hardcoded training/dataset parameters
- [x] Review `split_corpus` to identify dataset split parameters
- [x] Review checkpoint saving logic to identify checkpoint-related parameters
- [x] Review evaluation logic to identify evaluation-related parameters
- [x] Review sampling logic to identify sampling-related parameters
- [x] Document current default values for each hyperparameter
- [x] Identify reasonable value ranges/constraints for each hyperparameter
- [x] Categorize hyperparameters (Model Architecture, Training, Dataset, Evaluation, Sampling, Checkpointing)
- [x] Create summary document listing all findings

**Acceptance Criteria:**
- Complete list of all hyperparameters with current values
- Clear categorization of each hyperparameter
- Identification of which hyperparameters need to be added to config
- Documentation of any hyperparameters that should remain hardcoded (with justification)

**Files to Review:**
- `src/config.py` - Current config structure
- `src/model/transformer.py` - Model architecture defaults
- `main.py` - Training loop and hardcoded values
- `src/dataset.py` - Dataset split defaults
- `src/training/trainer.py` - Training-related defaults
- `src/training/checkpoint.py` - Checkpoint-related defaults
- `src/evaluation/evaluator.py` - Evaluation defaults
- `src/sampling/sampler.py` - Sampling defaults
- `README.md` - Current documentation

**Output:**
- Markdown document listing all hyperparameters with:
  - Current value
  - Current location (file/line)
  - Category
  - Whether it should be configurable
  - Suggested default value
  - Value constraints/notes

**Status:** ✅ **COMPLETE**

**Deliverables Created:**
- `docs/hyperparameter-audit.md` - Complete inventory of all hyperparameters with detailed analysis
- `docs/hyperparameter-audit-summary.md` - Actionable summary for Phase 2 Sessions 2-6

**Key Findings:**
- **8 hyperparameters** need to be added to config (all HIGH priority):
  - Model Architecture: `n_layers`, `d_model`, `n_heads`, `d_ff`, `dropout`
  - Dataset: `train_ratio`
  - Training: `max_steps`
  - Checkpointing: `checkpoint_cadence`
- **14 hyperparameters** already in config (no changes needed)
- All hardcoded values identified and documented with locations
- Value constraints and ranges documented for all hyperparameters

---

## Session 2: Extend Config System

**Goal:** Extend `TrainingConfig` to include all missing hyperparameters identified in Session 1.

**Dependencies:** Session 1 (Hyperparameter Audit)

**Deliverables:**
- Updated `TrainingConfig` class with all hyperparameters
- Updated `from_dict` method to handle new fields
- Updated `to_dict` method to include new fields
- Updated docstrings with comprehensive descriptions
- Type hints for all new fields

**Tasks:**
- [ ] Add model architecture hyperparameters to `TrainingConfig`:
  - [ ] `n_layers: int = 4`
  - [ ] `d_model: int = 256`
  - [ ] `n_heads: int = 4`
  - [ ] `d_ff: int = 1024`
  - [ ] `dropout: float = 0.1`
- [ ] Add dataset hyperparameters:
  - [ ] `train_ratio: float = 0.95`
- [ ] Add training loop hyperparameters:
  - [ ] `max_steps: int = 10000`
  - [ ] `checkpoint_cadence: int = 1000`
- [ ] Update `from_dict` to include new fields in `valid_keys`
- [ ] Update `to_dict` to serialize new fields
- [ ] Add comprehensive docstrings explaining each hyperparameter:
  - [ ] Purpose/effect
  - [ ] Default value
  - [ ] Reasonable value ranges
  - [ ] Notes on interactions with other hyperparameters
- [ ] Add validation logic (if needed) for value constraints
- [ ] Ensure backward compatibility (existing configs should still work)

**Acceptance Criteria:**
- All hyperparameters from Session 1 audit are included
- `TrainingConfig` can be instantiated with defaults
- `TrainingConfig.from_dict()` correctly parses all fields
- `TrainingConfig.to_dict()` correctly serializes all fields
- Docstrings are comprehensive and helpful
- No breaking changes to existing code
- Type hints are correct for all fields

**Files to Modify:**
- `src/config.py` - Extend `TrainingConfig` class

**Files to Review:**
- `src/config.py` - Current implementation
- Tests that use `TrainingConfig` - Ensure compatibility

**Design Considerations:**
- Should maintain backward compatibility with existing configs
- Consider grouping related hyperparameters (e.g., model architecture could be a nested dict, but keep flat for simplicity initially)
- Ensure all numeric types are correct (int vs float)
- Consider adding validation for value ranges (e.g., dropout between 0 and 1)

---

## Session 3: Create YAML Config Structure

**Goal:** Create example YAML configuration files demonstrating all hyperparameters and establish a standard config file format.

**Dependencies:** Session 2 (Extended Config System)

**Deliverables:**
- Default configuration file (`configs/default.yaml`)
- Example configuration files for different scenarios
- Documentation of YAML config file format
- Comments in YAML files explaining each hyperparameter

**Tasks:**
- [ ] Create `configs/default.yaml` with all hyperparameters:
  - [ ] Model architecture section with comments
  - [ ] Training hyperparameters section with comments
  - [ ] Dataset hyperparameters section with comments
  - [ ] Evaluation/sampling hyperparameters section with comments
  - [ ] Checkpoint hyperparameters section with comments
- [ ] Create `configs/small-model.yaml` example:
  - [ ] Smaller model (e.g., 2 layers, 128 dim)
  - [ ] Appropriate batch size for smaller model
- [ ] Create `configs/large-model.yaml` example:
  - [ ] Larger model (e.g., 6 layers, 512 dim)
  - [ ] Appropriate batch size for larger model
- [ ] Create `configs/fast-training.yaml` example:
  - [ ] Faster training settings (larger batch, fewer steps)
- [ ] Create `configs/detailed-eval.yaml` example:
  - [ ] More frequent evaluation and sampling
- [ ] Add comments to all YAML files explaining:
  - [ ] What each hyperparameter does
  - [ ] Default values
  - [ ] When to change them
  - [ ] Common value ranges
- [ ] Create `configs/README.md` documenting:
  - [ ] How to use config files
  - [ ] How to create custom configs
  - [ ] How configs are loaded
  - [ ] Examples of common modifications

**Acceptance Criteria:**
- At least 3 example config files created
- All config files are valid YAML
- All config files include helpful comments
- Default config matches current default behavior
- Example configs demonstrate different use cases
- README explains config system clearly

**Files to Create:**
- `configs/default.yaml`
- `configs/small-model.yaml`
- `configs/large-model.yaml`
- `configs/fast-training.yaml`
- `configs/detailed-eval.yaml`
- `configs/README.md`

**Design Considerations:**
- Use clear section headers/comments in YAML
- Group related hyperparameters together
- Include inline comments explaining each value
- Ensure YAML files are human-readable and well-formatted
- Consider using YAML anchors/aliases if there's repetition (but keep simple initially)

---

## Session 4: Update main.py Integration

**Goal:** Update `main.py` to use config for all hyperparameters, removing hardcoded values.

**Dependencies:** Session 2 (Extended Config System)

**Deliverables:**
- Updated `main.py` that reads all hyperparameters from config
- Removed hardcoded values
- Proper error handling for missing/invalid config values
- Clear logging of which config is being used

**Tasks:**
- [ ] Update `Transformer` instantiation to use config:
  - [ ] `n_layers` from config
  - [ ] `d_model` from config
  - [ ] `n_heads` from config
  - [ ] `d_ff` from config
  - [ ] `dropout` from config
- [ ] Update `split_corpus` call to use `config.train_ratio`
- [ ] Update checkpoint saving logic to use `config.checkpoint_cadence`
- [ ] Update `max_steps` handling:
  - [ ] Read from config by default
  - [ ] Allow CLI override (keep `--max-steps` argument)
  - [ ] Prefer CLI override if provided
- [ ] Add validation that required config values are present
- [ ] Add logging to show which config values are being used
- [ ] Ensure backward compatibility:
  - [ ] Default config created if none provided
  - [ ] Missing config values use defaults from `TrainingConfig`
- [ ] Update error messages to be helpful if config is invalid

**Acceptance Criteria:**
- No hardcoded hyperparameter values in `main.py`
- All hyperparameters flow from config to components
- CLI override for `max_steps` still works
- Default config is used if no config file provided
- Missing config values use sensible defaults
- Clear error messages for invalid configs
- Logging shows active config values

**Files to Modify:**
- `main.py` - Update to use config for all hyperparameters

**Files to Review:**
- `main.py` - Current implementation
- `src/config.py` - Config structure
- `src/model/transformer.py` - Model constructor signature

**Design Considerations:**
- Maintain CLI argument for `--max-steps` as convenience override
- Ensure config loading happens early in `main()`
- Consider logging config hash or summary for reproducibility
- Handle edge cases (missing config file, invalid values, etc.)

---

## Session 5: Update README Documentation

**Goal:** Update README with comprehensive hyperparameter documentation, including how to modify them and what values they take.

**Dependencies:** Session 3 (YAML Config Structure)

**Deliverables:**
- Updated README section on Configuration
- Comprehensive hyperparameter reference
- Examples of modifying hyperparameters
- Links to example config files

**Tasks:**
- [ ] Replace existing "Configuration" section with comprehensive guide:
  - [ ] Overview of config system
  - [ ] How to use config files
  - [ ] How to create custom configs
  - [ ] How to override via CLI (if applicable)
- [ ] Add detailed hyperparameter reference section:
  - [ ] Model Architecture hyperparameters:
    - [ ] `n_layers` - description, default, range, notes
    - [ ] `d_model` - description, default, range, notes
    - [ ] `n_heads` - description, default, range, notes
    - [ ] `d_ff` - description, default, range, notes
    - [ ] `dropout` - description, default, range, notes
  - [ ] Training hyperparameters:
    - [ ] `learning_rate` - description, default, range, notes
    - [ ] `weight_decay` - description, default, range, notes
    - [ ] `beta1`, `beta2` - description, default, range, notes
    - [ ] `batch_size` - description, default, range, notes
    - [ ] `max_steps` - description, default, range, notes
  - [ ] Dataset hyperparameters:
    - [ ] `max_seq_len` - description, default, range, notes
    - [ ] `train_ratio` - description, default, range, notes
  - [ ] Evaluation/Sampling hyperparameters:
    - [ ] `eval_cadence` - description, default, range, notes
    - [ ] `sampling_cadence` - description, default, range, notes
    - [ ] `sampling_temperature` - description, default, range, notes
    - [ ] `sampling_prompt` - description, default, range, notes
    - [ ] `sampling_max_length` - description, default, range, notes
  - [ ] Checkpointing hyperparameters:
    - [ ] `checkpoint_cadence` - description, default, range, notes
  - [ ] Other hyperparameters:
    - [ ] `seed` - description, default, range, notes
- [ ] Add examples section:
  - [ ] Example: Using default config
  - [ ] Example: Using custom config file
  - [ ] Example: Modifying specific hyperparameters
  - [ ] Example: Creating a small model config
  - [ ] Example: Creating a large model config
- [ ] Add troubleshooting section:
  - [ ] Common config errors
  - [ ] How to validate config files
  - [ ] How to check which config is active
- [ ] Update Quick Start section to mention config files
- [ ] Add links to example config files in `configs/` directory

**Acceptance Criteria:**
- README has comprehensive hyperparameter documentation
- All hyperparameters are documented with defaults and ranges
- Examples are clear and practical
- Documentation matches actual config system behavior
- Links to example config files work
- Troubleshooting section is helpful

**Files to Modify:**
- `README.md` - Update Configuration section and add hyperparameter reference

**Design Considerations:**
- Use clear formatting (tables, code blocks, etc.)
- Include practical examples
- Explain when/why to change each hyperparameter
- Link to example config files
- Keep documentation up-to-date with code

---

## Session 6: Testing & Validation

**Goal:** Ensure the config system works end-to-end, all hyperparameters are properly loaded, and the system is robust.

**Dependencies:** Sessions 2-4 (Config System, YAML Files, main.py Integration)

**Deliverables:**
- Updated tests for `TrainingConfig`
- Integration tests for config loading
- Validation that all hyperparameters work correctly
- Documentation of any edge cases or limitations

**Tasks:**
- [ ] Update existing `TrainingConfig` tests:
  - [ ] Test new hyperparameters are included
  - [ ] Test defaults match expected values
  - [ ] Test `from_dict` with new fields
  - [ ] Test `to_dict` includes new fields
- [ ] Add tests for config loading from YAML:
  - [ ] Test loading default config
  - [ ] Test loading custom config
  - [ ] Test missing fields use defaults
  - [ ] Test invalid YAML handling
  - [ ] Test invalid value types handling
- [ ] Add integration tests:
  - [ ] Test `main.py` uses config for model creation
  - [ ] Test `main.py` uses config for dataset split
  - [ ] Test `main.py` uses config for checkpoint cadence
  - [ ] Test CLI override still works
  - [ ] Test default config is used when no file provided
- [ ] Manual testing:
  - [ ] Load each example config file
  - [ ] Verify model is created with correct architecture
  - [ ] Verify training uses correct hyperparameters
  - [ ] Verify checkpointing uses correct cadence
  - [ ] Verify evaluation uses correct settings
- [ ] Test backward compatibility:
  - [ ] Old config files (missing new fields) still work
  - [ ] Default behavior unchanged when no config provided
- [ ] Document any limitations or known issues

**Acceptance Criteria:**
- All tests pass
- Config loading works for all example configs
- Integration tests verify end-to-end functionality
- Backward compatibility maintained
- Error handling is robust
- Documentation is accurate

**Files to Modify:**
- `tests/test_config.py` (or create if doesn't exist)
- `tests/test_integration.py` - Add config-related tests

**Files to Review:**
- All test files that use `TrainingConfig`
- `main.py` - Ensure testable

**Design Considerations:**
- Test both valid and invalid configs
- Test edge cases (empty config, missing fields, wrong types)
- Ensure tests are maintainable
- Consider property-based testing for value validation

---

## Success Criteria for Phase 2

1. ✅ All hyperparameters are configurable via YAML files
2. ✅ No hardcoded hyperparameter values in `main.py`
3. ✅ Comprehensive example config files exist
4. ✅ README has complete hyperparameter documentation
5. ✅ Config system is tested and validated
6. ✅ Backward compatibility is maintained
7. ✅ Users can easily modify hyperparameters without code changes

## Post-Phase 2 Considerations

After Phase 2 completion, consider:
- **Phase 3:** Advanced hyperparameter features (learning rate schedules, gradient clipping, etc.)
- **Phase 3+:** Hyperparameter search/optimization tools
- **Phase 3+:** Config validation schema (e.g., JSON Schema or Pydantic)
- **Phase 3+:** Config inheritance/merging (base config + override configs)

## Notes

- This phase focuses on centralization and documentation, not adding new hyperparameters
- Keep the config system simple and maintainable
- Prioritize clarity and ease of use over advanced features
- Ensure all changes are backward compatible
- Document everything thoroughly
