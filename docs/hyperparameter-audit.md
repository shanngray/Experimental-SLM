# Hyperparameter Audit - Complete Inventory

This document provides a comprehensive inventory of all hyperparameters in the codebase, their current locations, default values, and recommendations for configuration.

**Generated:** Phase 2 Session 1  
**Purpose:** Foundation for centralizing all hyperparameters in the configuration system

---

## Table of Contents

1. [Hyperparameters Already in Config](#hyperparameters-already-in-config)
2. [Model Architecture Hyperparameters](#model-architecture-hyperparameters)
3. [Training Hyperparameters](#training-hyperparameters)
4. [Dataset Hyperparameters](#dataset-hyperparameters)
5. [Evaluation Hyperparameters](#evaluation-hyperparameters)
6. [Sampling Hyperparameters](#sampling-hyperparameters)
7. [Checkpointing Hyperparameters](#checkpointing-hyperparameters)
8. [Other Hyperparameters](#other-hyperparameters)
9. [Hardcoded Values (Should Remain)](#hardcoded-values-should-remain)
10. [Summary and Recommendations](#summary-and-recommendations)

---

## Hyperparameters Already in Config

These hyperparameters are already present in `TrainingConfig` (`src/config.py`):

| Hyperparameter | Current Value | Location | Category | Configurable? |
|---------------|---------------|----------|----------|---------------|
| `learning_rate` | 3e-4 | `src/config.py:45` | Training | ✅ Yes |
| `weight_decay` | 0.1 | `src/config.py:46` | Training | ✅ Yes |
| `beta1` | 0.9 | `src/config.py:47` | Training | ✅ Yes |
| `beta2` | 0.95 | `src/config.py:48` | Training | ✅ Yes |
| `batch_size` | 16 | `src/config.py:49` | Training | ✅ Yes |
| `max_seq_len` | 256 | `src/config.py:50` | Dataset | ✅ Yes |
| `seed` | None | `src/config.py:51` | Other | ✅ Yes |
| `hooks` | None | `src/config.py:52` | Other | ✅ Yes |
| `eval_cadence` | None | `src/config.py:53` | Evaluation | ✅ Yes |
| `sampling_cadence` | None | `src/config.py:54` | Sampling | ✅ Yes |
| `sampling_temperature` | 1.0 | `src/config.py:55` | Sampling | ✅ Yes |
| `sampling_prompt` | "The" | `src/config.py:56` | Sampling | ✅ Yes |
| `sampling_max_length` | 100 | `src/config.py:57` | Sampling | ✅ Yes |
| `sampling_seed` | 42 | `src/config.py:58` | Sampling | ✅ Yes |

---

## Model Architecture Hyperparameters

These hyperparameters control the transformer model architecture and are currently hardcoded in `Transformer.__init__`:

| Hyperparameter | Current Value | Location | Category | Configurable? | Priority |
|---------------|---------------|----------|----------|---------------|----------|
| `n_layers` | 4 | `src/model/transformer.py:50` | Model Architecture | ❌ No (should be) | **HIGH** |
| `d_model` | 256 | `src/model/transformer.py:51` | Model Architecture | ❌ No (should be) | **HIGH** |
| `n_heads` | 4 | `src/model/transformer.py:52` | Model Architecture | ❌ No (should be) | **HIGH** |
| `d_ff` | 1024 | `src/model/transformer.py:53` | Model Architecture | ❌ No (should be) | **HIGH** |
| `dropout` | 0.1 | `src/model/transformer.py:54` | Model Architecture | ❌ No (should be) | **HIGH** |
| `vocab_size` | Dynamic | `main.py:213` | Model Architecture | ❌ No (derived) | **LOW** |
| `max_seq_len` | 256 | `src/model/transformer.py:49` | Model Architecture | ✅ Yes (already in config) | - |

### Detailed Analysis

#### `n_layers` (Number of Transformer Blocks)
- **Current Value:** 4
- **Location:** `src/model/transformer.py:50` (default parameter)
- **Usage:** `main.py:238,253` (hardcoded when creating Transformer)
- **Should be Configurable:** ✅ Yes
- **Suggested Default:** 4
- **Value Range:** Typically 2-12 for small models, 12-96+ for large models
- **Constraints:** Must be positive integer
- **Notes:** Controls model depth. More layers = more capacity but slower training/inference.

#### `d_model` (Model Dimension)
- **Current Value:** 256
- **Location:** `src/model/transformer.py:51` (default parameter)
- **Usage:** `main.py:238,253` (hardcoded when creating Transformer)
- **Should be Configurable:** ✅ Yes
- **Suggested Default:** 256
- **Value Range:** Typically 128-512 for small models, 512-4096+ for large models
- **Constraints:** Must be positive integer, typically a power of 2
- **Notes:** Controls model width. Must be divisible by `n_heads`.

#### `n_heads` (Number of Attention Heads)
- **Current Value:** 4
- **Location:** `src/model/transformer.py:52` (default parameter)
- **Usage:** `main.py:238,253` (hardcoded when creating Transformer)
- **Should be Configurable:** ✅ Yes
- **Suggested Default:** 4
- **Value Range:** Typically 2-8 for small models, 8-128+ for large models
- **Constraints:** Must be positive integer, `d_model % n_heads == 0` (d_model must be divisible by n_heads)
- **Notes:** Controls multi-head attention. More heads = more parallel attention patterns.

#### `d_ff` (Feed-Forward Hidden Dimension)
- **Current Value:** 1024
- **Location:** `src/model/transformer.py:53` (default parameter)
- **Usage:** `main.py:238,253` (hardcoded when creating Transformer)
- **Should be Configurable:** ✅ Yes
- **Suggested Default:** 1024
- **Value Range:** Typically 2-4x `d_model` (512-1024 for d_model=256)
- **Constraints:** Must be positive integer
- **Notes:** Controls MLP capacity. Often set to 4x `d_model` (1024 for d_model=256).

#### `dropout` (Dropout Probability)
- **Current Value:** 0.1
- **Location:** `src/model/transformer.py:54` (default parameter)
- **Usage:** `main.py:238,253` (hardcoded when creating Transformer)
- **Should be Configurable:** ✅ Yes
- **Suggested Default:** 0.1
- **Value Range:** 0.0 to 1.0 (typically 0.0-0.3)
- **Constraints:** `0.0 <= dropout <= 1.0`
- **Notes:** Regularization technique. Higher values = more regularization but may hurt training.

#### `vocab_size` (Vocabulary Size)
- **Current Value:** Dynamic (derived from tokenizer)
- **Location:** `main.py:213` (computed from tokenizer)
- **Usage:** `main.py:238,253` (passed to Transformer)
- **Should be Configurable:** ❌ No (derived from tokenizer)
- **Notes:** Automatically determined by tokenizer. Character-level ASCII tokenizer yields ~98 tokens.

---

## Training Hyperparameters

Training-related hyperparameters:

| Hyperparameter | Current Value | Location | Category | Configurable? | Priority |
|---------------|---------------|----------|----------|---------------|----------|
| `learning_rate` | 3e-4 | `src/config.py:45` | Training | ✅ Yes | - |
| `weight_decay` | 0.1 | `src/config.py:46` | Training | ✅ Yes | - |
| `beta1` | 0.9 | `src/config.py:47` | Training | ✅ Yes | - |
| `beta2` | 0.95 | `src/config.py:48` | Training | ✅ Yes | - |
| `batch_size` | 16 | `src/config.py:49` | Training | ✅ Yes | - |
| `max_steps` | 10000 | `main.py:187` | Training | ❌ No (should be) | **HIGH** |
| `seed` | None | `src/config.py:51` | Training | ✅ Yes | - |

### Detailed Analysis

#### `max_steps` (Maximum Training Steps)
- **Current Value:** 10000 (default), can be overridden via `--max-steps` CLI arg
- **Location:** `main.py:187` (hardcoded default)
- **Usage:** `main.py:276` (training loop condition)
- **Should be Configurable:** ✅ Yes
- **Suggested Default:** 10000
- **Value Range:** Typically 1000-100000+ depending on dataset size and model capacity
- **Constraints:** Must be positive integer
- **Notes:** Controls training duration. CLI override (`--max-steps`) should take precedence over config.

---

## Dataset Hyperparameters

Dataset-related hyperparameters:

| Hyperparameter | Current Value | Location | Category | Configurable? | Priority |
|---------------|---------------|----------|----------|---------------|----------|
| `max_seq_len` | 256 | `src/config.py:50` | Dataset | ✅ Yes | - |
| `train_ratio` | 0.95 | `main.py:218` | Dataset | ❌ No (should be) | **HIGH** |
| `context_length` | 256 | `src/dataset.py:73` | Dataset | ✅ Yes (uses `max_seq_len`) | - |

### Detailed Analysis

#### `train_ratio` (Train/Validation Split Ratio)
- **Current Value:** 0.95
- **Location:** `main.py:218` (hardcoded in `split_corpus` call)
- **Default in Function:** `src/dataset.py:12` (0.95)
- **Usage:** `main.py:218` (splitting corpus)
- **Should be Configurable:** ✅ Yes
- **Suggested Default:** 0.95
- **Value Range:** 0.0 to 1.0 (typically 0.8-0.95)
- **Constraints:** `0.0 < train_ratio < 1.0`
- **Notes:** Fraction of corpus used for training. Remaining fraction used for validation.

---

## Evaluation Hyperparameters

Evaluation-related hyperparameters:

| Hyperparameter | Current Value | Location | Category | Configurable? | Priority |
|---------------|---------------|----------|----------|---------------|----------|
| `eval_cadence` | None | `src/config.py:53` | Evaluation | ✅ Yes | - |

### Detailed Analysis

#### `eval_cadence` (Evaluation Frequency)
- **Current Value:** None (disabled by default)
- **Location:** `src/config.py:53`
- **Usage:** `src/training/trainer.py:317-322` (evaluation logic)
- **Should be Configurable:** ✅ Yes (already configurable)
- **Suggested Default:** None (disabled) or 500
- **Value Range:** Positive integer (steps between evaluations) or None to disable
- **Constraints:** Must be positive integer if set, or None
- **Notes:** How often to compute validation loss. None = disabled.

---

## Sampling Hyperparameters

Sampling-related hyperparameters (all already in config):

| Hyperparameter | Current Value | Location | Category | Configurable? | Priority |
|---------------|---------------|----------|----------|---------------|----------|
| `sampling_cadence` | None | `src/config.py:54` | Sampling | ✅ Yes | - |
| `sampling_temperature` | 1.0 | `src/config.py:55` | Sampling | ✅ Yes | - |
| `sampling_prompt` | "The" | `src/config.py:56` | Sampling | ✅ Yes | - |
| `sampling_max_length` | 100 | `src/config.py:57` | Sampling | ✅ Yes | - |
| `sampling_seed` | 42 | `src/config.py:58` | Sampling | ✅ Yes | - |

### Detailed Analysis

All sampling hyperparameters are already configurable. No changes needed.

---

## Checkpointing Hyperparameters

Checkpointing-related hyperparameters:

| Hyperparameter | Current Value | Location | Category | Configurable? | Priority |
|---------------|---------------|----------|----------|---------------|----------|
| `checkpoint_cadence` | 1000 | `main.py:286` | Checkpointing | ❌ No (should be) | **HIGH** |
| `checkpoint_dir` | "checkpoints" | `src/training/checkpoint.py:26` | Checkpointing | ❌ No (optional) | **LOW** |
| `checkpoint_name` | "checkpoint_step_{step}" | `src/training/checkpoint.py:55` | Checkpointing | ❌ No (derived) | **LOW** |

### Detailed Analysis

#### `checkpoint_cadence` (Checkpoint Save Frequency)
- **Current Value:** 1000
- **Location:** `main.py:286` (hardcoded in training loop)
- **Usage:** `main.py:286` (checkpoint saving condition)
- **Should be Configurable:** ✅ Yes
- **Suggested Default:** 1000
- **Value Range:** Positive integer (steps between checkpoints) or None to disable
- **Constraints:** Must be positive integer if set, or None
- **Notes:** How often to save checkpoints. None = only save at end/interrupt.

#### `checkpoint_dir` (Checkpoint Directory)
- **Current Value:** "checkpoints"
- **Location:** `src/training/checkpoint.py:26` (default parameter)
- **Usage:** `main.py:287` (passed to `save_checkpoint`)
- **Should be Configurable:** ⚠️ Optional (low priority)
- **Notes:** Directory for saving checkpoints. Can remain hardcoded for simplicity.

#### `checkpoint_name` (Checkpoint Naming)
- **Current Value:** "checkpoint_step_{step}" (derived)
- **Location:** `src/training/checkpoint.py:55` (derived from step)
- **Should be Configurable:** ❌ No (derived from step counter)
- **Notes:** Automatically generated from step counter. No need to make configurable.

---

## Other Hyperparameters

Other miscellaneous hyperparameters:

| Hyperparameter | Current Value | Location | Category | Configurable? | Priority |
|---------------|---------------|----------|----------|---------------|----------|
| `seed` | None | `src/config.py:51` | Other | ✅ Yes | - |
| `hooks` | None | `src/config.py:52` | Other | ✅ Yes | - |

### Detailed Analysis

All other hyperparameters are already configurable. No changes needed.

---

## Hardcoded Values (Should Remain)

These values are hardcoded but should remain hardcoded (not configurable):

| Value | Location | Reason |
|-------|----------|--------|
| Tokenizer type (character-level ASCII) | `src/tokenizer.py` | Tokenizer architecture is Phase 3+ consideration |
| Window stride (equals context_length) | `src/dataset.py:110` | Derived from context_length, not independent hyperparameter |
| Loss function (cross-entropy) | `src/training/loss.py` | Loss function choice is architectural, not hyperparameter |
| Optimizer type (AdamW) | `src/training/trainer.py:43` | Optimizer choice is architectural, not hyperparameter |
| Checkpoint file names (model.pt, optimizer.pt, etc.) | `src/training/checkpoint.py` | Internal implementation detail |
| Default data file preference (uni-alg-int.txt) | `main.py:101` | Convenience default, can be overridden via CLI |

---

## Summary and Recommendations

### Hyperparameters to Add to Config (Priority: HIGH)

1. **Model Architecture** (all HIGH priority):
   - `n_layers: int = 4`
   - `d_model: int = 256`
   - `n_heads: int = 4`
   - `d_ff: int = 1024`
   - `dropout: float = 0.1`

2. **Dataset**:
   - `train_ratio: float = 0.95`

3. **Training Loop**:
   - `max_steps: int = 10000`
   - `checkpoint_cadence: int = 1000`

### Hyperparameters Already in Config (No Changes Needed)

- All training optimizer hyperparameters (learning_rate, weight_decay, beta1, beta2, batch_size)
- All sampling hyperparameters (sampling_cadence, sampling_temperature, sampling_prompt, sampling_max_length, sampling_seed)
- Evaluation hyperparameters (eval_cadence)
- Dataset hyperparameters (max_seq_len)
- Other (seed, hooks)

### Value Constraints Summary

| Hyperparameter | Type | Constraints | Notes |
|---------------|------|-------------|-------|
| `n_layers` | int | `> 0` | Positive integer |
| `d_model` | int | `> 0`, `d_model % n_heads == 0` | Must be divisible by n_heads |
| `n_heads` | int | `> 0`, `d_model % n_heads == 0` | Must divide d_model evenly |
| `d_ff` | int | `> 0` | Positive integer |
| `dropout` | float | `0.0 <= dropout <= 1.0` | Probability range |
| `train_ratio` | float | `0.0 < train_ratio < 1.0` | Fraction between 0 and 1 |
| `max_steps` | int | `> 0` | Positive integer |
| `checkpoint_cadence` | int | `> 0` or `None` | Positive integer or None to disable |

### Implementation Notes

1. **Model Architecture Parameters**: These are currently hardcoded in `main.py` when creating the `Transformer`. They should be read from config and passed to the constructor.

2. **train_ratio**: Currently hardcoded in `main.py:218`. Should be read from config.

3. **max_steps**: Currently has a default of 10000 in `main.py:187` and can be overridden via CLI. Should be in config with CLI override taking precedence.

4. **checkpoint_cadence**: Currently hardcoded in `main.py:286`. Should be read from config.

5. **Validation**: Consider adding validation logic in `TrainingConfig` to enforce constraints (e.g., `d_model % n_heads == 0`, `0.0 <= dropout <= 1.0`).

---

## Next Steps (Phase 2 Sessions 2-6)

1. **Session 2**: Extend `TrainingConfig` to include all missing hyperparameters
2. **Session 3**: Create YAML config files with examples
3. **Session 4**: Update `main.py` to use config for all hyperparameters
4. **Session 5**: Update README with comprehensive hyperparameter documentation
5. **Session 6**: Test and validate the config system

---

**End of Audit Document**

