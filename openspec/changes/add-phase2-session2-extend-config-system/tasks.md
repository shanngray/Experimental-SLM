## 1. Implementation
- [x] 1.1 Add model architecture hyperparameters to `TrainingConfig`:
  - [x] `n_layers: int = 4`
  - [x] `d_model: int = 256`
  - [x] `n_heads: int = 4`
  - [x] `d_ff: int = 1024`
  - [x] `dropout: float = 0.1`
- [x] 1.2 Add dataset hyperparameter:
  - [x] `train_ratio: float = 0.95`
- [x] 1.3 Add training loop hyperparameters:
  - [x] `max_steps: int = 10000`
  - [x] `checkpoint_cadence: int = 1000`
- [x] 1.4 Update `from_dict` method:
  - [x] Add new fields to `valid_keys` set
- [x] 1.5 Update `to_dict` method:
  - [x] Include new fields in serialization
- [x] 1.6 Add comprehensive docstrings:
  - [x] Document purpose/effect of each new hyperparameter
  - [x] Document default values
  - [x] Document reasonable value ranges
  - [x] Document constraints (e.g., `d_model % n_heads == 0`, `0.0 <= dropout <= 1.0`)
  - [x] Document interactions with other hyperparameters
- [x] 1.7 Add validation logic (optional but recommended):
  - [x] Validate `d_model % n_heads == 0`
  - [x] Validate `0.0 <= dropout <= 1.0`
  - [x] Validate `0.0 < train_ratio < 1.0`
  - [x] Validate `checkpoint_cadence > 0` or `None`

## 2. Testing
- [x] 2.1 Test `TrainingConfig()` creates instance with all defaults
- [x] 2.2 Test `TrainingConfig.from_dict({})` uses defaults for missing fields
- [x] 2.3 Test `TrainingConfig.from_dict({"n_layers": 6})` correctly sets n_layers
- [x] 2.4 Test `TrainingConfig.to_dict()` includes all new fields
- [x] 2.5 Test backward compatibility: old configs (missing new fields) still work
- [x] 2.6 Test validation constraints (if implemented)

## 3. Documentation
- [x] 3.1 Verify docstrings are comprehensive and helpful
- [x] 3.2 Verify type hints are correct for all fields

