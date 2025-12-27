# Change: Add CLI-Based Inference Interface

## Why

The current system supports training and model management but lacks a standalone inference capability. Users need to run trained models interactively from the command line to generate text without starting a full training run. This is the first step toward broader inference capabilities including API servers and standardized output formats.

## What Changes

- Add `inference` subcommand to `main.py` for standalone text generation
- Support loading models from checkpoints or the model registry
- Enable interactive and single-shot inference modes
- Provide configurable sampling parameters (temperature, max length, seed)
- Design inference interface to support future output format adapters (OpenAI API, BAML, etc.)
- Lay groundwork for future standalone inference service

## Impact

- Affected specs: `main-entry`
- Affected code:
  - `main.py` - New inference subcommand handler
  - Reuses existing sampling functionality from `src/sampling/sampler.py`
  - Reuses existing model loading from `load_model_adapter()` function
- Future extensibility: Design supports adding output format adapters and inference service in later phases

