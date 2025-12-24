# Change: Add Main Entry Point for Training Script

## Why
The current `main.py` is a placeholder that only prints "Hello from experimental-slm!". All Phase 1 components are complete (tokenizer, dataset, dataloader, model, trainer, checkpointing, evaluation, sampling), but they need to be tied together into a working entry point. According to README.md and phase1-sessions.md Session 10, the project needs a simple integration script that orchestrates these existing components. This change will make the project runnable end-to-end as documented in the README, enabling users to train the SLM with a single command (`uv run python main.py`) and resume from checkpoints (`uv run python main.py --resume checkpoints/latest.pt`).

## What Changes
- Implement simple integration script in `main.py` that:
  - Loads configuration (from YAML file or uses TrainingConfig defaults)
  - Loads text data from file and initializes tokenizer
  - Creates train/val split and WindowDatasets
  - Creates DataLoaders for training and validation
  - Creates Transformer model and AdamW optimizer
  - Initializes Trainer (or loads from checkpoint if --resume provided)
  - Runs training loop: iterate over batches, call `trainer.training_step()`, save checkpoints periodically
  - Handles KeyboardInterrupt gracefully (save checkpoint before exit)
- Add command-line argument parsing:
  - `--resume <checkpoint_path>`: Resume training from checkpoint
  - `--config <config_path>`: Load configuration from YAML file (optional)
  - `--data <data_path>`: Specify data file path (defaults to `data/uni-alg-int.txt` or similar)
  - `--max-steps <N>`: Override maximum training steps from config
- Add basic error handling for missing files and invalid checkpoints
- Create comprehensive integration test in `tests/test_integration.py`:
  - End-to-end test: tiny corpus → train → checkpoint → resume → verify
  - Test checkpoint/resume produces identical results
  - Test loss decreases over time (smoke test)
  - Test reproducibility: same seed → same results
- Verify logging includes all required fields (run_id, config_hash, git_commit, step, loss, val_loss, sample_text)
- Ensure log format is parseable and structured
- Update documentation as needed

## Impact
- Affected specs: New `main-entry` capability specification
- Affected code:
  - `main.py` (complete rewrite - currently placeholder)
  - `tests/test_integration.py` (new - end-to-end integration test)
  - May need simple YAML config loader utility if not present
- Dependencies:
  - All Phase 1 sessions 1-9 components are already implemented and tested
  - Uses existing: Tokenizer, Dataset, DataLoader, Transformer, Trainer, TrainingConfig, checkpoint functions
- Future impact:
  - Enables end-to-end training runs as documented in README
  - Completes Phase 1 Session 10: Integration & Polish
  - Verifies full pipeline reproducibility and correctness
  - Foundation for future CLI enhancements

