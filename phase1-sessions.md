# Phase 1: Session Breakdown

This document breaks Phase 1 into logical coding sessions. Each session:
- Completes a modular, testable piece of functionality
- Includes unit tests
- Leaves the codebase in a clean, working state
- Can be completed in a single focused coding session

⸻

## Session Overview

| Session | Focus | Dependencies | Estimated Complexity |
|---------|-------|--------------|---------------------|
| 1 | Tokenizer Foundation | None | Low |
| 2 | Dataset Preparation | Session 1 | Low |
| 3 | Batching Infrastructure | Session 2 | Low |
| 4 | Model Core (Transformer Blocks) | None | Medium |
| 5 | Model Assembly | Session 4 | Medium |
| 6 | Basic Training Loop | Sessions 3, 5 | Medium |
| 7 | Checkpointing & Resume | Session 6 | Low |
| 8 | Hooks Infrastructure | Session 6 | Medium |
| 9 | Evaluation & Sampling | Sessions 5, 6 | Medium |
| 10 | Integration & Polish | All previous | Low |

⸻

---

## Session 1: Tokenizer Foundation

**Goal:** Implement character-level tokenizer with ASCII policy and vocab persistence.

**Deliverables:**
- `src/tokenizer.py` - Tokenizer class
- `src/normalize.py` - Text normalization functions
- `tests/test_tokenizer.py` - Comprehensive tests
- `tests/test_normalize.py` - Normalization tests

**Implementation Checklist:**
- [ ] `normalize_text()` implements ASCII policy (printable 32–126 + `\n` + `\t`)
- [ ] Unknown characters map to `<UNK>` placeholder
- [ ] `Tokenizer` class with `encode()` and `decode()` methods
- [ ] Vocab definition: `<PAD>=0`, `<UNK>=1`, then allowed ASCII chars
- [ ] Vocab saved to disk (JSON format)
- [ ] Vocab loadable from disk
- [ ] Round-trip test: `decode(encode(text))` matches normalized text (where possible)

**Tests Required:**
- [ ] Normalization handles edge cases (non-ASCII, control chars, empty strings, unicode)
- [ ] Encode produces correct token IDs
- [ ] Decode produces correct text
- [ ] Round-trip preserves normalized text
- [ ] Vocab save/load produces identical tokenization
- [ ] Special tokens (`<PAD>`, `<UNK>`) handled correctly

**Acceptance Criteria:**
- All tests pass
- Can tokenize sample text files
- Vocab file is human-readable JSON
- Code is clean, documented, and follows project style

**Files Created:**
```
src/
  tokenizer.py
  normalize.py
tests/
  test_tokenizer.py
  test_normalize.py
```

**Cleanup Before Next Session:**
- Run all tests
- Verify no linter errors
- Commit with message: "Session 1: Tokenizer foundation"

⸻

---

## Session 2: Dataset Preparation

**Goal:** Implement train/val split and sequence windowing from corpus.

**Dependencies:** Session 1 (Tokenizer)

**Deliverables:**
- `src/dataset.py` - Dataset classes and utilities
- `tests/test_dataset.py` - Dataset tests

**Implementation Checklist:**
- [ ] `split_corpus()` - Contiguous train/val split (e.g., 95%/5%)
- [ ] Split is deterministic (seed-based)
- [ ] `WindowDataset` - Creates (x, y) pairs from corpus
- [ ] Windows have shape [256] (context length)
- [ ] Sliding windows with stride 256 (non-overlapping)
- [ ] y is x shifted by 1 position (next-token prediction)
- [ ] Handles corpus boundaries correctly (drop incomplete last window)

**Tests Required:**
- [ ] Split produces expected sizes (95%/5% or configurable)
- [ ] Same seed → same split
- [ ] Window shapes are correct [256]
- [ ] y is correctly shifted (y[i] == x[i+1])
- [ ] Window boundaries handled (last incomplete window dropped)
- [ ] Can iterate through dataset without errors

**Acceptance Criteria:**
- All tests pass
- Can create dataset from text file
- Dataset produces correct sequence pairs
- Code is clean and documented

**Files Created:**
```
src/
  dataset.py
tests/
  test_dataset.py
```

**Cleanup Before Next Session:**
- Run all tests
- Verify no linter errors
- Commit with message: "Session 2: Dataset preparation"

⸻

---

## Session 3: Batching Infrastructure

**Goal:** Implement DataLoader that produces batched tensors ready for training.

**Dependencies:** Session 2 (Dataset)

**Deliverables:**
- `src/dataloader.py` - DataLoader implementation
- `tests/test_dataloader.py` - DataLoader tests

**Implementation Checklist:**
- [ ] `DataLoader` produces batches of shape [B, 256]
- [ ] Batch size B is configurable (default: 16)
- [ ] Tensors are dtype `int64`
- [ ] Handles last batch deterministically (drop incomplete batch)
- [ ] Supports iteration (can loop through batches)
- [ ] Optional: gradient accumulation support (defer if not needed yet)

**Tests Required:**
- [ ] Batch shapes are correct [B, 256]
- [ ] Batch size is configurable
- [ ] Last incomplete batch is dropped
- [ ] Can iterate through all batches
- [ ] Batches are deterministic (same seed → same batches)
- [ ] Handles empty dataset gracefully

**Acceptance Criteria:**
- All tests pass
- Can create DataLoader from Dataset
- Batches are ready for model input
- Code is clean and documented

**Files Created:**
```
src/
  dataloader.py
tests/
  test_dataloader.py
```

**Cleanup Before Next Session:**
- Run all tests
- Verify no linter errors
- Commit with message: "Session 3: Batching infrastructure"

⸻

---

## Session 4: Model Core (Transformer Blocks)

**Goal:** Implement core Transformer components (attention, MLP, layer norm).

**Dependencies:** None (pure model code, no training dependencies)

**Deliverables:**
- `src/model/attention.py` - Multi-head attention with causal masking
- `src/model/mlp.py` - Feed-forward network
- `src/model/layer_norm.py` - Layer normalization
- `src/model/transformer_block.py` - Single transformer block
- `tests/test_attention.py` - Attention tests
- `tests/test_mlp.py` - MLP tests
- `tests/test_transformer_block.py` - Block tests

**Implementation Checklist:**
- [ ] `MultiHeadAttention` with configurable heads
- [ ] Causal masking implemented (position i cannot attend to position > i)
- [ ] `MLP` with configurable d_ff
- [ ] `LayerNorm` implementation
- [ ] `TransformerBlock` combining attention + MLP + residuals
- [ ] All components initialized deterministically (seed-based)

**Tests Required:**
- [ ] Attention output shapes are correct
- [ ] Causal masking prevents future attention (manual mask check)
- [ ] MLP output shapes are correct
- [ ] Transformer block forward pass works
- [ ] Residual connections work correctly
- [ ] Initialization is deterministic

**Acceptance Criteria:**
- All tests pass
- Components can be instantiated and run forward pass
- Causal masking verified with test
- Code is clean, well-documented, modular

**Files Created:**
```
src/model/
  __init__.py
  attention.py
  mlp.py
  layer_norm.py
  transformer_block.py
tests/
  test_attention.py
  test_mlp.py
  test_transformer_block.py
```

**Cleanup Before Next Session:**
- Run all tests
- Verify no linter errors
- Commit with message: "Session 4: Model core (transformer blocks)"

⸻

---

## Session 5: Model Assembly

**Goal:** Assemble full decoder-only Transformer model with embeddings.

**Dependencies:** Session 4 (Model Core)

**Deliverables:**
- `src/model/embeddings.py` - Token and positional embeddings
- `src/model/transformer.py` - Full model assembly
- `tests/test_embeddings.py` - Embedding tests
- `tests/test_transformer.py` - Full model tests

**Implementation Checklist:**
- [ ] `TokenEmbedding` - Maps token IDs to embeddings
- [ ] `PositionalEmbedding` - Learned absolute positional embeddings
- [ ] `Transformer` - Full decoder-only model
- [ ] Model hyperparameters: n_layers=4, d_model=256, n_heads=4, d_ff=1024, dropout=0.1
- [ ] Forward pass returns logits [B, 256, vocab_size]
- [ ] Model initialization is deterministic

**Tests Required:**
- [ ] Embedding shapes are correct
- [ ] Positional embeddings are learnable parameters
- [ ] Model forward pass produces correct logit shapes [B, 256, vocab_size]
- [ ] Model can process batches from DataLoader
- [ ] Initialization is deterministic

**Acceptance Criteria:**
- All tests pass
- Model can process a batch end-to-end
- Logits are correct shape and dtype
- Code is clean and documented

**Files Created:**
```
src/model/
  embeddings.py
  transformer.py
tests/
  test_embeddings.py
  test_transformer.py
```

**Cleanup Before Next Session:**
- Run all tests
- Verify no linter errors
- Commit with message: "Session 5: Model assembly"

⸻

---

## Session 6: Basic Training Loop

**Goal:** Implement training step, loss computation, and optimizer setup.

**Dependencies:** Sessions 3 (DataLoader), 5 (Model)

**Deliverables:**
- `src/training/trainer.py` - Training loop
- `src/training/loss.py` - Loss computation
- `src/config.py` - Configuration management
- `tests/test_trainer.py` - Training tests

**Implementation Checklist:**
- [ ] `compute_loss()` - Cross-entropy next-token over all positions
- [ ] `Trainer` class with training step
- [ ] Training step: forward → loss → backward → optimizer step
- [ ] AdamW optimizer: lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95)
- [ ] Step counter increments correctly
- [ ] Basic logging (loss per step)
- [ ] Config system for hyperparameters

**Tests Required:**
- [ ] Loss computation is correct (manual check on known inputs)
- [ ] Training step completes without errors
- [ ] Optimizer updates model parameters
- [ ] Step counter increments
- [ ] Loss decreases on simple synthetic data (smoke test)

**Acceptance Criteria:**
- All tests pass
- Can run training for a few steps on tiny dataset
- Loss is computed and logged
- Code is clean and documented

**Files Created:**
```
src/training/
  __init__.py
  trainer.py
  loss.py
src/
  config.py
tests/
  test_trainer.py
```

**Cleanup Before Next Session:**
- Run all tests
- Verify no linter errors
- Commit with message: "Session 6: Basic training loop"

⸻

---

## Session 7: Checkpointing & Resume

**Goal:** Implement save/load checkpoints and resume training.

**Dependencies:** Session 6 (Training)

**Deliverables:**
- `src/training/checkpoint.py` - Checkpoint save/load
- `tests/test_checkpoint.py` - Checkpoint tests

**Implementation Checklist:**
- [ ] `save_checkpoint()` - Saves model, optimizer, config, vocab, step
- [ ] Checkpoint format: JSON metadata + binary weights (or PyTorch format)
- [ ] `load_checkpoint()` - Loads checkpoint
- [ ] Resume continues step count correctly
- [ ] Resume produces identical loss progression (test)

**Tests Required:**
- [ ] Save checkpoint creates files
- [ ] Load checkpoint restores model state
- [ ] Load checkpoint restores optimizer state
- [ ] Resume continues from correct step
- [ ] Resume produces identical loss (within tolerance)

**Acceptance Criteria:**
- All tests pass
- Can save and load checkpoints
- Resume works correctly
- Code is clean and documented

**Files Created:**
```
src/training/
  checkpoint.py
tests/
  test_checkpoint.py
```

**Cleanup Before Next Session:**
- Run all tests
- Verify no linter errors
- Commit with message: "Session 7: Checkpointing & resume"

⸻

---

## Session 8: Hooks Infrastructure

**Goal:** Implement minimal hook system for experiments.

**Dependencies:** Session 6 (Training)

**Deliverables:**
- `src/hooks/registry.py` - Hook registry
- `src/hooks/forward_hooks.py` - Forward hooks
- `src/hooks/update_hooks.py` - Update hooks
- `tests/test_hooks.py` - Hook tests

**Implementation Checklist:**
- [ ] Hook registry loads from config
- [ ] Registry supports toggling hooks on/off
- [ ] Forward hook can log activation stats (mean/std) without changing outputs
- [ ] Update hook exists (identity default)
- [ ] Update hook receives gradients and can transform them
- [ ] Every run logs: run_id, git_commit, config_hash, hook_list

**Tests Required:**
- [ ] Hook registry loads hooks from config
- [ ] Forward hook produces correct stats
- [ ] Forward hook doesn't change outputs
- [ ] Update hook identity doesn't change gradients
- [ ] Update hook can transform gradients
- [ ] Hook safety logging works

**Acceptance Criteria:**
- All tests pass
- Hooks can be registered and called
- Hooks don't break training
- Code is clean and documented

**Files Created:**
```
src/hooks/
  __init__.py
  registry.py
  forward_hooks.py
  update_hooks.py
tests/
  test_hooks.py
```

**Cleanup Before Next Session:**
- Run all tests
- Verify no linter errors
- Commit with message: "Session 8: Hooks infrastructure"

⸻

---

## Session 9: Evaluation & Sampling

**Goal:** Implement validation loss computation and text sampling.

**Dependencies:** Sessions 5 (Model), 6 (Training)

**Deliverables:**
- `src/evaluation/evaluator.py` - Validation evaluation
- `src/sampling/sampler.py` - Text sampling
- `tests/test_evaluator.py` - Evaluation tests
- `tests/test_sampler.py` - Sampling tests

**Implementation Checklist:**
- [ ] `compute_val_loss()` - Validation loss on validation set
- [ ] Evaluation cadence: every N steps (e.g., 200)
- [ ] `sample_text()` - Generates text from fixed prompt
- [ ] Sampling: temperature=1.0, top-k disabled (pure multinomial)
- [ ] Sampling uses fixed seed for reproducibility
- [ ] Generated text logged periodically

**Tests Required:**
- [ ] Val loss computed correctly
- [ ] Sampling produces text of correct length
- [ ] Sampling is reproducible (same seed → same output)
- [ ] Sampling uses correct temperature
- [ ] Generated text is valid (can be decoded)

**Acceptance Criteria:**
- All tests pass
- Can evaluate validation loss
- Can generate text samples
- Code is clean and documented

**Files Created:**
```
src/evaluation/
  __init__.py
  evaluator.py
src/sampling/
  __init__.py
  sampler.py
tests/
  test_evaluator.py
  test_sampler.py
```

**Cleanup Before Next Session:**
- Run all tests
- Verify no linter errors
- Commit with message: "Session 9: Evaluation & sampling"

⸻

---

## Session 10: Integration & Polish

**Goal:** Integrate all components, verify end-to-end, and polish.

**Dependencies:** All previous sessions

**Deliverables:**
- `src/main.py` - Main training script
- `tests/test_integration.py` - End-to-end integration test
- `README.md` - Usage documentation
- Any missing documentation

**Implementation Checklist:**
- [ ] End-to-end training script
- [ ] Integration test: tiny corpus → train → checkpoint → resume → verify
- [ ] Logging includes: run_id, config_hash, git_commit, step, loss, val_loss, sample text
- [ ] Log format is parseable (JSON or structured text)
- [ ] Documentation: how to run training, resume, modify config
- [ ] All docstrings complete

**Tests Required:**
- [ ] End-to-end test: full training pipeline works
- [ ] Checkpoint/resume produces identical results
- [ ] Loss decreases over time (smoke test)
- [ ] Generated samples improve over time (qualitative check)
- [ ] Reproducibility: same seed → same results

**Acceptance Criteria:**
- All tests pass (unit + integration)
- Can run full training pipeline
- Loss decreases reliably
- Results are reproducible
- Documentation is complete
- Codebase is clean and well-organized

**Files Created/Updated:**
```
src/
  main.py
tests/
  test_integration.py
README.md
```

**Final Cleanup:**
- Run full test suite
- Verify no linter errors
- Update phase1-checklist.md with completion status
- Commit with message: "Session 10: Integration & polish - Phase 1 complete"

⸻

---

## Session Workflow

### Before Starting a Session:
1. Review the session goals and dependencies
2. Ensure all dependencies are complete
3. Create a feature branch (optional but recommended)
4. Review relevant parts of `questions.md` and `phase1-checklist.md`

### During a Session:
1. Implement the core functionality
2. Write tests as you go (TDD recommended)
3. Run tests frequently
4. Keep code clean and documented

### After Completing a Session:
1. Run full test suite for the session
2. Fix any linter errors
3. Verify code is clean and documented
4. Commit with descriptive message
5. Update session checklist (mark items complete)
6. Take a break before next session

### If Stuck:
- Review the acceptance criteria
- Check if dependencies are truly complete
- Consider simplifying the implementation (MVP first)
- Document what's blocking you

⸻

---

## Success Metrics Per Session

Each session should achieve:
- ✅ All planned functionality implemented
- ✅ All tests passing
- ✅ Code is clean and documented
- ✅ No linter errors
- ✅ Codebase is in a working state
- ✅ Can demonstrate the functionality works

**Phase 1 is complete when:**
- All 10 sessions are done
- All acceptance criteria from `phase1-checklist.md` are met
- End-to-end training works
- Loss decreases reliably
- Results are reproducible
