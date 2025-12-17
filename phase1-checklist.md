# Phase 1 Checklist: Definition of Done

This is a practical "Definition of Done" checklist for Phase 1 implementation. Each item must be completed and verified before Phase 1 is considered complete.

⸻

## 1. Data + Tokenizer

### 1.1 Text Normalization
- [ ] `normalize_text()` implements ASCII policy (printable 32–126 + `\n` + `\t`)
- [ ] Unknown characters map to `<UNK>` token
- [ ] Unit test: verify normalization handles edge cases (non-ASCII, control chars, empty strings)

### 1.2 Tokenizer Implementation
- [ ] `Tokenizer.encode(str) -> List[int]` converts text to token IDs
- [ ] `Tokenizer.decode(List[int]) -> str` converts token IDs back to text
- [ ] Round-trip test: `decode(encode(text)) == normalize_text(text)` (where possible)
- [ ] Vocab definition: `<PAD>=0`, `<UNK>=1`, then allowed ASCII chars

### 1.3 Vocab Persistence
- [ ] Vocab saved to disk (human-readable format + machine-readable format)
- [ ] Vocab re-loadable from disk
- [ ] Unit test: save → load → encode produces same IDs

⸻

## 2. Dataset / Sequence Builder

### 2.1 Train/Val Split
- [ ] Contiguous split implemented deterministically (e.g., first 95% train, last 5% val)
- [ ] Split is reproducible (same seed → same split)
- [ ] Unit test: split produces expected sizes

### 2.2 Window Builder
- [ ] Produces (x, y) pairs with shape [256] from corpus
- [ ] Sliding windows with stride 256 (non-overlapping)
- [ ] y is x shifted by 1 position (next-token prediction)
- [ ] Unit test: shapes + shifting correctness
- [ ] Unit test: window boundaries handled correctly (last incomplete window)

⸻

## 3. Batching

### 3.1 Batch Creation
- [ ] Produces tensors x, y of shape [B, 256], dtype int64, on device
- [ ] Batch size B is configurable (default: 16, fallback: 8)
- [ ] Handles last batch behavior deterministically (decide: drop or pad; default: drop)

### 3.2 Gradient Accumulation (if needed)
- [ ] Gradient accumulation implemented for effective larger batch sizes
- [ ] Works correctly with optimizer updates

⸻

## 4. Model

### 4.1 Architecture
- [ ] Decoder-only Transformer implemented
- [ ] Hyperparameters: n_layers=4, d_model=256, n_heads=4, d_ff=1024, dropout=0.1
- [ ] Learned absolute positional embeddings implemented

### 4.2 Forward Pass
- [ ] Forward pass returns logits with shape [B, 256, vocab_size]
- [ ] Causal masking verified (test: position i cannot attend to position > i)
- [ ] Unit test: causal masking correctness (manual attention mask check)

### 4.3 Initialization
- [ ] Weights initialized deterministically (seed-based)
- [ ] Initialization scheme documented

⸻

## 5. Training

### 5.1 Training Step
- [ ] Training step: forward → loss → backward → optimizer step
- [ ] Loss: standard cross-entropy next-token over all positions
- [ ] Gradient norm logging (optional but recommended)
- [ ] Step counter increments correctly

### 5.2 Optimizer
- [ ] AdamW optimizer configured: lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95)
- [ ] Optimizer state tracked correctly

### 5.3 Checkpointing
- [ ] Checkpoint saves: model state, optimizer state, config, vocab, step number
- [ ] Checkpoint format is human-readable (JSON/YAML) + binary weights
- [ ] Resume works and continues step count correctly
- [ ] Test: save → load → continue training produces identical loss progression

⸻

## 6. Hooks (Minimal)

### 6.1 Hook Registry
- [ ] Hook registry loads from config
- [ ] Registry supports toggling hooks on/off

### 6.2 Forward Hooks
- [ ] Forward hook can log activation stats (mean/std) without changing outputs
- [ ] Hook doesn't break training when enabled
- [ ] Test: forward hook produces correct stats

### 6.3 Update Hooks
- [ ] Update hook exists (identity default)
- [ ] Update hook receives gradients and can transform them
- [ ] Test: identity hook doesn't change gradients

### 6.4 Hook Safety
- [ ] Every run logs: run_id, git_commit, config_hash, hook_list
- [ ] Config hash includes all relevant hyperparameters + hook config

⸻

## 7. Evaluation + Sampling

### 7.1 Validation Loss
- [ ] Val loss computed on validation set
- [ ] Evaluation cadence: every N steps (e.g., 200)
- [ ] Val loss logged properly

### 7.2 Text Sampling
- [ ] Sampling produces text from a fixed prompt
- [ ] Sampling parameters: temperature=1.0, top-k disabled (pure multinomial)
- [ ] Sampling uses fixed seed for reproducibility
- [ ] Generated text logged periodically

### 7.3 Logging
- [ ] Logs include: run_id, config_hash, git_commit, step, loss, val_loss, sample text
- [ ] Log format is parseable (JSON or structured text)
- [ ] Logs written to disk

⸻

## 8. Acceptance Criteria

### 8.1 Learning Verification
- [ ] On a tiny dataset, training loss decreases over time
- [ ] Loss curve shows downward trend (doesn't need to converge, just trend down)
- [ ] Val loss also decreases (may overfit later, that's okay)

### 8.2 Reproducibility
- [ ] Re-running with same seed reproduces same loss curve (within tiny floating tolerance, e.g., 1e-6)
- [ ] Same config → same results across runs

### 8.3 Qualitative Check
- [ ] Generated samples become less random over time
- [ ] Early samples are mostly random noise
- [ ] Later samples show some structure (even if not perfect)

⸻

## 9. Documentation

### 9.1 Code Documentation
- [ ] Core functions have docstrings
- [ ] Config parameters documented

### 9.2 Usage Documentation
- [ ] How to run training documented
- [ ] How to resume from checkpoint documented
- [ ] How to modify config documented

⸻

## 10. Testing

### 10.1 Unit Tests
- [ ] Tokenizer tests (normalization, encode/decode round-trip)
- [ ] Dataset tests (split, windowing)
- [ ] Model tests (forward pass shapes, causal masking)

### 10.2 Integration Test
- [ ] End-to-end test: tiny corpus → train for a few steps → checkpoint → resume → verify consistency

⸻

---

## Success Criteria Summary

Phase 1 is complete when:
1. ✅ Training pipeline runs end-to-end without errors
2. ✅ Loss decreases reliably over time
3. ✅ Results are reproducible (same seed → same results)
4. ✅ Hook infrastructure is in place (even if minimal)
5. ✅ Checkpoint/resume works correctly
6. ✅ Generated samples show improvement over time

**Phase 1 is NOT about:**
- Achieving perfect text generation
- Optimal hyperparameters
- Production-ready performance
- Comprehensive hook system

**Phase 1 IS about:**
- Proving the pipeline works
- Establishing a stable baseline
- Ensuring reproducibility
- Setting up infrastructure for experiments
