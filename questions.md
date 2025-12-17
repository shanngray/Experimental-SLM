# Questions & Decisions Log

This document captures open questions and decisions for the Experimental SLM project. Each question is marked with its phase status:
- **P1-blocker**: Must decide before Phase 1 coding starts (affects interfaces, data shapes, determinism)
- **P1-defer**: Can decide after baseline runs
- **P2/P3**: Future considerations

⸻

## 1. Tokenization & Input Representation

### 1.1 Tokenization Granularity **[P1-blocker]**

**Question:** What level of tokenization should we commit to initially?

**Options:**
1. Character-level (ASCII)
	• Entails: Very small vocab (≈128–256), simple implementation, maximal transparency.
	• Pros: Clean learning signal, easy to inspect, trivial tokenizer.
	• Cons: Longer effective sequence length, slower learning of semantics.
2. Simple word-level
	• Entails: Pre-tokenization by whitespace / punctuation.
	• Pros: Faster semantic learning.
	• Cons: Vocabulary explosion, brittle to OOV words, less "from first principles".
3. Subword (BPE / unigram)
	• Entails: Training a tokenizer, added complexity.
	• Pros: Best trade-off in production systems.
	• Cons: Adds abstraction and hides learning dynamics early.

**✅ Phase 1 Decision: Character-level (ASCII)**
- Why: Matches "from first principles", smallest moving parts, maximal transparency.
- Migration path: Define interface abstraction for future swap to subword/BPE.

⸻

### 1.2 ASCII Policy **[P1-blocker]**

**Question:** What exactly counts as "ASCII" for v1?

**Options:**
- Printable only (32–126)
- Printable + newline + tab
- Full 0–127 including control chars
- Reserve an <UNK> or <CTRL> bucket

**Entails:** Token ID stability, dataset normalization rules, decode fidelity.

**✅ Phase 1 Decision: Printable ASCII (32–126) + newline `\n` + tab `\t`**
- Entails: Deterministic vocab, no hidden control-char bugs, preserves line structure.

⸻

### 1.3 Unknown Character Handling **[P1-blocker]**

**Question:** How do we handle characters outside the ASCII policy?

**✅ Phase 1 Decision: Map anything outside policy → `<UNK>`**
- Entails: Can ingest "dirty" text without crashing; decode won't be perfect but training is stable.
- Vocab definition: `<PAD>=0`, `<UNK>=1`, then the allowed ASCII chars.

⸻

### 1.4 Tokenizer Evolution **[P1-defer]**

**Question:** Can we change tokenization without retraining from scratch?

**Options:**
- Accept retraining as inevitable
- Design a token-remapping or adapter layer
- Train a tokenizer-agnostic latent space (hard)

**This affects:** Long-term experimental continuity.

**Status:** Defer until after baseline works.

⸻

## 2. Data & Corpus Design

### 2.1 Dataset Choice **[P1-defer]**

**Question:** What text domains should we prioritise initially?

**Options:**
- Homogeneous (e.g. books, Wikipedia-like prose)
- Mixed small corpora
- Synthetic / toy grammars

**Trade-off:** Clean learning signals vs representational richness.

**Status:** Defer - any small corpus works for baseline validation.

⸻

### 2.2 Dataset Size **[P1-defer]**

**Question:** How small is "small" for v1?

**Options:**
- Tens of thousands of characters
- Millions of characters
- Incremental curriculum growth

**Entails:** Training stability, overfitting visibility, iteration speed.

**Status:** Defer - pick any size that shows learning curves.

⸻

### 2.3 Train/Val Split **[P1-blocker]**

**Question:** How do we split training and validation data?

**✅ Phase 1 Decision: Contiguous split (e.g., first 95% train, last 5% val)**
- Why: Avoids leakage from random shuffling on small corpora; simplest.

⸻

### 2.4 Curriculum Learning **[P1-defer]**

**Question:** Do we deliberately stage data complexity?

**Options:**
- Flat dataset
- Curriculum (simple → complex)
- Online data augmentation

**Relevance:** Especially important if you later test non-standard learning rules.

**Status:** Defer - flat dataset for Phase 1 baseline.

⸻

## 3. Model Architecture

### 3.1 Transformer Hyperparameters **[P1-blocker]**

**Question:** What are the baseline "tiny but meaningful" defaults?

**Parameters to decide:**
- Number of layers
- Attention heads
- d_model
- d_ff
- Positional encoding type

**Constraint:** Must be small enough for CPU, large enough to exhibit structure.

**✅ Phase 1 Decision:**
- Architecture: Decoder-only Transformer (already decided)
- n_layers=4, d_model=256, n_heads=4, d_ff=1024, dropout=0.1
- Entails: Small enough to run, large enough to show learning curves.

⸻

### 3.2 Positional Encoding **[P1-blocker]**

**Question:** How do we encode position?

**Options:**
- Sinusoidal
- Learned absolute embeddings
- Relative / rotary (RoPE)

**Trade-off:** Simplicity vs inductive bias vs future extensibility.

**✅ Phase 1 Decision: Learned absolute positional embeddings**
- Why: Easiest to implement/inspect; RoPE can come later.

⸻

### 3.3 Architectural Variants **[P1-defer]**

**Question:** When (if ever) do we deviate from the canonical decoder-only transformer?

**Options:**
- Keep transformer fixed as a control baseline
- Introduce variants as experimental branches
- Modularise attention / MLP blocks

**Risk:** Losing comparability across experiments.

**Status:** Defer - keep fixed for Phase 1 baseline.

⸻

## 4. Learning Dynamics & Backpropagation

(This is your core research surface.)

### 4.1 Backprop Modification Scope **[P1-defer]**

**Question:** At what level do we intervene in learning?

**Options:**
1. Gradient transforms (clipping, noise, projection)
2. Custom autograd functions on selected ops
3. Entirely alternative update rules

**Entails:** Increasing complexity, harder debugging, but maximal insight.

**Status:** Defer - standard backprop for Phase 1 baseline.

⸻

### 4.2 Optimizer Choice **[P1-blocker]**

**Question:** What is the baseline optimizer?

**Options:**
- SGD
- Adam
- AdamW
- Custom experimental optimizers

**✅ Phase 1 Decision: AdamW**
- Parameters: lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95)
- Why: Stable baseline. Will swap later via hooks.

⸻

### 4.3 Learning Rate Strategy **[P1-blocker]**

**Question:** Fixed LR or schedule?

**Options:**
- Constant
- Warmup + decay
- Adaptive / learned schedules

**✅ Phase 1 Decision: Fixed LR (no schedule for v1)**
- Option: Warmup later once baseline works.

⸻

### 4.4 Auxiliary Losses **[P1-defer]**

**Question:** Do we introduce regularizers or constraints early?

**Examples:**
- Activation sparsity
- Norm penalties
- Information bottlenecks

**Risk:** Confounding baseline behaviour too early.

**Status:** Defer - standard cross-entropy loss only for Phase 1.

⸻

### 4.5 Loss Function **[P1-blocker]**

**Question:** What loss function do we use?

**✅ Phase 1 Decision: Standard cross-entropy next-token over all positions**

⸻

## 5. Experiment Hooks & Safety

### 5.1 Hook Design **[P1-blocker]**

**Question:** How do we ensure hooks don't silently invalidate experiments?

**Options:**
- Strict config hashes per run
- Explicit experiment manifests
- Hook whitelisting / gating

**✅ Phase 1 Decision: Minimal viable version**
- Implement registry + toggles, but only ship:
  - Forward hooks (read-only logging first)
  - Update hook (gradient transform slot, identity by default)
- Why: Don't overbuild; prove the plumbing.
- Hook safety: Every run writes run_id, git_commit, config_hash, hook_list
- Entails: Prevents "silent" experimental drift.

⸻

### 5.2 Reproducibility **[P1-blocker]**

**Question:** What guarantees do we enforce?

**Considerations:**
- Random seeds
- Deterministic ops
- Versioned configs

**✅ Phase 1 Decision: Fixed seeds + log full config + hash of config**
- Entails: Makes A/B experiments credible.

⸻

### 5.3 Ablation Discipline **[P1-defer]**

**Question:** How do we enforce clean comparisons?

**Options:**
- One-change-per-run policy
- Automated ablation generation
- Baseline re-runs per experiment

**Status:** Defer - establish baseline first.

⸻

## 6. Training Loop & Systems Concerns

### 6.1 Batch Size Strategy **[P1-blocker]**

**Question:** Fixed or adaptive batch sizing?

**Options:**
- Static batch size
- Memory-aware dynamic batching

**✅ Phase 1 Decision: Fixed batch size B=16 (or 8 if slow), with gradient accumulation as fallback**
- Why: Predictable throughput on CPU.

⸻

### 6.2 Sequence Building & Context Length **[P1-blocker]**

**Question:** How do we build sequences from the corpus?

**✅ Phase 1 Decisions:**
- Context length: 256 (already decided)
- Windowing strategy: Sliding windows with stride 256 (non-overlapping) for v1
- Why: Simplest, fastest on CPU, easy to reason about.
- Option: stride <256 later if you want more samples (but compute rises).

⸻

### 6.3 Checkpointing **[P1-blocker]**

**Question:** How often and how heavy?

**Options:**
- Epoch-based
- Step-based
- Minimal vs full state (optimizer, hooks)

**Status:** TBD - decide during implementation based on iteration speed.

⸻

### 6.4 Device Abstraction **[P1-defer]**

**Question:** How early do we generalise beyond CPU?

**Options:**
- CPU-only until stable
- Early MPS abstraction
- Full device registry from day one

**Status:** Defer - CPU-only for Phase 1.

⸻

## 7. Evaluation & "What Does Good Mean?"

### 7.1 Quantitative Metrics **[P1-blocker]**

**Question:** What are the success criteria?

**Options:**
- Loss / perplexity targets
- Learning speed
- Stability under perturbation

**✅ Phase 1 Decision: Evaluation cadence**
- Evaluate val loss every N steps (e.g., 200) + generate a sample
- Success criteria: "Loss decreases reliably + samples become less random over time"
- Why: Phase 1 is pipeline correctness + baseline learning, not SOTA.

⸻

### 7.2 Qualitative Evaluation **[P1-blocker]**

**Question:** How often do we sample text, and how do we judge it?

**Options:**
- Fixed-interval sampling
- Human inspection only
- Automated heuristics (entropy, repetition)

**✅ Phase 1 Decision:**
- Sampling default: temperature=1.0, top-k disabled for v1 (pure multinomial), fixed prompt seed text
- Fixed-interval sampling during evaluation cadence.

⸻

### 7.3 Comparative Evaluation **[P1-defer]**

**Question:** Do we compare against anything external?

**Options:**
- No (self-contained research)
- Simple baselines
- Other tiny models

**Status:** Defer - self-contained for Phase 1.

⸻

## 8. Tooling, Logging & Introspection

### 8.1 Logging Depth **[P1-defer]**

**Question:** What internal states do we record?

**Options:**
- Loss only
- Activations
- Gradients
- Attention maps

**Trade-off:** Insight vs performance and storage.

**Status:** Defer - start minimal, expand based on needs.

⸻

### 8.2 Visualization **[P1-defer]**

**Question:** How do we inspect behaviour?

**Options:**
- Offline plots
- Live dashboards
- Text-only logs

**Status:** Defer - text logs sufficient for Phase 1.

⸻

### 8.3 Experiment Tracking **[P1-defer]**

**Question:** Minimal or full experiment management?

**Options:**
- Simple filesystem logs
- Lightweight custom tracker
- Third-party tools (likely overkill)

**Status:** Defer - simple filesystem logs for Phase 1.

⸻

## 9. Scope & Evolution

### 9.1 Lower-Level Components **[P3]**

**Question:** Do we ever drop below Python?

**Options:**
- Never (accept performance limits)
- Targeted C++/CUDA extensions later
- Rewrite critical paths only if forced

**Status:** Future consideration.

⸻

### 9.2 Scaling Path **[P2]**

**Question:** What signals justify scaling up?

**Examples:**
- Stable learning curves
- Reproducible experimental effects
- Clear research questions unlocked by scale

**Status:** Post-Phase 1 consideration.

⸻

### 9.3 Research Identity **[P3]**

**Question (meta):** Is this primarily:
- A pedagogical system?
- A learning-dynamics testbed?
- A prototype AGI substrate?

**Status:** Meta question, ongoing reflection.

⸻

---

## Summary: Phase 1 Required Decisions

All **P1-blocker** items above must be locked before coding starts. They define:
- Vocabulary + tensor shapes
- Dataset preparation
- Model config
- Training determinism
- Minimal experiment interface

Everything marked **P1-defer** can be decided after baseline runs.