# Adapting Soft Thinking to Experimental SLM

## Overview

**Soft Thinking** is a technique that enables language models to generate "soft" abstract concept tokens in a continuous concept space, rather than selecting discrete tokens at each step. This allows for richer intermediate reasoning representations.

### Core Concept

Instead of selecting a single next token during generation, Soft Thinking:
1. Samples top-k tokens with their probabilities during a "thinking" phase
2. Creates a weighted embedding by combining these top-k token embeddings
3. Feeds this soft embedding back into the model for the next step
4. Switches to discrete token selection when "thinking" is complete

This enables the model to explore multiple reasoning paths simultaneously in a continuous concept space.

## Key Implementation Components

Based on the SGLang Soft Thinking implementation, three main components need modification:

### 1. **Sampler Logic** (`src/sampling/sampler.py`)

**Current State:**
- Simple temperature-based multinomial sampling
- Outputs single discrete token ID per step
- No top-k/top-p filtering
- No entropy tracking

**Required Changes:**
- Add top-k, top-p, and min-p filtering
- Support dual-mode sampling: "soft thinking" vs "after thinking"
- Calculate entropy for early stopping decisions
- Return top-k probabilities and indices (not just single token)
- Track thinking mode state per sequence

**Key Features to Add:**
```python
# Filtering methods
- top_k_renorm_prob(probs, top_k)
- top_p_renorm_prob(probs, top_p)
- min_p filtering based on max_prob threshold

# Entropy calculation
entropy = -sum(probs * log(probs))

# Dual sampling modes
- soft_thinking_mode: Return top-k (prob, index) pairs
- after_thinking_mode: Return single sampled token

# Early stopping logic
- Switch from soft to discrete when entropy < threshold
```

### 2. **Embedding Layer** (`src/model/embeddings.py`)

**Current State:**
- `TokenEmbedding` only supports discrete token_ids input
- Standard `forward(token_ids)` → embeddings

**Required Changes:**
- Add `weighted_forward(topk_probs, topk_indices)` method
- Compute weighted sum: `sum(prob_i * embedding[token_i])`
- Handle probability normalization

**Implementation:**
```python
def weighted_forward(
    self, 
    topk_probs: torch.Tensor,  # [B, K]
    topk_indices: torch.Tensor  # [B, K]
) -> torch.Tensor:
    """
    Compute weighted embedding from top-k tokens.
    
    Returns: [B, d_model] weighted embedding
    """
    # Get embeddings for all top-k tokens: [B, K, d_model]
    topk_embeddings = self.embedding(topk_indices.long())
    
    # Normalize probabilities
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    
    # Weighted sum: [B, d_model]
    weighted_embedding = torch.sum(
        topk_probs.unsqueeze(-1) * topk_embeddings, 
        dim=1
    )
    
    return weighted_embedding
```

### 3. **Transformer Forward Pass** (`src/model/transformer.py`)

**Current State:**
- Always expects discrete `token_ids` as input
- Directly embeds token_ids via `token_embedding(token_ids)`

**Required Changes:**
- Accept optional `topk_probs` and `topk_indices` parameters
- Use weighted embedding when in soft thinking mode
- Fall back to standard embedding for discrete tokens

**Modified Forward Signature:**
```python
def forward(
    self, 
    token_ids: torch.Tensor = None,
    topk_probs: torch.Tensor = None,
    topk_indices: torch.Tensor = None
) -> torch.Tensor:
    """
    Forward pass supporting both discrete and soft embeddings.
    """
    # Soft thinking mode
    if topk_probs is not None and topk_indices is not None:
        token_embeds = self.token_embedding.weighted_forward(
            topk_probs, topk_indices
        )
        # Note: topk_probs/indices are for SINGLE position only
        # Need to handle sequence context separately
    # Standard discrete mode
    else:
        token_embeds = self.token_embedding(token_ids)
    
    # Rest of forward pass unchanged
    ...
```

## Implementation Considerations

### 1. **Inference vs Training**

**Inference-Only Approach (Recommended First):**
- Apply Soft Thinking only during generation/sampling
- No changes to training loop
- Easier to implement and test
- Can use existing trained models

**Training with Soft Thinking:**
- Would require loss computation on soft embeddings
- More complex gradient flow
- Potential benefits for learning better representations
- Requires retraining from scratch or fine-tuning

**Recommendation:** Start with inference-only implementation.

### 2. **Sequence Context Handling**

**Challenge:** Soft Thinking generates soft embeddings for the NEXT token, but the model needs full sequence context.

**Solution Options:**

**Option A: Append soft embedding to sequence**
```python
# Current sequence: [B, seq_len, d_model]
# Soft next embedding: [B, d_model]
# Combined: [B, seq_len+1, d_model]
x = torch.cat([sequence_embeds, soft_next_embed.unsqueeze(1)], dim=1)
```

**Option B: Replace last position (for iterative refinement)**
```python
# Replace last position with soft embedding
x[:, -1, :] = soft_next_embed
```

**Option C: Hybrid approach (used in SGLang)**
- Maintain explicit sequence of discrete tokens
- Only use soft embedding for current generation step
- Track both discrete history and soft current state

### 3. **Mode Switching Logic**

**Early Stopping Criteria:**
- Entropy threshold: Switch when `entropy < threshold`
- Step limit: Maximum number of soft thinking steps
- Confidence threshold: Switch when `max_prob > threshold`

**State Tracking:**
- Need to track per-sequence state: thinking vs discrete mode
- Store accumulated soft embeddings during thinking phase
- Clear state when switching to discrete mode

### 4. **Hyperparameters to Add**

```yaml
soft_thinking:
  # Soft thinking mode
  enabled: true
  max_topk: 10  # Number of top tokens to blend
  
  # During thinking
  thinking_top_k: 50
  thinking_top_p: 0.95
  thinking_min_p: 0.05
  
  # After thinking
  after_thinking_top_k: 10
  after_thinking_top_p: 0.9
  after_thinking_min_p: 0.1
  
  # Early stopping
  entropy_threshold: 2.0
  max_thinking_steps: 5
  
  # Optional noise (not used in paper)
  dirichlet_alpha: 0.0
```

### 5. **Positional Embeddings**

**Issue:** Soft embeddings don't have explicit positions.

**Solutions:**
- Add positional embedding to soft embedding based on sequence position
- Use relative positional encoding (RoPE) which is implicit
- Your model uses learned absolute positional embeddings

**For your implementation:**
```python
# Add positional embedding for the new position
soft_embed = weighted_forward(topk_probs, topk_indices)
pos_embed = self.pos_embedding(current_position)
soft_embed = soft_embed + pos_embed
```

### 6. **Generation Loop Restructuring**

**Current Loop:**
```python
for step in range(max_length):
    logits = model(token_ids)
    next_token = sample(logits[-1])
    token_ids = append(token_ids, next_token)
```

**Soft Thinking Loop:**
```python
for step in range(max_length):
    if soft_mode:
        logits = model(current_context, soft_embedding=prev_soft_embed)
        topk_probs, topk_indices, entropy = soft_sample(logits[-1])
        
        # Check early stopping
        if entropy < threshold:
            soft_mode = False
            next_token = discrete_sample(topk_probs, topk_indices)
            token_ids = append(token_ids, next_token)
        else:
            # Continue soft thinking
            prev_soft_embed = (topk_probs, topk_indices)
    else:
        # Normal discrete sampling
        logits = model(token_ids)
        next_token = sample(logits[-1])
        token_ids = append(token_ids, next_token)
```

## Implementation Plan

### Phase 1: Core Infrastructure (Minimal Changes)

1. **Extend TokenEmbedding**
   - Add `weighted_forward()` method
   - Add unit tests for weighted embedding computation
   - Verify normalization and shape handling

2. **Extend Sampler**
   - Add filtering functions: `top_k_renorm_prob`, `top_p_renorm_prob`, `min_p_filter`
   - Add entropy calculation
   - Add `sample_topk()` function that returns (topk_probs, topk_indices, entropy)
   - Keep existing `sample_text()` for backward compatibility

3. **Add Configuration**
   - Extend `src/config.py` with soft thinking parameters
   - Add config schema validation
   - Add example config: `configs/soft-thinking.yaml`

### Phase 2: Transformer Integration

4. **Modify Transformer.forward()**
   - Add optional `topk_probs` and `topk_indices` parameters
   - Add conditional logic for weighted vs standard embedding
   - Handle positional embedding addition
   - Maintain backward compatibility

5. **Create Soft Thinking Sampler**
   - New function: `sample_text_soft_thinking()`
   - Implement mode switching logic
   - Implement early stopping based on entropy
   - Track thinking steps per generation

### Phase 3: Testing & Validation

6. **Unit Tests**
   - Test weighted embedding computation
   - Test filtering functions
   - Test entropy calculation
   - Test mode switching logic

7. **Integration Tests**
   - Test full generation with soft thinking
   - Compare outputs: discrete vs soft thinking
   - Measure token efficiency (tokens saved via early stopping)
   - Validate generation quality

8. **Evaluation**
   - Run on test prompts
   - Measure reasoning quality (if applicable benchmarks)
   - Analyze entropy patterns
   - Tune hyperparameters

### Phase 4: Optimization & Extensions

9. **Performance Optimization**
   - Profile generation speed
   - Optimize weighted embedding computation
   - Cache frequently used embeddings

10. **Advanced Features**
    - Implement per-token thinking budget
    - Add visualization of soft thinking process
    - Add hooks for analyzing soft embeddings
    - Consider training-time soft thinking (future)

## Differences from SGLang Implementation

### Simplifications for Experimental SLM:

1. **No Tensor Parallelism**: Your model runs on single GPU
   - Skip `weighted_forward_tp()` implementation
   - No vocab sharding logic needed
   - Simpler implementation

2. **No CUDA Graphs**: Focus on correctness first
   - Add later for performance if needed

3. **Simpler Batching**: Single sequence generation
   - No per-sequence mode tracking in batch
   - Easier debugging

4. **No Dirichlet Noise**: Not used in paper
   - Can skip this feature initially

### Additional Considerations for Your Model:

1. **Vocabulary Size**: Your model has smaller vocab
   - Faster softmax computation
   - Less memory for top-k storage

2. **Model Size**: Smaller model (256d vs 1024d+)
   - Faster embedding lookups
   - Lower memory overhead

3. **Sequence Length**: Shorter sequences (256 vs 4096+)
   - Less context to manage
   - Simpler position handling

## Expected Benefits

1. **Token Efficiency**: 10-30% fewer tokens via early stopping
2. **Reasoning Quality**: Smoother exploration of solution space
3. **Controllability**: Adjustable thinking depth via hyperparameters
4. **Analysis**: Entropy patterns reveal model confidence

## Risks & Mitigation

### Risk 1: Degraded Generation Quality
- **Mitigation**: Make soft thinking optional, allow fallback to discrete
- **Testing**: Compare outputs extensively

### Risk 2: Slower Generation Speed
- **Mitigation**: Profile and optimize critical paths
- **Testing**: Measure tokens/second before and after

### Risk 3: Hyperparameter Sensitivity
- **Mitigation**: Provide sensible defaults from SGLang paper
- **Testing**: Grid search on small test set

### Risk 4: Complex Debugging
- **Mitigation**: Add extensive logging and visualization
- **Testing**: Unit test each component independently

## Success Metrics

1. **Functional**: Model generates coherent text with soft thinking enabled
2. **Efficiency**: 10%+ reduction in generated tokens for equivalent quality
3. **Quality**: Generated text quality maintained or improved
4. **Performance**: Generation speed within 2x of discrete sampling
5. **Robustness**: Works across different prompts and domains

## Next Steps

1. Review this plan and identify any concerns or questions
2. Set up development branch for soft thinking implementation
3. Start with Phase 1: Core Infrastructure
4. Implement and test each component incrementally
5. Document findings and adapt plan as needed

## References

- [Soft Thinking Paper](https://arxiv.org/abs/2505.15778)
- [SGLang Soft Thinking Implementation](https://github.com/eric-ai-lab/Soft-Thinking)
- [SGLang Modifications README](https://github.com/eric-ai-lab/Soft-Thinking/blob/main/sglang_soft_thinking_pkg/README.md)
