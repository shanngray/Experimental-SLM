# Design: CLI-Based Inference Interface

## Context

The current system has all necessary components for text generation (model loading, tokenization, sampling) but lacks a user-facing inference interface. Users currently must either:
1. Run a full training session to see text generation (via sampling during training)
2. Write custom scripts to load models and generate text

This change adds a CLI inference interface that makes the model immediately usable for text generation tasks.

## Goals / Non-Goals

### Goals
- Provide simple CLI interface for text generation from trained models
- Support both checkpoint files and registry models as sources
- Enable interactive and single-shot inference modes
- Allow configuration of sampling parameters (temperature, length, seed)
- Design interface to support future extensions (API formats, services)

### Non-Goals (for this change)
- OpenAI API format output (future work)
- BAML integration (future work)
- Standalone inference service/server (future work)
- Batch inference processing (future work)
- Streaming output (future work)
- Multi-model serving (future work)

## Decisions

### Decision 1: CLI Subcommand Structure

**Choice**: Add `inference` as a new subcommand to `main.py`

**Rationale**:
- Consistent with existing subcommands (import-model, list-models, etc.)
- Natural separation from training mode
- Allows distinct argument sets and help text
- Easy to discover via `python main.py --help`

**Alternative considered**: Separate `inference.py` script
- Rejected: Would fragment the CLI interface and duplicate model loading logic

### Decision 2: Model Source Specification

**Choice**: Support two mutually exclusive model sources:
- `--checkpoint <path>` - Load from checkpoint directory
- `--model-name <name>` - Load from model registry

**Rationale**:
- Checkpoints are common during development/experimentation
- Registry models are cleaner for production use
- Mutually exclusive to avoid ambiguity
- Reuses existing `load_model_adapter()` function

**Alternative considered**: Auto-detect based on path format
- Rejected: Ambiguous, harder to understand, error-prone

### Decision 3: Interactive vs Single-Shot Mode

**Choice**: Support both modes via `--interactive` flag:
- Default (no flag): Single-shot mode - read prompt from `--prompt` arg or stdin, generate once, exit
- With `--interactive`: Continuous loop - prompt user repeatedly until quit

**Rationale**:
- Single-shot mode is scriptable and composable with Unix tools
- Interactive mode is convenient for exploration and demos
- Flag makes behavior explicit

**Alternative considered**: Always interactive
- Rejected: Not scriptable, harder to use in pipelines

### Decision 4: Sampling Parameters

**Choice**: Expose sampling parameters as CLI arguments:
- `--temperature` (default: 1.0)
- `--max-length` (default: 100)
- `--seed` (default: None for variety, or specify for reproducibility)

**Rationale**:
- Direct control over generation quality/creativity
- Matches existing `sample_text()` function signature
- Common pattern in ML inference tools

### Decision 5: Output Format (Current)

**Choice**: Plain text output to stdout

**Rationale**:
- Simplest implementation for initial version
- Unix-friendly (composable with pipes, redirection)
- Matches user expectations for CLI tools

**Future extension**: Output format adapters can be added as flags:
- `--format text` (default)
- `--format openai` (future: JSON in OpenAI API format)
- `--format baml` (future: BAML-compatible output)

## Architecture

### Current Implementation

```
CLI Args → handle_inference() → Load Model → Interactive Loop / Single-Shot
                                                      ↓
                                              sample_text() → stdout
```

### Components
1. **Argument Parser**: New `inference` subparser with all options
2. **handle_inference()**: Main inference orchestration function
3. **Model Loading**: Reuse `load_model_adapter()` from training code
4. **Interactive Loop**: Simple REPL-style prompt loop
5. **Single-Shot**: Read prompt, generate once, exit

### Future Extension Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    CLI / API Gateway Layer                    │
│  (main.py inference | HTTP server | gRPC server | etc.)      │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  Inference Service Layer                      │
│  - Model loading & caching                                    │
│  - Request batching (future)                                  │
│  - Multi-model management (future)                            │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  Output Format Adapters                       │
│  - TextAdapter (current: plain text)                          │
│  - OpenAIAdapter (future: OpenAI API format)                  │
│  - BAMLAdapter (future: BAML format)                          │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  Core Generation Engine                       │
│  - sample_text() and variants                                 │
│  - Streaming support (future)                                 │
└──────────────────────────────────────────────────────────────┘
```

### Extension Points (For Future Work)

1. **Output Format Adapters**: 
   - Interface: `format_output(prompt: str, generated_text: str, metadata: dict) -> str`
   - Adapters convert raw generation output to target format
   - Specified via `--format` flag

2. **Inference Service**:
   - HTTP/gRPC server wrapping inference logic
   - Model management and caching
   - Request queuing and batching
   - Authentication and rate limiting

3. **Streaming Interface**:
   - Token-by-token streaming for real-time output
   - WebSocket or SSE for web clients
   - Requires refactoring `sample_text()` to yield tokens

## Risks / Trade-offs

### Risk: Model Loading Time
- **Issue**: Loading large models from checkpoint can be slow
- **Mitigation**: 
  - Show loading progress indicators
  - Document model loading performance
  - Future: Add model caching/warm start capability

### Risk: Memory Usage
- **Issue**: Holding model in memory for interactive mode
- **Mitigation**:
  - Single-shot mode releases memory immediately
  - Document memory requirements
  - Future: Add model unloading after timeout

### Trade-off: Simplicity vs Features
- **Current**: Simple single-model, single-prompt interface
- **Future**: May need refactoring to support batch inference, streaming, etc.
- **Decision**: Start simple, extend later based on actual needs

## Migration Plan

No migration needed - this is a new capability. Existing training and model management commands are unaffected.

## Open Questions

1. **Q**: Should we support reading prompts from files?
   - **A**: Defer to future work. Can use stdin redirection for now (`python main.py inference --checkpoint ... < prompts.txt`)

2. **Q**: Should we support multiple prompts in single-shot mode?
   - **A**: No, keep single-shot simple. Use interactive mode or scripting for multiple prompts.

3. **Q**: Should sampling parameters be saved in checkpoint and reused?
   - **A**: No, sampling is typically experiment-specific. Users should specify parameters explicitly.

4. **Q**: Should we add prompt templates or few-shot examples?
   - **A**: Defer to future work. Keep initial implementation focused on core generation.

