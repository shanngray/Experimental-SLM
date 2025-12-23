# Change: Add Phase 1 Session 8 - Hooks Infrastructure

## Why
Phase 1 Session 8 implements a minimal hook system for experiments. This infrastructure enables low-level experimentation by allowing hooks to intercept and observe (or modify) training behavior without changing core training code. Forward hooks can log activation statistics (mean/std) for debugging and analysis, while update hooks can transform gradients for experimental learning rules. The hook system is essential for the project's experimental flexibility goal, enabling modifications to learning dynamics, backpropagation behavior, and architectural assumptions without invasive changes to the core training loop. This session provides the foundation for future experimental work and maintains the project's emphasis on transparency and inspectability.

## What Changes
- Add hooks infrastructure with registry system for managing hooks
- Implement forward hooks that can log activation statistics without modifying outputs
- Implement update hooks that can receive and transform gradients
- Hook registry loads hooks from configuration and supports toggling hooks on/off
- Every training run logs: run_id, git_commit, config_hash, hook_list for reproducibility and experiment tracking
- Comprehensive test coverage for hook registration, execution, and safety

## Impact
- Affected specs: New `hooks` capability specification
- Affected code:
  - `src/hooks/__init__.py` (new)
  - `src/hooks/registry.py` (new)
  - `src/hooks/forward_hooks.py` (new)
  - `src/hooks/update_hooks.py` (new)
  - `tests/test_hooks.py` (new)
  - `src/training/trainer.py` (modified - integrate hooks)
  - `src/config.py` (modified - add hook configuration)
- Dependencies: 
  - Session 6 (Training) - provides Trainer infrastructure for hook integration
- Future impact: 
  - Enables experimental modifications to learning dynamics
  - Foundation for alternative learning rules and backpropagation modifications
  - Supports activation analysis and debugging
  - Enables gradient transformation experiments

