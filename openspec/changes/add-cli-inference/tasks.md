# Implementation Tasks

## 1. Core Implementation
- [ ] 1.1 Add `inference` subcommand parser to `main.py` argument parser
- [ ] 1.2 Implement `handle_inference()` function to orchestrate inference flow
- [ ] 1.3 Add checkpoint/model loading logic for inference mode
- [ ] 1.4 Implement interactive mode (continuous prompt loop)
- [ ] 1.5 Implement single-shot mode (one prompt, exit)
- [ ] 1.6 Add sampling parameter arguments (--temperature, --max-length, --seed)
- [ ] 1.7 Add model source arguments (--checkpoint or --model-name)

## 2. Testing
- [ ] 2.1 Add integration test for inference subcommand
- [ ] 2.2 Test inference from checkpoint
- [ ] 2.3 Test inference from model registry
- [ ] 2.4 Test interactive mode
- [ ] 2.5 Test single-shot mode
- [ ] 2.6 Test sampling parameter variations
- [ ] 2.7 Test error handling (missing model, invalid params)

## 3. Documentation
- [ ] 3.1 Update README.md with inference usage examples
- [ ] 3.2 Add inference command to help/usage documentation
- [ ] 3.3 Document sampling parameters and their effects
- [ ] 3.4 Add examples for common inference workflows

## 4. Future Extensibility (Design Only)
- [ ] 4.1 Document extension points for output format adapters
- [ ] 4.2 Document architecture for future inference service
- [ ] 4.3 Add notes on OpenAI API format compatibility
- [ ] 4.4 Add notes on BAML integration approach

