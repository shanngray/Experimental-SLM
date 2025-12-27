## ADDED Requirements

### Requirement: CLI Inference Interface
The system SHALL provide a CLI inference interface that allows users to run text generation from trained models without starting a full training session.

#### Scenario: Inference subcommand exists
- **WHEN** `python main.py inference --help` is executed
- **THEN** help text is displayed showing inference options
- **AND** available arguments include model source, sampling parameters, and mode flags

#### Scenario: Inference with checkpoint
- **WHEN** `python main.py inference --checkpoint <path> --prompt "Hello"` is executed
- **THEN** model is loaded from the specified checkpoint directory
- **AND** text is generated from the prompt
- **AND** generated text is output to stdout
- **AND** program exits after generation completes

#### Scenario: Inference with registry model
- **WHEN** `python main.py inference --model-name <name> --prompt "Hello"` is executed
- **THEN** model is loaded from the registry by name
- **AND** text is generated from the prompt
- **AND** generated text is output to stdout
- **AND** program exits after generation completes

### Requirement: Inference Modes
The system SHALL support both interactive and single-shot inference modes for different use cases.

#### Scenario: Single-shot mode with prompt argument
- **WHEN** inference is invoked with `--prompt <text>` and no `--interactive` flag
- **THEN** text is generated once from the specified prompt
- **AND** output is written to stdout
- **AND** program exits immediately after generation

#### Scenario: Single-shot mode with stdin
- **WHEN** inference is invoked without `--prompt` and no `--interactive` flag
- **THEN** prompt text is read from stdin
- **AND** text is generated once from the stdin prompt
- **AND** output is written to stdout
- **AND** program exits immediately after generation

#### Scenario: Interactive mode
- **WHEN** inference is invoked with `--interactive` flag
- **THEN** a continuous prompt loop is started
- **AND** user is repeatedly prompted for input
- **AND** each prompt generates text output
- **AND** loop continues until user quits (e.g., "quit", "exit", or Ctrl+D)
- **AND** interactive mode provides friendly prompts and instructions

### Requirement: Inference Sampling Parameters
The system SHALL support configurable sampling parameters for inference to control generation quality and reproducibility.

#### Scenario: Temperature parameter
- **WHEN** inference is invoked with `--temperature <value>`
- **THEN** sampling uses the specified temperature value
- **AND** default temperature is 1.0 if not specified
- **AND** temperature affects generation randomness as expected

#### Scenario: Max length parameter
- **WHEN** inference is invoked with `--max-length <n>`
- **THEN** generation produces at most n tokens
- **AND** default max_length is 100 if not specified

#### Scenario: Seed parameter for reproducibility
- **WHEN** inference is invoked with `--seed <n>`
- **THEN** generation uses the specified random seed
- **AND** identical prompts with same seed produce identical outputs
- **AND** default seed is None (non-deterministic) if not specified

### Requirement: Inference Error Handling
The system SHALL handle common error cases gracefully during inference.

#### Scenario: Missing model source
- **WHEN** inference is invoked without `--checkpoint` or `--model-name`
- **THEN** an error message is displayed indicating model source is required
- **AND** program exits with non-zero exit code

#### Scenario: Invalid checkpoint path
- **WHEN** inference is invoked with `--checkpoint <invalid-path>`
- **THEN** an error message is displayed indicating checkpoint not found
- **AND** program exits with non-zero exit code

#### Scenario: Invalid model name
- **WHEN** inference is invoked with `--model-name <unknown-name>`
- **THEN** an error message is displayed indicating model not found in registry
- **AND** available models are suggested
- **AND** program exits with non-zero exit code

#### Scenario: Mutually exclusive model sources
- **WHEN** inference is invoked with both `--checkpoint` and `--model-name`
- **THEN** an error message is displayed indicating sources are mutually exclusive
- **AND** program exits with non-zero exit code

#### Scenario: Invalid sampling parameters
- **WHEN** inference is invoked with invalid parameter values (e.g., negative temperature)
- **THEN** an error message is displayed indicating parameter constraints
- **AND** program exits with non-zero exit code

### Requirement: Inference Output Format
The system SHALL output generated text in a clear and user-friendly format with extensibility for future output adapters.

#### Scenario: Plain text output
- **WHEN** inference generates text
- **THEN** output is written to stdout as plain text
- **AND** output includes the full generated text (prompt + continuation)
- **AND** output is human-readable and properly formatted

#### Scenario: Interactive mode output formatting
- **WHEN** inference runs in interactive mode
- **THEN** prompts and outputs are clearly distinguished
- **AND** user prompts are labeled (e.g., "Prompt: ")
- **AND** generated text is labeled (e.g., "Generated: ")
- **AND** interface is user-friendly and intuitive

#### Scenario: Future extensibility for output formats
- **WHEN** considering future extensions
- **THEN** design supports adding output format adapters (e.g., `--format openai`)
- **AND** current implementation uses plain text format as default
- **AND** architecture allows adding JSON, OpenAI API, BAML, and other formats later

