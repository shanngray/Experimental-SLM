Project Outline: Building a Small Language Model from Scratch

1. Project Goal

Build a small, locally runnable language model from first principles in order to:
	•	Deeply understand how language models work end-to-end
	•	Experiment with architectural and algorithmic changes
	•	Innovate at both high and low levels of the training and learning process

The emphasis is on learning and experimentation, not on competing with large-scale production models.

⸻

2. Design Philosophy
	•	From scratch, but not from nothing
	•	Do not start from an existing pretrained language model
	•	Avoid re-implementing basic numerical or tensor primitives unnecessarily
	•	Use well-established libraries where appropriate, but fully understand and control how they are used
	•	Transparency over abstraction
	•	Prefer explicit implementations over opaque “magic” abstractions
	•	Ensure each major component (tokenization, model architecture, training loop, backprop, etc.) is inspectable and modifiable
	•	Experimental flexibility
	•	The system must allow low-level experimentation (e.g. modifying learning dynamics, backpropagation behavior, or architectural assumptions)

⸻

3. Programming Language & Core Framework

Chosen Language
	•	Python

Rationale:
	•	Fast iteration and experimentation
	•	Strong ecosystem for ML research
	•	Easy to get AI assistance for development
	•	Widely used for modern language model research

Open consideration:
	•	If performance or experimentation constraints emerge later, selective lower-level components (e.g. CUDA extensions, C++ bindings) could be explored — but this is explicitly out of scope for early phases.

⸻

Chosen ML Framework
	•	PyTorch

Rationale:
	•	Highly flexible and “pythonic”
	•	Dynamic computation graphs make debugging and experimentation easier
	•	Well-suited for research and architectural innovation
	•	Supports custom autograd functions and low-level overrides

Planned Usage Pattern:
	•	Use PyTorch for:
	•	Tensor operations
	•	GPU acceleration (if available)
	•	Baseline autograd and optimization infrastructure
	•	Write custom code where:
	•	Backpropagation behavior is modified
	•	Learning rules deviate from standard gradient descent
	•	Architectural experiments require non-standard behavior

This will be a hybrid approach: leveraging PyTorch’s infrastructure while retaining the ability to replace or override critical mechanisms.

⸻

4. Model Scope & Scalability

Initial Scope
	•	Start with a very small prototype
	•	Minimal parameter count
	•	Minimal vocabulary
	•	Designed to run comfortably on local hardware

Long-Term Intent
	•	Architecture should be scalable by design
	•	Model size can grow incrementally
	•	Scaling limited primarily by:
	•	Local hardware
	•	Affordable rented compute (if used later)

Scalability is a design constraint, even for the smallest prototype.

⸻

5. Tokenization & Input Representation

Initial Approach
	•	Use a very simple character or token set
	•	Likely start with ASCII
	•	Exclude emojis, Unicode complexity, etc.

Open Questions
	•	Tokenization strategy:
	•	Character-level vs subword vs simple word-level
	•	Vocabulary size trade-offs
	•	How difficult it will be to:
	•	Expand the character/token set later
	•	Change tokenization without retraining from scratch

These decisions will be intentionally conservative early on to reduce complexity and isolate learning behavior.

⸻

6. Training Data

Initial Data
	•	Use readily available, simple text datasets
	•	Focus is not on dataset quality or coverage, but on:
	•	Learning dynamics
	•	Model behavior
	•	Experimental control

Open Questions
	•	What domains or styles of text (if any) should be prioritized?
	•	How dataset choice might influence later experiments

⸻

7. Learning & Experimental Goals

Status: Open / To Be Defined

This section is intentionally left open for now.

Future goals may include (but are not yet committed to):
	•	Experimenting with alternative learning rules
	•	Modifying backpropagation mechanics
	•	Exploring architectural biases
	•	Studying scaling behavior from very small models upward

These goals will be refined once a baseline model exists.

⸻

8. Open Questions (Explicitly Tracked)
	1.	Do we ever want or need lower-level (non-Python) components?
	2.	Final tokenization strategy and migration path
	3.	How experimental deviations from standard backprop will be structured
	4.	Evaluation criteria (what does “learning better” mean in this project?)
	5.	Tooling for inspection, logging, and visualization of internal states

⸻
