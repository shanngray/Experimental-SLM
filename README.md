# Experimental SLM

A small, locally runnable language model built from first principles for learning and experimentation.

## Project Overview

This project aims to build a small language model from scratch to deeply understand how language models work end-to-end. The emphasis is on learning and experimentation, not on competing with large-scale production models.

**Goals:**
- Deeply understand how language models work end-to-end
- Experiment with architectural and algorithmic changes
- Innovate at both high and low levels of the training and learning process

## Setup Instructions

### Prerequisites

- **Python 3.13** or higher
- **uv** - Fast Python package installer and resolver

### Installing uv

If you don't have `uv` installed, you can install it using one of the following methods:

#### macOS/Linux (using curl)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### macOS (using Homebrew)
```bash
brew install uv
```

#### Windows (using PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Using pip
```bash
pip install uv
```

After installation, restart your terminal or run `source ~/.bashrc` (or equivalent) to ensure `uv` is in your PATH.

### Setting Up the Project

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd experimental-slm
   ```

2. **Install dependencies using uv**:
   ```bash
   uv sync
   ```
   
   This command will:
   - Create a virtual environment (if one doesn't exist)
   - Install Python 3.13 (if not already installed)
   - Install all project dependencies (including PyTorch)
   - Make the project available in the environment

3. **Activate the virtual environment** (if needed):
   
   After running `uv sync`, you can activate the environment:
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```
   
   Alternatively, you can run commands directly with `uv run`:
   ```bash
   uv run python main.py
   ```

4. **Verify installation**:
   ```bash
   uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

### Adding New Dependencies

To add a new dependency to the project:

1. **Add it to pyproject.toml**:
   ```bash
   uv add <package-name>
   ```
   
   Or manually edit `pyproject.toml` and add it to the `dependencies` list, then run:
   ```bash
   uv sync
   ```

2. **Remove a dependency**:
   ```bash
   uv remove <package-name>
   ```

### Common uv Commands

- `uv sync` - Install all dependencies and sync the environment
- `uv add <package>` - Add a new dependency
- `uv remove <package>` - Remove a dependency
- `uv run <command>` - Run a command in the project's virtual environment
- `uv pip list` - List installed packages

## Project Structure

*[Placeholder: Detailed project structure explanation will be added here]*

The project follows a modular structure:
- `src/` - Source code modules
- `tests/` - Test files
- `configs/` - YAML configuration files
- `data/` - Dataset storage
- `logs/` - Training logs
- `checkpoints/` - Model checkpoints

## Quick Start

*[Placeholder: Quick start guide will be added here]*

## Development Workflow

*[Placeholder: Development workflow documentation will be added here]*

## Architecture Overview

*[Placeholder: Architecture overview will be added here]*

## Contributing

*[Placeholder: Contributing guidelines will be added here]*

## License

*[Placeholder: License information will be added here]*
