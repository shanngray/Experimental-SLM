---
alwaysApply: true
---
# Cursor Rules: Use `uv` for All Python Operations

This project uses **uv** exclusively for Python package management, virtual environment, and execution. **NEVER** run Python commands directly.

## Package Management
- ✅ `uv sync` - Install/sync dependencies
- ✅ `uv add <package>` - Add dependency
- ✅ `uv add --dev <package>` - Add dev dependency
- ✅ `uv remove <package>` - Remove dependency
- ❌ NEVER use `pip install`, `pip uninstall`, or `python -m pip`

## Running Python
- ✅ `uv run python main.py` - Run scripts
- ✅ `uv run python -c "code"` - Run inline code
- ✅ `uv run python -m <module>` - Run modules
- ❌ NEVER use `python` directly

## Testing
- ✅ `uv run pytest` - Run all tests
- ✅ `uv run pytest tests/test_file.py` - Run specific tests
- ❌ NEVER use `pytest` or `python -m pytest` directly

## Virtual Environment
- ✅ `uv sync` automatically manages `.venv/`
- ❌ NEVER manually create venv or use `python -m venv`

## Examples

**CORRECT:**
```bash
uv sync
uv run python main.py
uv run pytest
uv add numpy
```

**WRONG:**
```bash
python main.py
pytest
pip install numpy
```

**Rule: Always prefix Python commands with `uv run`. Always use `uv` for package management.**
