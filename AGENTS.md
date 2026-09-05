# Agent Instructions

This repository uses `uv` for package management and environment isolation.

## Running Commands

All commands (tests, formatting, linting, type-checking) should be executed using `uv` to ensure consistency and correct environment resolution.

### Executing Tests

To run the unit tests in the virtual environment in parallel (using `pytest-xdist`), use:
```bash
uv run pytest -n auto
```

To run a specific test file:
```bash
uv run pytest src/tests/stats/test_patsy.py
```

### Linting and Formatting

The project uses `ruff` for linting and formatting. Although `ruff` may be installed system-wide, run it via `uv run` to ensure consistency:
```bash
uv run ruff check
uv run ruff format
```

### Type Checking

Type checking is performed using `ty`. To run the type checker:
```bash
uv run ty check
```
