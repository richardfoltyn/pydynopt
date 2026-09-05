# Agent Instructions

This repository uses `uv` for package management and environment isolation. Do not install new packages unless explicitly requested.

## Running Commands

All commands (tests, formatting, linting, type-checking) must be executed using `uv`.
Ruff and ty are pinned project development dependencies. Always invoke them with
`uv run`; do not use system executables or install them with `uv tool install`.

### Testing

Run unit tests in parallel:
```bash
uv run pytest -n auto
```

Run a specific test file:
```bash
uv run pytest src/tests/stats/test_patsy.py
```

### Linting, Formatting, and Type Checking

- Only run Ruff and address `ty` issues on `.py` files modified by the agent.
- `ruff check` does not format code even with `fix = true`. Always run `ruff format`.

**Apply fixes and formatting to changed files in this order:**
```bash
uv run ruff check --fix <changed .py paths>
uv run ruff format <changed .py paths>
```

**Verify changes without modifying:**
```bash
uv run ruff check --no-fix <changed .py paths>
uv run ruff format --check <changed .py paths>
uv run ty check <changed .py paths>
```

---

## Python Guidelines

### Target Version & Style
- Target Python 3.13+. Do not use compatibility shims or Python 3.14+ syntax.
- String quotes: Single quotes (`'...'`), matching `pyproject.toml`.
- Module docstrings: Describe module purpose; format author as `Author: Firstname Lastname`.
- Avoid decorative banner comments (e.g., `# ----------`) at the module level.
- Avoid complex expressions in `return` statements.
- Prefix Pandas DataFrames with `df_` (new code only).
- Prefer short local variable names (e.g., `lb` instead of `lower_bound`).

### Docstrings
- Use NumPy docstring style with reST formatting for inline code (``code``).
- Keep explanations concise; use `-` for bullet points.
- Omit types from docstrings (handled by type annotations) unless describing array shapes/dimensions.
- Omit docstrings for trivial methods (`__repr__`, `__str__`, attribute-only `__init__`) and `@overload` signatures.
- Single return value: omit return variable name in `Returns` section.
- Multiple return values: include return variable names in `Returns` section.

### Code Structure
- Single-purpose functions; avoid monolithic functions.
- Do not create separate 1–2 line helper functions unless reused across multiple call sites.
- Follow EAFP; avoid redundant parameter validation when caller and callee are internally controlled.

### Type Annotations
- Annotate all function arguments and return types.
- Do not add runtime `isinstance` checks for arguments already type-annotated.
- Use modern union syntax: `|` (not `Union`), `| None` (not `Optional`).
- Prefer collection types from `collections.abc` over `typing`.
- Use abstract collections for inputs (e.g., `Sequence`), concrete for return values (e.g., `list`).
- Never use `cast()` or `TYPE_CHECKING`. Fix the underlying design instead.
- Use standard `# type: ignore` (not tool-specific `# ty: ignore[...]`).
- Use `from __future__ import annotations` only when postponed evaluation of annotations is required.
