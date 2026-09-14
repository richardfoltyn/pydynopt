#!/bin/bash
set -euo pipefail

run_quiet() {
    local output
    output="$(mktemp)"
    if ! "$@" >"${output}" 2>&1; then
        tail -80 "${output}"
        rm -f "${output}"
        return 1
    fi
    rm -f "${output}"
}

readonly PYTHON_FILES=(
    '.auto/benchmark.py'
    'src/pydynopt/interpolate/numba/linear.py'
    'src/pydynopt/interpolate/numba/search.py'
)

run_quiet uv run pytest -q -n auto src/tests/interpolation
run_quiet uv run ruff check --no-fix "${PYTHON_FILES[@]}"
run_quiet uv run ruff format --check "${PYTHON_FILES[@]}"
run_quiet uv run ty check "${PYTHON_FILES[@]}"
