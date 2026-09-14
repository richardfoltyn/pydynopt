#!/bin/bash
set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

readonly CPU_LIST='0-7'
readonly PYTHON_FILES=(
    '.auto/benchmark.py'
    'src/pydynopt/interpolate/numba/linear.py'
    'src/pydynopt/interpolate/numba/search.py'
)

taskset --cpu-list "${CPU_LIST}" uv run python -m py_compile "${PYTHON_FILES[@]}"
taskset --cpu-list "${CPU_LIST}" uv run python .auto/benchmark.py
