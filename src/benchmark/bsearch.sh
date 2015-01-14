#!/bin/bash

VENV_ROOT=$HOME/virtualenv/hpc
PDO_ROOT=$HOME/repos/pydynopt/src

. $VENV_ROOT/bin/activate

GRID_MAX=10
GRID_MIN=1
GRID_N=100
NEEDLE=34.5

SETUP="
import numpy as np
from pydynopt.utils import _bsearch
stack = np.linspace(${GRID_MIN}, ${GRID_MAX}, ${GRID_N})
needle = ${NEEDLE}
"

STMT="_bsearch(stack, needle)"

python3 -m timeit -s "${SETUP}" "${STMT}"