#!/bin/bash

VENV_ROOT=$HOME/virtualenv/hpc
PDO_ROOT=$HOME/repos/pydynopt/src

. $VENV_ROOT/bin/activate

SETUP="
import numpy as np
from cython import double
from pydynopt.utils import _bsearch
stack = np.linspace(0, 10, 100)
needle = 34.32
"

STMT="_bsearch(stack, needle)"

python3 -m timeit -s "${SETUP}" "${STMT}"