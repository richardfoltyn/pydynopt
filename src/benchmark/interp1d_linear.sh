#!/bin/bash

VENV_ROOT=$HOME/virtualenv/hpc
PDO_ROOT=$HOME/repos/pydynopt/src

. $VENV_ROOT/bin/activate


SETUP="
import numpy as np
from pydynopt.interpolate import _interp1d_linear_impl
xp = np.linspace(0, 10, 100)
fp = np.exp(xp)/10 + np.sin(xp)
x = 43.567
"

STMT="np.interp(x, xp, fp)"
python3 -m timeit -s "${SETUP}" "${STMT}"


STMT="_interp1d_linear_impl(x, xp, fp)"
python3 -m timeit -s "${SETUP}" "${STMT}"